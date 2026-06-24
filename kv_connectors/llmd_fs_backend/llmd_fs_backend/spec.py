# Copyright 2025 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterator

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingManager,
    OffloadingSpec,
)
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

from llmd_fs_backend import _logger as logger
from llmd_fs_backend.file_mapper import FileMapper
from llmd_fs_backend.manager import SharedStorageOffloadingManager
from llmd_fs_backend.mediums import SharedStorageLoadStoreSpec
from llmd_fs_backend.worker import (
    DEFAULT_MAX_STAGING_MEMORY_GB,
    DEFAULT_MAX_WRITE_QUEUED_SECONDS,
    DEFAULT_READ_PREFERRING_WORKERS_RATIO,
    DEFAULT_THREADS_PER_GPU,
    StorageOffloadingHandlers,
)

DEFAULT_STORAGE_BLOCK_SIZE = 256

# Backends served by the NIXL engine (vs the C++ POSIX/GDS engine).
NIXL_BACKENDS = ("OBJ", "MEMOS")


class SharedStorageOffloadingSpec(OffloadingSpec):
    """
    OffloadingSpec for shared storage backend (e.g., mounted NFS, PVC).
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        # Hide "block_size" from the base class to bypass the uniformity
        # assertion on hybrid models (we derive the factor ourselves below).
        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        extra_config = kv_transfer_config.kv_connector_extra_config
        hidden_block_size = extra_config.pop("block_size", None)
        try:
            super().__init__(vllm_config, kv_cache_config)
        finally:
            if hidden_block_size is not None:
                extra_config["block_size"] = hidden_block_size

        self._manager: OffloadingManager | None = None
        # worker-side
        self._handlers: StorageOffloadingHandlers | None = None

        self.threads_per_gpu = int(
            self.extra_config.get("threads_per_gpu", DEFAULT_THREADS_PER_GPU)
        )
        shared_storage_path = self.extra_config.get(
            "shared_storage_path", "/tmp/shared-kv"
        )
        self.max_staging_memory_gb = int(
            self.extra_config.get(
                "max_staging_memory_gb", DEFAULT_MAX_STAGING_MEMORY_GB
            )
        )  # Max staging CPU buffer in GB

        self.offloaded_block_size = int(
            self.extra_config.get("block_size", DEFAULT_STORAGE_BLOCK_SIZE)
        )

        # hash_block_size = GCD of all groups' block sizes (the granularity at
        # which Request.block_hashes are computed); use it instead of
        # cache_config.block_size which can be larger on hybrid models (e.g. DSv4).
        assert self.offloaded_block_size % self.hash_block_size == 0, (
            "offloaded_block_size must be a multiple of hash_block_size"
        )
        self.gpu_blocks_per_file = self.offloaded_block_size // self.hash_block_size

        # DOCA MEMOS stores each object as a single NVMe KV value; the controller
        # caps the value size per op. Shrink gpu_blocks_per_file (tokens/block) so
        # a packed object fits. Runs on every rank before the block layout is
        # baked in below, so scheduler and workers stay in agreement.
        if self.extra_config.get("backend") == "MEMOS":
            self._scale_block_size_for_memos(kv_cache_config)

        # Derive block_size_factor from file layout instead of base class.
        self.block_size_factor = self.gpu_blocks_per_file

        self.read_preferring_ratio = float(
            self.extra_config.get(
                "read_preferring_ratio", DEFAULT_READ_PREFERRING_WORKERS_RATIO
            )
        )
        self.max_write_queued_seconds = float(
            self.extra_config.get(
                "max_write_queued_seconds", DEFAULT_MAX_WRITE_QUEUED_SECONDS
            )
        )

        parallel_config = vllm_config.parallel_config
        tp_size = parallel_config.tensor_parallel_size
        pp_size = parallel_config.pipeline_parallel_size
        pcp_size = parallel_config.prefill_context_parallel_size
        assert parallel_config.world_size == tp_size * pp_size * pcp_size

        self.file_mapper = FileMapper.from_vllm_config(
            root_dir=shared_storage_path,
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            gpu_blocks_per_file=self.gpu_blocks_per_file,
        )
        self.file_mapper.write_run_config()

    def _memos_object_bytes_per_block(self, kv_cache_config: KVCacheConfig) -> int:
        """Bytes one offloaded GPU block (hash_block_size tokens) occupies across
        all layers — i.e. the per-block size of a packed DOCA MEMOS object value.

        ``KVCacheSpec.page_size_bytes`` is per layer for ``spec.block_size``
        tokens; scale to one hash_block_size-token block and sum over each group's
        layers. Mirrors the worker's ``sum(ref.page_size_bytes ...)`` sizing.
        """
        total = 0
        for group in kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            per_layer = spec.page_size_bytes * self.hash_block_size // spec.block_size
            total += per_layer * len(group.layer_names)
        return total

    def _scale_block_size_for_memos(self, kv_cache_config: KVCacheConfig) -> None:
        """Shrink ``gpu_blocks_per_file`` so a packed object fits the device's
        advertised max value size. Warns and leaves the configured block size
        unchanged if the limit or the per-block size can't be determined."""
        from llmd_nixl.memos_backend import query_max_value_size

        max_value_size = query_max_value_size(self.extra_config)
        if not max_value_size:
            logger.warning(
                "DOCA MEMOS: max value size unavailable; keeping configured "
                "offloaded block_size=%d tokens (%d GPU blocks/object)",
                self.offloaded_block_size,
                self.gpu_blocks_per_file,
            )
            return

        try:
            per_block_bytes = self._memos_object_bytes_per_block(kv_cache_config)
        except Exception:
            logger.warning(
                "DOCA MEMOS: failed to compute per-block bytes; keeping "
                "configured block size",
                exc_info=True,
            )
            return
        if per_block_bytes <= 0:
            logger.warning(
                "DOCA MEMOS: per-block bytes computed as %d; keeping configured "
                "block size",
                per_block_bytes,
            )
            return

        max_blocks = max_value_size // per_block_bytes
        if max_blocks < 1:
            logger.warning(
                "DOCA MEMOS: a single GPU block (%d bytes) exceeds the device max "
                "value size (%d bytes); capping at 1 block/object — transfers may "
                "fail until block_size or the device limit changes",
                per_block_bytes,
                max_value_size,
            )
            max_blocks = 1

        if self.gpu_blocks_per_file <= max_blocks:
            logger.info(
                "DOCA MEMOS: offloaded block fits device max value size "
                "(%d GPU blocks/object x %d bytes <= %d bytes)",
                self.gpu_blocks_per_file,
                per_block_bytes,
                max_value_size,
            )
            return

        old_blocks = self.gpu_blocks_per_file
        self.gpu_blocks_per_file = max_blocks
        self.offloaded_block_size = self.gpu_blocks_per_file * self.hash_block_size
        logger.warning(
            "DOCA MEMOS: scaling offloaded block size to fit device max value "
            "size (%d bytes, %d bytes/block): gpu_blocks_per_file %d -> %d, "
            "offloaded block_size %d -> %d tokens",
            max_value_size,
            per_block_bytes,
            old_blocks,
            self.gpu_blocks_per_file,
            old_blocks * self.hash_block_size,
            self.offloaded_block_size,
        )

    def get_manager(self) -> OffloadingManager:
        assert self.vllm_config.parallel_config.rank == 0, "Scheduler rank should be 0"
        if not self._manager:
            backend = self.extra_config.get("backend", "POSIX")
            if backend in NIXL_BACKENDS:
                from llmd_nixl.manager import NixlStorageOffloadingManager

                self.extra_config.setdefault("storage_medium", "OBJECT_STORE")
                self._manager = NixlStorageOffloadingManager(
                    file_mapper=self.file_mapper,
                    extra_config=self.extra_config,
                )
            else:
                self.extra_config.setdefault("storage_medium", "SHARED_STORAGE")
                self._manager = SharedStorageOffloadingManager(
                    file_mapper=self.file_mapper,
                    extra_config=self.extra_config,
                )
        return self._manager

    def get_handlers(
        self,
        kv_caches: CanonicalKVCaches,
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if not self._handlers:
            backend = self.extra_config.get("backend", "POSIX")
            if backend in NIXL_BACKENDS:
                from llmd_nixl.worker import NixlStorageOffloadingHandlers

                handlers_cls = NixlStorageOffloadingHandlers
            else:
                handlers_cls = StorageOffloadingHandlers
            self._handlers = handlers_cls(
                file_mapper=self.file_mapper,
                gpu_blocks_per_file=self.gpu_blocks_per_file,
                gpu_block_size=self.hash_block_size,
                kv_caches=kv_caches,
                threads_per_gpu=self.threads_per_gpu,
                max_staging_memory_gb=self.max_staging_memory_gb,
                read_preferring_ratio=self.read_preferring_ratio,
                max_write_queued_seconds=self.max_write_queued_seconds,
                extra_config=self.extra_config,
            )

        assert self._handlers is not None
        yield (
            GPULoadStoreSpec,
            SharedStorageLoadStoreSpec,
            self._handlers.gpu_to_storage_handler,
        )
        yield (
            SharedStorageLoadStoreSpec,
            GPULoadStoreSpec,
            self._handlers.storage_to_gpu_handler,
        )
