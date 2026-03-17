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

import torch
from collections.abc import Iterator
from typing import Optional

from vllm.config import VllmConfig
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from llmd_s3_backend.manager import S3OffloadingManager
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.attention.backend import AttentionBackend

from llmd_s3_backend.worker import (
    GPUS3OffloadingHandler,
    S3GPUOffloadingHandler,
    DEFAULT_MAX_STAGING_MEMORY_GB,
    DEFAULT_MAX_THREADS_PER_GPU,
)

from vllm.v1.kv_offload.worker.worker import OffloadingHandler
from llmd_s3_backend.mediums import S3LoadStoreSpec


class S3OffloadingSpec(OffloadingSpec):
    """
    OffloadingSpec for S3 backend.
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config=None):
        """
        Initialize S3 offloading spec from vLLM configuration.

        Configuration is read from ``vllm_config.kv_transfer_config
        .kv_connector_extra_config``. The following keys are supported:

        Args:
            vllm_config: Top-level vLLM configuration object.
            kv_cache_config: Optional KV cache configuration override.

        Extra-config keys (via ``kv_connector_extra_config``):
            s3_bucket (str): **Required.** S3 bucket name.
            s3_prefix (str): S3 key prefix for KV objects.
                Default: ``"kv-cache"``.
            s3_region (str | None): AWS region.
                Default: from environment or ``"us-east-1"``.
            s3_endpoint_url (str | None): Custom endpoint URL for
                S3-compatible services (e.g. Ceph).
            s3_addressing_style (str): S3 addressing style.
                One of ``"auto"``, ``"path"``, ``"virtual"``.
                Default: ``"auto"``.
            s3_profile_name (str | None): AWS named profile from
                credentials file.
            threads_per_gpu (int): Worker threads per GPU for async
                transfers. Capped at 64. Default: ``64``.
            max_staging_memory_gb (int): Maximum pinned staging memory
                in GB. Default: ``150``.
            enable_presence_cache (bool): Enable LRU presence cache
                backed by a manifest for cross-instance sync.
                Default: ``False``.
            manifest_prefix (str): S3 prefix for manifest files.
                Default: ``"manifests"``.
            compaction_threshold (int): Number of delta files that
                triggers manifest compaction. Default: ``100``.
            compaction_interval_hours (int): Minimum hours between
                automatic compactions. Default: ``24``.
            cache_max_size (int | None): Maximum entries in the LRU
                presence cache. ``None`` for unbounded.
                Default: ``1_000_000``.
            io_driver (str): I/O driver for S3 transfers.
                One of ``"auto"``, ``"crt"``, ``"io_uring"``,
                ``"cuobject"``. Default: ``"auto"``.
            iouring_queue_depth (int): io_uring submission queue depth.
                Default: ``1024``.
            iouring_num_workers (int): io_uring worker thread count.
                Default: ``16``.
            pinned_buffer_size_mb (int): Size of each pinned buffer
                in MB. Default: ``128``.
            pinned_buffer_pool_size (int): Number of pinned buffers
                in the pool. Default: ``64``.

        Raises:
            ValueError: If ``s3_bucket`` is missing or ``io_driver``
                is not a recognized value.
        """
        super().__init__(vllm_config, kv_cache_config)

        self._num_blocks: Optional[int] = None
        self._manager: Optional[OffloadingManager] = None

        # S3-specific configuration
        self.s3_bucket = self.extra_config.get("s3_bucket")
        if not self.s3_bucket:
            raise ValueError("s3_bucket is required in extra_config")

        self.s3_prefix = self.extra_config.get("s3_prefix", "kv-cache")
        self.s3_region = self.extra_config.get("s3_region")
        self.s3_endpoint_url = self.extra_config.get("s3_endpoint_url")
        self.s3_addressing_style = self.extra_config.get(
            "s3_addressing_style", "auto"
        )
        self.s3_profile_name = self.extra_config.get("s3_profile_name")

        # General configuration
        self.threads_per_gpu = int(
            self.extra_config.get("threads_per_gpu", DEFAULT_MAX_THREADS_PER_GPU)
        )
        self.max_staging_memory_gb = self.extra_config.get(
            "max_staging_memory_gb", DEFAULT_MAX_STAGING_MEMORY_GB
        )
        
        # Presence cache configuration (optional)
        self.enable_presence_cache = self.extra_config.get("enable_presence_cache", False)
        self.manifest_prefix = self.extra_config.get("manifest_prefix", "manifests")
        self.compaction_threshold = int(self.extra_config.get("compaction_threshold", 100))
        self.compaction_interval_hours = int(self.extra_config.get("compaction_interval_hours", 24))
        
        # LRU cache configuration
        cache_max_size = self.extra_config.get("cache_max_size", 1_000_000)
        self.cache_max_size = None if cache_max_size is None else int(cache_max_size)
        
        # I/O driver selection
        self.io_driver = self.extra_config.get("io_driver", "auto")
        if self.io_driver not in ("auto", "crt", "io_uring", "cuobject"):
            raise ValueError(
                f"Invalid io_driver '{self.io_driver}'. "
                f"Must be one of: auto, crt, io_uring, cuobject"
            )
        
        # io_uring configuration (used when io_driver is 'io_uring' or auto-selected)
        self.iouring_queue_depth = int(self.extra_config.get("iouring_queue_depth", 1024))
        self.iouring_num_workers = int(self.extra_config.get("iouring_num_workers", 16))
        self.pinned_buffer_size_mb = int(self.extra_config.get("pinned_buffer_size_mb", 128))
        self.pinned_buffer_pool_size = int(self.extra_config.get("pinned_buffer_pool_size", 64))
        
        # Resolve io_driver if set to 'auto'
        self.resolved_io_driver = self._select_io_driver()

        self.gpu_blocks_per_file = int(
            self.offloaded_block_size / self.gpu_block_size
        )
        assert (
            self.offloaded_block_size % self.gpu_block_size == 0
        ), "offloaded_block_size must be a multiple of gpu_block_size"

        self._gpu_to_s3: Optional[OffloadingHandler] = None
        self._s3_to_gpu: Optional[OffloadingHandler] = None
    def _select_io_driver(self) -> str:
        """
        Select the I/O driver based on the io_driver configuration.
        
        Auto-selection logic:
        1. cuobject: If available (future implementation)
        2. io_uring: If kernel supports it (Linux 5.1+)
        3. crt: Fallback (always available)
        
        Returns:
            str: The selected driver name ('crt', 'io_uring', or 'cuobject')
        """
        if self.io_driver != "auto":
            # User explicitly specified a driver
            if self.io_driver == "cuobject":
                raise NotImplementedError(
                    "cuobject driver is not yet implemented. "
                    "This driver will enable GPU-direct storage with RDMA, bypassing CPU for data transfers. "
                    "For now, please use 'io_uring' (Linux only, experimental zero-copy) or 'crt' (default, stable). "
                    "Set io_driver='auto' to automatically select the best available driver."
                )
            return self.io_driver
        
        # Auto-selection logic
        # TODO: Check for cuobject support when implemented
        
        # Check for io_uring support
        try:
            import platform
            if platform.system() == "Linux":
                # Try to import io_uring modules to check availability
                from llmd_s3_backend.iouring_ops import IoUringContext
                
                # Create a temporary context to check kernel support
                try:
                    ctx = IoUringContext(queue_depth=2)
                    ctx.close()
                    return "io_uring"
                except Exception:
                    # io_uring not available, fall back to crt
                    pass
        except ImportError:
            # io_uring modules not available
            pass
        
        # Fallback to CRT (always available)
        return "crt"


    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            self._manager = S3OffloadingManager(
                model_name=self.vllm_config.model_config.model,
                tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
                tp_rank=self.vllm_config.parallel_config.rank,
                dtype=self.vllm_config.cache_config.cache_dtype,
                bucket=self.s3_bucket,
                prefix=self.s3_prefix,
                region=self.s3_region,
                endpoint_url=self.s3_endpoint_url,
                addressing_style=self.s3_addressing_style,
                profile_name=self.s3_profile_name,
                enable_presence_cache=self.enable_presence_cache,
                manifest_prefix=self.manifest_prefix,
                compaction_threshold=self.compaction_threshold,
                compaction_interval_hours=self.compaction_interval_hours,
                cache_max_size=self.cache_max_size,
            )
        return self._manager

    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:

        if not self._gpu_to_s3 or not self._s3_to_gpu:
            self._gpu_to_s3 = GPUS3OffloadingHandler(
                model_name=self.vllm_config.model_config.model,
                tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
                tp_rank=self.vllm_config.parallel_config.rank,
                kv_caches=kv_caches,
                gpu_blocks_per_file=self.gpu_blocks_per_file,
                dtype=self.vllm_config.cache_config.cache_dtype,
                threads_per_gpu=self.threads_per_gpu,
                max_staging_memory_gb=self.max_staging_memory_gb,
                bucket=self.s3_bucket,
                prefix=self.s3_prefix,
                region=self.s3_region,
                endpoint_url=self.s3_endpoint_url,
                addressing_style=self.s3_addressing_style,
                profile_name=self.s3_profile_name,
                attn_backends=attn_backends,
            )

            self._s3_to_gpu = S3GPUOffloadingHandler(
                model_name=self.vllm_config.model_config.model,
                tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
                tp_rank=self.vllm_config.parallel_config.rank,
                dtype=self.vllm_config.cache_config.cache_dtype,
                gpu_blocks_per_file=self.gpu_blocks_per_file,
                kv_caches=kv_caches,
                threads_per_gpu=self.threads_per_gpu,
                max_staging_memory_gb=self.max_staging_memory_gb,
                bucket=self.s3_bucket,
                prefix=self.s3_prefix,
                region=self.s3_region,
                endpoint_url=self.s3_endpoint_url,
                addressing_style=self.s3_addressing_style,
                profile_name=self.s3_profile_name,
                attn_backends=attn_backends,
                # I/O driver configuration
                io_driver=self.resolved_io_driver,
                iouring_queue_depth=self.iouring_queue_depth,
                iouring_num_workers=self.iouring_num_workers,
                pinned_buffer_size_mb=self.pinned_buffer_size_mb,
                pinned_buffer_pool_size=self.pinned_buffer_pool_size,
            )

        yield GPULoadStoreSpec, S3LoadStoreSpec, self._gpu_to_s3
        yield S3LoadStoreSpec, GPULoadStoreSpec, self._s3_to_gpu

# Made with Bob
