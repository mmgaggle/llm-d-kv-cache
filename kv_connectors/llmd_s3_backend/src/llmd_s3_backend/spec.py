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
from vllm.attention.backends.abstract import AttentionBackend

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

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)

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

        self.gpu_blocks_per_file = int(
            self.offloaded_block_size / self.gpu_block_size
        )
        assert (
            self.offloaded_block_size % self.gpu_block_size == 0
        ), "offloaded_block_size must be a multiple of gpu_block_size"

        self._gpu_to_s3: Optional[OffloadingHandler] = None
        self._s3_to_gpu: Optional[OffloadingHandler] = None

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
            )

        yield GPULoadStoreSpec, S3LoadStoreSpec, self._gpu_to_s3
        yield S3LoadStoreSpec, GPULoadStoreSpec, self._s3_to_gpu

# Made with Bob
