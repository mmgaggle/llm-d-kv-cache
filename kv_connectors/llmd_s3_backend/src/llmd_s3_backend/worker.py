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

import io
import math
import torch
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, Dict, List, Tuple
from vllm.attention.backends.abstract import AttentionBackend
from vllm.logger import init_logger
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferSpec,
    TransferResult,
)
from llmd_s3_backend.s3_client import S3ClientWrapper

logger = init_logger(__name__)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
DEFAULT_MAX_STAGING_MEMORY_GB = 150
DEFAULT_MAX_THREADS_PER_GPU = 64


class S3OffloadingHandler(OffloadingHandler):
    """Base handler with common helpers for S3 offloading."""

    def __init__(
        self,
        model_name: str,
        tp_size: int,
        tp_rank: int,
        dtype: torch.dtype,
        gpu_blocks_per_file: int,
        threads_per_gpu: int,
        attn_backends: dict[str, type[AttentionBackend]],
        bucket: str,
        prefix: str = "kv-cache",
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        addressing_style: str = "auto",
        profile_name: Optional[str] = None,
        max_staging_memory_gb: int = DEFAULT_MAX_STAGING_MEMORY_GB,
    ):
        self.model_name = model_name
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.dtype = dtype
        self.gpu_blocks_per_file = gpu_blocks_per_file
        self.threads_per_gpu = min(
            threads_per_gpu, DEFAULT_MAX_THREADS_PER_GPU
        )
        self.max_staging_memory_gb = max_staging_memory_gb
        self.attn_backends = attn_backends

        # Initialize S3 client
        self.s3_client = S3ClientWrapper(
            bucket=bucket,
            region=region,
            endpoint_url=endpoint_url,
            addressing_style=addressing_style,
            profile_name=profile_name,
        )

        # Build base S3 key
        dtype_str = str(dtype).replace("torch.", "")
        self.base_key = (
            f"{prefix}/{model_name}/tp_{tp_size}/rank_{tp_rank}/{dtype_str}"
        )

        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=self.threads_per_gpu)
        self.pending_futures: Dict[int, Future] = {}
        self.completed_jobs: List[TransferResult] = []
        self.lock = threading.Lock()

        # CUDA streams for async GPU operations
        self.h2d_stream = torch.cuda.Stream()
        self.d2h_stream = torch.cuda.Stream()

    def _get_s3_key(self, block_hash: int) -> str:
        """Generate S3 key for a given block hash."""
        if isinstance(block_hash, bytes):
            block_hash = int.from_bytes(block_hash, "little")
        block_hash_hex = f"{block_hash & ((1 << 64) - 1):016x}"
        subfolder1, subfolder2 = block_hash_hex[:3], block_hash_hex[3:5]
        return f"{self.base_key}/{subfolder1}/{subfolder2}/{block_hash_hex}.bin"

    def compute_buffer_size_mb(
        self,
        tensors,
        gpu_blocks_per_file,
        layers_before_num_blocks,
        num_blocks_idx,
        safety=1.0,
        min_mb=32,
        max_mb=None,
    ):
        """Estimate staging memory size in MB."""
        ref = tensors[0]
        per_block = ref.index_select(
            num_blocks_idx, torch.tensor([0], device=ref.device)
        ).squeeze(num_blocks_idx)

        per_block_elems = per_block.numel()
        block_elems = per_block_elems * gpu_blocks_per_file
        total_elems = (
            block_elems * len(tensors) if layers_before_num_blocks else block_elems
        )
        total_bytes = total_elems * ref.element_size()
        mb = math.ceil(total_bytes / (1024 * 1024) * safety)
        if min_mb:
            mb = max(mb, min_mb)
        if max_mb:
            mb = min(mb, max_mb)
        return mb

    def get_finished(self) -> list[TransferResult]:
        """Poll finished async transfers."""
        with self.lock:
            finished = self.completed_jobs.copy()
            self.completed_jobs.clear()
            return finished

    def get_kv_cache_parameters(self, gpu_caches: dict[str, torch.Tensor]):
        """Determine KV cache layout parameters."""
        list_num_blocks_idx = []
        list_kv_before_num_blocks = []
        list_layers_before_num_blocks = []

        for layer_name, gpu_tensor in gpu_caches.items():
            gpu_shape = gpu_tensor.shape
            attn_backend = self.attn_backends[layer_name]

            test_shape = attn_backend.get_kv_cache_shape(
                num_blocks=1234, block_size=16, num_kv_heads=8, head_size=256
            )

            if len(gpu_shape) != len(test_shape):
                assert len(gpu_shape) == len(test_shape) + 1
                num_blocks_idx = 0
                kv_before_num_blocks = False
                layers_before_num_blocks = False
            elif test_shape[0] == 1234:
                num_blocks_idx = 0
                kv_before_num_blocks = False
                layers_before_num_blocks = True
            else:
                assert test_shape[0] == 2
                assert test_shape[1] == 1234
                assert gpu_shape[0] == 2
                num_blocks_idx = 1
                kv_before_num_blocks = True
                layers_before_num_blocks = True

            list_num_blocks_idx.append(num_blocks_idx)
            list_kv_before_num_blocks.append(kv_before_num_blocks)
            list_layers_before_num_blocks.append(layers_before_num_blocks)

        return (
            list_num_blocks_idx,
            list_kv_before_num_blocks,
            list_layers_before_num_blocks,
        )

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)


# ----------------------------------------------------------------------
# GPU → S3 (PUT)
# ----------------------------------------------------------------------
class GPUS3OffloadingHandler(S3OffloadingHandler):
    """Handler for writing KV blocks from GPU tensors to S3."""

    def __init__(
        self,
        model_name: str,
        tp_size: int,
        tp_rank: int,
        kv_caches: Dict[str, torch.Tensor],
        gpu_blocks_per_file: int,
        attn_backends: Dict[str, type[AttentionBackend]],
        dtype: torch.dtype,
        bucket: str,
        prefix: str = "kv-cache",
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        addressing_style: str = "auto",
        profile_name: Optional[str] = None,
        threads_per_gpu: Optional[int] = None,
        max_staging_memory_gb: float = DEFAULT_MAX_STAGING_MEMORY_GB,
    ):
        super().__init__(
            model_name,
            tp_size,
            tp_rank,
            dtype,
            gpu_blocks_per_file,
            threads_per_gpu or DEFAULT_MAX_THREADS_PER_GPU,
            attn_backends,
            bucket,
            prefix,
            region,
            endpoint_url,
            addressing_style,
            profile_name,
            max_staging_memory_gb,
        )

        self.src_tensors = list(kv_caches.values())

        logger.info(
            f"GPUS3OffloadingHandler: tp_rank={self.tp_rank}, "
            f"threads_per_gpu={self.threads_per_gpu}, "
            f"bucket={bucket}, base_key={self.base_key}"
        )

    def _put_blocks_to_s3(
        self, job_id: int, s3_key: str, tensors: List[torch.Tensor], block_ids: List[int]
    ) -> Tuple[int, bool]:
        """Upload blocks to S3 (runs in thread pool)."""
        try:
            # Extract blocks from GPU tensors
            blocks_data = []
            for tensor in tensors:
                # Assuming shape: (2, num_blocks, num_heads, block_size, head_size)
                # or similar - extract specified blocks
                selected_blocks = tensor[:, block_ids, :, :, :]
                blocks_data.append(selected_blocks.cpu().numpy())

            # Serialize to bytes
            buffer = io.BytesIO()
            import numpy as np
            np.savez_compressed(buffer, *blocks_data)
            data = buffer.getvalue()

            # Upload to S3
            self.s3_client.put_object(s3_key, data)

            logger.debug(
                f"[PUT] job_id={job_id} uploaded {len(data)} bytes to {s3_key}"
            )
            return (job_id, True)
        except Exception as e:
            logger.error(f"[PUT] job_id={job_id} failed: {e}")
            return (job_id, False)

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        """Launch async PUT transfers from GPU tensors to S3."""
        src_spec, dst_spec = spec
        if dst_spec is None or len(dst_spec.block_hashes) == 0:
            return True

        for i, block_hash in enumerate(dst_spec.block_hashes):
            start = i * self.gpu_blocks_per_file
            end = min((i + 1) * self.gpu_blocks_per_file, len(src_spec.block_ids))
            if start >= len(src_spec.block_ids):
                break

            block_ids = src_spec.block_ids[start:end]
            s3_key = self._get_s3_key(block_hash)

            # Submit to thread pool
            future = self.executor.submit(
                self._put_blocks_to_s3, job_id, s3_key, self.src_tensors, block_ids
            )

            def callback(fut, jid=job_id):
                result = fut.result()
                with self.lock:
                    self.completed_jobs.append(result)

            future.add_done_callback(callback)

            with self.lock:
                self.pending_futures[job_id] = future

        return True


# ----------------------------------------------------------------------
# S3 → GPU (GET)
# ----------------------------------------------------------------------
class S3GPUOffloadingHandler(S3OffloadingHandler):
    """Handler for reading KV blocks from S3 back into GPU."""

    def __init__(
        self,
        model_name: str,
        tp_size: int,
        tp_rank: int,
        dtype: torch.dtype,
        gpu_blocks_per_file: int,
        kv_caches: Dict[str, torch.Tensor],
        attn_backends: Dict[str, type[AttentionBackend]],
        bucket: str,
        prefix: str = "kv-cache",
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        addressing_style: str = "auto",
        profile_name: Optional[str] = None,
        threads_per_gpu: Optional[int] = None,
        max_staging_memory_gb: float = DEFAULT_MAX_STAGING_MEMORY_GB,
    ):
        super().__init__(
            model_name,
            tp_size,
            tp_rank,
            dtype,
            gpu_blocks_per_file,
            threads_per_gpu or DEFAULT_MAX_THREADS_PER_GPU,
            attn_backends,
            bucket,
            prefix,
            region,
            endpoint_url,
            addressing_style,
            profile_name,
            max_staging_memory_gb,
        )

        self.dst_tensors = list(kv_caches.values())

        logger.info(
            f"S3GPUOffloadingHandler: tp_rank={self.tp_rank}, "
            f"threads_per_gpu={self.threads_per_gpu}, "
            f"bucket={bucket}, base_key={self.base_key}"
        )

    def _get_blocks_from_s3(
        self, job_id: int, s3_key: str, tensors: List[torch.Tensor], block_ids: List[int]
    ) -> Tuple[int, bool]:
        """Download blocks from S3 (runs in thread pool)."""
        try:
            # Download from S3
            data = self.s3_client.get_object(s3_key)

            # Deserialize
            import numpy as np
            buffer = io.BytesIO(data)
            loaded = np.load(buffer)
            blocks_data = [loaded[f"arr_{i}"] for i in range(len(loaded.files))]

            # Copy to GPU tensors
            for tensor, block_data in zip(tensors, blocks_data):
                block_tensor = torch.from_numpy(block_data).to(
                    device=tensor.device, dtype=tensor.dtype
                )
                # Copy blocks to specified positions
                tensor[:, block_ids, :, :, :] = block_tensor

            logger.debug(
                f"[GET] job_id={job_id} downloaded {len(data)} bytes from {s3_key}"
            )
            return (job_id, True)
        except Exception as e:
            logger.error(f"[GET] job_id={job_id} failed: {e}")
            return (job_id, False)

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        """Launch async GET transfers from S3 to GPU tensors."""
        src_spec, dst_spec = spec
        if src_spec is None or len(src_spec.block_hashes) == 0:
            return True

        first_len = (
            len(dst_spec.block_ids) % self.gpu_blocks_per_file
            or self.gpu_blocks_per_file
        )
        start = 0

        for i, block_hash in enumerate(src_spec.block_hashes):
            if i == 0:
                size = first_len
            else:
                size = self.gpu_blocks_per_file

            end = min(start + size, len(dst_spec.block_ids))
            block_ids = dst_spec.block_ids[start:end]
            s3_key = self._get_s3_key(block_hash)

            # Submit to thread pool
            future = self.executor.submit(
                self._get_blocks_from_s3, job_id, s3_key, self.dst_tensors, block_ids
            )

            def callback(fut, jid=job_id):
                result = fut.result()
                with self.lock:
                    self.completed_jobs.append(result)

            future.add_done_callback(callback)

            with self.lock:
                self.pending_futures[job_id] = future

            start += size

        return True

# Made with Bob
