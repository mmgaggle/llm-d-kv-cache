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
from collections.abc import Iterable
from typing import Optional

from vllm.v1.core.kv_cache_utils import BlockHash
from llmd_s3_backend.mediums import S3LoadStoreSpec
from llmd_s3_backend.s3_client import S3ClientWrapper
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingManager,
    PrepareStoreOutput,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class S3OffloadingManager(OffloadingManager):
    """
    S3OffloadingManager manages KV offloading to S3 object storage.
    """

    def __init__(
        self,
        model_name: str,
        tp_size: int,
        tp_rank: int,
        dtype: torch.dtype,
        bucket: str,
        prefix: str = "kv-cache",
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        addressing_style: str = "auto",
        profile_name: Optional[str] = None,
    ) -> None:
        """
        Initialize S3 offloading manager.

        Args:
            model_name: Model name for organizing S3 keys
            tp_size: Tensor parallel size
            tp_rank: Tensor parallel rank
            dtype: Data type for KV cache
            bucket: S3 bucket name
            prefix: S3 key prefix (default: "kv-cache")
            region: AWS region
            endpoint_url: Custom S3 endpoint URL
            addressing_style: S3 addressing style
            profile_name: AWS profile name
        """
        # Basic metadata about the model and tensor parallelism
        self.model_name = model_name
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.dtype = dtype
        self.bucket = bucket
        self.prefix = prefix

        # Initialize S3 client
        self.s3_client = S3ClientWrapper(
            bucket=bucket,
            region=region,
            endpoint_url=endpoint_url,
            addressing_style=addressing_style,
            profile_name=profile_name,
        )

        # Build base S3 key prefix for this model/rank
        dtype_str = str(dtype).replace("torch.", "")
        self.base_key = f"{prefix}/{model_name}/tp_{tp_size}/rank_{tp_rank}/{dtype_str}"

        logger.info(
            f"S3OffloadingManager initialized: bucket={bucket}, "
            f"base_key={self.base_key}"
        )

    def _get_s3_key(self, block_hash: BlockHash) -> str:
        """
        Generate S3 key for a given block hash.
        Uses same hierarchical structure as filesystem connector.
        """
        if isinstance(block_hash, bytes):
            block_hash = int.from_bytes(block_hash, "little")
        block_hash_hex = f"{block_hash & ((1 << 64) - 1):016x}"
        subfolder1, subfolder2 = block_hash_hex[:3], block_hash_hex[3:5]
        return f"{self.base_key}/{subfolder1}/{subfolder2}/{block_hash_hex}.bin"

    # ----------------------------------------------------------------------
    # Lookup
    # ----------------------------------------------------------------------
    def lookup(self, block_hashes: Iterable[BlockHash]) -> int:
        """
        Return how many consecutive blocks from the start are already offloaded.
        """
        hit_count = 0
        for block_hash in block_hashes:
            s3_key = self._get_s3_key(block_hash)
            if not self.s3_client.object_exists(s3_key):
                break
            hit_count += 1
        return hit_count

    # ----------------------------------------------------------------------
    # Load
    # ----------------------------------------------------------------------
    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        """
        For S3, loading is stateless - return specs that point to S3 objects.
        """
        return S3LoadStoreSpec(block_hashes)

    def touch(self, block_hashes: Iterable[BlockHash]):
        """
        Update access times if desired.
        S3 version does nothing here as updates are handled by the worker.
        """
        pass

    def complete_load(self, block_hashes: Iterable[BlockHash]):
        """Stateless load - no post-load action needed."""
        pass

    # ----------------------------------------------------------------------
    # Store
    # ----------------------------------------------------------------------
    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> Optional[PrepareStoreOutput]:
        """
        Prepare storing new blocks.
        S3 always accepts new blocks. Eviction is not needed.
        If an object already exists, the worker handles it.
        """
        block_hashes_to_store = list(block_hashes)

        # Set up store spec
        store_spec = S3LoadStoreSpec(block_hashes_to_store)

        return PrepareStoreOutput(
            block_hashes_to_store=block_hashes_to_store,
            store_spec=store_spec,
            block_hashes_evicted=[],  # no eviction needed
        )

    def complete_store(
        self, block_hashes: Iterable[BlockHash], success: bool = True
    ):
        """
        For S3, storing is stateless - no action needed.
        """
        pass

# Made with Bob
