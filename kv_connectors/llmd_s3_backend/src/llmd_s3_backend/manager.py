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
import threading
import time
from collections.abc import Iterable
from typing import Optional

from vllm.v1.core.kv_cache_utils import BlockHash
from llmd_s3_backend.mediums import S3LoadStoreSpec
from llmd_s3_backend.s3_client import S3ClientWrapper
from llmd_s3_backend.lru_cache import LRUPresenceCache
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingManager,
    PrepareStoreOutput,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    from llmd_s3_backend.manifest import ManifestManager
    MANIFEST_AVAILABLE = True
except ImportError:
    MANIFEST_AVAILABLE = False
    logger.warning("Manifest support not available")


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
        enable_presence_cache: bool = False,
        manifest_prefix: str = "manifests",
        compaction_threshold: int = 100,
        compaction_interval_hours: int = 24,
        cache_max_size: Optional[int] = 1_000_000,
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
            enable_presence_cache: Enable presence cache with manifest (default: False)
            manifest_prefix: S3 prefix for manifest files (default: "manifests")
            compaction_threshold: Number of delta files before compaction (default: 100)
            compaction_interval_hours: Hours between compactions (default: 24)
            cache_max_size: Maximum number of blocks in presence cache (default: 1,000,000)
                           Set to None for unbounded cache
        """
        # Basic metadata about the model and tensor parallelism
        self.model_name = model_name
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.dtype = dtype
        self.bucket = bucket
        self.prefix = prefix
        self.enable_presence_cache = enable_presence_cache

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

        # Optional presence cache with manifest
        self._presence_cache: Optional[LRUPresenceCache] = None
        self._manifest_manager: Optional[ManifestManager] = None
        self._cache_max_size = cache_max_size
        
        if enable_presence_cache:
            if not MANIFEST_AVAILABLE:
                logger.warning("Presence cache requested but manifest module not available")
            else:
                self._init_presence_cache(
                    manifest_prefix=manifest_prefix,
                    compaction_threshold=compaction_threshold,
                    compaction_interval_hours=compaction_interval_hours,
                )

        logger.info(
            f"S3OffloadingManager initialized: bucket={bucket}, "
            f"base_key={self.base_key}, presence_cache={enable_presence_cache}"
        )
    
    def _init_presence_cache(
        self,
        manifest_prefix: str,
        compaction_threshold: int,
        compaction_interval_hours: int,
    ):
        """Initialize presence cache with manifest support."""
        try:
            # Initialize LRU cache
            self._presence_cache = LRUPresenceCache(max_size=self._cache_max_size)
            
            # Initialize manifest manager
            self._manifest_manager = ManifestManager(
                s3_client=self.s3_client,
                model_name=self.model_name,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                dtype=str(self.dtype).replace("torch.", ""),
                manifest_prefix=manifest_prefix,
                compaction_threshold=compaction_threshold,
                compaction_interval_hours=compaction_interval_hours,
            )
            
            # Load manifest and pre-warm cache
            manifest = self._manifest_manager.load_manifest()
            self._presence_cache.update(manifest.keys())
            
            stats = self._presence_cache.get_stats()
            logger.info(
                f"Pre-warmed presence cache with {stats['size']} blocks "
                f"(max_size={stats['max_size']})"
            )
            
            # Start background refresh thread
            self._refresh_thread = threading.Thread(
                target=self._refresh_cache_loop,
                daemon=True
            )
            self._refresh_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to initialize presence cache: {e}")
            self._presence_cache = LRUPresenceCache(max_size=self._cache_max_size)
    
    def _refresh_cache_loop(self):
        """Periodically refresh presence cache from manifest."""
        while True:
            try:
                time.sleep(300)  # Refresh every 5 minutes
                
                if self._manifest_manager and self._presence_cache:
                    manifest = self._manifest_manager.load_manifest()
                    current_keys = self._presence_cache.get_keys()
                    new_blocks = set(manifest.keys()) - current_keys
                    if new_blocks:
                        self._presence_cache.update(new_blocks)
                        stats = self._presence_cache.get_stats()
                        logger.info(
                            f"Refreshed cache: added {len(new_blocks)} new blocks, "
                            f"size={stats['size']}, evictions={stats['evictions']}, "
                            f"hit_rate={stats['hit_rate']:.2%}"
                        )
                            
            except Exception as e:
                logger.error(f"Error refreshing cache: {e}")
                time.sleep(60)  # Wait before retry

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
        Uses presence cache if enabled, otherwise falls back to HEAD requests.
        """
        hit_count = 0
        for block_hash in block_hashes:
            block_hash_str = str(block_hash) if not isinstance(block_hash, str) else block_hash
            
            # Check presence cache first if enabled
            if self._presence_cache is not None:
                if block_hash_str in self._presence_cache:
                    hit_count += 1
                    continue
            
            # Not in cache or cache disabled - check S3
            s3_key = self._get_s3_key(block_hash)
            if not self.s3_client.object_exists(s3_key):
                break  # Miss - stop checking
            
            # Found in S3 - add to cache if enabled
            if self._presence_cache is not None:
                self._presence_cache.add(block_hash_str)
            
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
        Update presence cache and manifest when blocks are stored.
        """
        if not success:
            return
        
        block_hashes_list = list(block_hashes)
        
        # Update presence cache if enabled
        if self._presence_cache is not None:
            block_hash_strs = [
                str(bh) if not isinstance(bh, str) else bh
                for bh in block_hashes_list
            ]
            self._presence_cache.update(block_hash_strs)
        
        # Update manifest if enabled
        if self._manifest_manager is not None:
            s3_keys = [self._get_s3_key(bh) for bh in block_hashes_list]
            self._manifest_manager.queue_add_blocks(block_hashes_list, s3_keys)

# Made with Bob
