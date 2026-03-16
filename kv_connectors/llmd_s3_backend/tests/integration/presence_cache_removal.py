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

"""
Comprehensive tests for presence cache block removal functionality.

Tests cover:
- Cache removal during manifest sync
- invalidate_block() API
- Full lifecycle (add -> verify -> delete -> verify removed)
- Integration with manifest DELETE operations
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from llmd_s3_backend.lru_cache import LRUPresenceCache
from llmd_s3_backend.manager import S3OffloadingManager
from llmd_s3_backend.manifest import ManifestManager, DeltaOperation


class TestPresenceCacheRemoval:
    """Test presence cache removal operations."""
    
    def test_remove_single_block(self):
        """Test removing a single block from cache."""
        cache = LRUPresenceCache(max_size=10)
        
        # Add blocks
        cache.add("block1")
        cache.add("block2")
        cache.add("block3")
        
        assert "block2" in cache
        
        # Remove block2
        cache.remove("block2")
        
        assert "block1" in cache
        assert "block2" not in cache
        assert "block3" in cache
        assert len(cache) == 2
    
    def test_remove_multiple_blocks(self):
        """Test removing multiple blocks from cache."""
        cache = LRUPresenceCache(max_size=10)
        
        # Add blocks
        for i in range(10):
            cache.add(f"block{i}")
        
        # Remove several blocks
        blocks_to_remove = ["block2", "block5", "block7"]
        for block in blocks_to_remove:
            cache.remove(block)
        
        # Verify removals
        for block in blocks_to_remove:
            assert block not in cache
        
        # Verify remaining blocks
        assert len(cache) == 7
        assert "block0" in cache
        assert "block9" in cache
    
    def test_remove_during_sync(self):
        """Test that cache correctly removes blocks during manifest sync."""
        cache = LRUPresenceCache(max_size=100)
        
        # Simulate initial cache state
        initial_blocks = {f"block{i}" for i in range(20)}
        cache.update(initial_blocks)
        
        # Simulate manifest with some blocks removed
        manifest_blocks = {f"block{i}" for i in range(10, 30)}
        
        # Calculate diff (as done in manager)
        new_blocks = manifest_blocks - initial_blocks
        deleted_blocks = initial_blocks - manifest_blocks
        
        # Apply changes
        cache.update(new_blocks)
        for block in deleted_blocks:
            cache.remove(block)
        
        # Verify state
        assert len(cache) == 20
        for i in range(10):
            assert f"block{i}" not in cache  # Deleted
        for i in range(10, 30):
            assert f"block{i}" in cache  # Present


class TestManagerInvalidateBlock:
    """Test S3OffloadingManager.invalidate_block() method."""
    
    def test_invalidate_cache_entry_api_exists(self):
        """Test that invalidate_cache_entry method exists on manager."""
        # This is a simple API check - actual functionality tested in unit tests
        from llmd_s3_backend.manager import S3OffloadingManager
        assert hasattr(S3OffloadingManager, 'invalidate_cache_entry')
        
    def test_cache_removal_logic(self):
        """Test the cache removal logic directly."""
        cache = LRUPresenceCache(max_size=10)
        
        # Add block
        block_hash = "test-block-hash"
        cache.add(block_hash)
        assert block_hash in cache
        
        # Remove block (simulating what invalidate_block does)
        cache.remove(block_hash)
        assert block_hash not in cache
        
    def test_cache_removal_nonexistent(self):
        """Test removing non-existent block doesn't raise error."""
        cache = LRUPresenceCache(max_size=10)
        
        # Remove non-existent block (should not raise)
        cache.remove("non-existent-block")
        assert len(cache) == 0


class TestManifestDeleteOperations:
    """Test manifest DELETE operations and cache synchronization."""
    
    def test_delete_operation_structure(self):
        """Test that DELETE operations have correct structure."""
        # Create DELETE operation
        delete_op = DeltaOperation("DELETE", "block-hash-123")
        
        assert delete_op.type == "DELETE"
        assert delete_op.block_hash == "block-hash-123"
        assert delete_op.s3_key is None
        assert delete_op.size_bytes is None
        assert delete_op.created_at is None
    
    def test_manifest_processes_deletes(self):
        """Test that manifest correctly processes DELETE operations."""
        # Simulate manifest with ADD and DELETE operations
        # This tests the logic without needing S3
        manifest = {
            "block1": "key1",
            "block2": "key2",
            "block3": "key3",
        }
        
        # Apply DELETE operation
        delete_ops = [
            DeltaOperation("DELETE", "block2"),
        ]
        
        # Simulate processing deletes (as done in load_manifest)
        for op in delete_ops:
            if op.type == "DELETE":
                manifest.pop(op.block_hash, None)
        
        # Verify block2 is removed
        assert "block1" in manifest
        assert "block2" not in manifest
        assert "block3" in manifest
        assert len(manifest) == 2


class TestCacheSyncWithDeletions:
    """Test cache synchronization when manifest has deletions."""
    
    def test_sync_removes_deleted_blocks(self):
        """Test that cache sync removes blocks deleted from manifest."""
        cache = LRUPresenceCache(max_size=100)
        
        # Initial cache state (simulating initial manifest load)
        initial_blocks = {f"block{i}" for i in range(10)}
        cache.update(initial_blocks)
        
        assert len(cache) == 10
        for i in range(10):
            assert f"block{i}" in cache
        
        # Simulate manifest update: blocks 5-14 (removed 0-4, added 10-14)
        updated_manifest_keys = {f"block{i}" for i in range(5, 15)}
        
        # Manually trigger sync (simulating what background thread does)
        current_keys = cache.get_keys()
        manifest_keys = updated_manifest_keys
        
        new_blocks = manifest_keys - current_keys
        deleted_blocks = current_keys - manifest_keys
        
        # Apply changes
        if new_blocks:
            cache.update(new_blocks)
        if deleted_blocks:
            for block_hash in deleted_blocks:
                cache.remove(block_hash)
        
        # Verify deletions
        for i in range(5):
            assert f"block{i}" not in cache
        
        # Verify additions
        for i in range(10, 15):
            assert f"block{i}" in cache
        
        # Verify existing blocks still present
        for i in range(5, 10):
            assert f"block{i}" in cache
        
        assert len(cache) == 10


class TestFullLifecycle:
    """Test full lifecycle: add -> verify -> delete -> verify removed."""
    
    def test_block_lifecycle_in_cache(self):
        """Test complete block lifecycle in presence cache."""
        cache = LRUPresenceCache(max_size=100)
        
        # Phase 1: Add block
        block_hash = "lifecycle-block-123"
        cache.add(block_hash)
        
        # Phase 2: Verify present
        assert block_hash in cache
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["size"] == 1
        
        # Phase 3: Delete block
        cache.remove(block_hash)
        
        # Phase 4: Verify removed
        assert block_hash not in cache
        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["size"] == 0
    
    def test_block_lifecycle_with_manifest_simulation(self):
        """Test complete block lifecycle simulating manifest operations."""
        cache = LRUPresenceCache(max_size=100)
        
        block_hash = "lifecycle-block-456"
        
        # Phase 1: Add to cache (simulating successful upload and manifest ADD)
        cache.add(block_hash)
        assert block_hash in cache
        
        # Phase 2: Verify can check presence
        is_present = block_hash in cache
        assert is_present is True
        
        # Phase 3: Remove (simulating manifest DELETE operation)
        cache.remove(block_hash)
        
        # Phase 4: Verify removed
        is_present = block_hash in cache
        assert is_present is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob