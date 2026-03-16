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

"""Tests for LRU presence cache."""

import pytest
import threading
import time
from llmd_s3_backend.lru_cache import LRUPresenceCache


class TestLRUPresenceCache:
    """Test LRU cache basic functionality."""
    
    def test_basic_operations(self):
        """Test basic add/contains operations."""
        cache = LRUPresenceCache(max_size=3)
        
        # Add items
        cache.add("key1")
        cache.add("key2")
        cache.add("key3")
        
        # Check presence
        assert "key1" in cache
        assert "key2" in cache
        assert "key3" in cache
        assert "key4" not in cache
        
        # Check size
        assert len(cache) == 3
    
    def test_lru_eviction(self):
        """Test that least recently used items are evicted."""
        cache = LRUPresenceCache(max_size=3)
        
        # Fill cache
        cache.add("key1")
        cache.add("key2")
        cache.add("key3")
        
        # Access key1 to make it recently used
        _ = "key1" in cache
        
        # Add key4, should evict key2 (least recently used)
        cache.add("key4")
        
        assert "key1" in cache  # Recently accessed
        assert "key2" not in cache  # Evicted
        assert "key3" in cache
        assert "key4" in cache
        
        stats = cache.get_stats()
        assert stats["evictions"] == 1
    
    def test_update_multiple(self):
        """Test updating cache with multiple keys."""
        cache = LRUPresenceCache(max_size=5)
        
        keys = ["key1", "key2", "key3"]
        cache.update(keys)
        
        assert len(cache) == 3
        for key in keys:
            assert key in cache
    
    def test_update_with_eviction(self):
        """Test that update respects max_size."""
        cache = LRUPresenceCache(max_size=3)
        
        # Add initial keys
        cache.add("key1")
        cache.add("key2")
        
        # Update with more keys than remaining space
        cache.update(["key3", "key4", "key5"])
        
        # Should have evicted key1 and key2
        assert "key1" not in cache
        assert "key2" not in cache
        assert "key3" in cache
        assert "key4" in cache
        assert "key5" in cache
        
        stats = cache.get_stats()
        assert stats["evictions"] == 2
    
    def test_remove(self):
        """Test removing items from cache."""
        cache = LRUPresenceCache(max_size=3)
        
        cache.add("key1")
        cache.add("key2")
        
        assert "key1" in cache
        cache.remove("key1")
        assert "key1" not in cache
        assert len(cache) == 1
    
    def test_clear(self):
        """Test clearing the cache."""
        cache = LRUPresenceCache(max_size=3)
        
        cache.update(["key1", "key2", "key3"])
        assert len(cache) == 3
        
        cache.clear()
        assert len(cache) == 0
        
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["evictions"] == 0
    
    def test_unbounded_cache(self):
        """Test cache with no size limit."""
        cache = LRUPresenceCache(max_size=None)
        
        # Add many items
        for i in range(1000):
            cache.add(f"key{i}")
        
        assert len(cache) == 1000
        
        stats = cache.get_stats()
        assert stats["evictions"] == 0
        assert stats["max_size"] is None
    
    def test_access_order_update(self):
        """Test that accessing items updates their order."""
        cache = LRUPresenceCache(max_size=3)
        
        cache.add("key1")
        cache.add("key2")
        cache.add("key3")
        
        # Access key1 multiple times
        _ = "key1" in cache
        _ = "key1" in cache
        
        # Add key4, should evict key2 (oldest unaccessed)
        cache.add("key4")
        
        assert "key1" in cache
        assert "key2" not in cache
        assert "key3" in cache
        assert "key4" in cache
    
    def test_duplicate_add(self):
        """Test that adding duplicate keys doesn't increase size."""
        cache = LRUPresenceCache(max_size=3)
        
        cache.add("key1")
        cache.add("key1")
        cache.add("key1")
        
        assert len(cache) == 1
        
        stats = cache.get_stats()
        assert stats["evictions"] == 0


class TestLRUCacheStats:
    """Test cache statistics tracking."""
    
    def test_hit_miss_tracking(self):
        """Test that hits and misses are tracked correctly."""
        cache = LRUPresenceCache(max_size=3)
        
        cache.add("key1")
        
        # Hits
        _ = "key1" in cache
        _ = "key1" in cache
        
        # Misses
        _ = "key2" in cache
        _ = "key3" in cache
        
        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 0.5
    
    def test_eviction_tracking(self):
        """Test that evictions are tracked correctly."""
        cache = LRUPresenceCache(max_size=2)
        
        cache.add("key1")
        cache.add("key2")
        cache.add("key3")  # Evicts key1
        cache.add("key4")  # Evicts key2
        
        stats = cache.get_stats()
        assert stats["evictions"] == 2
        assert stats["size"] == 2
    
    def test_stats_after_clear(self):
        """Test that stats are reset after clear."""
        cache = LRUPresenceCache(max_size=3)
        
        cache.add("key1")
        _ = "key1" in cache
        _ = "key2" in cache
        
        cache.clear()
        
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["evictions"] == 0
        assert stats["size"] == 0


class TestLRUCacheThreadSafety:
    """Test thread safety of LRU cache."""
    
    def test_concurrent_adds(self):
        """Test concurrent additions from multiple threads."""
        cache = LRUPresenceCache(max_size=1000)
        num_threads = 10
        items_per_thread = 100
        
        def add_items(thread_id):
            for i in range(items_per_thread):
                cache.add(f"thread{thread_id}_key{i}")
        
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=add_items, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All items should be in cache
        assert len(cache) == num_threads * items_per_thread
    
    def test_concurrent_reads_writes(self):
        """Test concurrent reads and writes."""
        cache = LRUPresenceCache(max_size=100)
        
        # Pre-populate cache
        for i in range(50):
            cache.add(f"key{i}")
        
        def reader():
            for _ in range(100):
                _ = f"key{_ % 50}" in cache
        
        def writer():
            for i in range(50, 100):
                cache.add(f"key{i}")
                time.sleep(0.001)
        
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=reader))
        threads.append(threading.Thread(target=writer))
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have 100 items (some may have been evicted)
    
    def test_concurrent_removals(self):
        """Test concurrent removals from multiple threads."""
        cache = LRUPresenceCache(max_size=1000)
        
        # Pre-populate cache
        for i in range(100):
            cache.add(f"key{i}")
        
        def remove_items(start, end):
            for i in range(start, end):
                cache.remove(f"key{i}")
        
        threads = []
        # 5 threads each removing 20 items
        for i in range(5):
            t = threading.Thread(target=remove_items, args=(i*20, (i+1)*20))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All 100 items should be removed
        assert len(cache) == 0
    
    def test_concurrent_add_remove(self):
        """Test concurrent additions and removals."""
        cache = LRUPresenceCache(max_size=100)
        
        # Pre-populate with some items
        for i in range(50):
            cache.add(f"key{i}")
        
        def adder():
            for i in range(50, 100):
                cache.add(f"key{i}")
                time.sleep(0.001)
        
        def remover():
            for i in range(0, 50):
                cache.remove(f"key{i}")
                time.sleep(0.001)
        
        threads = [
            threading.Thread(target=adder),
            threading.Thread(target=remover),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have ~50 items (the newly added ones)
        assert 40 <= len(cache) <= 60  # Allow some variance due to timing
        assert len(cache) <= 100


class TestLRUCacheBulkOperations:
    """Test bulk operations on LRU cache."""
    
    def test_get_keys(self):
        """Test getting all keys from cache."""
        cache = LRUPresenceCache(max_size=10)
        
        keys = [f"key{i}" for i in range(5)]
        cache.update(keys)
        
        cached_keys = cache.get_keys()
        assert cached_keys == set(keys)
    
    def test_bulk_check(self):
        """Test checking multiple keys at once."""
        cache = LRUPresenceCache(max_size=10)
        
        cache.update(["key1", "key2", "key3"])
        
        found = cache.bulk_check(["key1", "key2", "key4", "key5"])
        
        assert found == {"key1", "key2"}
        
        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 2
    
    def test_bulk_check_updates_order(self):
        """Test that bulk_check updates access order."""
        cache = LRUPresenceCache(max_size=3)
        
        cache.update(["key1", "key2", "key3"])
        
        # Bulk check key1 and key2
        cache.bulk_check(["key1", "key2"])
        
        # Add key4, should evict key3 (least recently accessed)
        cache.add("key4")
        
        assert "key1" in cache
        assert "key2" in cache
        assert "key3" not in cache
        assert "key4" in cache


class TestLRUCacheEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_size_cache(self):
        """Test cache with size 0."""
        cache = LRUPresenceCache(max_size=0)
        
        cache.add("key1")
        
        # Should immediately evict
        assert len(cache) == 0
        assert "key1" not in cache
    
    def test_size_one_cache(self):
        """Test cache with size 1."""
        cache = LRUPresenceCache(max_size=1)
        
        cache.add("key1")
        assert "key1" in cache
        
        cache.add("key2")
        assert "key1" not in cache
        assert "key2" in cache
        
        stats = cache.get_stats()
        assert stats["evictions"] == 1
    
    def test_empty_update(self):
        """Test updating with empty list."""
        cache = LRUPresenceCache(max_size=3)
        
        cache.update([])
        assert len(cache) == 0
    
    def test_remove_nonexistent(self):
        """Test removing key that doesn't exist."""
        cache = LRUPresenceCache(max_size=3)
        
        cache.add("key1")
        cache.remove("key2")  # Should not raise error
        
        assert len(cache) == 1
        assert "key1" in cache


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
