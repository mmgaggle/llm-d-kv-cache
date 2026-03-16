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
LRU cache implementation for presence cache with configurable size limits.

This module provides a thread-safe LRU (Least Recently Used) cache that
automatically evicts the least recently accessed items when the cache
reaches its maximum size.
"""

from collections import OrderedDict
from typing import Iterable, Optional, Set
import threading


class LRUPresenceCache:
    """
    Thread-safe LRU cache for tracking block presence in S3.
    
    Uses OrderedDict to maintain access order. When the cache reaches
    max_size, the least recently used items are evicted.
    
    Args:
        max_size: Maximum number of items to cache (default: 1,000,000)
                  Set to None for unbounded cache (original behavior)
    """
    
    def __init__(self, max_size: Optional[int] = 1_000_000):
        self.max_size = max_size
        self._cache: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache (updates access order)."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return True
            self._misses += 1
            return False
    
    def add(self, key: str) -> None:
        """Add a single key to the cache."""
        with self._lock:
            if key in self._cache:
                # Already exists, just update access order
                self._cache.move_to_end(key)
            else:
                # New key
                self._cache[key] = None
                self._evict_if_needed()
    
    def update(self, keys: Iterable[str]) -> None:
        """Add multiple keys to the cache."""
        with self._lock:
            for key in keys:
                if key in self._cache:
                    self._cache.move_to_end(key)
                else:
                    self._cache[key] = None
            self._evict_if_needed()
    
    def remove(self, key: str) -> None:
        """Remove a key from the cache."""
        with self._lock:
            self._cache.pop(key, None)
    
    def clear(self) -> None:
        """Clear all items from the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
    
    def _evict_if_needed(self) -> None:
        """Evict least recently used items if cache exceeds max_size."""
        if self.max_size is None:
            return
        
        while len(self._cache) > self.max_size:
            # Remove oldest item (first in OrderedDict)
            self._cache.popitem(last=False)
            self._evictions += 1
    
    def __len__(self) -> int:
        """Return number of items in cache."""
        with self._lock:
            return len(self._cache)
    
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            dict with keys: size, max_size, hits, misses, evictions, hit_rate
        """
        with self._lock:
            total_accesses = self._hits + self._misses
            hit_rate = self._hits / total_accesses if total_accesses > 0 else 0.0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": hit_rate,
            }
    
    def get_keys(self) -> Set[str]:
        """
        Get a snapshot of all keys currently in cache.
        
        Note: This creates a copy and does not update access order.
        """
        with self._lock:
            return set(self._cache.keys())
    
    def bulk_check(self, keys: Iterable[str]) -> Set[str]:
        """
        Check multiple keys at once, returning the set of keys that exist.
        
        This is more efficient than checking keys individually when you need
        to check many keys. Updates access order for all found keys.
        
        Args:
            keys: Keys to check
            
        Returns:
            Set of keys that exist in the cache
        """
        with self._lock:
            found = set()
            for key in keys:
                if key in self._cache:
                    self._cache.move_to_end(key)
                    found.add(key)
                    self._hits += 1
                else:
                    self._misses += 1
            return found

# Made with Bob
