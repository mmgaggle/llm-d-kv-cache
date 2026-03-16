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

"""Tests for pinned buffer pool."""

import pytest
import torch
import threading
import time
from llmd_s3_backend.pinned_buffers import (
    PinnedBufferPool,
    PinnedBuffer,
    PinnedBufferView,
)


class TestPinnedBufferPool:
    """Test pinned buffer pool functionality."""
    
    def test_pool_creation(self):
        """Test creating a buffer pool."""
        pool = PinnedBufferPool(buffer_size_mb=1, num_buffers=4)
        
        stats = pool.get_stats()
        assert stats["total_buffers"] == 4
        assert stats["available"] == 4
        assert stats["in_use"] == 0
        assert stats["utilization"] == 0.0
        assert stats["buffer_size_mb"] == 1.0
        assert stats["total_memory_mb"] == 4.0
    
    def test_acquire_release(self):
        """Test acquiring and releasing buffers."""
        pool = PinnedBufferPool(buffer_size_mb=1, num_buffers=2)
        
        # Acquire first buffer
        buf1 = pool.acquire(timeout=1.0)
        assert buf1 is not None
        assert buf1.in_use is True
        
        stats = pool.get_stats()
        assert stats["in_use"] == 1
        assert stats["available"] == 1
        
        # Acquire second buffer
        buf2 = pool.acquire(timeout=1.0)
        assert buf2 is not None
        assert buf2.buffer_id != buf1.buffer_id
        
        stats = pool.get_stats()
        assert stats["in_use"] == 2
        assert stats["available"] == 0
        
        # Release first buffer
        pool.release(buf1)
        stats = pool.get_stats()
        assert stats["in_use"] == 1
        assert stats["available"] == 1
        
        # Release second buffer
        pool.release(buf2)
        stats = pool.get_stats()
        assert stats["in_use"] == 0
        assert stats["available"] == 2
    
    def test_pool_exhaustion(self):
        """Test behavior when pool is exhausted."""
        pool = PinnedBufferPool(buffer_size_mb=1, num_buffers=2)
        
        # Acquire all buffers
        buf1 = pool.acquire(timeout=1.0)
        buf2 = pool.acquire(timeout=1.0)
        
        # Try to acquire when exhausted
        buf3 = pool.acquire(timeout=0.1)
        assert buf3 is None
        
        # Release one and try again
        pool.release(buf1)
        buf3 = pool.acquire(timeout=1.0)
        assert buf3 is not None
    
    def test_concurrent_access(self):
        """Test thread-safe concurrent access."""
        pool = PinnedBufferPool(buffer_size_mb=1, num_buffers=10)
        acquired_buffers = []
        lock = threading.Lock()
        
        def worker():
            buf = pool.acquire(timeout=2.0)
            if buf:
                with lock:
                    acquired_buffers.append(buf.buffer_id)
                time.sleep(0.01)  # Simulate work
                pool.release(buf)
        
        # Launch multiple threads
        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All threads should have acquired buffers
        assert len(acquired_buffers) == 20
        
        # Pool should be back to full capacity
        stats = pool.get_stats()
        assert stats["available"] == 10
        assert stats["in_use"] == 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_pinned_memory(self):
        """Test that buffers are actually pinned."""
        pool = PinnedBufferPool(buffer_size_mb=1, num_buffers=1, device="cpu")
        
        buf = pool.acquire()
        assert buf is not None
        
        # Check if tensor is pinned
        assert buf.tensor.is_pinned()
        
        pool.release(buf)


class TestPinnedBufferView:
    """Test pinned buffer view functionality."""
    
    def test_view_creation(self):
        """Test creating a view into a buffer."""
        pool = PinnedBufferPool(buffer_size_mb=1, num_buffers=1)
        buf = pool.acquire()
        
        view = PinnedBufferView(buf, size=1024)
        assert view.size == 1024
        assert view.buffer == buf
        
        pool.release(buf)
    
    def test_view_size_validation(self):
        """Test that view size is validated."""
        pool = PinnedBufferPool(buffer_size_mb=1, num_buffers=1)
        buf = pool.acquire()
        
        # Should raise error if view size exceeds buffer size
        with pytest.raises(ValueError):
            PinnedBufferView(buf, size=buf.size_bytes + 1)
        
        pool.release(buf)
    
    def test_copy_from_bytes(self):
        """Test copying data from bytes into buffer."""
        pool = PinnedBufferPool(buffer_size_mb=1, num_buffers=1)
        buf = pool.acquire()
        
        test_data = b"Hello, pinned memory!" * 100
        view = PinnedBufferView(buf, size=len(test_data))
        
        view.copy_from_bytes(test_data)
        retrieved = view.as_bytes()
        
        assert retrieved == test_data
        
        pool.release(buf)
    
    def test_as_tensor(self):
        """Test getting tensor view."""
        pool = PinnedBufferPool(buffer_size_mb=1, num_buffers=1)
        buf = pool.acquire()
        
        view = PinnedBufferView(buf, size=1024)
        tensor = view.as_tensor()
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == 1024
        assert tensor.dtype == torch.uint8
        
        pool.release(buf)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_copy_to_gpu(self):
        """Test copying data to GPU."""
        pool = PinnedBufferPool(buffer_size_mb=1, num_buffers=1, device="cpu")
        buf = pool.acquire()
        
        # Create test data
        test_data = b"GPU transfer test" * 100
        view = PinnedBufferView(buf, size=len(test_data))
        view.copy_from_bytes(test_data)
        
        # Create GPU tensor
        gpu_tensor = torch.empty(len(test_data), dtype=torch.uint8, device="cuda")
        
        # Copy to GPU
        view.copy_to_gpu(gpu_tensor, non_blocking=True)
        torch.cuda.synchronize()
        
        # Verify data
        cpu_result = gpu_tensor.cpu().numpy().tobytes()
        assert cpu_result == test_data
        
        pool.release(buf)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_async_gpu_copy_with_stream(self):
        """Test async GPU copy with CUDA stream."""
        pool = PinnedBufferPool(buffer_size_mb=1, num_buffers=1, device="cpu")
        buf = pool.acquire()
        
        test_data = b"Async GPU test" * 100
        view = PinnedBufferView(buf, size=len(test_data))
        view.copy_from_bytes(test_data)
        
        # Create GPU tensor and stream
        gpu_tensor = torch.empty(len(test_data), dtype=torch.uint8, device="cuda")
        stream = torch.cuda.Stream()
        
        # Async copy
        view.copy_to_gpu(gpu_tensor, stream=stream, non_blocking=True)
        stream.synchronize()
        
        # Verify
        cpu_result = gpu_tensor.cpu().numpy().tobytes()
        assert cpu_result == test_data
        
        pool.release(buf)


class TestPinnedBufferPerformance:
    """Performance tests for pinned buffers."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_pinned_vs_pageable_transfer_speed(self):
        """Compare pinned vs pageable memory transfer speeds."""
        size_mb = 64
        size_bytes = size_mb * 1024 * 1024
        
        # Pinned memory
        pool = PinnedBufferPool(buffer_size_mb=size_mb, num_buffers=1, device="cpu")
        buf = pool.acquire()
        view = PinnedBufferView(buf, size=size_bytes)
        
        gpu_tensor_pinned = torch.empty(size_bytes, dtype=torch.uint8, device="cuda")
        
        # Warm up
        view.copy_to_gpu(gpu_tensor_pinned, non_blocking=True)
        torch.cuda.synchronize()
        
        # Time pinned transfer
        start = time.time()
        for _ in range(10):
            view.copy_to_gpu(gpu_tensor_pinned, non_blocking=True)
        torch.cuda.synchronize()
        pinned_time = time.time() - start
        
        pool.release(buf)
        
        # Pageable memory
        pageable_tensor = torch.empty(size_bytes, dtype=torch.uint8, device="cpu")
        gpu_tensor_pageable = torch.empty(size_bytes, dtype=torch.uint8, device="cuda")
        
        # Warm up
        gpu_tensor_pageable.copy_(pageable_tensor)
        torch.cuda.synchronize()
        
        # Time pageable transfer
        start = time.time()
        for _ in range(10):
            gpu_tensor_pageable.copy_(pageable_tensor)
        torch.cuda.synchronize()
        pageable_time = time.time() - start
        
        # Pinned should be faster
        speedup = pageable_time / pinned_time
        print(f"\nPinned memory speedup: {speedup:.2f}x")
        print(f"Pinned: {pinned_time:.3f}s, Pageable: {pageable_time:.3f}s")
        
        # Expect at least 1.5x speedup (conservative, usually 2-3x)
        assert speedup > 1.5, f"Expected speedup > 1.5x, got {speedup:.2f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

# Made with Bob
