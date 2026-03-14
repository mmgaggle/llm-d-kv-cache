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
Pinned (page-locked) memory buffer pool for zero-copy GPU transfers.

Pinned memory allows direct DMA transfers between CPU and GPU without
going through the OS page cache, providing 3-4x faster transfer speeds.
"""

import torch
import threading
import queue
from typing import Optional
from dataclasses import dataclass
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class PinnedBuffer:
    """A pinned memory buffer with metadata."""
    tensor: torch.Tensor  # The actual pinned memory
    buffer_id: int        # Unique identifier
    size_bytes: int       # Buffer size in bytes
    in_use: bool = False  # Whether buffer is currently allocated


class PinnedBufferPool:
    """
    Pool of pre-allocated pinned (page-locked) memory buffers.
    
    Pinned memory enables fast DMA transfers to GPU without OS paging overhead.
    Pre-allocation amortizes the cost of pinning memory across many operations.
    
    Thread-safe for concurrent access from multiple workers.
    """
    
    def __init__(
        self,
        buffer_size_mb: int = 128,
        num_buffers: int = 64,
        device: str = "cpu"
    ):
        """
        Initialize pinned buffer pool.
        
        Args:
            buffer_size_mb: Size of each buffer in megabytes
            num_buffers: Number of buffers to pre-allocate
            device: Device to allocate on ("cpu" for pinned host memory)
        """
        self.buffer_size_bytes = buffer_size_mb * 1024 * 1024
        self.num_buffers = num_buffers
        self.device = device
        
        # Pre-allocate all buffers
        self.buffers = []
        self.available = queue.Queue()
        self.lock = threading.Lock()
        
        logger.info(
            f"Allocating {num_buffers} pinned buffers of "
            f"{buffer_size_mb} MB each ({num_buffers * buffer_size_mb} MB total)"
        )
        
        try:
            for i in range(num_buffers):
                # Allocate buffer and pin it
                tensor = torch.empty(
                    self.buffer_size_bytes,
                    dtype=torch.uint8,
                    device=device
                )
                
                if device == "cpu" and torch.cuda.is_available():
                    tensor = tensor.pin_memory()
                
                buffer = PinnedBuffer(
                    tensor=tensor,
                    buffer_id=i,
                    size_bytes=self.buffer_size_bytes,
                    in_use=False
                )
                
                self.buffers.append(buffer)
                self.available.put(buffer)
            
            logger.info(
                f"Successfully allocated {num_buffers} pinned buffers "
                f"({self._get_total_memory_mb():.1f} MB total)"
            )
            
        except Exception as e:
            logger.error(f"Failed to allocate pinned buffers: {e}")
            raise
    
    def acquire(self, timeout: Optional[float] = None) -> Optional[PinnedBuffer]:
        """
        Acquire a buffer from the pool.
        
        Args:
            timeout: Maximum time to wait for a buffer (None = wait forever)
            
        Returns:
            PinnedBuffer if available, None if timeout
        """
        try:
            buffer = self.available.get(timeout=timeout)
            with self.lock:
                buffer.in_use = True
            return buffer
        except queue.Empty:
            logger.warning("Buffer pool exhausted, no buffers available")
            return None
    
    def release(self, buffer: PinnedBuffer):
        """
        Return a buffer to the pool.
        
        Args:
            buffer: Buffer to return
        """
        with self.lock:
            if not buffer.in_use:
                logger.warning(f"Releasing buffer {buffer.buffer_id} that was not in use")
                return
            buffer.in_use = False
        
        self.available.put(buffer)
    
    def get_stats(self) -> dict:
        """Get pool statistics."""
        with self.lock:
            in_use = sum(1 for b in self.buffers if b.in_use)
            available = self.num_buffers - in_use
            
            return {
                "total_buffers": self.num_buffers,
                "in_use": in_use,
                "available": available,
                "utilization": in_use / self.num_buffers if self.num_buffers > 0 else 0,
                "buffer_size_mb": self.buffer_size_bytes / (1024 * 1024),
                "total_memory_mb": self._get_total_memory_mb(),
            }
    
    def _get_total_memory_mb(self) -> float:
        """Get total memory allocated in MB."""
        return (self.buffer_size_bytes * self.num_buffers) / (1024 * 1024)
    
    def __del__(self):
        """Clean up buffers on destruction."""
        logger.info(f"Releasing {self.num_buffers} pinned buffers")
        # Tensors will be automatically freed by PyTorch


class PinnedBufferView:
    """
    A view into a pinned buffer for a specific data size.
    
    Allows working with a subset of a larger buffer without copying.
    """
    
    def __init__(self, buffer: PinnedBuffer, size: int):
        """
        Create a view into a buffer.
        
        Args:
            buffer: The underlying pinned buffer
            size: Size of data in bytes (must be <= buffer.size_bytes)
        """
        if size > buffer.size_bytes:
            raise ValueError(
                f"View size {size} exceeds buffer size {buffer.size_bytes}"
            )
        
        self.buffer = buffer
        self.size = size
        self._tensor_view = buffer.tensor[:size]
    
    def as_tensor(self) -> torch.Tensor:
        """Get tensor view of the data."""
        return self._tensor_view
    
    def as_bytes(self) -> bytes:
        """Get bytes view of the data (copies to Python bytes)."""
        return self._tensor_view.cpu().numpy().tobytes()
    
    def copy_from_bytes(self, data: bytes):
        """
        Copy data from bytes into the buffer.
        
        Args:
            data: Bytes to copy (must be <= size)
        """
        if len(data) > self.size:
            raise ValueError(
                f"Data size {len(data)} exceeds view size {self.size}"
            )
        
        # Copy into tensor
        import numpy as np
        arr = np.frombuffer(data, dtype=np.uint8)
        self._tensor_view[:len(data)].copy_(torch.from_numpy(arr))
    
    def copy_to_gpu(
        self,
        dst: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
        non_blocking: bool = True
    ):
        """
        Copy data to GPU tensor.
        
        Args:
            dst: Destination GPU tensor
            stream: CUDA stream for async copy
            non_blocking: Whether to use non-blocking transfer
        """
        if stream is not None:
            with torch.cuda.stream(stream):
                dst.copy_(self._tensor_view, non_blocking=non_blocking)
        else:
            dst.copy_(self._tensor_view, non_blocking=non_blocking)


# Example usage and testing
if __name__ == "__main__":
    import time
    
    # Create a small pool for testing
    pool = PinnedBufferPool(buffer_size_mb=64, num_buffers=4)
    
    print("Pool stats:", pool.get_stats())
    
    # Acquire and release buffers
    buffers = []
    for i in range(3):
        buf = pool.acquire(timeout=1.0)
        if buf:
            print(f"Acquired buffer {buf.buffer_id}")
            buffers.append(buf)
    
    print("Pool stats after acquiring 3:", pool.get_stats())
    
    # Release buffers
    for buf in buffers:
        pool.release(buf)
        print(f"Released buffer {buf.buffer_id}")
    
    print("Pool stats after releasing:", pool.get_stats())
    
    # Test buffer view
    buf = pool.acquire()
    if buf:
        view = PinnedBufferView(buf, size=1024)
        test_data = b"Hello, pinned memory!" * 50
        view.copy_from_bytes(test_data)
        retrieved = view.as_bytes()[:len(test_data)]
        assert retrieved == test_data, "Data mismatch!"
        print("Buffer view test passed!")
        pool.release(buf)

# Made with Bob
