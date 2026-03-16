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
Tests for io_uring buffer registration and zero-copy operations.

These tests validate the buffer registration API and IORING_OP_READ_FIXED
functionality for achieving true zero-copy performance.
"""

import os
import sys
import pytest
import tempfile

# Skip all tests if not on Linux
pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="io_uring buffer registration only available on Linux"
)

try:
    from llmd_s3_backend.iouring_ops import IoUringContext, OpType
    from llmd_s3_backend.pinned_buffers import PinnedBufferPool
    IOURING_AVAILABLE = True
except ImportError:
    IOURING_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="io_uring not available")


@pytest.fixture(scope="module")
def check_buffer_registration_support():
    """
    Check if buffer registration is supported by the kernel.
    
    Skips all tests in the module if buffer registration is not supported.
    This is expected in minimal VM kernels (e.g., Podman on macOS).
    """
    if not IOURING_AVAILABLE:
        pytest.skip("io_uring not available")
    
    ctx = IoUringContext()
    try:
        if not ctx.supports_buffer_registration():
            pytest.skip(
                "Buffer registration (IORING_REGISTER_BUFFERS) not supported by kernel. "
                "This is expected in minimal VM kernels like Podman on macOS (Fedora CoreOS). "
                "Tests will pass on production Linux systems with full kernel support (Linux 5.1+)."
            )
    finally:
        ctx.close()


@pytest.mark.usefixtures("check_buffer_registration_support")
class TestBufferRegistration:
    """Test buffer registration with io_uring."""
    
    def test_register_single_buffer(self):
        """Test registering a single buffer."""
        ctx = IoUringContext(queue_depth=128)
        
        # Create a test buffer
        buffer = bytearray(4096)  # 4KB page-aligned
        mv = memoryview(buffer)
        
        # Register buffer
        buffer_ids = ctx.register_buffers([mv])
        
        assert len(buffer_ids) == 1
        assert buffer_ids[0] == 0
        assert len(ctx.registered_buffers) == 1
        assert 0 in ctx.buffer_map
        
        # Cleanup
        ctx.unregister_buffers()
        assert len(ctx.registered_buffers) == 0
        ctx.close()
    
    def test_register_multiple_buffers(self):
        """Test registering multiple buffers."""
        ctx = IoUringContext(queue_depth=128)
        
        # Create multiple test buffers
        buffers = [bytearray(4096) for _ in range(4)]
        memviews = [memoryview(buf) for buf in buffers]
        
        # Register buffers
        buffer_ids = ctx.register_buffers(memviews)
        
        assert len(buffer_ids) == 4
        assert buffer_ids == [0, 1, 2, 3]
        assert len(ctx.registered_buffers) == 4
        
        # Verify all buffers are tracked
        for i in range(4):
            assert i in ctx.buffer_map
            addr, size = ctx.buffer_map[i]
            assert size == 4096
        
        # Cleanup
        ctx.unregister_buffers()
        ctx.close()
    
    def test_register_buffers_twice_fails(self):
        """Test that registering buffers twice raises an error."""
        ctx = IoUringContext(queue_depth=128)
        
        buffer = bytearray(4096)
        mv = memoryview(buffer)
        
        # First registration succeeds
        ctx.register_buffers([mv])
        
        # Second registration should fail
        with pytest.raises(ValueError, match="already registered"):
            ctx.register_buffers([mv])
        
        ctx.close()
    
    def test_unregister_empty_buffers(self):
        """Test that unregistering with no buffers is safe."""
        ctx = IoUringContext(queue_depth=128)
        
        # Should not raise
        ctx.unregister_buffers()
        
        ctx.close()
    
    def test_prep_read_fixed_validation(self):
        """Test validation in prep_read_fixed."""
        ctx = IoUringContext(queue_depth=128)
        
        buffer = bytearray(4096)
        mv = memoryview(buffer)
        buffer_ids = ctx.register_buffers([mv])
        buffer_id = buffer_ids[0]
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x" * 1024)
            temp_file = f.name
        
        try:
            fd = os.open(temp_file, os.O_RDONLY)
            
            # Valid read should not raise
            user_data = ctx.prep_read_fixed(
                fd=fd,
                buffer_id=buffer_id,
                offset=0,
                length=1024,
                file_offset=0
            )
            assert user_data > 0
            
            # Invalid buffer ID should raise
            with pytest.raises(ValueError, match="not registered"):
                ctx.prep_read_fixed(
                    fd=fd,
                    buffer_id=999,
                    offset=0,
                    length=1024,
                    file_offset=0
                )
            
            # Read exceeding buffer size should raise
            with pytest.raises(ValueError, match="exceed buffer size"):
                ctx.prep_read_fixed(
                    fd=fd,
                    buffer_id=buffer_id,
                    offset=0,
                    length=8192,  # Larger than 4096 buffer
                    file_offset=0
                )
            
            os.close(fd)
        finally:
            os.unlink(temp_file)
            ctx.close()
    
    def test_close_unregisters_buffers(self):
        """Test that close() automatically unregisters buffers."""
        ctx = IoUringContext(queue_depth=128)
        
        buffer = bytearray(4096)
        mv = memoryview(buffer)
        ctx.register_buffers([mv])
        
        assert len(ctx.registered_buffers) == 1
        
        # Close should unregister
        ctx.close()
        
        # Verify cleanup (check attributes exist first)
        if hasattr(ctx, 'registered_buffers'):
            assert len(ctx.registered_buffers) == 0


@pytest.mark.usefixtures("check_buffer_registration_support")
class TestPinnedBufferPoolRegistration:
    """Test PinnedBufferPool integration with buffer registration."""
    
    def test_pinned_buffer_pool_creation(self):
        """Test creating a pinned buffer pool."""
        pool = PinnedBufferPool(buffer_size_mb=1, num_buffers=2)
        
        assert pool.buffer_size == 1 * 1024 * 1024
        assert pool.num_buffers == 2
        assert len(pool.buffers) == 2
        
        # Verify buffers are pinned
        for buf in pool.buffers:
            assert buf.is_pinned()
    
    def test_acquire_and_release(self):
        """Test acquiring and releasing buffers."""
        pool = PinnedBufferPool(buffer_size_mb=1, num_buffers=2)
        
        # Acquire buffer
        buffer1 = pool.acquire(timeout=1.0)
        assert buffer1 is not None
        assert buffer1.size == 1 * 1024 * 1024
        
        # Acquire second buffer
        buffer2 = pool.acquire(timeout=1.0)
        assert buffer2 is not None
        
        # Pool should be exhausted
        with pytest.raises(TimeoutError):
            pool.acquire(timeout=0.1)
        
        # Release and re-acquire
        pool.release(buffer1)
        buffer3 = pool.acquire(timeout=1.0)
        assert buffer3 is not None


@pytest.mark.skipif(not IOURING_AVAILABLE, reason="io_uring not available")
@pytest.mark.usefixtures("check_buffer_registration_support")
class TestZeroCopyFileRead:
    """Test zero-copy file reading with registered buffers."""
    
    def test_zero_copy_read_small_file(self):
        """Test zero-copy read of a small file."""
        # Create test file
        test_data = b"Hello, zero-copy world!" * 100
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(test_data)
            temp_file = f.name
        
        try:
            # Setup io_uring with registered buffer
            ctx = IoUringContext(queue_depth=128)
            buffer = bytearray(len(test_data))
            mv = memoryview(buffer)
            buffer_ids = ctx.register_buffers([mv])
            
            # Open file and read with zero-copy
            fd = os.open(temp_file, os.O_RDONLY)
            
            user_data = ctx.prep_read_fixed(
                fd=fd,
                buffer_id=buffer_ids[0],
                offset=0,
                length=len(test_data),
                file_offset=0
            )
            
            # Submit and wait for completion
            ctx.submit()
            cqe = ctx.wait_cqe()
            
            # Verify read succeeded
            assert cqe.res == len(test_data)
            assert bytes(buffer) == test_data
            
            ctx.cqe_seen(cqe)
            os.close(fd)
            ctx.close()
        finally:
            os.unlink(temp_file)
    
    def test_zero_copy_read_large_file(self):
        """Test zero-copy read of a larger file (1MB)."""
        # Create 1MB test file
        test_data = os.urandom(1024 * 1024)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(test_data)
            temp_file = f.name
        
        try:
            # Setup io_uring with 2MB buffer (to hold the file)
            ctx = IoUringContext(queue_depth=128)
            buffer = bytearray(2 * 1024 * 1024)
            mv = memoryview(buffer)
            buffer_ids = ctx.register_buffers([mv])
            
            # Open file and read with zero-copy
            fd = os.open(temp_file, os.O_RDONLY)
            
            user_data = ctx.prep_read_fixed(
                fd=fd,
                buffer_id=buffer_ids[0],
                offset=0,
                length=len(test_data),
                file_offset=0
            )
            
            # Submit and wait for completion
            ctx.submit()
            cqe = ctx.wait_cqe()
            
            # Verify read succeeded
            assert cqe.res == len(test_data)
            assert bytes(buffer[:len(test_data)]) == test_data
            
            ctx.cqe_seen(cqe)
            os.close(fd)
            ctx.close()
        finally:
            os.unlink(temp_file)

# Made with Bob
