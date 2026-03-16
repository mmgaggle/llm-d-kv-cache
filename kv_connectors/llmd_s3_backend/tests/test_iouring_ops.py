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
Tests for io_uring operations.

These tests verify the io_uring integration works correctly.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llmd_s3_backend.iouring_ops import (
    IOURING_AVAILABLE,
    IoUringContext,
    IoUringSocket,
    OpType,
    CompletionEvent,
)


@pytest.mark.skipif(not IOURING_AVAILABLE, reason="liburing not available")
def test_iouring_context_creation():
    """Test creating an io_uring context."""
    ctx = IoUringContext(queue_depth=32)
    assert ctx.queue_depth == 32
    assert ctx.ring is not None
    assert ctx.stats.submissions == 0
    assert ctx.stats.completions == 0
    ctx.close()


@pytest.mark.skipif(not IOURING_AVAILABLE, reason="liburing not available")
def test_iouring_user_data_allocation():
    """Test user_data allocation."""
    ctx = IoUringContext(queue_depth=32)
    
    # Allocate user data for different operation types
    ud1 = ctx.allocate_user_data(OpType.CONNECT)
    ud2 = ctx.allocate_user_data(OpType.SEND)
    ud3 = ctx.allocate_user_data(OpType.RECV)
    
    # Verify they're unique
    assert ud1 != ud2
    assert ud2 != ud3
    assert ud1 != ud3
    
    # Verify op_type is encoded
    assert (ud1 & 0xFF) == OpType.CONNECT
    assert (ud2 & 0xFF) == OpType.SEND
    assert (ud3 & 0xFF) == OpType.RECV
    
    ctx.close()


@pytest.mark.skipif(not IOURING_AVAILABLE, reason="liburing not available")
def test_iouring_stats():
    """Test statistics tracking."""
    ctx = IoUringContext(queue_depth=32)
    
    # Initial stats
    assert ctx.stats.submissions == 0
    assert ctx.stats.completions == 0
    assert ctx.stats.errors == 0
    assert ctx.stats.bytes_sent == 0
    assert ctx.stats.bytes_received == 0
    
    ctx.close()


@pytest.mark.skipif(not IOURING_AVAILABLE, reason="liburing not available")
def test_completion_event():
    """Test CompletionEvent dataclass."""
    # Success event
    event = CompletionEvent(
        op_type=OpType.SEND,
        result=1024,
        user_data=0x100 | OpType.SEND
    )
    assert event.success
    assert event.error_code == 0
    assert event.op_type == OpType.SEND
    
    # Error event
    error_event = CompletionEvent(
        op_type=OpType.RECV,
        result=-5,  # -EAGAIN
        user_data=0x200 | OpType.RECV
    )
    assert not error_event.success
    assert error_event.error_code == 5


@pytest.mark.skipif(not IOURING_AVAILABLE, reason="liburing not available")
def test_iouring_socket_creation():
    """Test IoUringSocket creation."""
    ctx = IoUringContext(queue_depth=32)
    sock = IoUringSocket(ctx)
    
    assert sock.ctx is ctx
    assert sock.fd is None
    assert not sock.connected
    assert len(sock._pending_ops) == 0
    
    sock.close()
    ctx.close()


def test_iouring_availability():
    """Test that we can detect io_uring availability."""
    # This test should pass on both Linux and macOS
    if sys.platform.startswith('linux'):
        # On Linux, liburing should be available in the container
        assert IOURING_AVAILABLE, "liburing should be available on Linux"
    else:
        # On macOS, liburing won't be available
        assert not IOURING_AVAILABLE, "liburing should not be available on macOS"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
