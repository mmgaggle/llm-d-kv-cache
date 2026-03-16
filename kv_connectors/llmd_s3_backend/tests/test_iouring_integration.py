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
Integration tests for io_uring zero-copy path with worker.py.

These tests verify the complete integration of io_uring with the
S3GPUOffloadingHandler, including:
- Configuration parsing
- Pool initialization
- Zero-copy downloads
- Fallback to CRT
- Error handling
- Resource cleanup

NOTE: These tests require Linux with io_uring support to run the
zero-copy path. On other platforms, they verify fallback behavior.
"""

import pytest
import sys
import io
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List

# Check if we're on Linux for io_uring tests
IS_LINUX = sys.platform.startswith('linux')

# Try to import io_uring components
try:
    from llmd_s3_backend.iouring_pool import IoUringPool, IoUringConfig
    from llmd_s3_backend.pinned_buffers import PinnedBufferPool, PinnedBuffer
    from llmd_s3_backend.s3_auth import S3SigV4Signer
    IOURING_AVAILABLE = True
except ImportError:
    IOURING_AVAILABLE = False
    IoUringPool = None
    PinnedBufferPool = None
    IoUringConfig = None
    S3SigV4Signer = None


class TestIoUringConfiguration:
    """Test io_uring configuration in spec.py."""
    
    def test_default_configuration(self):
        """Test default io_uring configuration values."""
        from llmd_s3_backend.spec import S3OffloadingSpec
        
        # Create spec with minimal config
        config = {
            "s3_bucket": "test-bucket",
            "block_size": 256,
        }
        
        spec = S3OffloadingSpec(config)
        
        # Verify defaults
        assert spec.enable_iouring == False
        assert spec.iouring_queue_depth == 1024
        assert spec.iouring_num_workers == 16
        assert spec.pinned_buffer_size_mb == 128
        assert spec.pinned_buffer_pool_size == 64
    
    def test_custom_configuration(self):
        """Test custom io_uring configuration."""
        from llmd_s3_backend.spec import S3OffloadingSpec
        
        config = {
            "s3_bucket": "test-bucket",
            "block_size": 256,
            "enable_iouring": True,
            "iouring_queue_depth": 2048,
            "iouring_num_workers": 32,
            "pinned_buffer_size_mb": 256,
            "pinned_buffer_pool_size": 128,
        }
        
        spec = S3OffloadingSpec(config)
        
        assert spec.enable_iouring == True
        assert spec.iouring_queue_depth == 2048
        assert spec.iouring_num_workers == 32
        assert spec.pinned_buffer_size_mb == 256
        assert spec.pinned_buffer_pool_size == 128


class TestWorkerInitialization:
    """Test S3GPUOffloadingHandler initialization with io_uring."""
    
    @pytest.fixture
    def mock_torch(self):
        """Mock torch module."""
        with patch('llmd_s3_backend.worker.torch') as mock:
            mock.dtype = Mock()
            mock.float16 = Mock()
            mock.cuda = Mock()
            mock.cuda.Stream = Mock(return_value=Mock())
            yield mock
    
    @pytest.fixture
    def mock_kv_caches(self, mock_torch):
        """Mock KV cache tensors."""
        mock_tensor = Mock()
        mock_tensor.shape = (2, 1000, 32, 128, 64)  # (kv, blocks, heads, block_size, head_dim)
        mock_tensor.device = Mock()
        mock_tensor.dtype = mock_torch.float16
        
        return {
            "layer_0": mock_tensor,
            "layer_1": mock_tensor,
        }
    
    @pytest.fixture
    def mock_attn_backends(self):
        """Mock attention backends."""
        mock_backend = Mock()
        mock_backend.get_kv_cache_shape = Mock(
            return_value=(2, 1234, 32, 128, 64)
        )
        
        return {
            "layer_0": mock_backend,
            "layer_1": mock_backend,
        }
    
    def test_initialization_iouring_disabled(
        self, mock_torch, mock_kv_caches, mock_attn_backends
    ):
        """Test initialization with io_uring disabled."""
        from llmd_s3_backend.worker import S3GPUOffloadingHandler
        
        with patch('llmd_s3_backend.worker.S3ClientWrapper'):
            handler = S3GPUOffloadingHandler(
                model_name="test-model",
                tp_size=1,
                tp_rank=0,
                dtype=mock_torch.float16,
                gpu_blocks_per_file=256,
                kv_caches=mock_kv_caches,
                attn_backends=mock_attn_backends,
                bucket="test-bucket",
                enable_iouring=False,
            )
            
            assert handler.enable_iouring == False
            assert handler.iouring_pool is None
            assert handler.pinned_buffer_pool is None
    
    @pytest.mark.skipif(not IS_LINUX, reason="io_uring only available on Linux")
    @pytest.mark.skipif(not IOURING_AVAILABLE, reason="io_uring components not available")
    def test_initialization_iouring_enabled_linux(
        self, mock_torch, mock_kv_caches, mock_attn_backends
    ):
        """Test initialization with io_uring enabled on Linux."""
        from llmd_s3_backend.worker import S3GPUOffloadingHandler
        
        with patch('llmd_s3_backend.worker.S3ClientWrapper'), \
             patch('llmd_s3_backend.worker.boto3') as mock_boto3, \
             patch('llmd_s3_backend.worker.PinnedBufferPool') as mock_pool, \
             patch('llmd_s3_backend.worker.IoUringPool') as mock_iouring:
            
            # Mock boto3 credentials
            mock_session = Mock()
            mock_creds = Mock()
            mock_creds.access_key = "test_key"
            mock_creds.secret_key = "test_secret"
            mock_creds.token = None  # No session token
            mock_session.get_credentials = Mock(return_value=mock_creds)
            mock_boto3.Session = Mock(return_value=mock_session)
            
            handler = S3GPUOffloadingHandler(
                model_name="test-model",
                tp_size=1,
                tp_rank=0,
                dtype=mock_torch.float16,
                gpu_blocks_per_file=256,
                kv_caches=mock_kv_caches,
                attn_backends=mock_attn_backends,
                bucket="test-bucket",
                enable_iouring=True,
                iouring_queue_depth=1024,
                iouring_num_workers=16,
                pinned_buffer_size_mb=128,
                pinned_buffer_pool_size=64,
            )
            
            # Verify io_uring was initialized
            assert handler.enable_iouring == True
            mock_pool.assert_called_once()
            mock_iouring.assert_called_once()
    
    def test_initialization_iouring_fallback_non_linux(
        self, mock_torch, mock_kv_caches, mock_attn_backends
    ):
        """Test that io_uring falls back to CRT on non-Linux."""
        from llmd_s3_backend.worker import S3GPUOffloadingHandler
        
        with patch('llmd_s3_backend.worker.S3ClientWrapper'), \
             patch('llmd_s3_backend.worker.IOURING_AVAILABLE', False):
            
            handler = S3GPUOffloadingHandler(
                model_name="test-model",
                tp_size=1,
                tp_rank=0,
                dtype=mock_torch.float16,
                gpu_blocks_per_file=256,
                kv_caches=mock_kv_caches,
                attn_backends=mock_attn_backends,
                bucket="test-bucket",
                enable_iouring=True,  # Request io_uring
            )
            
            # Should fall back to CRT
            assert handler.enable_iouring == False
            assert handler.iouring_pool is None


class TestZeroCopyDownload:
    """Test zero-copy download path."""
    
    @pytest.fixture
    def mock_handler(self):
        """Create mock handler with io_uring enabled."""
        handler = Mock()
        handler.enable_iouring = True
        handler.iouring_pool = Mock()
        handler.pinned_buffer_pool = Mock()
        handler.s3_client = Mock()
        return handler
    
    @pytest.mark.skipif(not IOURING_AVAILABLE, reason="io_uring components not available")
    def test_zerocopy_success(self, mock_handler):
        """Test successful zero-copy download."""
        from llmd_s3_backend.worker import S3GPUOffloadingHandler
        
        # Create test data
        test_data = np.random.rand(2, 10, 32, 128, 64).astype(np.float16)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, test_data)
        test_bytes = buffer.getvalue()
        
        # Mock pinned buffer
        mock_buffer = Mock()
        mock_buffer.size_bytes = len(test_bytes) * 2
        mock_buffer.tensor = Mock()
        mock_buffer.tensor.__getitem__ = Mock(return_value=Mock())
        mock_buffer.tensor.__getitem__.return_value.cpu = Mock(return_value=Mock())
        mock_buffer.tensor.__getitem__.return_value.cpu.return_value.numpy = Mock(
            return_value=np.frombuffer(test_bytes, dtype=np.uint8)
        )
        
        mock_handler.pinned_buffer_pool.acquire = Mock(return_value=mock_buffer)
        mock_handler.iouring_pool.get_object_zerocopy = Mock(return_value=len(test_bytes))
        
        # Mock tensors
        mock_tensors = [Mock(), Mock()]
        for tensor in mock_tensors:
            tensor.device = Mock()
            tensor.dtype = Mock()
        
        # Call the method
        result = S3GPUOffloadingHandler._get_blocks_zerocopy(
            mock_handler,
            job_id=1,
            s3_key="test/key.bin",
            tensors=mock_tensors,
            block_ids=[0, 1, 2]
        )
        
        # Verify success
        assert result == (1, True)
        mock_handler.pinned_buffer_pool.acquire.assert_called_once()
        mock_handler.iouring_pool.get_object_zerocopy.assert_called_once()
        mock_handler.pinned_buffer_pool.release.assert_called_once()
    
    def test_zerocopy_fallback_on_error(self, mock_handler):
        """Test fallback to CRT when zero-copy fails."""
        from llmd_s3_backend.worker import S3GPUOffloadingHandler
        
        # Make zero-copy fail
        mock_handler.pinned_buffer_pool.acquire = Mock(
            side_effect=RuntimeError("Buffer pool exhausted")
        )
        
        # Mock CRT path
        test_data = np.random.rand(2, 10, 32, 128, 64).astype(np.float16)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, test_data)
        mock_handler.s3_client.get_object = Mock(return_value=buffer.getvalue())
        
        mock_tensors = [Mock(), Mock()]
        for tensor in mock_tensors:
            tensor.device = Mock()
            tensor.dtype = Mock()
            tensor.__setitem__ = Mock()
        
        # Call the method - should fall back to CRT
        with patch('llmd_s3_backend.worker.torch'):
            result = S3GPUOffloadingHandler._get_blocks_from_s3(
                mock_handler,
                job_id=1,
                s3_key="test/key.bin",
                tensors=mock_tensors,
                block_ids=[0, 1, 2]
            )
        
        # Should succeed via CRT fallback
        assert result == (1, True)
        mock_handler.s3_client.get_object.assert_called_once()


class TestMultipathing:
    """Test multipathing support."""
    
    @pytest.mark.skipif(not IOURING_AVAILABLE, reason="io_uring components not available")
    def test_multipath_endpoint_parsing(self):
        """Test parsing of comma-separated endpoints."""
        from llmd_s3_backend.worker import S3GPUOffloadingHandler
        
        with patch('llmd_s3_backend.worker.S3ClientWrapper'), \
             patch('llmd_s3_backend.worker.boto3') as mock_boto3, \
             patch('llmd_s3_backend.worker.torch'), \
             patch('llmd_s3_backend.worker.PinnedBufferPool'), \
             patch('llmd_s3_backend.worker.IoUringPool') as mock_iouring:
            
            # Mock credentials
            mock_session = Mock()
            mock_creds = Mock()
            mock_creds.access_key = "test"
            mock_creds.secret_key = "test"
            mock_creds.token = None
            mock_session.get_credentials = Mock(return_value=mock_creds)
            mock_boto3.Session = Mock(return_value=mock_session)
            
            handler = S3GPUOffloadingHandler(
                model_name="test",
                tp_size=1,
                tp_rank=0,
                dtype=Mock(),
                gpu_blocks_per_file=256,
                kv_caches={"layer_0": Mock()},
                attn_backends={"layer_0": Mock()},
                bucket="test-bucket",
                endpoint_url="10.0.1.5:443,10.0.1.6:443,10.0.1.7:443",
                enable_iouring=True,
            )
            
            # Verify endpoints were parsed
            call_args = mock_iouring.call_args
            endpoints = call_args[1]['endpoints']
            assert len(endpoints) == 3
            assert "10.0.1.5:443" in endpoints
            assert "10.0.1.6:443" in endpoints
            assert "10.0.1.7:443" in endpoints


class TestResourceCleanup:
    """Test proper resource cleanup."""
    
    @pytest.mark.skipif(not IOURING_AVAILABLE, reason="io_uring components not available")
    def test_cleanup_on_delete(self):
        """Test that resources are cleaned up on handler deletion."""
        from llmd_s3_backend.worker import S3GPUOffloadingHandler
        
        with patch('llmd_s3_backend.worker.S3ClientWrapper'), \
             patch('llmd_s3_backend.worker.boto3') as mock_boto3, \
             patch('llmd_s3_backend.worker.torch'), \
             patch('llmd_s3_backend.worker.PinnedBufferPool') as mock_pool, \
             patch('llmd_s3_backend.worker.IoUringPool') as mock_iouring:
            
            # Mock credentials
            mock_session = Mock()
            mock_creds = Mock()
            mock_creds.access_key = "test"
            mock_creds.secret_key = "test"
            mock_creds.token = None
            mock_session.get_credentials = Mock(return_value=mock_creds)
            mock_boto3.Session = Mock(return_value=mock_session)
            
            # Create mock pool instance
            mock_pool_instance = Mock()
            mock_pool.return_value = mock_pool_instance
            
            mock_iouring_instance = Mock()
            mock_iouring.return_value = mock_iouring_instance
            
            handler = S3GPUOffloadingHandler(
                model_name="test",
                tp_size=1,
                tp_rank=0,
                dtype=Mock(),
                gpu_blocks_per_file=256,
                kv_caches={"layer_0": Mock()},
                attn_backends={"layer_0": Mock()},
                bucket="test-bucket",
                enable_iouring=True,
            )
            
            # Delete handler
            del handler
            
            # Verify cleanup was called
            mock_iouring_instance.close.assert_called_once()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
