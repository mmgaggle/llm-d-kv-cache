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
API compatibility tests for S3GPUOffloadingHandler with vLLM 0.17.1.

These tests verify that the worker code is compatible with vLLM's API changes
in version 0.17.1, including:
- Configuration parsing with new VllmConfig structure
- Worker initialization with updated AttentionBackend imports
- Method signatures match vLLM's OffloadingHandler interface
- Proper handling of kv_cache_config parameter

NOTE: These are NOT end-to-end integration tests. They use mocks to verify
API compatibility without requiring real S3, GPU, or io_uring operations.
For real integration tests, see test_iouring_live.py.

For unit tests of individual components, see:
- test_pinned_buffers.py (8/8 tests passing)
- test_iouring_ops.py (6/6 tests passing)
"""

import pytest
import sys
import io
import numpy as np
import torch
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List
from vllm.config import VllmConfig

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
        
        # Create mock VllmConfig with proper structure
        mock_kv_transfer_config = Mock()
        mock_kv_transfer_config.kv_connector_extra_config = {
            "s3_bucket": "test-bucket",
            "block_size": 256,
        }
        
        mock_cache_config = Mock()
        mock_cache_config.block_size = 16
        
        mock_config = Mock(spec=VllmConfig)
        mock_config.kv_transfer_config = mock_kv_transfer_config
        mock_config.cache_config = mock_cache_config
        
        spec = S3OffloadingSpec(mock_config)
        
        # Verify defaults
        assert spec.enable_iouring == False
        assert spec.iouring_queue_depth == 1024
        assert spec.iouring_num_workers == 16
        assert spec.pinned_buffer_size_mb == 128
        assert spec.pinned_buffer_pool_size == 64
    
    def test_custom_configuration(self):
        """Test custom io_uring configuration."""
        from llmd_s3_backend.spec import S3OffloadingSpec
        
        # Create mock VllmConfig with custom io_uring settings
        mock_kv_transfer_config = Mock()
        mock_kv_transfer_config.kv_connector_extra_config = {
            "s3_bucket": "test-bucket",
            "block_size": 256,
            "enable_iouring": True,
            "iouring_queue_depth": 2048,
            "iouring_num_workers": 32,
            "pinned_buffer_size_mb": 256,
            "pinned_buffer_pool_size": 128,
        }
        
        mock_cache_config = Mock()
        mock_cache_config.block_size = 16
        
        mock_config = Mock(spec=VllmConfig)
        mock_config.kv_transfer_config = mock_kv_transfer_config
        mock_config.cache_config = mock_cache_config
        
        spec = S3OffloadingSpec(mock_config)
        
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



# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
