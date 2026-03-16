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
Tests for io_driver parameter and auto-selection logic.
"""

import pytest
import platform
from unittest.mock import Mock, patch, MagicMock
from llmd_s3_backend.spec import S3OffloadingSpec


def create_mock_vllm_config(extra_config: dict):
    """Create a mock VllmConfig with the given extra_config."""
    config = Mock()
    config.model_config.model = "test-model"
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.rank = 0
    config.cache_config.cache_dtype = "float16"
    config.cache_config.block_size = 16
    config.cache_config.num_gpu_blocks = 1000
    
    # Mock kv_transfer_config to return extra_config properly
    config.kv_transfer_config = Mock()
    config.kv_transfer_config.kv_connector_extra_config = extra_config
    
    return config


@pytest.fixture
def base_extra_config():
    """Base configuration for S3OffloadingSpec."""
    return {
        "s3_bucket": "test-bucket",
        "s3_prefix": "test-prefix",
        "s3_region": "us-east-1",
        "block_size": 256,
    }


class TestIoDriverValidation:
    """Test io_driver parameter validation."""
    
    def test_valid_drivers(self, base_extra_config):
        """Test that all valid driver values are accepted."""
        valid_drivers = ["auto", "crt", "io_uring", "cuobject"]
        
        for driver in valid_drivers:
            config = base_extra_config.copy()
            config["io_driver"] = driver
            mock_config = create_mock_vllm_config(config)
            
            if driver == "cuobject":
                # cuobject should raise NotImplementedError during auto-selection
                with pytest.raises(NotImplementedError, match="cuobject driver is not yet implemented"):
                    S3OffloadingSpec(mock_config, None)
            else:
                # Other drivers should work
                spec = S3OffloadingSpec(mock_config, None)
                assert spec.io_driver == driver
    
    def test_invalid_driver(self, base_extra_config):
        """Test that invalid driver values raise ValueError."""
        config = base_extra_config.copy()
        config["io_driver"] = "invalid_driver"
        mock_config = create_mock_vllm_config(config)
        
        with pytest.raises(ValueError, match="Invalid io_driver 'invalid_driver'"):
            S3OffloadingSpec(mock_config, None)
    
    def test_default_driver(self, base_extra_config):
        """Test that default driver is 'auto'."""
        mock_config = create_mock_vllm_config(base_extra_config)
        spec = S3OffloadingSpec(mock_config, None)
        assert spec.io_driver == "auto"


class TestAutoSelection:
    """Test io_driver auto-selection logic."""
    
    @patch('platform.system')
    @patch('llmd_s3_backend.iouring_ops.IoUringContext')
    def test_auto_selects_iouring_on_linux(self, mock_ctx_class, mock_platform,
                                           base_extra_config):
        """Test that auto selects io_uring on Linux when available."""
        # Mock Linux system
        mock_platform.return_value = "Linux"
        
        # Mock successful io_uring context creation
        mock_ctx = Mock()
        mock_ctx_class.return_value = mock_ctx
        
        config = base_extra_config.copy()
        config["io_driver"] = "auto"
        mock_config = create_mock_vllm_config(config)
        
        spec = S3OffloadingSpec(mock_config, None)
        
        assert spec.io_driver == "auto"
        assert spec.resolved_io_driver == "io_uring"
        mock_ctx.close.assert_called_once()
    
    @patch('platform.system')
    def test_auto_selects_crt_on_non_linux(self, mock_platform,
                                           base_extra_config):
        """Test that auto selects crt on non-Linux systems."""
        # Mock non-Linux system
        mock_platform.return_value = "Darwin"  # macOS
        
        config = base_extra_config.copy()
        config["io_driver"] = "auto"
        mock_config = create_mock_vllm_config(config)
        
        spec = S3OffloadingSpec(mock_config, None)
        
        assert spec.io_driver == "auto"
        assert spec.resolved_io_driver == "crt"
    
    @patch('platform.system')
    @patch('llmd_s3_backend.iouring_ops.IoUringContext')
    def test_auto_falls_back_to_crt_on_iouring_error(self, mock_ctx_class, mock_platform,
                                                      base_extra_config):
        """Test that auto falls back to crt if io_uring initialization fails."""
        # Mock Linux system
        mock_platform.return_value = "Linux"
        
        # Mock io_uring context creation failure
        mock_ctx_class.side_effect = Exception("io_uring not supported")
        
        config = base_extra_config.copy()
        config["io_driver"] = "auto"
        mock_config = create_mock_vllm_config(config)
        
        spec = S3OffloadingSpec(mock_config, None)
        
        assert spec.io_driver == "auto"
        assert spec.resolved_io_driver == "crt"
    
    @patch('platform.system')
    def test_auto_falls_back_to_crt_on_import_error(self, mock_platform,
                                                     base_extra_config):
        """Test that auto falls back to crt if io_uring modules not available."""
        # Mock Linux system
        mock_platform.return_value = "Linux"
        
        # Patch the import to raise ImportError
        with patch.dict('sys.modules', {'llmd_s3_backend.iouring_ops': None}):
            config = base_extra_config.copy()
            config["io_driver"] = "auto"
            mock_config = create_mock_vllm_config(config)
            
            spec = S3OffloadingSpec(mock_config, None)
            
            assert spec.io_driver == "auto"
            assert spec.resolved_io_driver == "crt"


class TestExplicitDriverSelection:
    """Test explicit io_driver selection."""
    
    def test_explicit_crt_selection(self, base_extra_config):
        """Test explicit crt driver selection."""
        config = base_extra_config.copy()
        config["io_driver"] = "crt"
        mock_config = create_mock_vllm_config(config)
        
        spec = S3OffloadingSpec(mock_config, None)
        
        assert spec.io_driver == "crt"
        assert spec.resolved_io_driver == "crt"
    
    @patch('platform.system')
    def test_explicit_iouring_selection(self, mock_platform, base_extra_config):
        """Test explicit io_uring driver selection."""
        # Mock Linux system
        mock_platform.return_value = "Linux"
        
        config = base_extra_config.copy()
        config["io_driver"] = "io_uring"
        mock_config = create_mock_vllm_config(config)
        
        spec = S3OffloadingSpec(mock_config, None)
        
        assert spec.io_driver == "io_uring"
        assert spec.resolved_io_driver == "io_uring"
    
    def test_explicit_cuobject_raises_not_implemented(self, base_extra_config):
        """Test that explicit cuobject selection raises NotImplementedError."""
        config = base_extra_config.copy()
        config["io_driver"] = "cuobject"
        mock_config = create_mock_vllm_config(config)
        
        with pytest.raises(NotImplementedError, match="cuobject driver is not yet implemented"):
            S3OffloadingSpec(mock_config, None)


class TestIoUringConfiguration:
    """Test io_uring configuration parameters."""
    
    def test_default_iouring_config(self, base_extra_config):
        """Test default io_uring configuration values."""
        mock_config = create_mock_vllm_config(base_extra_config)
        spec = S3OffloadingSpec(mock_config, None)
        
        assert spec.iouring_queue_depth == 1024
        assert spec.iouring_num_workers == 16
        assert spec.pinned_buffer_size_mb == 128
        assert spec.pinned_buffer_pool_size == 64
    
    def test_custom_iouring_config(self, base_extra_config):
        """Test custom io_uring configuration values."""
        config = base_extra_config.copy()
        config.update({
            "iouring_queue_depth": 2048,
            "iouring_num_workers": 32,
            "pinned_buffer_size_mb": 256,
            "pinned_buffer_pool_size": 128,
        })
        mock_config = create_mock_vllm_config(config)
        
        spec = S3OffloadingSpec(mock_config, None)
        
        assert spec.iouring_queue_depth == 2048
        assert spec.iouring_num_workers == 32
        assert spec.pinned_buffer_size_mb == 256
        assert spec.pinned_buffer_pool_size == 128


class TestRealSystemAutoSelection:
    """Test auto-selection on the actual system (integration test)."""
    
    def test_auto_selection_on_current_system(self, base_extra_config):
        """Test auto-selection logic on the current system."""
        config = base_extra_config.copy()
        config["io_driver"] = "auto"
        mock_config = create_mock_vllm_config(config)
        
        spec = S3OffloadingSpec(mock_config, None)
        
        # Verify the selection is valid
        assert spec.resolved_io_driver in ["crt", "io_uring"]
        
        # On Linux, should try io_uring first
        if platform.system() == "Linux":
            # If io_uring modules are available, should select io_uring
            try:
                from llmd_s3_backend.iouring_ops import IoUringContext
                # If we can import, check if kernel supports it
                try:
                    ctx = IoUringContext(queue_depth=2)
                    ctx.close()
                    # Kernel supports io_uring
                    assert spec.resolved_io_driver == "io_uring"
                except Exception:
                    # Kernel doesn't support io_uring (e.g., Podman VM)
                    assert spec.resolved_io_driver == "crt"
            except ImportError:
                # io_uring modules not available
                assert spec.resolved_io_driver == "crt"
        else:
            # On non-Linux, should always select crt
            assert spec.resolved_io_driver == "crt"
    
    def test_explicit_crt_always_works(self, base_extra_config):
        """Test that explicit crt selection always works."""
        config = base_extra_config.copy()
        config["io_driver"] = "crt"
        mock_config = create_mock_vllm_config(config)
        
        spec = S3OffloadingSpec(mock_config, None)
        
        assert spec.resolved_io_driver == "crt"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
