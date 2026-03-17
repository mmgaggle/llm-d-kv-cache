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

"""Shared pytest fixtures for S3 backend unit tests.

Fixtures defined here are automatically available to all test modules
in this directory.  For importable helper functions (e.g. tensor
builders, hash generators) see :mod:`helpers`.
"""

import pytest
from unittest.mock import Mock, MagicMock


@pytest.fixture
def mock_s3_client():
    """A MagicMock standing in for :class:`S3ClientWrapper`."""
    client = MagicMock()
    client.put_object = MagicMock()
    client.get_object = MagicMock()
    return client


@pytest.fixture
def mock_attn_backends():
    """Mock attention backends reporting shape ``(2, N, 8, 16, 256)``."""
    backend = Mock()
    backend.get_kv_cache_shape = Mock(return_value=(2, 1234, 8, 16, 256))
    return {"layer_0": backend, "layer_1": backend}


@pytest.fixture
def mock_cuda_stream():
    """Mock CUDA stream object."""
    return Mock()
