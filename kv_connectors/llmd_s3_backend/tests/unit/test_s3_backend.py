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

import pytest
import torch
import struct
import hashlib
from unittest.mock import Mock, patch, MagicMock
from vllm.v1.core.kv_cache_utils import BlockHash

from llmd_s3_backend.manager import S3OffloadingManager
from llmd_s3_backend.mediums import S3LoadStoreSpec
from llmd_s3_backend.s3_client import S3ClientWrapper


# ----------------------------
# Helper functions
# ----------------------------
def get_prefix_hash(token_ids):
    """Generate a stable 64-bit hash for a list of token IDs."""
    buf = bytearray()
    for t in token_ids:
        buf += struct.pack("<I", int(t) & 0xFFFFFFFF)
    digest_int = int.from_bytes(hashlib.sha256(buf).digest()[:8], "big")
    return BlockHash((digest_int & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little"))


# ----------------------------
# Tests
# ----------------------------
class TestS3LoadStoreSpec:
    """Test S3LoadStoreSpec class."""

    def test_init(self):
        """Test initialization with block hashes."""
        hashes = [get_prefix_hash(range(100, 117)), get_prefix_hash(range(200, 217))]
        spec = S3LoadStoreSpec(hashes)
        assert len(spec.block_hashes) == 2
        assert spec.block_hashes == hashes

    def test_medium(self):
        """Test medium identifier."""
        assert S3LoadStoreSpec.medium() == "S3"

    def test_repr(self):
        """Test string representation."""
        hashes = [get_prefix_hash(range(100, 117))]
        spec = S3LoadStoreSpec(hashes)
        assert repr(spec) == repr(hashes)


class TestS3ClientWrapper:
    """Test S3ClientWrapper class."""

    @patch("llmd_s3_backend.s3_client.boto3.Session")
    def test_init_default(self, mock_session):
        """Test initialization with default parameters."""
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        client = S3ClientWrapper(bucket="test-bucket")

        assert client.bucket == "test-bucket"
        assert client.region == "us-east-1"
        assert client.addressing_style == "auto"
        mock_session.assert_called_once()

    @patch("llmd_s3_backend.s3_client.boto3.Session")
    def test_init_custom(self, mock_session):
        """Test initialization with custom parameters."""
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        client = S3ClientWrapper(
            bucket="test-bucket",
            region="us-west-2",
            endpoint_url="http://localhost:9000",
            addressing_style="path",
            profile_name="test-profile",
        )

        assert client.bucket == "test-bucket"
        assert client.region == "us-west-2"
        assert client.endpoint_url == "http://localhost:9000"
        assert client.addressing_style == "path"

    @patch("llmd_s3_backend.s3_client.boto3.Session")
    def test_object_exists_true(self, mock_session):
        """Test object_exists when object exists."""
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client
        mock_client.head_object.return_value = {}

        client = S3ClientWrapper(bucket="test-bucket")
        assert client.object_exists("test-key") is True
        mock_client.head_object.assert_called_once_with(
            Bucket="test-bucket", Key="test-key"
        )

    @patch("llmd_s3_backend.s3_client.boto3.Session")
    def test_object_exists_false(self, mock_session):
        """Test object_exists when object doesn't exist."""
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        # Simulate 404 error
        error = Exception()
        error.response = {"Error": {"Code": "404"}}
        mock_client.exceptions.ClientError = Exception
        mock_client.head_object.side_effect = error

        client = S3ClientWrapper(bucket="test-bucket")
        assert client.object_exists("test-key") is False

    @patch("llmd_s3_backend.s3_client.boto3.Session")
    def test_object_exists_reraises_non_404(self, mock_session):
        """Test that object_exists reraises non-404 errors."""
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        error = Exception("Access Denied")
        error.response = {"Error": {"Code": "403"}}
        mock_client.exceptions.ClientError = Exception
        mock_client.head_object.side_effect = error

        client = S3ClientWrapper(bucket="test-bucket")
        with pytest.raises(Exception, match="Access Denied"):
            client.object_exists("test-key")

    @patch("llmd_s3_backend.s3_client.boto3.Session")
    def test_put_object(self, mock_session):
        """Test uploading data to S3."""
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        client = S3ClientWrapper(bucket="test-bucket")
        client.put_object("path/to/key.bin", b"binary-data")

        mock_client.put_object.assert_called_once_with(
            Bucket="test-bucket", Key="path/to/key.bin", Body=b"binary-data"
        )

    @patch("llmd_s3_backend.s3_client.boto3.Session")
    def test_get_object(self, mock_session):
        """Test downloading data from S3."""
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        mock_body = MagicMock()
        mock_body.read.return_value = b"returned-data"
        mock_client.get_object.return_value = {"Body": mock_body}

        client = S3ClientWrapper(bucket="test-bucket")
        data = client.get_object("path/to/key.bin")

        assert data == b"returned-data"
        mock_client.get_object.assert_called_once_with(
            Bucket="test-bucket", Key="path/to/key.bin"
        )

    @patch("llmd_s3_backend.s3_client.boto3.Session")
    def test_get_object_with_etag(self, mock_session):
        """Test downloading data with ETag."""
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        mock_body = MagicMock()
        mock_body.read.return_value = b"data"
        mock_client.get_object.return_value = {
            "Body": mock_body,
            "ETag": '"abc123"',
        }

        client = S3ClientWrapper(bucket="test-bucket")
        data, etag = client.get_object_with_etag("key")

        assert data == b"data"
        assert etag == "abc123"  # Quotes stripped

    @patch("llmd_s3_backend.s3_client.boto3.Session")
    def test_put_object_if_match(self, mock_session):
        """Test conditional PUT with ETag."""
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        client = S3ClientWrapper(bucket="test-bucket")
        client.put_object_if_match("key", b"new-data", "abc123")

        mock_client.put_object.assert_called_once_with(
            Bucket="test-bucket", Key="key", Body=b"new-data", IfMatch="abc123"
        )

    @patch("llmd_s3_backend.s3_client.boto3.Session")
    def test_delete_object(self, mock_session):
        """Test deleting an object from S3."""
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        client = S3ClientWrapper(bucket="test-bucket")
        client.delete_object("path/to/key.bin")

        mock_client.delete_object.assert_called_once_with(
            Bucket="test-bucket", Key="path/to/key.bin"
        )

    @patch("llmd_s3_backend.s3_client.boto3.Session")
    def test_list_objects(self, mock_session):
        """Test listing objects with a prefix."""
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "prefix/a.bin"}, {"Key": "prefix/b.bin"}]},
            {"Contents": [{"Key": "prefix/c.bin"}]},
        ]

        client = S3ClientWrapper(bucket="test-bucket")
        keys = client.list_objects("prefix/")

        assert keys == ["prefix/a.bin", "prefix/b.bin", "prefix/c.bin"]
        mock_client.get_paginator.assert_called_once_with("list_objects_v2")
        mock_paginator.paginate.assert_called_once_with(
            Bucket="test-bucket", Prefix="prefix/"
        )

    @patch("llmd_s3_backend.s3_client.boto3.Session")
    def test_list_objects_empty(self, mock_session):
        """Test listing objects when prefix matches nothing."""
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{}]  # No 'Contents' key

        client = S3ClientWrapper(bucket="test-bucket")
        keys = client.list_objects("empty/")

        assert keys == []


class TestS3OffloadingManager:
    """Test S3OffloadingManager class."""

    @patch("llmd_s3_backend.manager.S3ClientWrapper")
    def test_init(self, mock_client_class):
        """Test manager initialization."""
        manager = S3OffloadingManager(
            model_name="test-model",
            tp_size=2,
            tp_rank=0,
            dtype=torch.float16,
            bucket="test-bucket",
            prefix="test-prefix",
            region="us-west-2",
        )

        assert manager.model_name == "test-model"
        assert manager.tp_size == 2
        assert manager.tp_rank == 0
        assert manager.dtype == torch.float16
        assert manager.bucket == "test-bucket"
        assert manager.prefix == "test-prefix"
        assert "test-prefix/test-model/tp_2/rank_0/float16" in manager.base_key

    @patch("llmd_s3_backend.manager.S3ClientWrapper")
    def test_get_s3_key(self, mock_client_class):
        """Test S3 key generation."""
        manager = S3OffloadingManager(
            model_name="test-model",
            tp_size=1,
            tp_rank=0,
            dtype=torch.float16,
            bucket="test-bucket",
        )

        block_hash = get_prefix_hash(range(100, 117))
        key = manager._get_s3_key(block_hash)

        assert key.startswith("kv-cache/test-model/tp_1/rank_0/float16/")
        assert key.endswith(".bin")

    @patch("llmd_s3_backend.manager.S3ClientWrapper")
    def test_lookup(self, mock_client_class):
        """Test lookup method."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.object_exists.side_effect = [True, True, False]

        manager = S3OffloadingManager(
            model_name="test-model",
            tp_size=1,
            tp_rank=0,
            dtype=torch.float16,
            bucket="test-bucket",
        )

        hashes = [
            get_prefix_hash(range(100, 117)),
            get_prefix_hash(range(200, 217)),
            get_prefix_hash(range(300, 317)),
        ]

        hit_count = manager.lookup(hashes)
        assert hit_count == 2

    @patch("llmd_s3_backend.manager.S3ClientWrapper")
    def test_prepare_load(self, mock_client_class):
        """Test prepare_load method."""
        manager = S3OffloadingManager(
            model_name="test-model",
            tp_size=1,
            tp_rank=0,
            dtype=torch.float16,
            bucket="test-bucket",
        )

        hashes = [get_prefix_hash(range(100, 117))]
        spec = manager.prepare_load(hashes)

        assert isinstance(spec, S3LoadStoreSpec)
        assert spec.block_hashes == hashes

    @patch("llmd_s3_backend.manager.S3ClientWrapper")
    def test_prepare_store(self, mock_client_class):
        """Test prepare_store method."""
        manager = S3OffloadingManager(
            model_name="test-model",
            tp_size=1,
            tp_rank=0,
            dtype=torch.float16,
            bucket="test-bucket",
        )

        hashes = [get_prefix_hash(range(100, 117))]
        output = manager.prepare_store(hashes)

        assert output is not None
        assert len(output.block_hashes_to_store) == 1
        assert len(output.block_hashes_evicted) == 0
        assert isinstance(output.store_spec, S3LoadStoreSpec)

    @patch("llmd_s3_backend.manager.S3ClientWrapper")
    def test_touch_is_noop(self, mock_client_class):
        """Test that touch is a no-op for S3."""
        manager = S3OffloadingManager(
            model_name="test-model",
            tp_size=1,
            tp_rank=0,
            dtype=torch.float16,
            bucket="test-bucket",
        )

        hashes = [get_prefix_hash(range(100, 117))]
        # Should not raise
        manager.touch(hashes)

    @patch("llmd_s3_backend.manager.S3ClientWrapper")
    def test_complete_load_is_noop(self, mock_client_class):
        """Test that complete_load is a no-op for S3."""
        manager = S3OffloadingManager(
            model_name="test-model",
            tp_size=1,
            tp_rank=0,
            dtype=torch.float16,
            bucket="test-bucket",
        )

        hashes = [get_prefix_hash(range(100, 117))]
        # Should not raise
        manager.complete_load(hashes)

    @patch("llmd_s3_backend.manager.S3ClientWrapper")
    def test_complete_store_updates_presence_cache(self, mock_client_class):
        """Test that complete_store adds blocks to presence cache."""
        manager = S3OffloadingManager(
            model_name="test-model",
            tp_size=1,
            tp_rank=0,
            dtype=torch.float16,
            bucket="test-bucket",
        )

        # Manually set up a presence cache (bypass manifest init)
        from llmd_s3_backend.lru_cache import LRUPresenceCache
        manager._presence_cache = LRUPresenceCache(max_size=100)

        hashes = [get_prefix_hash(range(100, 117)), get_prefix_hash(range(200, 217))]
        manager.complete_store(hashes, success=True)

        assert len(manager._presence_cache) == 2
        for h in hashes:
            assert str(h) in manager._presence_cache

    @patch("llmd_s3_backend.manager.S3ClientWrapper")
    def test_complete_store_skips_on_failure(self, mock_client_class):
        """Test that complete_store does nothing when success=False."""
        manager = S3OffloadingManager(
            model_name="test-model",
            tp_size=1,
            tp_rank=0,
            dtype=torch.float16,
            bucket="test-bucket",
        )

        from llmd_s3_backend.lru_cache import LRUPresenceCache
        manager._presence_cache = LRUPresenceCache(max_size=100)

        hashes = [get_prefix_hash(range(100, 117))]
        manager.complete_store(hashes, success=False)

        assert len(manager._presence_cache) == 0

    @patch("llmd_s3_backend.manager.S3ClientWrapper")
    def test_complete_store_no_cache(self, mock_client_class):
        """Test that complete_store works when presence cache is disabled."""
        manager = S3OffloadingManager(
            model_name="test-model",
            tp_size=1,
            tp_rank=0,
            dtype=torch.float16,
            bucket="test-bucket",
            enable_presence_cache=False,
        )

        hashes = [get_prefix_hash(range(100, 117))]
        # Should not raise
        manager.complete_store(hashes, success=True)

    @patch("llmd_s3_backend.manager.S3ClientWrapper")
    def test_complete_store_updates_manifest(self, mock_client_class):
        """Test that complete_store queues blocks in manifest manager."""
        manager = S3OffloadingManager(
            model_name="test-model",
            tp_size=1,
            tp_rank=0,
            dtype=torch.float16,
            bucket="test-bucket",
        )

        mock_manifest = MagicMock()
        manager._manifest_manager = mock_manifest

        hashes = [get_prefix_hash(range(100, 117))]
        manager.complete_store(hashes, success=True)

        mock_manifest.queue_add_blocks.assert_called_once()
        call_args = mock_manifest.queue_add_blocks.call_args
        assert call_args[0][0] == hashes

    @patch("llmd_s3_backend.manager.S3ClientWrapper")
    def test_invalidate_cache_entry(self, mock_client_class):
        """Test lazy invalidation of a presence cache entry."""
        manager = S3OffloadingManager(
            model_name="test-model",
            tp_size=1,
            tp_rank=0,
            dtype=torch.float16,
            bucket="test-bucket",
        )

        from llmd_s3_backend.lru_cache import LRUPresenceCache
        manager._presence_cache = LRUPresenceCache(max_size=100)

        # Add an entry, then invalidate it
        manager._presence_cache.add("block_abc")
        assert "block_abc" in manager._presence_cache

        manager.invalidate_cache_entry("block_abc")
        assert "block_abc" not in manager._presence_cache

    @patch("llmd_s3_backend.manager.S3ClientWrapper")
    def test_invalidate_cache_entry_nonexistent(self, mock_client_class):
        """Test invalidating a key that isn't in the cache."""
        manager = S3OffloadingManager(
            model_name="test-model",
            tp_size=1,
            tp_rank=0,
            dtype=torch.float16,
            bucket="test-bucket",
        )

        from llmd_s3_backend.lru_cache import LRUPresenceCache
        manager._presence_cache = LRUPresenceCache(max_size=100)

        # Should not raise
        manager.invalidate_cache_entry("nonexistent")

    @patch("llmd_s3_backend.manager.S3ClientWrapper")
    def test_invalidate_cache_entry_no_cache(self, mock_client_class):
        """Test invalidation when presence cache is disabled."""
        manager = S3OffloadingManager(
            model_name="test-model",
            tp_size=1,
            tp_rank=0,
            dtype=torch.float16,
            bucket="test-bucket",
            enable_presence_cache=False,
        )

        # Should not raise
        manager.invalidate_cache_entry("block_abc")

    @patch("llmd_s3_backend.manager.S3ClientWrapper")
    def test_lookup_with_presence_cache(self, mock_client_class):
        """Test lookup uses presence cache when enabled."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        manager = S3OffloadingManager(
            model_name="test-model",
            tp_size=1,
            tp_rank=0,
            dtype=torch.float16,
            bucket="test-bucket",
        )

        from llmd_s3_backend.lru_cache import LRUPresenceCache
        manager._presence_cache = LRUPresenceCache(max_size=100)

        hashes = [
            get_prefix_hash(range(100, 117)),
            get_prefix_hash(range(200, 217)),
        ]

        # Pre-populate cache with first hash
        manager._presence_cache.add(str(hashes[0]))

        # Second hash not in cache, found in S3
        mock_client.object_exists.return_value = True

        hit_count = manager.lookup(hashes)
        assert hit_count == 2

        # First hash should have been a cache hit (no S3 call needed for it)
        # Second hash triggers an S3 HEAD and gets added to cache
        assert mock_client.object_exists.call_count == 1
        assert str(hashes[1]) in manager._presence_cache

    @patch("llmd_s3_backend.manager.S3ClientWrapper")
    def test_lookup_stops_at_first_miss(self, mock_client_class):
        """Test that lookup stops counting at the first miss."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.object_exists.side_effect = [True, False, True]

        manager = S3OffloadingManager(
            model_name="test-model",
            tp_size=1,
            tp_rank=0,
            dtype=torch.float16,
            bucket="test-bucket",
        )

        hashes = [
            get_prefix_hash(range(100, 117)),
            get_prefix_hash(range(200, 217)),
            get_prefix_hash(range(300, 317)),
        ]

        hit_count = manager.lookup(hashes)
        assert hit_count == 1
        # Should not check the third hash after the second misses
        assert mock_client.object_exists.call_count == 2
