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

import io
import pytest
import json
import time
from unittest.mock import Mock, MagicMock, patch
import fastavro
from llmd_s3_backend.manifest import (
    ManifestManager,
    ManifestEntry,
    DeltaOperation,
    Snapshot,
    Delta,
    ManifestPointer,
    MANIFEST_POINTER_SCHEMA,
    SNAPSHOT_SCHEMA,
    DELTA_SCHEMA,
)


def serialize_avro(schema, record):
    """Helper to serialize Avro for tests."""
    output = io.BytesIO()
    fastavro.schemaless_writer(output, schema, record)
    return output.getvalue()


class TestManifestDataClasses:
    """Test manifest data classes."""
    
    def test_manifest_entry(self):
        """Test ManifestEntry creation."""
        entry = ManifestEntry(
            block_hash="abc123",
            s3_key="kv-cache/model/abc/12/abc123.bin",
            size_bytes=524288,
            created_at="2024-01-15T10:00:00Z"
        )
        assert entry.block_hash == "abc123"
        assert entry.size_bytes == 524288
    
    def test_delta_operation_add(self):
        """Test DeltaOperation for ADD."""
        op = DeltaOperation(
            type="ADD",
            block_hash="abc123",
            s3_key="kv-cache/model/abc/12/abc123.bin",
            size_bytes=524288
        )
        assert op.type == "ADD"
        assert op.block_hash == "abc123"
    
    def test_delta_operation_delete(self):
        """Test DeltaOperation for DELETE."""
        op = DeltaOperation(
            type="DELETE",
            block_hash="abc123"
        )
        assert op.type == "DELETE"
        assert op.s3_key is None
    
    def test_snapshot(self):
        """Test Snapshot creation."""
        snapshot = Snapshot(
            snapshot_id="snapshot-001",
            timestamp="2024-01-15T10:00:00Z",
            model="test-model",
            tp_size=4,
            tp_rank=0,
            dtype="float16",
            block_count=2,
            blocks=[
                ManifestEntry("hash1", "key1", 524288, "2024-01-15T10:00:00Z"),
                ManifestEntry("hash2", "key2", 524288, "2024-01-15T10:00:00Z"),
            ]
        )
        assert snapshot.block_count == 2
        assert len(snapshot.blocks) == 2
    
    def test_delta(self):
        """Test Delta creation."""
        delta = Delta(
            delta_id="delta-001",
            base_snapshot="snapshot-001",
            timestamp="2024-01-15T10:05:00Z",
            operations=[
                DeltaOperation("ADD", "hash3", "key3", 524288)
            ]
        )
        assert delta.delta_id == "delta-001"
        assert len(delta.operations) == 1
    
    def test_manifest_pointer(self):
        """Test ManifestPointer creation."""
        pointer = ManifestPointer(
            current_snapshot="snapshot-001",
            delta_files=["delta-001", "delta-002"],
            last_compaction="2024-01-15T10:00:00Z",
            version=2
        )
        assert pointer.version == 2
        assert len(pointer.delta_files) == 2


class TestManifestManager:
    """Test ManifestManager class."""
    
    @pytest.fixture
    def mock_s3_client(self):
        """Create mock S3 client."""
        client = Mock()
        client.bucket = "test-bucket"
        return client
    
    @pytest.fixture
    def manifest_manager(self, mock_s3_client):
        """Create ManifestManager with mocked S3."""
        return ManifestManager(
            s3_client=mock_s3_client,
            model_name="test-model",
            tp_size=4,
            tp_rank=0,
            dtype="float16",
            manifest_prefix="manifests",
            compaction_threshold=10,
            compaction_interval_hours=1,
            delta_batch_size=5,
            delta_batch_timeout=1,
        )
    
    def test_init(self, manifest_manager):
        """Test ManifestManager initialization."""
        assert manifest_manager.model_name == "test-model"
        assert manifest_manager.tp_size == 4
        assert manifest_manager.compaction_threshold == 10
    
    def test_get_keys(self, manifest_manager):
        """Test S3 key generation."""
        pointer_key = manifest_manager._get_pointer_key()
        assert pointer_key == "manifests/current-snapshot.avro"
        
        snapshot_key = manifest_manager._get_snapshot_key("snapshot-001")
        assert snapshot_key == "manifests/snapshot-001.avro"
        
        delta_key = manifest_manager._get_delta_key("delta-001")
        assert delta_key == "manifests/delta-001.avro"
    
    def test_load_manifest_empty(self, manifest_manager, mock_s3_client):
        """Test loading manifest when none exists."""
        mock_s3_client.object_exists.return_value = False
        
        manifest = manifest_manager.load_manifest()
        
        assert manifest == {}
        mock_s3_client.object_exists.assert_called_once()
    
    def test_load_manifest_with_snapshot(self, manifest_manager, mock_s3_client):
        """Test loading manifest with snapshot only."""
        # Mock pointer (Avro)
        pointer_dict = {
            "current_snapshot": "snapshot-001",
            "delta_files": [],
            "last_compaction": "2024-01-15T10:00:00Z",
            "version": 0
        }
        pointer_avro = serialize_avro(MANIFEST_POINTER_SCHEMA, pointer_dict)
        
        # Mock snapshot (Avro)
        snapshot_dict = {
            "snapshot_id": "snapshot-001",
            "timestamp": "2024-01-15T10:00:00Z",
            "model": "test-model",
            "tp_size": 4,
            "tp_rank": 0,
            "dtype": "float16",
            "block_count": 2,
            "blocks": [
                {
                    "block_hash": "hash1",
                    "s3_key": "key1",
                    "size_bytes": 524288,
                    "created_at": "2024-01-15T10:00:00Z"
                },
                {
                    "block_hash": "hash2",
                    "s3_key": "key2",
                    "size_bytes": 524288,
                    "created_at": "2024-01-15T10:00:00Z"
                }
            ]
        }
        snapshot_avro = serialize_avro(SNAPSHOT_SCHEMA, snapshot_dict)
        
        def mock_exists(key):
            return True
        
        def mock_get(key):
            if "current-snapshot" in key:
                return pointer_avro
            elif "snapshot-001" in key:
                return snapshot_avro
            return b''
        
        mock_s3_client.object_exists.side_effect = mock_exists
        mock_s3_client.get_object.side_effect = mock_get
        
        manifest = manifest_manager.load_manifest()
        
        assert len(manifest) == 2
        assert manifest["hash1"] == "key1"
        assert manifest["hash2"] == "key2"
    
    def test_load_manifest_with_deltas(self, manifest_manager, mock_s3_client):
        """Test loading manifest with snapshot and deltas."""
        # Mock pointer with deltas (Avro)
        pointer_dict = {
            "current_snapshot": "snapshot-001",
            "delta_files": ["delta-001"],
            "last_compaction": "2024-01-15T10:00:00Z",
            "version": 1
        }
        pointer_avro = serialize_avro(MANIFEST_POINTER_SCHEMA, pointer_dict)
        
        # Mock snapshot (Avro)
        snapshot_dict = {
            "snapshot_id": "snapshot-001",
            "timestamp": "2024-01-15T10:00:00Z",
            "model": "test-model",
            "tp_size": 4,
            "tp_rank": 0,
            "dtype": "float16",
            "block_count": 1,
            "blocks": [
                {"block_hash": "hash1", "s3_key": "key1", "size_bytes": 524288, "created_at": "2024-01-15T10:00:00Z"}
            ]
        }
        snapshot_avro = serialize_avro(SNAPSHOT_SCHEMA, snapshot_dict)
        
        # Mock delta with ADD and DELETE (Avro)
        delta_dict = {
            "delta_id": "delta-001",
            "base_snapshot": "snapshot-001",
            "timestamp": "2024-01-15T11:00:00Z",
            "operations": [
                {"type": "ADD", "block_hash": "hash2", "s3_key": "key2", "size_bytes": 524288, "created_at": "2024-01-15T11:00:00Z"},
                {"type": "DELETE", "block_hash": "hash1", "s3_key": None, "size_bytes": None, "created_at": None}
            ]
        }
        delta_avro = serialize_avro(DELTA_SCHEMA, delta_dict)
        
        def mock_get(key):
            if "current-snapshot" in key:
                return pointer_avro
            elif "snapshot-001" in key:
                return snapshot_avro
            elif "delta-001" in key:
                return delta_avro
            return b''
        
        mock_s3_client.object_exists.return_value = True
        mock_s3_client.get_object.side_effect = mock_get
        
        manifest = manifest_manager.load_manifest()
        
        # hash1 should be deleted, hash2 should be added
        assert len(manifest) == 1
        assert "hash1" not in manifest
        assert manifest["hash2"] == "key2"
    
    def test_queue_add_blocks(self, manifest_manager):
        """Test queuing blocks to add."""
        block_hashes = ["hash1", "hash2"]
        s3_keys = ["key1", "key2"]
        
        manifest_manager.queue_add_blocks(block_hashes, s3_keys)
        
        # Check queue has operations
        assert manifest_manager._delta_queue.qsize() == 2
    
    def test_queue_delete_blocks(self, manifest_manager):
        """Test queuing blocks to delete."""
        block_hashes = ["hash1", "hash2"]
        
        manifest_manager.queue_delete_blocks(block_hashes)
        
        # Check queue has operations
        assert manifest_manager._delta_queue.qsize() == 2
    
    def test_write_delta_batch(self, manifest_manager, mock_s3_client):
        """Test writing a delta batch with conditional PUT."""
        operations = [
            DeltaOperation("ADD", "hash1", "key1", 524288, "2024-01-15T10:00:00Z"),
            DeltaOperation("ADD", "hash2", "key2", 524288, "2024-01-15T10:00:00Z"),
        ]
        
        # Mock pointer read with ETag (Avro)
        pointer_dict = {
            "current_snapshot": "snapshot-001",
            "delta_files": [],
            "last_compaction": "2024-01-15T10:00:00Z",
            "version": 0
        }
        pointer_avro = serialize_avro(MANIFEST_POINTER_SCHEMA, pointer_dict)
        mock_s3_client.object_exists.return_value = True
        mock_s3_client.get_object_with_etag.return_value = (pointer_avro, "test-etag-123")
        
        manifest_manager._write_delta_batch(operations)
        
        # Verify delta was written and pointer updated with conditional PUT
        assert mock_s3_client.put_object.call_count == 1  # Delta file
        assert mock_s3_client.put_object_if_match.call_count == 1  # Pointer with ETag
    
    def test_should_compact_threshold(self, manifest_manager, mock_s3_client):
        """Test compaction trigger by delta count."""
        # Mock pointer with many deltas (Avro)
        pointer_dict = {
            "current_snapshot": "snapshot-001",
            "delta_files": [f"delta-{i}" for i in range(15)],  # > threshold of 10
            "last_compaction": "2024-01-15T10:00:00Z",
            "version": 15
        }
        pointer_avro = serialize_avro(MANIFEST_POINTER_SCHEMA, pointer_dict)
        mock_s3_client.object_exists.return_value = True
        mock_s3_client.get_object.return_value = pointer_avro
        
        should_compact = manifest_manager._should_compact()
        
        assert should_compact is True
    
    def test_should_compact_time(self, manifest_manager, mock_s3_client):
        """Test compaction trigger by time."""
        from datetime import datetime, timedelta
        
        # Mock pointer with old compaction time (Avro)
        old_time = (datetime.utcnow() - timedelta(hours=25)).isoformat()
        pointer_dict = {
            "current_snapshot": "snapshot-001",
            "delta_files": ["delta-001"],  # < threshold
            "last_compaction": old_time,  # > 24 hours ago
            "version": 1
        }
        pointer_avro = serialize_avro(MANIFEST_POINTER_SCHEMA, pointer_dict)
        mock_s3_client.object_exists.return_value = True
        mock_s3_client.get_object.return_value = pointer_avro
        
        should_compact = manifest_manager._should_compact()
        
        assert should_compact is True
    
    def test_compact(self, manifest_manager, mock_s3_client):
        """Test compaction process."""
        # Mock load_manifest to return test data
        with patch.object(manifest_manager, 'load_manifest') as mock_load:
            mock_load.return_value = {
                "hash1": "key1",
                "hash2": "key2",
                "hash3": "key3"
            }
            
            manifest_manager.compact()
            
            # Verify snapshot and pointer were written
            put_calls = [call for call in mock_s3_client.put_object.call_args_list]
            assert len(put_calls) == 2  # Snapshot + pointer
            
            # Verify snapshot is Avro binary (not JSON)
            snapshot_call = put_calls[0]
            snapshot_bytes = snapshot_call[0][1]
            assert isinstance(snapshot_bytes, bytes)
            # Avro binary starts with specific bytes, not '{' like JSON
            assert snapshot_bytes[0] != ord('{')


class TestPresenceCacheIntegration:
    """Test presence cache integration with manager."""
    
    @pytest.fixture
    def mock_s3_client(self):
        """Create mock S3 client."""
        client = Mock()
        client.bucket = "test-bucket"
        client.object_exists = Mock(return_value=False)
        return client
    
    def test_manager_with_presence_cache_disabled(self, mock_s3_client):
        """Test manager without presence cache."""
        from llmd_s3_backend.manager import S3OffloadingManager
        import torch
        
        manager = S3OffloadingManager(
            model_name="test-model",
            tp_size=4,
            tp_rank=0,
            dtype=torch.float16,
            bucket="test-bucket",
            enable_presence_cache=False,
        )
        manager.s3_client = mock_s3_client
        
        assert manager._presence_cache is None
        assert manager._manifest_manager is None
    
    @patch('llmd_s3_backend.manager.MANIFEST_AVAILABLE', True)
    @patch('llmd_s3_backend.manager.ManifestManager')
    def test_manager_with_presence_cache_enabled(self, mock_manifest_class, mock_s3_client):
        """Test manager with presence cache enabled."""
        from llmd_s3_backend.manager import S3OffloadingManager
        import torch
        
        # Mock manifest manager
        mock_manifest = Mock()
        mock_manifest.load_manifest.return_value = {
            "hash1": "key1",
            "hash2": "key2"
        }
        mock_manifest_class.return_value = mock_manifest
        
        manager = S3OffloadingManager(
            model_name="test-model",
            tp_size=4,
            tp_rank=0,
            dtype=torch.float16,
            bucket="test-bucket",
            enable_presence_cache=True,
        )
        manager.s3_client = mock_s3_client
        
        # Verify cache was pre-warmed
        assert manager._presence_cache is not None
        assert len(manager._presence_cache) == 2
        assert "hash1" in manager._presence_cache
        assert "hash2" in manager._presence_cache


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
