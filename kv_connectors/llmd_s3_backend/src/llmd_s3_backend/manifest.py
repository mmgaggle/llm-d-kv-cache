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
Iceberg-style manifest system for tracking cache block presence.

Provides:
- Snapshot manifests (base state)
- Delta files (MOR - Merge-On-Read)
- Automatic compaction
- Multi-instance coordination
- Avro serialization for compact storage
- Conditional PUT for pointer updates (optimistic concurrency control)
"""

import io
import json
import time
import threading
import queue
from datetime import datetime
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, asdict
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.logger import init_logger
import fastavro
from botocore.exceptions import ClientError

logger = init_logger(__name__)

# Avro schemas
MANIFEST_ENTRY_SCHEMA = {
    "type": "record",
    "name": "ManifestEntry",
    "fields": [
        {"name": "block_hash", "type": "string"},
        {"name": "s3_key", "type": "string"},
        {"name": "size_bytes", "type": "int"},
        {"name": "created_at", "type": "string"},
    ]
}

DELTA_OPERATION_SCHEMA = {
    "type": "record",
    "name": "DeltaOperation",
    "fields": [
        {"name": "type", "type": "string"},
        {"name": "block_hash", "type": "string"},
        {"name": "s3_key", "type": ["null", "string"], "default": None},
        {"name": "size_bytes", "type": ["null", "int"], "default": None},
        {"name": "created_at", "type": ["null", "string"], "default": None},
    ]
}

SNAPSHOT_SCHEMA = {
    "type": "record",
    "name": "Snapshot",
    "fields": [
        {"name": "snapshot_id", "type": "string"},
        {"name": "timestamp", "type": "string"},
        {"name": "model", "type": "string"},
        {"name": "tp_size", "type": "int"},
        {"name": "tp_rank", "type": "int"},
        {"name": "dtype", "type": "string"},
        {"name": "block_count", "type": "int"},
        {"name": "blocks", "type": {"type": "array", "items": MANIFEST_ENTRY_SCHEMA}},
    ]
}

DELTA_SCHEMA = {
    "type": "record",
    "name": "Delta",
    "fields": [
        {"name": "delta_id", "type": "string"},
        {"name": "base_snapshot", "type": "string"},
        {"name": "timestamp", "type": "string"},
        {"name": "operations", "type": {"type": "array", "items": DELTA_OPERATION_SCHEMA}},
    ]
}

MANIFEST_POINTER_SCHEMA = {
    "type": "record",
    "name": "ManifestPointer",
    "fields": [
        {"name": "current_snapshot", "type": "string"},
        {"name": "delta_files", "type": {"type": "array", "items": "string"}},
        {"name": "last_compaction", "type": "string"},
        {"name": "version", "type": "int"},
    ]
}


@dataclass
class ManifestEntry:
    """Single block entry in manifest."""
    block_hash: str
    s3_key: str
    size_bytes: int
    created_at: str


@dataclass
class DeltaOperation:
    """Single operation in a delta file."""
    type: str  # "ADD" or "DELETE"
    block_hash: str
    s3_key: Optional[str] = None
    size_bytes: Optional[int] = None
    created_at: Optional[str] = None


@dataclass
class Snapshot:
    """Snapshot manifest."""
    snapshot_id: str
    timestamp: str
    model: str
    tp_size: int
    tp_rank: int
    dtype: str
    block_count: int
    blocks: List[ManifestEntry]


@dataclass
class Delta:
    """Delta file (MOR)."""
    delta_id: str
    base_snapshot: str
    timestamp: str
    operations: List[DeltaOperation]


@dataclass
class ManifestPointer:
    """Pointer to current snapshot and delta files."""
    current_snapshot: str
    delta_files: List[str]
    last_compaction: str
    version: int


class ManifestManager:
    """
    Manages Iceberg-style manifests for cache block tracking.
    
    Features:
    - Snapshot manifests (base state)
    - Delta files (incremental updates)
    - Automatic compaction
    - Multi-instance safe
    """
    
    def __init__(
        self,
        s3_client,
        model_name: str,
        tp_size: int,
        tp_rank: int,
        dtype: str,
        manifest_prefix: str = "manifests",
        compaction_threshold: int = 100,
        compaction_interval_hours: int = 24,
        delta_batch_size: int = 1000,
        delta_batch_timeout: int = 10,
    ):
        self.s3_client = s3_client
        self.model_name = model_name
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.dtype = dtype
        self.manifest_prefix = manifest_prefix
        self.compaction_threshold = compaction_threshold
        self.compaction_interval_hours = compaction_interval_hours
        self.delta_batch_size = delta_batch_size
        self.delta_batch_timeout = delta_batch_timeout
        
        # Delta write queue
        self._delta_queue: queue.Queue = queue.Queue()
        self._delta_writer_thread = threading.Thread(
            target=self._delta_writer_loop,
            daemon=True
        )
        self._delta_writer_thread.start()
        
        # Compaction thread
        self._compaction_thread = threading.Thread(
            target=self._compaction_loop,
            daemon=True
        )
        self._compaction_thread.start()
        
        logger.info(f"ManifestManager initialized: prefix={manifest_prefix}")
    
    def _get_instance_prefix(self) -> str:
        """
        Get instance-specific prefix for manifest files.
        
        Scopes manifests by model, tp_size, tp_rank, and dtype to prevent
        different vLLM instances from loading irrelevant cache blocks.
        
        Returns:
            str: Prefix like "manifests/model-name/tp_4/rank_0/float16"
        """
        return f"{self.manifest_prefix}/{self.model_name}/tp_{self.tp_size}/rank_{self.tp_rank}/{self.dtype}"
    
    def _get_pointer_key(self) -> str:
        """Get S3 key for manifest pointer (scoped to instance)."""
        return f"{self._get_instance_prefix()}/current-snapshot.avro"
    
    def _get_snapshot_key(self, snapshot_id: str) -> str:
        """Get S3 key for snapshot (scoped to instance)."""
        return f"{self._get_instance_prefix()}/{snapshot_id}.avro"
    
    def _get_delta_key(self, delta_id: str) -> str:
        """Get S3 key for delta file (scoped to instance)."""
        return f"{self._get_instance_prefix()}/{delta_id}.avro"
    
    def _serialize_avro(self, schema: dict, record: dict) -> bytes:
        """Serialize a record to Avro bytes."""
        output = io.BytesIO()
        fastavro.schemaless_writer(output, schema, record)
        return output.getvalue()
    
    def _deserialize_avro(self, schema: dict, data: bytes) -> dict:
        """Deserialize Avro bytes to a record."""
        input_stream = io.BytesIO(data)
        return fastavro.schemaless_reader(input_stream, schema)
    
    def load_manifest(self) -> Dict[str, str]:
        """
        Load current manifest (snapshot + deltas).
        Returns dict mapping block_hash -> s3_key.
        """
        try:
            # Read pointer
            pointer_key = self._get_pointer_key()
            if not self.s3_client.object_exists(pointer_key):
                logger.info("No manifest found, starting fresh")
                return {}
            
            pointer_data = self.s3_client.get_object(pointer_key)
            pointer_dict = self._deserialize_avro(MANIFEST_POINTER_SCHEMA, pointer_data)
            pointer = ManifestPointer(**pointer_dict)
            
            # Load snapshot
            manifest: Dict[str, str] = {}
            snapshot_key = self._get_snapshot_key(pointer.current_snapshot)
            if self.s3_client.object_exists(snapshot_key):
                snapshot_data = self.s3_client.get_object(snapshot_key)
                snapshot_dict = self._deserialize_avro(SNAPSHOT_SCHEMA, snapshot_data)
                for entry in snapshot_dict["blocks"]:
                    manifest[entry["block_hash"]] = entry["s3_key"]
                logger.info(f"Loaded snapshot {pointer.current_snapshot} with {len(manifest)} blocks")
            
            # Apply deltas (MOR)
            for delta_file in pointer.delta_files:
                delta_key = self._get_delta_key(delta_file)
                if self.s3_client.object_exists(delta_key):
                    delta_data = self.s3_client.get_object(delta_key)
                    delta_dict = self._deserialize_avro(DELTA_SCHEMA, delta_data)
                    for op in delta_dict["operations"]:
                        if op["type"] == "ADD" and op["s3_key"]:
                            manifest[op["block_hash"]] = op["s3_key"]
                        elif op["type"] == "DELETE":
                            manifest.pop(op["block_hash"], None)
            
            logger.info(f"Loaded manifest with {len(manifest)} total blocks ({len(pointer.delta_files)} deltas)")
            return manifest
            
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            return {}
    
    def queue_add_blocks(self, block_hashes: List[BlockHash], s3_keys: List[str]):
        """Queue blocks to be added to manifest (async)."""
        for block_hash, s3_key in zip(block_hashes, s3_keys):
            self._delta_queue.put(DeltaOperation(
                type="ADD",
                block_hash=str(block_hash),
                s3_key=s3_key,
                size_bytes=524288,  # Standard block size
                created_at=datetime.utcnow().isoformat()
            ))
    
    def queue_delete_blocks(self, block_hashes: List[BlockHash]):
        """Queue blocks to be deleted from manifest (async)."""
        for block_hash in block_hashes:
            self._delta_queue.put(DeltaOperation(
                type="DELETE",
                block_hash=str(block_hash)
            ))
    
    def _delta_writer_loop(self):
        """Background thread to write delta files in batches."""
        batch: List[DeltaOperation] = []
        
        while True:
            try:
                # Collect operations for batch
                try:
                    op = self._delta_queue.get(timeout=self.delta_batch_timeout)
                    batch.append(op)
                    
                    # Write batch if full
                    if len(batch) >= self.delta_batch_size:
                        self._write_delta_batch(batch)
                        batch = []
                        
                except queue.Empty:
                    # Timeout - write partial batch if any
                    if batch:
                        self._write_delta_batch(batch)
                        batch = []
                        
            except Exception as e:
                logger.error(f"Error in delta writer loop: {e}")
                time.sleep(1)
    
    def _write_delta_batch(self, operations: List[DeltaOperation]):
        """Write a batch of operations to a delta file."""
        try:
            # Create delta
            delta_id = f"delta-{int(time.time() * 1000)}"
            delta_dict = {
                "delta_id": delta_id,
                "base_snapshot": "",  # Will be filled from pointer
                "timestamp": datetime.utcnow().isoformat(),
                "operations": [asdict(op) for op in operations]
            }
            
            # Write delta file (Avro)
            delta_key = self._get_delta_key(delta_id)
            delta_bytes = self._serialize_avro(DELTA_SCHEMA, delta_dict)
            self.s3_client.put_object(delta_key, delta_bytes)
            
            # Update pointer
            self._append_delta_to_pointer(delta_id)
            
            logger.info(f"Wrote delta {delta_id} with {len(operations)} operations")
            
        except Exception as e:
            logger.error(f"Failed to write delta batch: {e}")
    
    def _append_delta_to_pointer(self, delta_id: str):
        """
        Append delta file to pointer using conditional PUT (optimistic concurrency control).
        
        Uses ETag-based conditional PUT to prevent race conditions when multiple
        instances update the pointer simultaneously.
        """
        max_retries = 5
        for attempt in range(max_retries):
            try:
                pointer_key = self._get_pointer_key()
                
                # Read current pointer with ETag
                if self.s3_client.object_exists(pointer_key):
                    pointer_data, etag = self.s3_client.get_object_with_etag(pointer_key)
                    pointer_dict = self._deserialize_avro(MANIFEST_POINTER_SCHEMA, pointer_data)
                else:
                    # Create initial pointer (no ETag for new object)
                    pointer_dict = {
                        "current_snapshot": "snapshot-initial",
                        "delta_files": [],
                        "last_compaction": datetime.utcnow().isoformat(),
                        "version": 0
                    }
                    etag = None
                
                # Append delta
                pointer_dict["delta_files"].append(delta_id)
                pointer_dict["version"] += 1
                
                # Write updated pointer with conditional PUT (Avro)
                pointer_bytes = self._serialize_avro(MANIFEST_POINTER_SCHEMA, pointer_dict)
                
                if etag:
                    # Conditional PUT - only succeeds if ETag matches
                    self.s3_client.put_object_if_match(pointer_key, pointer_bytes, etag)
                else:
                    # Initial write - no condition needed
                    self.s3_client.put_object(pointer_key, pointer_bytes)
                
                logger.info(f"Updated pointer with delta {delta_id} (version {pointer_dict['version']})")
                return
                
            except ClientError as e:
                # Check if it's a precondition failed (412) - another instance updated
                if e.response.get('Error', {}).get('Code') == 'PreconditionFailed':
                    logger.info(f"Pointer updated by another instance, retrying (attempt {attempt + 1})")
                    time.sleep(0.05 * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    # Other error - propagate
                    raise
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Failed to update pointer (attempt {attempt + 1}): {e}")
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"Failed to update pointer after {max_retries} attempts: {e}")
                    raise
    
    def _compaction_loop(self):
        """Background thread to periodically compact deltas into snapshots."""
        while True:
            try:
                time.sleep(3600)  # Check every hour
                
                if self._should_compact():
                    logger.info("Starting compaction...")
                    self.compact()
                    
            except Exception as e:
                logger.error(f"Error in compaction loop: {e}")
    
    def _should_compact(self) -> bool:
        """Check if compaction is needed."""
        try:
            pointer_key = self._get_pointer_key()
            if not self.s3_client.object_exists(pointer_key):
                return False
            
            pointer_data = self.s3_client.get_object(pointer_key)
            pointer_dict = self._deserialize_avro(MANIFEST_POINTER_SCHEMA, pointer_data)
            
            # Compact if too many deltas
            if len(pointer_dict["delta_files"]) >= self.compaction_threshold:
                return True
            
            # Compact if too much time has passed
            last_compaction = datetime.fromisoformat(pointer_dict["last_compaction"])
            hours_since = (datetime.utcnow() - last_compaction).total_seconds() / 3600
            if hours_since >= self.compaction_interval_hours:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking compaction: {e}")
            return False
    
    def compact(self):
        """Compact delta files into a new snapshot."""
        try:
            # Load full manifest
            manifest = self.load_manifest()
            
            # Create new snapshot
            snapshot_id = f"snapshot-{int(time.time())}"
            snapshot_dict = {
                "snapshot_id": snapshot_id,
                "timestamp": datetime.utcnow().isoformat(),
                "model": self.model_name,
                "tp_size": self.tp_size,
                "tp_rank": self.tp_rank,
                "dtype": self.dtype,
                "block_count": len(manifest),
                "blocks": [
                    {
                        "block_hash": block_hash,
                        "s3_key": s3_key,
                        "size_bytes": 524288,
                        "created_at": datetime.utcnow().isoformat()
                    }
                    for block_hash, s3_key in manifest.items()
                ]
            }
            
            # Write snapshot (Avro)
            snapshot_key = self._get_snapshot_key(snapshot_id)
            snapshot_bytes = self._serialize_avro(SNAPSHOT_SCHEMA, snapshot_dict)
            self.s3_client.put_object(snapshot_key, snapshot_bytes)
            
            # Update pointer (clear deltas, Avro)
            pointer_dict = {
                "current_snapshot": snapshot_id,
                "delta_files": [],
                "last_compaction": datetime.utcnow().isoformat(),
                "version": 0
            }
            pointer_key = self._get_pointer_key()
            pointer_bytes = self._serialize_avro(MANIFEST_POINTER_SCHEMA, pointer_dict)
            self.s3_client.put_object(pointer_key, pointer_bytes)
            
            logger.info(f"Compacted {len(manifest)} blocks into snapshot {snapshot_id}")
            
        except Exception as e:
            logger.error(f"Failed to compact: {e}")

# Made with Bob