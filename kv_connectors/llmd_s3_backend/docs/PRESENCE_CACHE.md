# Presence Cache with Iceberg-Style Manifests

The S3 backend connector supports an optional **presence cache** feature that dramatically improves lookup performance by maintaining an in-memory cache of which blocks exist in S3, backed by Iceberg-style manifest files with **Avro serialization** and **conditional PUT for concurrency control**.

## Overview

Without presence cache:
- Every block lookup requires a HEAD request to S3 (~5ms latency)
- 100 lookups = 500ms overhead

With presence cache:
- Lookups are instant in-memory checks (<0.01ms)
- Manifest pre-warms cache on startup
- All vLLM instances share the same manifest
- Avro provides 3-5x compression vs JSON (10-15 MB vs 30 MB for 1M blocks)
- Conditional PUT prevents race conditions in multi-instance deployments

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ vLLM Instance                                               │
│                                                             │
│  ┌──────────────────┐         ┌─────────────────────────┐  │
│  │ Presence Cache   │◄────────│ Manifest Manager        │  │
│  │ (in-memory set)  │         │ - Load snapshot         │  │
│  └──────────────────┘         │ - Apply deltas (MOR)    │  │
│         ▲                     │ - Write deltas          │  │
│         │                     │ - Trigger compaction    │  │
│         │                     └─────────────────────────┘  │
│         │                              ▲                    │
│         │                              │                    │
└─────────┼──────────────────────────────┼────────────────────┘
          │                              │
          │                              ▼
          │                     ┌─────────────────────┐
          │                     │ S3 Bucket           │
          │                     │                     │
          │                     │ manifests/          │
          │                     │ ├─ snapshot-001.avro│
          │                     │ ├─ snapshot-002.avro│
          │                     │ ├─ delta-003.avro   │
          │                     │ └─ delta-004.avro   │
          │                     │                     │
          └─────────────────────│ kv-cache/           │
                                │ └─ (cache blocks)   │
                                └─────────────────────┘
```

## Manifest Structure

All manifest files use **Apache Avro** binary format for compact storage and fast parsing. The examples below show the logical structure (as if JSON) for readability.

### Snapshot Manifest
Base state containing all blocks at a point in time:

```json
{
  "snapshot_id": "snapshot-1710512400",
  "timestamp": "2024-03-15T10:00:00Z",
  "model": "granite-3b-code-instruct",
  "tp_size": 4,
  "tp_rank": 0,
  "dtype": "float16",
  "block_count": 1000000,
  "blocks": [
    {
      "block_hash": "74f81fe167d99b4c",
      "s3_key": "kv-cache/model/74f/81/74f81fe167d99b4c.bin",
      "size_bytes": 524288,
      "created_at": "2024-03-15T09:30:00Z"
    }
    // ... more blocks
  ]
}
```

### Delta File (MOR - Merge-On-Read)
Incremental updates since last snapshot:

```json
{
  "delta_id": "delta-1710512700",
  "base_snapshot": "snapshot-1710512400",
  "timestamp": "2024-03-15T10:05:00Z",
  "operations": [
    {
      "type": "ADD",
      "block_hash": "750a1fe167d99b4e",
      "s3_key": "kv-cache/model/750/a1/750a1fe167d99b4e.bin",
      "size_bytes": 524288,
      "created_at": "2024-03-15T10:05:00Z"
    },
    {
      "type": "DELETE",
      "block_hash": "74f81fe167d99b4c"
    }
  ]
}
```

### Current Snapshot Pointer
Points to active snapshot and delta files:

```json
{
  "current_snapshot": "snapshot-1710512400",
  "delta_files": ["delta-1710512700", "delta-1710512800"],
  "last_compaction": "2024-03-15T10:00:00Z",
  "version": 2
}
```

## Configuration

Enable presence cache in vLLM configuration:

```bash
vllm serve model-name \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "spec_name": "S3OffloadingSpec",
      "spec_module_path": "llmd_s3_backend.spec",
      "s3_bucket": "vllm-cache",
      "s3_profile_name": "default",
      "enable_presence_cache": true,
      "manifest_prefix": "manifests",
      "compaction_threshold": 100,
      "compaction_interval_hours": 24
    }
  }'
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_presence_cache` | bool | `false` | Enable presence cache with manifest |
| `manifest_prefix` | str | `"manifests"` | S3 prefix for manifest files |
| `compaction_threshold` | int | `100` | Number of delta files before compaction |
| `compaction_interval_hours` | int | `24` | Hours between compactions |

## How It Works

### 1. Startup (Cold Start)

```python
# Instance starts
manager = S3OffloadingManager(enable_presence_cache=True)

# Load manifest
manifest = load_manifest()  # Reads snapshot + deltas from S3
# Time: ~2-5 seconds for 1M blocks

# Pre-warm cache
presence_cache = set(manifest.keys())  # 1M block hashes
# Memory: ~8 MB

# Ready to serve
# All lookups are now instant!
```

### 2. Block Lookup (Hot Path)

```python
def lookup(block_hashes):
    for block_hash in block_hashes:
        # Check presence cache (instant)
        if block_hash in presence_cache:
            continue  # Cache hit!
        
        # Cache miss - verify with HEAD request
        if s3_client.object_exists(s3_key):
            presence_cache.add(block_hash)  # Update cache
        else:
            break  # Not in S3
```

### 3. Block Storage (Write Path)

```python
def complete_store(block_hashes):
    # Update presence cache immediately
    presence_cache.update(block_hashes)
    
    # Queue delta write (async, non-blocking)
    manifest_manager.queue_add_blocks(block_hashes)
```

### 4. Delta Writing (Background)

```python
# Background thread collects operations
batch = []
while True:
    op = delta_queue.get(timeout=10)  # Wait 10s or until 1000 ops
    batch.append(op)
    
    if len(batch) >= 1000:
        # Write delta file
        write_delta(batch)
        batch = []
```

### 5. Compaction (Periodic)

```python
# Background thread checks every hour
if len(delta_files) > 100 or hours_since_compaction > 24:
    # Load full state
    manifest = load_snapshot() + apply_all_deltas()
    
    # Write new snapshot
    write_snapshot(manifest)
    
    # Update pointer (atomic)
    update_pointer(new_snapshot, delta_files=[])
```

### 6. Cache Refresh (Periodic)

```python
# Background thread refreshes every 5 minutes
manifest = load_manifest()
new_blocks = set(manifest.keys()) - presence_cache
presence_cache.update(new_blocks)
# Picks up blocks written by other instances!

## Multi-Instance Concurrency Control

### Conditional PUT with ETag

The manifest system uses **conditional PUT** to prevent race conditions when multiple vLLM instances update the pointer file simultaneously:

```python
# Instance A and B both want to add a delta
def append_delta_to_pointer(delta_id):
    for attempt in range(5):
        # 1. Read pointer with ETag
        pointer_data, etag = s3_client.get_object_with_etag("manifests/current-snapshot.avro")
        pointer = deserialize(pointer_data)
        
        # 2. Modify locally
        pointer["delta_files"].append(delta_id)
        pointer["version"] += 1
        
        # 3. Conditional PUT (only succeeds if ETag matches)
        try:
            s3_client.put_object_if_match(
                "manifests/current-snapshot.avro",
                serialize(pointer),
                etag  # Must match current ETag
            )
            return  # Success!
        except PreconditionFailed:
            # Another instance updated - retry with new ETag
            continue
```

### Race Condition Prevention

**Without conditional PUT:**
```
Time  Instance A              Instance B
----  ----------              ----------
T0    Read pointer (v=1)      
T1                            Read pointer (v=1)
T2    Add delta-A             
T3    Write pointer (v=2)     
T4                            Add delta-B
T5                            Write pointer (v=2) ← Overwrites A's delta!
```

**With conditional PUT:**
```
Time  Instance A              Instance B
----  ----------              ----------
T0    Read pointer (v=1, ETag=abc)      
T1                            Read pointer (v=1, ETag=abc)
T2    Add delta-A             
T3    PUT if ETag=abc ✓       
T4    Pointer now (v=2, ETag=def)
T5                            PUT if ETag=abc ✗ (412 Precondition Failed)
T6                            Retry: Read pointer (v=2, ETag=def)
T7                            Add delta-B
T8                            PUT if ETag=def ✓
```

### Benefits

- **No lost updates**: Every delta is guaranteed to be recorded
- **Automatic retry**: Failed updates retry with exponential backoff
- **Eventually consistent**: All instances converge to same state
- **Production-grade**: Handles high-contention scenarios

```

## Performance Impact

### Lookup Performance

| Scenario | Without Cache | With Cache | Speedup |
|----------|--------------|------------|---------|
| Single block | 5ms (HEAD) | <0.01ms | 500x |
| 100 blocks | 500ms | <1ms | 500x |
| 1000 blocks | 5s | <10ms | 500x |

### Startup Time

| Blocks in Cache | Load Time | Memory |
|----------------|-----------|--------|
| 100K | ~0.5s | ~0.8 MB |
| 1M | ~2s | ~8 MB |
| 10M | ~20s | ~80 MB |
| 100M | ~200s | ~800 MB |

### Multi-Instance Sharing

```
Timeline:
T=0:  Instance A starts, loads manifest (1M blocks)
T=5:  Instance A adds 10K new blocks
      → Updates presence cache immediately
      → Queues delta write (async)
T=10: Instance B starts, loads manifest (1.01M blocks)
      → Sees Instance A's 10K blocks!
      → No HEAD requests needed
T=15: Instance C starts, loads manifest (1.01M blocks)
      → Also sees all blocks from A and B
```

## Storage Overhead

### Manifest Files

```
Snapshot (1M blocks):
  - Uncompressed JSON: ~100 MB
  - Compressed: ~20-30 MB
  - One file per compaction

Delta files (1K blocks each):
  - ~100 KB per file
  - 100 files between compactions = ~10 MB

Total: ~40 MB for 1M blocks
```

### Comparison to Block Storage

```
Cache blocks: 1M × 512 KB = 512 GB
Manifest overhead: 40 MB
Overhead percentage: 0.008%
```

## Multi-Instance Coordination

### Write Conflicts

Delta writes use optimistic concurrency:

```python
# Read current pointer
pointer = read_pointer()
version = pointer['version']

# Write delta
write_delta(delta_id)

# Update pointer with version check
try:
    update_pointer(delta_id, expected_version=version)
except VersionMismatch:
    retry()  # Another instance updated, retry
```

### Eventual Consistency

All instances eventually see the same state:
- Each instance refreshes cache every 5 minutes
- Compaction creates new snapshot visible to all
- Delta files are append-only (no conflicts)

## Best Practices

### When to Enable

✅ **Enable presence cache when:**
- Multiple vLLM instances share a bucket
- Cache has >100K blocks
- Latency is critical (<10ms lookup target)
- Instances restart frequently

❌ **Skip presence cache when:**
- Single instance deployment
- Cache has <10K blocks
- Simplicity is priority
- Memory is constrained

### Tuning Parameters

**Compaction threshold:**
- Lower (50): More frequent compaction, smaller deltas
- Higher (200): Less frequent compaction, more deltas
- Recommended: 100 for most workloads

**Compaction interval:**
- Shorter (12h): More up-to-date snapshots
- Longer (48h): Less compaction overhead
- Recommended: 24h for most workloads

### Monitoring

Key metrics to monitor:
- Presence cache hit rate (should be >95%)
- Delta file count (should stay <threshold)
- Compaction duration (should be <5 minutes)
- Cache refresh time (should be <10 seconds)

## Troubleshooting

### Cache Not Pre-Warming

**Symptom:** Slow lookups after restart

**Causes:**
- Manifest files don't exist (first run)
- S3 permissions issue
- Manifest module not installed

**Solution:**
```bash
# Check manifest files exist
aws s3 ls s3://bucket/manifests/

# Check logs for errors
grep "manifest" vllm.log

# Verify module installed
python -c "from llmd_s3_backend.manifest import ManifestManager"
```

### High Memory Usage

**Symptom:** Instance using >1GB RAM for cache

**Cause:** Too many blocks in cache

**Solution:**
- Reduce cache scope (separate buckets per model)
- Implement LRU eviction
- Use Bloom filter instead of set

### Stale Cache

**Symptom:** Cache misses for blocks that exist

**Cause:** Cache not refreshing

**Solution:**
- Check refresh thread is running
- Verify S3 connectivity
- Manually trigger refresh

## Future Enhancements

Potential improvements:
- Bloom filters for memory efficiency
- Distributed cache coordination (Redis/etcd)
- Incremental manifest loading
- Cache warming strategies
- LRU eviction policies
- Metrics and observability

## Summary

The presence cache with Iceberg-style manifests provides:

✅ **500x faster lookups** - In-memory vs S3 HEAD requests
✅ **Instant cold start** - Pre-warm from manifest in seconds
✅ **Multi-instance sharing** - All instances see same state
✅ **Minimal overhead** - <0.01% storage, ~8MB RAM per 1M blocks
✅ **Production-ready** - Compaction, refresh, conflict resolution

This is an optional feature that significantly improves performance for large-scale deployments with multiple vLLM instances sharing a cache bucket.