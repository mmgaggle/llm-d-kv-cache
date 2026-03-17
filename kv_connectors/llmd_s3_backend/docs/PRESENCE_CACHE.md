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
          │                     │ └─ model/tp_4/rank_0/float16/│
          │                     │    ├─ snapshot-001.avro      │
          │                     │    ├─ snapshot-002.avro      │
          │                     │    ├─ delta-003.avro         │
          │                     │    └─ delta-004.avro         │
          │                     │                     │
          └─────────────────────│ kv-cache/           │
                                │ └─ (cache blocks)   │
                                └─────────────────────┘
```

## S3 Bucket Structure

Here's what the complete S3 bucket looks like with manifests, deltas, and cache blocks:

```
s3://my-vllm-bucket/
│
├── manifests/                                    # Manifest root directory
│   │
│   ├── llama3-70b/                              # Model: Llama3-70B
│   │   └── tp_8/                                # Tensor Parallelism: 8
│   │       ├── rank_0/                          # Rank 0
│   │       │   └── float16/                     # Data type: float16
│   │       │       ├── current-snapshot.avro    # ← POINTER FILE (points to active snapshot + deltas)
│   │       │       ├── snapshot-1710512400.avro # Base snapshot (1M blocks)
│   │       │       ├── snapshot-1710598800.avro # New snapshot after compaction
│   │       │       ├── delta-1710512700.avro    # Delta 1 (100 ADD/DELETE ops)
│   │       │       ├── delta-1710512800.avro    # Delta 2 (150 ADD/DELETE ops)
│   │       │       ├── delta-1710512900.avro    # Delta 3 (200 ADD/DELETE ops)
│   │       │       └── delta-1710513000.avro    # Delta 4 (50 ADD/DELETE ops)
│   │       │
│   │       ├── rank_1/                          # Rank 1 (separate manifest)
│   │       │   └── float16/
│   │       │       ├── current-snapshot.avro
│   │       │       ├── snapshot-1710512400.avro
│   │       │       └── delta-1710512700.avro
│   │       │
│   │       └── rank_2/                          # Rank 2 (separate manifest)
│   │           └── float16/
│   │               └── ...
│   │
│   └── qwen3-32b/                               # Different model
│       └── tp_4/                                # Different TP size
│           └── rank_0/
│               └── bfloat16/                    # Different dtype
│                   ├── current-snapshot.avro
│                   ├── snapshot-1710515000.avro
│                   └── delta-1710515100.avro
│
└── kv-cache/                                    # Actual KV cache blocks
    ├── llama3-70b/
    │   └── tp_8/
    │       └── rank_0/
    │           └── float16/
    │               ├── 74f/                     # Hash-based prefix
    │               │   └── 81/
    │               │       └── 74f81fe167d99b4c.bin  # Cache block (512 KB)
    │               ├── 750/
    │               │   └── a1/
    │               │       └── 750a1fe167d99b4e.bin
    │               └── ...                      # 1M+ cache block files
    │
    └── qwen3-32b/
        └── tp_4/
            └── rank_0/
                └── bfloat16/
                    └── ...                      # Separate cache blocks
```

### File Relationships

**1. Pointer File** (`current-snapshot.avro`)
```json
{
  "current_snapshot": "snapshot-1710512400",
  "delta_files": [
    "delta-1710512700",
    "delta-1710512800",
    "delta-1710512900",
    "delta-1710513000"
  ],
  "last_compaction": "2024-03-15T10:00:00Z",
  "version": 5
}
```
- **Purpose**: Points to the active snapshot and all delta files
- **Updated**: Every time a new delta is written (using conditional PUT with ETag)
- **Size**: ~1 KB

**2. Snapshot File** (`snapshot-1710512400.avro`)
```json
{
  "snapshot_id": "snapshot-1710512400",
  "timestamp": "2024-03-15T10:00:00Z",
  "model": "llama3-70b",
  "tp_size": 8,
  "tp_rank": 0,
  "dtype": "float16",
  "block_count": 1000000,
  "blocks": [
    {
      "block_hash": "74f81fe167d99b4c",
      "s3_key": "kv-cache/llama3-70b/tp_8/rank_0/float16/74f/81/74f81fe167d99b4c.bin",
      "size_bytes": 524288,
      "created_at": "2024-03-15T09:30:00Z"
    },
    // ... 999,999 more blocks
  ]
}
```
- **Purpose**: Base state containing all blocks at a point in time
- **Created**: During compaction (merges snapshot + all deltas)
- **Size**: ~10-15 MB (Avro compressed) for 1M blocks

**3. Delta Files** (`delta-1710512700.avro`)
```json
{
  "delta_id": "delta-1710512700",
  "base_snapshot": "snapshot-1710512400",
  "timestamp": "2024-03-15T10:05:00Z",
  "operations": [
    {
      "type": "ADD",
      "block_hash": "750a1fe167d99b4e",
      "s3_key": "kv-cache/llama3-70b/tp_8/rank_0/float16/750/a1/750a1fe167d99b4e.bin",
      "size_bytes": 524288,
      "created_at": "2024-03-15T10:05:00Z"
    },
    {
      "type": "DELETE",
      "block_hash": "74f81fe167d99b4c"
    },
    // ... more operations
  ]
}
```
- **Purpose**: Incremental changes since last snapshot (MOR - Merge-On-Read)
- **Created**: Batched writes (every 1000 ops or 10 seconds)
- **Size**: ~100 KB per delta (for 1000 operations)

### Read Path (Loading Manifest)

```
1. Read pointer file
   └─> current-snapshot.avro
       ├─ current_snapshot: "snapshot-1710512400"
       └─ delta_files: ["delta-1710512700", "delta-1710512800", ...]

2. Read snapshot file
   └─> snapshot-1710512400.avro
       └─ Load 1,000,000 blocks into presence cache

3. Apply delta files (in order)
   ├─> delta-1710512700.avro (100 ops: 80 ADD, 20 DELETE)
   ├─> delta-1710512800.avro (150 ops: 120 ADD, 30 DELETE)
   ├─> delta-1710512900.avro (200 ops: 180 ADD, 20 DELETE)
   └─> delta-1710513000.avro (50 ops: 40 ADD, 10 DELETE)

4. Final presence cache state
   └─> 1,000,000 + 420 - 80 = 1,000,340 blocks
```

### Write Path (Adding Blocks)

**Important**: Delta files are **batched** - one delta file per 1000 blocks (or 10 seconds), NOT one per block!

```
Timeline of writing 2500 blocks:

T=0s: Block 1 written
  1. Store to S3: kv-cache/.../block001.bin
  2. Update cache: presence_cache.add("block001")
  3. Queue op: delta_queue.put({"type": "ADD", "block_hash": "block001"})
  
T=1s: Blocks 2-999 written
  └─> Same process, operations queued

T=2s: Block 1000 written
  └─> Batch threshold reached!
  
  4. Background thread creates delta file
     └─> manifests/.../delta-1710513100.avro (contains 1000 ADD operations)
  
  5. Update pointer file (conditional PUT with ETag)
     Before:
     {
       "current_snapshot": "snapshot-1710512400",
       "delta_files": [],
       "version": 1
     }
     
     After:
     {
       "current_snapshot": "snapshot-1710512400",
       "delta_files": ["delta-1710513100"],  ← Added
       "version": 2  ← Incremented
     }

T=3s: Blocks 1001-1999 written
  └─> Operations queued

T=4s: Block 2000 written
  └─> Second batch threshold reached!
  
  6. Create second delta file
     └─> manifests/.../delta-1710513200.avro (1000 more operations)
  
  7. Update pointer file again
     {
       "current_snapshot": "snapshot-1710512400",
       "delta_files": [
         "delta-1710513100",
         "delta-1710513200"  ← Added
       ],
       "version": 3
     }

T=5s: Blocks 2001-2500 written (only 500 blocks)
  └─> Operations queued, waiting...

T=15s: 10-second timeout reached
  └─> Even though only 500 ops, write delta anyway
  
  8. Create third delta file
     └─> manifests/.../delta-1710513300.avro (500 operations)
  
  9. Update pointer file
     {
       "current_snapshot": "snapshot-1710512400",
       "delta_files": [
         "delta-1710513100",
         "delta-1710513200",
         "delta-1710513300"  ← Added
       ],
       "version": 4
     }
```

**Result**: 2500 blocks written, but only **3 delta files** and **3 pointer updates** (not 2500!).

**Batching Benefits:**
- ✅ Reduces S3 API calls (3 writes instead of 2500)
- ✅ Reduces pointer file contention (3 updates instead of 2500)
- ✅ More efficient Avro compression
- ✅ Better performance under high write load

## Lifecycle Reconciliation

The system handles S3 lifecycle policies that expire/delete old cache blocks using a **two-tier approach**:

### 1. Lazy Invalidation (Primary Strategy)

**On GetObject Failure (404)**:
- Worker detects object not found
- Calls `manager.invalidate_cache_entry(block_hash)`
- Block removed from presence cache immediately
- vLLM treats as cache miss and continues

```python
# In worker._get_blocks()
try:
    data = self.s3_client.get_object(s3_key)
    # ... process data ...
except Exception as e:
    if e.response.get('Error', {}).get('Code') == '404':
        # Lazy invalidation: remove stale entry
        self.manager.invalidate_cache_entry(block_hash)
    # Treat as cache miss
    return (job_id, False)
```

**Benefits**:
- ✅ Zero cost (no extra API calls)
- ✅ Immediate invalidation on access
- ✅ Simple implementation
- ✅ Works for any deletion (lifecycle, manual, etc.)

**Tradeoff**:
- ⚠️ Presence cache may contain stale entries until accessed
- ⚠️ First access to expired block incurs GetObject cost + cache miss

### 2. LIST-Based Reconciliation (During Compaction)

**Periodic Cleanup** (every 24 hours or 100 deltas):
- Use S3 LIST API to get all existing blocks
- Filter manifest to only include existing blocks
- Much more efficient than HEAD per block

```python
def compact(self):
    manifest = self.load_manifest()  # 1,000,000 blocks
    
    # LIST all objects with prefix (1000 per page)
    existing_keys = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page['Contents']:
            existing_keys.add(obj['Key'])
    
    # Filter manifest
    reconciled = {h: k for h, k in manifest.items() if k in existing_keys}
    # 1,000,000 - 50,000 = 950,000 blocks
```

**Performance**:
- LIST: O(blocks/1000) API calls
  - 1M blocks = 1,000 LIST calls (~$0.005)
- HEAD: O(blocks) API calls
  - 1M blocks = 1,000,000 HEAD calls (~$400) ❌

**Cost Comparison** (1M blocks, 50k expired):
| Method | API Calls | Cost | Time |
|--------|-----------|------|------|
| LIST | 1,000 | $0.005 | ~10 seconds |
| HEAD | 1,000,000 | $400 | ~2.8 hours |
| Lazy | 50,000 (on access) | $0.02 | Distributed over time |

### Combined Strategy

```
Normal Operation:
  ├─> vLLM requests block
  ├─> Presence cache: "exists" ✅
  ├─> Worker: GetObject
  ├─> S3: 404 (expired)
  └─> Lazy invalidation: remove from cache

Compaction (every 24h):
  ├─> LIST all blocks in S3
  ├─> Filter manifest to existing blocks

### Cross-Instance Synchronization

**Problem**: Multiple vLLM instances with independent presence caches

```
Instance A: Compacts manifest, removes 50k expired blocks
Instance B: Still has 50k stale entries in cache
Instance C: Still has 50k stale entries in cache
```

**Solution**: Periodic manifest sync (every 5 minutes)

```python
def _refresh_cache_loop(self):
    while True:
        time.sleep(300)  # 5 minutes
        
        # Load latest manifest (includes compaction changes)
        manifest = load_manifest()
        
        # Full sync: add new blocks AND remove deleted blocks
        new_blocks = manifest_keys - current_keys
        deleted_blocks = current_keys - manifest_keys
        
        presence_cache.update(new_blocks)
        for block in deleted_blocks:
            presence_cache.remove(block)
```

**Timeline Example**:

```
T=0:00  Instance A compacts, removes 50k expired blocks
        ├─> Manifest updated in S3
        └─> Instance A cache: 950k blocks ✅

T=0:00  Instance B & C still have stale caches
        └─> Cache: 1M blocks (50k stale) ⚠️

T=0:30  Instance B accesses expired block
        ├─> GetObject → 404
        ├─> Lazy invalidation
        └─> Cache: 999,999 blocks (49,999 stale)

T=5:00  Instance B & C periodic sync
        ├─> Load manifest from S3
        ├─> Detect 50k deleted blocks
        ├─> Remove from cache
        └─> Cache: 950k blocks ✅

Result: All instances converged within 5 minutes
```

**Convergence Guarantees**:
- **Immediate**: Lazy invalidation on access (per-block)
- **5 minutes**: Periodic sync (bulk cleanup)
- **24 hours**: Compaction (authoritative source)

**Tradeoffs**:
- ⚠️ Up to 5-minute window with stale entries
- ✅ Acceptable: lazy invalidation handles access
- ✅ No coordination protocol needed
- ✅ Eventually consistent

  ├─> Write new snapshot
  └─> Presence cache refreshed on next load
```

**Result**: Best of both worlds
- Immediate cleanup on access (lazy)
- Periodic bulk cleanup (LIST)
- Minimal cost and latency

### Lifecycle Policy Example

```json
{
  "Rules": [
    {
      "Id": "ExpireOldCacheBlocks",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "kv-cache/"
      },
      "Expiration": {
        "Days": 30
      }
    }
  ]
}
```

With this policy:
- Cache blocks older than 30 days are automatically deleted
- Compaction removes stale references every 24 hours
- Presence cache stays accurate


## Immutability and Concurrency Control

The manifest system follows an **immutable data structure** pattern with a single mutable pointer:

### Immutable Files (Write-Once, Never Modified)

✅ **Snapshot files**: `snapshot-{timestamp}.avro`
- Named with unique timestamp
- Never modified after creation
- Can be safely read by multiple instances
- Old snapshots can be deleted after grace period

✅ **Delta files**: `delta-{timestamp}.avro`
- Named with unique timestamp
- Never modified after creation
- Append-only from system perspective
- Can be safely read by multiple instances

✅ **Cache blocks**: `{hash}.bin`
- Named with content hash
- Immutable (content-addressed storage)
- No collision risk (hash uniqueness)

### Mutable File (Single Point of Coordination)

⚠️ **Pointer file**: `current-snapshot.avro`
- **ONLY** file that gets updated
- Uses **conditional PUT** with ETag for concurrency control
- Atomic updates prevent race conditions
- Optimistic concurrency control (retry on conflict)

### Concurrency Control Example

```
Instance A and Instance B both want to add a delta:

T=0: Both read pointer (version=5, ETag="abc123")
     {
       "current_snapshot": "snapshot-1710512400",
       "delta_files": ["delta-1", "delta-2", "delta-3"],
       "version": 5
     }

T=1: Instance A writes delta-4.avro (immutable, no conflict)
     Instance B writes delta-5.avro (immutable, no conflict)

T=2: Instance A tries to update pointer
     PUT current-snapshot.avro
     If-Match: "abc123"  ← Must match current ETag
     Body: {
       "delta_files": ["delta-1", "delta-2", "delta-3", "delta-4"],
       "version": 6
     }
     ✅ SUCCESS (ETag matched, pointer updated, new ETag="def456")

T=3: Instance B tries to update pointer
     PUT current-snapshot.avro
     If-Match: "abc123"  ← Stale ETag!
     Body: {
       "delta_files": ["delta-1", "delta-2", "delta-3", "delta-5"],
       "version": 6
     }
     ❌ FAIL (412 Precondition Failed - ETag mismatch)

T=4: Instance B retries
     - Re-reads pointer (version=6, ETag="def456")
     - Sees delta-4 already added by Instance A
     - Updates to add delta-5
     PUT current-snapshot.avro
     If-Match: "def456"  ← Current ETag
     Body: {
       "delta_files": ["delta-1", "delta-2", "delta-3", "delta-4", "delta-5"],
       "version": 7
     }
     ✅ SUCCESS
```

### Benefits of This Design

✅ **No File Locks**: Immutable files don't need locking
✅ **High Concurrency**: Multiple instances can write deltas simultaneously
✅ **Conflict-Free Reads**: Readers never block writers
✅ **Automatic Retry**: Failed pointer updates retry with exponential backoff
✅ **Eventually Consistent**: All instances converge to same state
✅ **Collision-Free**: Timestamp-based names prevent overwrites

### Compaction Process

```
Trigger: 100 delta files OR 24 hours since last compaction

1. Load full state
   ├─> Read snapshot-1710512400.avro (1M blocks)
   └─> Apply all 100 delta files (10K operations)
   
2. Create new snapshot
   └─> snapshot-1710598800.avro (1,010,000 blocks after merging)

3. Update pointer (atomic operation)
   ├─> current_snapshot: "snapshot-1710598800"
   ├─> delta_files: []  ← Clear delta list
   └─> version: 106

4. Old files remain (for rollback/debugging)
   ├─> snapshot-1710512400.avro (can be deleted after grace period)
   └─> delta-*.avro files (can be deleted after grace period)
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
| `cache_max_size` | int | `1000000` | Maximum blocks in LRU cache (null for unbounded) |

### Manifest Scoping

Manifest files are **automatically scoped per vLLM instance** based on:
- Model name
- Tensor parallelism size (tp_size)
- Tensor parallelism rank (tp_rank)
- Data type (dtype)

This ensures each instance only loads relevant cache blocks, preventing memory waste from loading blocks for different models or configurations.

**Example manifest paths:**
```
manifests/llama3-70b/tp_8/rank_0/float16/current-snapshot.avro
manifests/llama3-70b/tp_8/rank_1/float16/current-snapshot.avro
manifests/qwen3-32b/tp_4/rank_0/bfloat16/current-snapshot.avro
```

### Manifest Sharing in Kubernetes

Instances with **identical configurations share the same manifest**, making this perfect for scaled Kubernetes deployments:

**Example: 3 replicas of Llama3-70B with TP=8, rank 0, float16**

```
┌─────────────────────────────────────────────────────────────┐
│ Kubernetes Deployment (replicas: 3)                        │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Pod 1        │  │ Pod 2        │  │ Pod 3        │     │
│  │ rank_0       │  │ rank_0       │  │ rank_0       │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                 │              │
│         └─────────────────┼─────────────────┘              │
│                           │                                │
└───────────────────────────┼────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │ Shared Manifest             │
              │ manifests/llama3-70b/       │
              │   tp_8/rank_0/float16/      │
              │   └─ current-snapshot.avro  │
              └─────────────────────────────┘
```

All 3 pods:
- ✅ **Share the same manifest** (same model/tp_size/tp_rank/dtype)
- ✅ **Coordinate writes** using conditional PUT (ETag-based concurrency control)
- ✅ **See each other's blocks** via periodic cache refresh (every 5 minutes)
- ✅ **Avoid redundant HEAD requests** for blocks written by other pods

**But different configurations get separate manifests:**

```
Llama3-70B TP=8 rank 0 float16:  manifests/llama3-70b/tp_8/rank_0/float16/
Llama3-70B TP=8 rank 1 float16:  manifests/llama3-70b/tp_8/rank_1/float16/
Llama3-70B TP=4 rank 0 float16:  manifests/llama3-70b/tp_4/rank_0/float16/
Qwen3-32B TP=4 rank 0 bfloat16:  manifests/qwen3-32b/tp_4/rank_0/bfloat16/
```

### Benefits of Scoped Manifests

✅ **Memory Efficiency**: Each instance only caches relevant blocks
✅ **Higher Hit Rate**: LRU cache focuses on frequently accessed blocks for that config
✅ **Automatic Sharing**: K8s replicas with identical configs share manifests
✅ **Isolation**: Different models/configs don't pollute each other's caches
✅ **Scalability**: Add replicas without increasing per-instance memory usage

### Multi-Model Deployment Example

```yaml
# Deployment 1: Llama3-70B with TP=8 (8 ranks × 3 replicas = 24 pods)
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: llama3-70b-tp8
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: vllm
        env:
        - name: VLLM_TP_SIZE
          value: "8"
        # Each rank (0-7) gets 3 replicas sharing the same manifest
        # Total: 8 unique manifests (one per rank)

# Deployment 2: Qwen3-32B with TP=4 (4 ranks × 2 replicas = 8 pods)
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qwen3-32b-tp4
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: vllm
        env:
        - name: VLLM_TP_SIZE
          value: "4"
        # Each rank (0-3) gets 2 replicas sharing the same manifest
        # Total: 4 unique manifests (one per rank)
```

**Result**: 32 total pods, but only 12 unique manifests (8 for Llama3 + 4 for Qwen3). Each pod only loads blocks relevant to its configuration, preventing memory waste from loading blocks for different models or tensor parallelism configurations.

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
- Configure `cache_max_size` to limit memory usage
- Default is 1M blocks (~8 MB RAM)
- Set lower for memory-constrained environments
- Set to `null` for unbounded cache (original behavior)

**Example:**
```json
{
  "cache_max_size": 500000  // Limit to 500K blocks (~4 MB)
}
```

### Stale Cache

**Symptom:** Cache misses for blocks that exist

**Cause:** Cache not refreshing

**Solution:**
- Check refresh thread is running
- Verify S3 connectivity
- Manually trigger refresh

## LRU Eviction Policy

The presence cache now implements **LRU (Least Recently Used) eviction** to prevent unbounded memory growth:

### How It Works

- Cache has configurable `cache_max_size` (default: 1,000,000 blocks)
- When cache is full, least recently accessed blocks are evicted
- Access order is updated on every lookup
- Thread-safe implementation using OrderedDict

### Configuration

```json
{
  "enable_presence_cache": true,
  "cache_max_size": 1000000  // Max blocks (default: 1M)
}
```

Set `cache_max_size` to `null` for unbounded cache (not recommended for production).

### Memory Usage

| cache_max_size | Memory Usage | Use Case |
|----------------|--------------|----------|
| 100,000 | ~0.8 MB | Small deployments |
| 1,000,000 | ~8 MB | Default (recommended) |
| 10,000,000 | ~80 MB | Large deployments |
| null | Unbounded | Testing only |

### Statistics

The cache tracks performance metrics:

```python
stats = cache.get_stats()
# {
#   "size": 950000,
#   "max_size": 1000000,
#   "hits": 5000000,
#   "misses": 50000,
#   "evictions": 100000,
#   "hit_rate": 0.99
# }
```

These stats are logged during cache refresh:
```
Refreshed cache: added 1000 new blocks, size=950000, evictions=100000, hit_rate=99.00%
```

### Benefits

✅ **Bounded memory** - Prevents OOM in long-running deployments
✅ **Automatic eviction** - No manual cache management needed
✅ **High hit rate** - LRU keeps frequently accessed blocks
✅ **Thread-safe** - Safe for concurrent access
✅ **Observable** - Built-in statistics tracking

## Future Enhancements

Potential improvements:
- Bloom filters for memory efficiency
- Distributed cache coordination (Redis/etcd)
- Incremental manifest loading
- Cache warming strategies
- Adaptive cache sizing based on workload
- Metrics export (Prometheus/OpenTelemetry)

## Summary

The presence cache with Iceberg-style manifests provides:

✅ **500x faster lookups** - In-memory vs S3 HEAD requests
✅ **Instant cold start** - Pre-warm from manifest in seconds
✅ **Multi-instance sharing** - All instances see same state
✅ **Minimal overhead** - <0.01% storage, ~8MB RAM per 1M blocks
✅ **Production-ready** - Compaction, refresh, conflict resolution

This is an optional feature that significantly improves performance for large-scale deployments with multiple vLLM instances sharing a cache bucket.

### 3. S3 Event Notifications (Future Enhancement)

**Real-Time Invalidation** via S3 bucket notifications:

```json
{
  "LambdaFunctionConfigurations": [{
    "Events": ["s3:ObjectRemoved:*"],
    "Filter": {
      "Key": {"FilterRules": [{"Name": "prefix", "Value": "kv-cache/"}]}
    },
    "LambdaFunctionArn": "arn:aws:lambda:region:account:function:invalidate-cache"
  }]
}
```

**Architecture**:
```
S3 Lifecycle → Delete Object
  ↓
S3 Event Notification
  ↓
SQS Queue (buffer)
  ↓
vLLM Instances (poll queue)
  ↓
Batch Invalidate Cache Entries
```

**Benefits**:
- ✅ Real-time invalidation (seconds, not hours)
- ✅ Zero GetObject failures
- ✅ No LIST API costs
- ✅ Scales to millions of blocks

**Implementation Complexity**: Medium
- Requires SQS queue setup
- vLLM needs background thread to poll queue
- Batch processing for efficiency

**Future Work**: Can be added as optional feature without breaking existing lazy/LIST approach.
