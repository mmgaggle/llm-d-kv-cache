# io_uring Zero-Copy S3 Integration Design

## Overview

This document describes the design for integrating Linux io_uring with the S3 backend to achieve zero-copy data transfer for KV cache blocks.

## Goals

1. **Zero-copy data path**: S3 → Kernel → Pinned Memory → GPU (no intermediate copies)
2. **Hybrid architecture**: io_uring for GetObject/PutObject, CRT client for control operations
3. **Graceful degradation**: Fall back to CRT client if io_uring unavailable
4. **Maintain multipathing**: Support multiple S3 endpoints via DNS round-robin
5. **Production ready**: Proper error handling, retries, monitoring

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    S3OffloadingHandler                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              S3ClientWrapper                         │   │
│  │  ┌────────────────┐         ┌──────────────────┐    │   │
│  │  │  CRT Client    │         │  IoUringPool     │    │   │
│  │  │  (Control)     │         │  (Data Path)     │    │   │
│  │  │                │         │                  │    │   │
│  │  │ - HeadObject   │         │ - GetObject      │    │   │
│  │  │ - ListObjects  │         │ - PutObject      │    │   │
│  │  │ - DeleteObject │         │                  │    │   │
│  │  │ - Manifests    │         │ Zero-copy to     │    │   │
│  │  │                │         │ pinned buffers   │    │   │
│  │  └────────────────┘         └──────────────────┘    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Pinned Buffer  │
                  │  Pool (CUDA)    │
                  └─────────────────┘
                           │
                           ▼
                      GPU Memory
```

### Data Flow: GetObject with io_uring

```
1. Request arrives for block_hash
   ↓
2. Allocate pinned buffer from pool
   ↓
3. Build HTTP GET request with SigV4 auth
   ↓
4. Establish HTTP/2 connection to S3 endpoint
   ↓
5. Submit io_uring read operation
   - Target: pinned buffer
   - Source: socket FD
   - Size: 62.5 MiB (for Qwen3-32B)
   ↓
6. io_uring completion (zero-copy complete)
   ↓
7. Trigger async GPU DMA from pinned buffer
   ↓
8. Return buffer to pool when GPU transfer complete
```

## Implementation Plan

### Phase 1: Foundation (Week 1)

**Files to create:**
- `src/llmd_s3_backend/iouring_pool.py` - io_uring connection pool
- `src/llmd_s3_backend/pinned_buffers.py` - CUDA pinned buffer management
- `src/llmd_s3_backend/s3_auth.py` - SigV4 signing for io_uring path

**Key components:**

1. **IoUringPool class:**
   ```python
   class IoUringPool:
       def __init__(self, queue_depth=1024, num_workers=16):
           self.ring = io_uring.IoUring(queue_depth)
           self.connections = {}  # endpoint -> [socket_fds]
           self.pinned_buffers = PinnedBufferPool()
       
       async def get_object_zerocopy(self, endpoint, key, pinned_buffer):
           # 1. Get or create HTTP/2 connection
           # 2. Build HTTP GET request
           # 3. Submit io_uring read to pinned buffer
           # 4. Wait for completion
           # 5. Return bytes read
   ```

2. **PinnedBufferPool class:**
   ```python
   class PinnedBufferPool:
       def __init__(self, buffer_size=65536*1024, num_buffers=64):
           # Pre-allocate pinned CUDA buffers
           self.buffers = [
               torch.cuda.ByteTensor(buffer_size).pin_memory()
               for _ in range(num_buffers)
           ]
           self.available = queue.Queue()
       
       def acquire(self) -> torch.Tensor:
           return self.available.get()
       
       def release(self, buffer: torch.Tensor):
           self.available.put(buffer)
   ```

3. **S3 SigV4 Signing:**
   ```python
   class S3SigV4Signer:
       def sign_request(self, method, endpoint, key, headers):
           # Implement AWS SigV4 signing
           # Return signed headers
   ```

### Phase 2: Integration (Week 2)

**Modify existing files:**
- `src/llmd_s3_backend/s3_client.py` - Add io_uring methods
- `src/llmd_s3_backend/worker.py` - Use pinned buffers

**Changes to S3ClientWrapper:**

```python
class S3ClientWrapper:
    def __init__(self, bucket, region=None, endpoint_url=None, 
                 addressing_style="auto", profile_name=None,
                 enable_iouring=True):
        # Existing CRT client
        self.crt_client = S3Client(...)
        
        # New io_uring pool (optional)
        self.iouring_pool = None
        if enable_iouring and self._check_iouring_support():
            self.iouring_pool = IoUringPool()
    
    def get_object(self, key: str, pinned_buffer=None) -> bytes:
        """Get object with optional zero-copy to pinned buffer."""
        if pinned_buffer is not None and self.iouring_pool:
            try:
                return self.iouring_pool.get_object_zerocopy(
                    self.endpoint_url, key, pinned_buffer
                )
            except Exception as e:
                logger.warning(f"io_uring failed, falling back to CRT: {e}")
        
        # Fallback to CRT client
        return self.crt_client.get_object(Bucket=self.bucket, Key=key)['Body'].read()
```

**Changes to S3GPUOffloadingHandler:**

```python
class S3GPUOffloadingHandler(S3OffloadingHandler):
    def __init__(self, ...):
        super().__init__(...)
        
        # Pre-allocate pinned buffer pool
        self.pinned_buffers = PinnedBufferPool(
            buffer_size=self._estimate_block_size(),
            num_buffers=self.threads_per_gpu * 2
        )
    
    def _get_blocks_from_s3(self, job_id, s3_key, tensors, block_ids):
        """Download with zero-copy if possible."""
        try:
            # Acquire pinned buffer
            pinned_buffer = self.pinned_buffers.acquire()
            
            # Zero-copy download
            bytes_read = self.s3_client.get_object(s3_key, pinned_buffer)
            
            # Deserialize from pinned buffer
            blocks_data = self._deserialize_from_pinned(pinned_buffer, bytes_read)
            
            # Async GPU transfer (already in pinned memory!)
            with torch.cuda.stream(self.h2d_stream):
                for tensor, block_data in zip(tensors, blocks_data):
                    block_tensor = torch.from_numpy(block_data)
                    # block_tensor is already pinned, fast DMA
                    tensor[:, block_ids, :, :, :] = block_tensor.to(
                        device=tensor.device, non_blocking=True
                    )
            
            # Release buffer back to pool
            self.pinned_buffers.release(pinned_buffer)
            
            return (job_id, True)
            
        except Exception as e:
            logger.error(f"Zero-copy failed: {e}")
            # Fallback to standard path
            return self._get_blocks_from_s3_standard(job_id, s3_key, tensors, block_ids)
```

### Phase 3: Testing & Benchmarking (Week 3)

**Test files to create:**
- `tests/test_iouring_pool.py` - Unit tests for io_uring
- `tests/test_pinned_buffers.py` - Buffer pool tests
- `tests/benchmark_iouring.py` - Performance comparison

**Benchmarks to run:**
1. Latency: io_uring vs CRT for single 62.5 MiB block
2. Throughput: Concurrent transfers (16, 32, 64 threads)
3. CPU usage: io_uring vs CRT under load
4. Memory efficiency: Pinned buffer pool vs on-demand allocation

### Phase 4: Production Hardening (Week 4)

**Features to add:**
1. Connection pooling and reuse
2. Automatic retry with exponential backoff
3. Health checks and circuit breakers
4. Metrics and monitoring (Prometheus)
5. Configuration options (enable/disable, tuning)
6. Documentation and examples

## Configuration

New configuration options in `spec.py`:

```python
@dataclass
class S3OffloadingSpec:
    # Existing fields...
    
    # io_uring configuration
    enable_iouring: bool = True
    iouring_queue_depth: int = 1024
    iouring_num_workers: int = 16
    
    # Pinned buffer configuration
    pinned_buffer_size_mb: int = 128  # Per buffer
    pinned_buffer_pool_size: int = 64  # Total buffers
```

## Performance Expectations

Based on your blog post results and io_uring characteristics:

| Metric | CRT Client | io_uring (Expected) | Improvement |
|--------|-----------|---------------------|-------------|
| Single block latency | ~15ms | ~8ms | 1.9x |
| Throughput (16 threads) | 60 GB/s | 90 GB/s | 1.5x |
| CPU usage | 40% | 15% | 2.7x |
| Memory copies | 2 | 0 | ∞ |

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| io_uring bugs | High | Fallback to CRT client |
| Platform compatibility | Medium | Runtime detection, graceful disable |
| HTTP/2 complexity | High | Use existing libraries (httpx) for framing |
| Memory pressure | Medium | Configurable buffer pool size |
| S3 auth issues | High | Thorough testing with real S3 |

## Dependencies

New Python packages needed:
- `liburing` or `python-liburing` - io_uring bindings
- `httpx[http2]` - HTTP/2 client for connection management
- `cryptography` - For SigV4 signing

Add to `pyproject.toml`:
```toml
[project.optional-dependencies]
iouring = [
    "liburing>=0.7.0",
    "httpx[http2]>=0.27.0",
    "cryptography>=42.0.0",
]
```

## Success Criteria

1. ✅ Zero-copy verified (no memcpy in data path)
2. ✅ Performance improvement >1.5x vs CRT
3. ✅ Graceful fallback working
4. ✅ All tests passing
5. ✅ Production deployment successful

## Timeline

- Week 1: Foundation (io_uring pool, pinned buffers, auth)
- Week 2: Integration (modify s3_client, worker)
- Week 3: Testing & benchmarking
- Week 4: Production hardening

**Total: 4 weeks to production-ready implementation**

## References

- [io_uring documentation](https://kernel.dk/io_uring.pdf)
- [AWS SigV4 signing](https://docs.aws.amazon.com/general/latest/gr/signature-version-4.html)
- [CUDA pinned memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory)
- [Your blog post](https://ceph.io/en/news/blog/2025/vllm-kv-caching/)