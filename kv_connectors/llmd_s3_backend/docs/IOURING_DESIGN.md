# io_uring Zero-Copy S3 Integration Design

> **Implementation Status: Prototype**
>
> | Component | File | Status |
> |-----------|------|--------|
> | io_uring bindings | `iouring_ops.py` | Implemented — liburing FFI, buffer registration, `prep_read_fixed` |
> | Connection pool | `iouring_pool.py` | **Prototype** — uses standard sockets, not actual io_uring for data transfer |
> | Pinned buffers | `pinned_buffers.py` | Implemented — allocation, acquire/release, GPU copy |
> | SigV4 signing | `s3_auth.py` | Implemented — full AWS SigV4 request signing |
> | Worker integration | `worker.py` | Partial — `_get_blocks_zerocopy` coded but calls prototype pool; CRT fallback is the active path |
> | True zero-copy (Phase 5) | — | Design only — `IORING_OP_READ_FIXED` to registered buffers not yet wired end-to-end |
>
> The CRT-based path (`io_driver="crt"`) is production-ready. The io_uring path
> can be enabled with `io_driver="io_uring"` but transparently falls back to CRT
> if unavailable or on error.

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
┌────────────────────────────────────────────────────────────┐
│                    S3OffloadingHandler                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              S3ClientWrapper                        │   │
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
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
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
- `tests/unit/test_iouring_ops.py` - Unit tests for io_uring
- `tests/unit/test_pinned_buffers.py` - Buffer pool tests
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

## Phase 5: True Zero-Copy with Registered Buffers

### Current Status (Prototype)

The current implementation (commits b9043b0 through 75db034) provides a working io_uring prototype but **does not achieve true zero-copy**. Performance testing revealed:

- **10MB objects**: 0.97x-1.05x speedup (acceptable)
- **64MB objects**: 0.40x speedup (2.5x slower than boto3) ⚠️

**Root cause**: The prototype uses `IORING_OP_RECV` which copies data from kernel space to userspace. For large objects (64MB KV cache blocks), this copy becomes a significant bottleneck.

### True Zero-Copy Architecture

To achieve true zero-copy, we need to use **registered buffers** with `IORING_OP_READ_FIXED`:

```
┌─────────────────────────────────────────────────────────────┐
│                    Kernel Space                             │
│                                                             │
│  ┌──────────────┐         ┌──────────────┐                  │
│  │  Socket      │  DMA    │  Registered  │                  │
│  │  Buffer      │────────▶│  Buffer      │                  │
│  │              │         │  (Pinned)    │                  │
│  └──────────────┘         └──────────────┘                  │
│                                  │                          │
└──────────────────────────────────┼──────────────────────────┘
                                   │ Zero-copy
                                   │ (no memcpy)
                                   ▼
                          ┌──────────────┐
                          │  Userspace   │
                          │  Pinned      │
                          │  Buffer      │
                          └──────────────┘
                                   │
                                   ▼
                              GPU Memory
```

### Implementation Steps

#### 1. Buffer Registration API

Add buffer registration to `IoUringContext` in `iouring_ops.py`:

```python
class IoUringContext:
    def __init__(self, queue_depth: int = 128, flags: int = 0):
        self.ring = liburing.io_uring(queue_depth, flags)
        self.registered_buffers = []
        self.buffer_map = {}  # buffer_id -> (address, size)
    
    def register_buffers(self, buffers: List[memoryview]) -> List[int]:
        """
        Register buffers with the kernel for zero-copy I/O.
        
        Args:
            buffers: List of memory views to register
            
        Returns:
            List of buffer IDs for use with IORING_OP_READ_FIXED
        """
        # Convert to iovec array
        iovecs = []
        for buf in buffers:
            iov = liburing.iovec()
            iov.iov_base = buf.obj  # Get underlying pointer
            iov.iov_len = len(buf)
            iovecs.append(iov)
        
        # Register with kernel
        ret = liburing.io_uring_register_buffers(
            self.ring,
            iovecs,
            len(iovecs)
        )
        if ret < 0:
            raise OSError(f"Failed to register buffers: {os.strerror(-ret)}")
        
        # Track registered buffers
        buffer_ids = []
        for i, buf in enumerate(buffers):
            buffer_id = len(self.registered_buffers)
            self.registered_buffers.append(buf)
            self.buffer_map[buffer_id] = (buf.obj, len(buf))
            buffer_ids.append(buffer_id)
        
        return buffer_ids
    
    def unregister_buffers(self):
        """Unregister all buffers."""
        if self.registered_buffers:
            liburing.io_uring_unregister_buffers(self.ring)
            self.registered_buffers.clear()
            self.buffer_map.clear()
```

#### 2. Fixed Buffer Read Operation

Add `IORING_OP_READ_FIXED` support:

```python
class IoUringContext:
    def prep_read_fixed(
        self,
        fd: int,
        buffer_id: int,
        offset: int,
        length: int,
        file_offset: int = 0
    ) -> int:
        """
        Prepare a fixed buffer read operation (zero-copy).
        
        Args:
            fd: File descriptor to read from
            buffer_id: ID of registered buffer
            offset: Offset within the buffer
            length: Number of bytes to read
            file_offset: Offset in the file (for file I/O)
            
        Returns:
            User data ID for tracking completion
        """
        sqe = self.get_sqe()
        if sqe is None:
            raise RuntimeError("No SQE available")
        
        # Get buffer info
        if buffer_id not in self.buffer_map:
            raise ValueError(f"Buffer {buffer_id} not registered")
        
        buf_addr, buf_size = self.buffer_map[buffer_id]
        if offset + length > buf_size:
            raise ValueError(f"Read exceeds buffer size")
        
        # Prepare fixed buffer read
        liburing.io_uring_prep_read_fixed(
            sqe,
            fd,
            buf_addr + offset,  # Target address in registered buffer
            length,
            file_offset,
            buffer_id  # Index in registered buffer array
        )
        
        # Set user data for tracking
        user_data = self.allocate_user_data()
        liburing.io_uring_sqe_set_data(sqe, user_data)
        
        self.stats.submissions += 1
        return user_data
```

#### 3. Update PinnedBufferPool

Modify `pinned_buffers.py` to support buffer registration:

```python
class PinnedBufferPool:
    def __init__(
        self,
        buffer_size_mb: int = 128,
        num_buffers: int = 64,
        register_with_iouring: bool = True
    ):
        self.buffer_size = buffer_size_mb * 1024 * 1024
        self.num_buffers = num_buffers
        
        # Allocate pinned buffers
        self.buffers = []
        for _ in range(num_buffers):
            # Allocate page-aligned memory for DMA
            buf = torch.empty(
                self.buffer_size,
                dtype=torch.uint8,
                pin_memory=True
            )
            self.buffers.append(buf)
        
        # Track buffer registration
        self.registered = False
        self.buffer_ids = []
        self.iouring_ctx = None
    
    def register_with_iouring(self, iouring_ctx: IoUringContext):
        """Register buffers with io_uring for zero-copy."""
        if self.registered:
            raise RuntimeError("Buffers already registered")
        
        # Get memory views of pinned buffers
        memviews = []
        for buf in self.buffers:
            # Get underlying memory view
            mv = memoryview(buf.numpy())
            memviews.append(mv)
        
        # Register with kernel
        self.buffer_ids = iouring_ctx.register_buffers(memviews)
        self.registered = True
        self.iouring_ctx = iouring_ctx
        
        logger.info(
            f"Registered {len(self.buffer_ids)} buffers "
            f"({self.buffer_size / (1024*1024):.1f} MB each) "
            f"with io_uring for zero-copy"
        )
    
    def get_buffer_id(self, buffer: PinnedBufferView) -> int:
        """Get io_uring buffer ID for a pinned buffer."""
        if not self.registered:
            raise RuntimeError("Buffers not registered with io_uring")
        
        # Find buffer index
        for i, buf in enumerate(self.buffers):
            if buffer._buffer is buf:
                return self.buffer_ids[i]
        
        raise ValueError("Buffer not found in pool")
```

#### 4. Update IoUringPool

Modify `iouring_pool.py` to use fixed buffer reads:

```python
class IoUringPool:
    def __init__(
        self,
        signer: S3SigV4Signer,
        endpoints: List[str],
        bucket: str,
        buffer_pool: PinnedBufferPool,
        config: IoUringConfig,
        use_https: bool = True
    ):
        # ... existing initialization ...
        
        # Register buffers with io_uring for zero-copy
        if buffer_pool:
            for worker in self.workers:
                buffer_pool.register_with_iouring(worker.ctx)
            logger.info("Enabled zero-copy with registered buffers")
    
    def get_object_zerocopy(
        self,
        key: str,
        pinned_buffer: PinnedBufferView,
        timeout: float = 30.0
    ) -> int:
        """
        Download S3 object with true zero-copy to pinned buffer.
        
        Uses IORING_OP_READ_FIXED for kernel-to-userspace zero-copy.
        """
        # ... existing connection setup ...
        
        # Get buffer ID for zero-copy read
        buffer_id = self.buffer_pool.get_buffer_id(pinned_buffer)
        
        # Submit fixed buffer read (zero-copy!)
        user_data = worker.ctx.prep_read_fixed(
            fd=sock.fileno(),
            buffer_id=buffer_id,
            offset=0,
            length=pinned_buffer.size,
            file_offset=0
        )
        
        # Submit and wait for completion
        worker.ctx.submit()
        event = worker.ctx.wait_completion(user_data, timeout)
        
        if not event.success:
            raise IOError(f"Zero-copy read failed: {event.error_code}")
        
        return event.result  # Bytes read
```

### Memory Alignment Requirements

For DMA to work efficiently, buffers must be page-aligned:

```python
def allocate_aligned_buffer(size: int) -> torch.Tensor:
    """Allocate page-aligned pinned buffer for DMA."""
    # Round up to page size (4KB)
    page_size = 4096
    aligned_size = ((size + page_size - 1) // page_size) * page_size
    
    # Allocate with alignment
    buf = torch.empty(
        aligned_size,
        dtype=torch.uint8,
        pin_memory=True
    )
    
    # Verify alignment
    addr = buf.data_ptr()
    assert addr % page_size == 0, "Buffer not page-aligned"
    
    return buf
```

### Testing Strategy

#### Unit Tests

```python
def test_buffer_registration():
    """Test buffer registration with io_uring."""
    ctx = IoUringContext(queue_depth=128)
    pool = PinnedBufferPool(buffer_size_mb=64, num_buffers=4)
    
    # Register buffers
    pool.register_with_iouring(ctx)
    
    # Verify registration
    assert pool.registered
    assert len(pool.buffer_ids) == 4
    
    # Cleanup
    ctx.unregister_buffers()

def test_zero_copy_read():
    """Test zero-copy read with IORING_OP_READ_FIXED."""
    # Create test file
    test_data = b"x" * (64 * 1024 * 1024)  # 64MB
    with open("/tmp/test.bin", "wb") as f:
        f.write(test_data)
    
    # Setup io_uring with registered buffers
    ctx = IoUringContext(queue_depth=128)
    pool = PinnedBufferPool(buffer_size_mb=128, num_buffers=1)
    pool.register_with_iouring(ctx)
    
    # Open file
    fd = os.open("/tmp/test.bin", os.O_RDONLY)
    
    # Zero-copy read
    buffer = pool.acquire()
    buffer_id = pool.get_buffer_id(buffer)
    
    user_data = ctx.prep_read_fixed(
        fd=fd,
        buffer_id=buffer_id,
        offset=0,
        length=len(test_data),
        file_offset=0
    )
    
    ctx.submit()
    event = ctx.wait_completion(user_data, timeout=5.0)
    
    # Verify
    assert event.success
    assert event.result == len(test_data)
    assert buffer.as_bytes()[:len(test_data)] == test_data
    
    # Cleanup
    os.close(fd)
    pool.release(buffer)
    ctx.unregister_buffers()
```

#### Performance Benchmarks

```python
def benchmark_zero_copy_vs_standard():
    """Compare zero-copy vs standard read performance."""
    test_sizes = [1, 10, 64, 128]  # MB
    
    for size_mb in test_sizes:
        # Create test data
        size = size_mb * 1024 * 1024
        test_data = os.urandom(size)
        
        # Benchmark standard read
        start = time.time()
        for _ in range(10):
            with open("/tmp/test.bin", "rb") as f:
                _ = f.read()
        standard_time = time.time() - start
        
        # Benchmark zero-copy read
        ctx = IoUringContext(queue_depth=128)
        pool = PinnedBufferPool(buffer_size_mb=size_mb * 2, num_buffers=1)
        pool.register_with_iouring(ctx)
        
        start = time.time()
        for _ in range(10):
            buffer = pool.acquire()
            buffer_id = pool.get_buffer_id(buffer)
            fd = os.open("/tmp/test.bin", os.O_RDONLY)
            
            user_data = ctx.prep_read_fixed(fd, buffer_id, 0, size, 0)
            ctx.submit()
            ctx.wait_completion(user_data, timeout=5.0)
            
            os.close(fd)
            pool.release(buffer)
        
        zerocopy_time = time.time() - start
        
        speedup = standard_time / zerocopy_time
        print(f"{size_mb}MB: {speedup:.2f}x speedup")
        
        # Expect >1.5x for large files
        if size_mb >= 64:
            assert speedup > 1.5, f"Expected >1.5x, got {speedup:.2f}x"
```

### Expected Performance Improvements

With true zero-copy implementation:

| Object Size | Current (Prototype) | With Registered Buffers | Improvement |
|-------------|---------------------|-------------------------|-------------|
| 10 MB       | 1.0x                | 1.2x                    | 1.2x        |
| 64 MB       | 0.4x ⚠️             | 1.8x                    | 4.5x        |
| 128 MB      | 0.2x ⚠️             | 2.5x                    | 12.5x       |

### Implementation Timeline

- **Week 1**: Buffer registration API and unit tests
- **Week 2**: Integrate with IoUringPool and PinnedBufferPool
- **Week 3**: Performance testing and optimization
- **Week 4**: Production deployment and monitoring

### Success Criteria

1. ✅ Buffers successfully registered with io_uring
2. ✅ `IORING_OP_READ_FIXED` operations working
3. ✅ Zero memcpy verified (strace/perf analysis)
4. ✅ >1.5x speedup for 64MB objects
5. ✅ >2.0x speedup for 128MB objects
6. ✅ All existing tests still passing

### References

- [io_uring registered buffers](https://kernel.dk/io_uring.pdf) - Section 5.3
- [IORING_OP_READ_FIXED documentation](https://man.archlinux.org/man/io_uring_prep_read_fixed.3.en)
- [Linux DMA requirements](https://www.kernel.org/doc/Documentation/DMA-API-HOWTO.txt)
- [Your blog post](https://ceph.io/en/news/blog/2025/vllm-kv-caching/)
