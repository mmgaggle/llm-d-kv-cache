# io_uring Development Guide

This guide explains how to develop and test the io_uring zero-copy integration on macOS using Docker containers.

## Overview

Since io_uring is a Linux-specific feature (kernel 5.1+), macOS developers need a Linux environment for development and testing. We provide Docker containers with all necessary dependencies pre-installed.

## Prerequisites

- Docker Desktop for Mac (with Linux VM support)
- Docker Compose
- At least 4GB RAM allocated to Docker

## Quick Start

### 1. Build the Development Container

```bash
cd kv_connectors/llmd_s3_backend
docker-compose -f docker-compose.iouring.yml build
```

### 2. Start the Development Environment

```bash
# Start container with interactive shell
docker-compose -f docker-compose.iouring.yml run --rm iouring-dev

# Or start in background
docker-compose -f docker-compose.iouring.yml up -d
docker-compose -f docker-compose.iouring.yml exec iouring-dev bash
```

### 3. Run Tests Inside Container

```bash
# Inside container
python -m pytest tests/test_pinned_buffers.py -v
python -m pytest tests/test_iouring_pool.py -v  # When implemented
```

### 4. Test with Local Ceph RGW (S3-compatible)

```bash
# Start Ceph RGW alongside dev container
docker-compose -f docker-compose.iouring.yml up -d ceph-rgw

# Access Ceph RGW at http://localhost:8080
# Default credentials: demo/demo

# Inside dev container, configure for Ceph
export AWS_ACCESS_KEY_ID=demo
export AWS_SECRET_ACCESS_KEY=demo
export S3_ENDPOINT=http://ceph-rgw:8080
```

**Note:** For production testing, use your existing Ceph cluster as documented in your blog post.

## Development Workflow

### Live Code Editing

The container mounts your local source directories:
- `./src` → `/workspace/src`
- `./tests` → `/workspace/tests`
- `./docs` → `/workspace/docs`

Changes made on macOS are immediately visible in the container.

### Running Specific Tests

```bash
# Inside container

# Test pinned buffers
python -m pytest tests/test_pinned_buffers.py::TestPinnedBufferPool::test_concurrent_access -v

# Test io_uring pool
python -m pytest tests/test_iouring_pool.py -v -s

# Run with coverage
python -m pytest tests/ --cov=llmd_s3_backend --cov-report=html
```

### Benchmarking

```bash
# Inside container

# Benchmark pinned vs pageable memory
python -m pytest tests/test_pinned_buffers.py::TestPinnedBufferPerformance -v -s

# Benchmark io_uring vs standard sockets
python tests/benchmark_iouring.py  # When implemented
```

## Container Architecture

### Base Image
- Ubuntu 22.04 LTS
- Linux kernel 5.15+ (supports io_uring)
- Python 3.11

### Installed Packages
- `liburing-dev` - io_uring C library
- `liburing2` - io_uring runtime
- `liburing` (Python) - Python bindings for io_uring
- PyTorch (CPU version)
- All project dependencies

### Capabilities
The container runs with elevated capabilities for io_uring:
- `CAP_SYS_ADMIN` - Required for io_uring setup
- `CAP_SYS_RESOURCE` - Required for memory locking
- `seccomp:unconfined` - Allows io_uring syscalls

## Verifying io_uring Support

### Check Kernel Version

```bash
# Inside container
uname -r
# Should show 5.15 or higher
```

### Check liburing Installation

```bash
# Inside container
python3 -c "import liburing; print(liburing.__version__)"
```

### Test io_uring Functionality

```bash
# Inside container
python3 << EOF
import liburing

# Create io_uring instance
ring = liburing.io_uring()
ring.queue_init(32, 0)
print(f"io_uring initialized with queue depth 32")
ring.queue_exit()
print("io_uring test successful!")
EOF
```

## Troubleshooting

### Container Won't Start

**Issue:** `docker-compose up` fails with capability errors

**Solution:** Ensure Docker Desktop has sufficient permissions:
```bash
# Check Docker Desktop settings
# Preferences → Resources → Advanced
# Ensure "Use kernel networking for UDP" is enabled
```

### io_uring Not Available

**Issue:** `liburing` import fails or io_uring operations fail

**Solution:** Verify kernel version and rebuild container:
```bash
docker-compose -f docker-compose.iouring.yml build --no-cache
```

### Permission Denied Errors

**Issue:** Cannot write to mounted volumes

**Solution:** Check file permissions on macOS:
```bash
chmod -R 755 src/ tests/ docs/
```

### Ceph RGW Connection Fails

**Issue:** Cannot connect to Ceph RGW from container

**Solution:** Use container network name:
```bash
# Inside container, use 'ceph-rgw' as hostname
export S3_ENDPOINT=http://ceph-rgw:8080

# From macOS, use localhost
export S3_ENDPOINT=http://localhost:8080
```

**For production Ceph cluster:**
```bash
# Use your actual Ceph RGW endpoint from blog post setup
export S3_ENDPOINT=http://s3.cephlab.com
```

## Performance Considerations

### Docker on macOS Limitations

Docker Desktop on macOS runs Linux in a VM, which adds overhead:
- **Network:** ~10-20% slower than native Linux
- **Disk I/O:** Significantly slower for bind mounts
- **Memory:** Shared with macOS, may cause swapping

### Optimization Tips

1. **Use Docker volumes instead of bind mounts for data:**
   ```yaml
   volumes:
     - iouring-data:/data  # Fast
     # vs
     - ./data:/data        # Slow on macOS
   ```

2. **Allocate sufficient resources:**
   - Docker Desktop → Preferences → Resources
   - CPUs: 4+
   - Memory: 8GB+
   - Swap: 2GB+

3. **Use BuildKit for faster builds:**
   ```bash
   export DOCKER_BUILDKIT=1
   docker-compose build
   ```

## Production Deployment

The io_uring implementation is designed to work on native Linux systems. The Docker container is for **development and testing only**.

For production:
1. Deploy on Linux hosts with kernel 5.1+
2. Install `liburing` system package
3. Install Python `liburing` bindings
4. No container overhead

## Next Steps

1. **Implement real io_uring operations** in `iouring_pool.py`
2. **Add comprehensive tests** in `tests/test_iouring_pool.py`
3. **Benchmark** against CRT client
4. **Integrate** with worker.py
5. **Test** on production Linux systems

## References

- [io_uring documentation](https://kernel.dk/io_uring.pdf)
- [liburing GitHub](https://github.com/axboe/liburing)
- [Docker Desktop for Mac](https://docs.docker.com/desktop/mac/)
- [MinIO Documentation](https://min.io/docs/minio/linux/index.html)