# Testing Guide for llmd-s3-backend

This guide covers how to test the S3 backend connector for vLLM KV cache offloading.

## Prerequisites

- Python 3.12 (required for vLLM compatibility)
- S3-compatible storage (AWS S3, Ceph, MinIO, etc.)
- GPU (for vLLM testing)

## Setup

### 1. Install Python 3.12

On macOS with Homebrew:
```bash
brew install python@3.12
```

### 2. Create Virtual Environment

```bash
cd kv_connectors/llmd_s3_backend
/opt/homebrew/bin/python3.12 -m venv venv
source venv/bin/activate
```

### 3. Install Package with Dependencies

```bash
pip install -e ".[dev]"
```

This installs:
- Core dependencies (boto3, torch, numpy)
- vLLM (for integration testing)
- Testing tools (pytest, black, ruff, moto)

## Running Unit Tests

Unit tests use mocked S3 operations and don't require actual S3 connectivity.

### Run All Tests

```bash
# From the llmd_s3_backend directory
source venv/bin/activate
pytest tests/ -v
```

### Run Specific Test Classes

```bash
# Test S3LoadStoreSpec
pytest tests/test_s3_backend.py::TestS3LoadStoreSpec -v

# Test S3ClientWrapper
pytest tests/test_s3_backend.py::TestS3ClientWrapper -v

# Test S3OffloadingManager
pytest tests/test_s3_backend.py::TestS3OffloadingManager -v
```

### Run with Coverage

```bash
pytest tests/ --cov=llmd_s3_backend --cov-report=html
```

## Testing with Real S3 Storage

### AWS S3

1. Configure AWS credentials:
```bash
aws configure
# Or set environment variables:
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-west-2
```

2. Create test bucket:
```bash
aws s3 mb s3://my-kv-cache-test-bucket
```

3. Run connectivity test:
```bash
source venv/bin/activate
python << 'EOF'
from llmd_s3_backend.s3_client import S3ClientWrapper

client = S3ClientWrapper(
    bucket="my-kv-cache-test-bucket",
    region="us-west-2"
)

print(f"✓ Connected to bucket: {client.bucket}")
print(f"  Region: {client.region}")

# Test object operations
test_key = "test/connectivity.txt"
exists = client.object_exists(test_key)
print(f"  Test object exists: {exists}")
EOF
```

### Ceph S3

1. Configure Ceph credentials in `~/.aws/credentials`:
```ini
[zgw]
aws_access_key_id = your_ceph_access_key
aws_secret_access_key = your_ceph_secret_key
region = us-east-1
```

2. Configure endpoint in `~/.aws/config`:
```ini
[profile zgw]
region = us-east-1
s3 =
    endpoint_url = http://your-ceph-endpoint:port
    addressing_style = path
```

3. Create test bucket (if needed):
```bash
aws --profile zgw s3 mb s3://vllm
```

4. Run connectivity test:
```bash
source venv/bin/activate
python << 'EOF'
from llmd_s3_backend.s3_client import S3ClientWrapper

client = S3ClientWrapper(
    bucket="vllm",
    profile_name="zgw"
)

print(f"✓ Connected to Ceph S3")
print(f"  Bucket: {client.bucket}")
print(f"  Region: {client.region}")
print(f"  Endpoint: {client.endpoint_url}")

# Test object operations
test_key = "test/connectivity.txt"
exists = client.object_exists(test_key)
print(f"  Test object exists: {exists}")
EOF
```

Or use the provided test script:
```bash
source venv/bin/activate
python test_ceph.py
```

### Local Ceph Container (Development Testing)

For local development and testing, you can use a single-container Ceph setup:

1. Clone and start the Ceph container:
```bash
git clone https://github.com/likid0/zgw-posix.git
cd zgw-posix
# Follow the repository's README for setup instructions
```

2. Configure AWS CLI for the local Ceph instance:
```bash
# Add to ~/.aws/credentials
[zgw]
aws_access_key_id = your_access_key
aws_secret_access_key = your_secret_key
region = us-east-1

# Add to ~/.aws/config
[profile zgw]
region = us-east-1
s3 =
    endpoint_url = http://localhost:8000
    addressing_style = path
```

3. Create test bucket:
```bash
aws --profile zgw s3 mb s3://vllm
```

4. Verify connectivity:
```bash
source venv/bin/activate
python test_ceph.py
```

## Testing with vLLM

### Download Model (Optional but Recommended)

To see download progress and avoid timeouts, pre-download the model:

```bash
source venv/bin/activate
./download_model.sh
```

This will:
- Download the Granite 3B model with progress indicators
- Cache it locally in `~/.cache/huggingface/hub` (or `$HF_HOME`)
- Allow vLLM to start faster on subsequent runs

You can also download a different model:
```bash
MODEL=ibm-granite/granite-8b-code-instruct ./download_model.sh
```

### Small Model Test (Granite 3B)

1. Ensure virtual environment is activated:
```bash
source venv/bin/activate
```

2. (Optional) Download model first to see progress:
```bash
./download_model.sh
```

3. Start vLLM server with S3 backend:

**For AWS S3:**
```bash
vllm serve ibm-granite/granite-3b-code-instruct \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "spec_name": "S3OffloadingSpec",
      "spec_module_path": "llmd_s3_backend.spec",
      "s3_bucket": "my-kv-cache-bucket",
      "s3_region": "us-west-2",
      "block_size": 256,
      "threads_per_gpu": 64
    }
  }' \
  --distributed-executor-backend mp \
  --max-model-len 2048
```

**For Ceph S3:**
```bash
vllm serve ibm-granite/granite-3b-code-instruct \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "spec_name": "S3OffloadingSpec",
      "spec_module_path": "llmd_s3_backend.spec",
      "s3_bucket": "vllm",
      "s3_profile_name": "zgw",
      "block_size": 256,
      "threads_per_gpu": 64
    }
  }' \
  --distributed-executor-backend mp \
  --max-model-len 2048
```

3. Test the server:
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ibm-granite/granite-3b-code-instruct",
    "prompt": "Write a Python function to calculate fibonacci numbers:",
    "max_tokens": 100
  }'
```

### Verify S3 Operations

Check that KV cache blocks are being stored in S3:

**AWS S3:**
```bash
aws s3 ls s3://my-kv-cache-bucket/kv-cache/ --recursive
```

**Ceph S3:**
```bash
aws --profile zgw s3 ls s3://vllm/kv-cache/ --recursive
```

You should see objects with paths like:
```
kv-cache/granite-3b-code-instruct/tp_1/rank_0/float16/abc/de/abcdef0123456789.bin

### End-to-End Validation

Use the provided validation script to test the complete workflow:

```bash
source venv/bin/activate
./validate_s3_cache.sh
```

This script will:
1. Start vLLM server with S3 backend (Ceph configuration)
2. Wait for server to be ready
3. Send test requests to the OpenAI API endpoint
4. Check vLLM logs for errors
5. Verify cache blocks are stored in S3
6. Stop and restart vLLM to test cold cache retrieval
7. Send another request to verify cache blocks are loaded from S3

**Manual validation steps:**

1. Start vLLM server (in one terminal):
```bash
source venv/bin/activate
vllm serve ibm-granite/granite-3b-code-instruct \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "spec_name": "S3OffloadingSpec",
      "spec_module_path": "llmd_s3_backend.spec",
      "s3_bucket": "vllm",
      "s3_profile_name": "zgw",
      "block_size": 256,
      "threads_per_gpu": 64
    }
  }' \
  --distributed-executor-backend mp \
  --max-model-len 2048 \
  --port 8000
```

2. Send test requests (in another terminal):
```bash
# First request - will create cache blocks
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ibm-granite/granite-3b-code-instruct",
    "prompt": "Write a Python function to calculate fibonacci numbers:",
    "max_tokens": 100
  }'

# Second request with same prompt - should reuse cache
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ibm-granite/granite-3b-code-instruct",
    "prompt": "Write a Python function to calculate fibonacci numbers:",
    "max_tokens": 100
  }'
```

3. Check vLLM logs for S3 operations:
```bash
# Look for lines indicating S3 store/load operations
# Should see messages about offloading blocks to S3
```

4. Verify cache blocks in S3:
```bash
aws --profile zgw s3 ls s3://vllm/kv-cache/ --recursive
```

5. Stop vLLM (Ctrl+C in the server terminal)

6. Restart vLLM with the same configuration

7. Send the same request again - should load cache blocks from S3:
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ibm-granite/granite-3b-code-instruct",
    "prompt": "Write a Python function to calculate fibonacci numbers:",
    "max_tokens": 100
  }'
```

8. Check logs for cache retrieval messages

```

## Configuration Parameters

### S3 Backend Parameters

- `s3_bucket` (required): S3 bucket name
- `s3_prefix` (default: "kv-cache"): S3 key prefix
- `s3_region` (default: from env or "us-east-1"): AWS region
- `s3_endpoint_url`: Custom endpoint for S3-compatible services
- `s3_addressing_style` (default: "auto"): "auto", "path", or "virtual"
- `s3_profile_name`: AWS profile name from credentials file
- `block_size` (default: 256): Number of GPU blocks per S3 object
- `threads_per_gpu` (default: 64): I/O threads per GPU
- `max_staging_memory_gb`: Staging memory limit in GB

### Environment Variables

- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `AWS_REGION`: AWS region
- `STORAGE_CONNECTOR_DEBUG`: Enable debug logs

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'llmd_s3_backend'`:
```bash
# Ensure you're in the virtual environment
source venv/bin/activate

# Reinstall in editable mode
pip install -e .
```

### S3 Connection Issues

1. Verify credentials:
```bash
aws s3 ls --profile zgw  # For Ceph with profile
aws s3 ls                 # For default AWS credentials
```

2. Check endpoint configuration:
```bash
aws configure get s3.endpoint_url --profile zgw
```

3. Enable debug logging:
```bash
export STORAGE_CONNECTOR_DEBUG=1
```

### vLLM Issues

1. Check vLLM version:
```bash
pip show vllm
# Should be >= 0.11.0
```

2. Verify GPU availability:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

3. Check vLLM logs for S3 operations:
```bash
# Look for lines containing "S3" or "offload"
```

## Code Formatting and Linting

```bash
# Format code
make format

# Check formatting
make lint

# Or manually:
black src/ tests/
ruff check src/ tests/
```

## Clean Up

```bash
# Remove test objects from S3
aws s3 rm s3://my-kv-cache-bucket/kv-cache/ --recursive

# Or with Ceph
aws --profile zgw s3 rm s3://vllm/kv-cache/ --recursive

# Deactivate virtual environment
deactivate