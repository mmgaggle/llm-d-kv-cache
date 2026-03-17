# Integration Tests

This directory contains integration tests for testing component interactions.

**Important:** Not all tests here require live S3! See below for details.

## Test Files

### Live S3 Tests (Require Ceph/S3 Endpoint)

- **test_ceph.py** - ⚠️ **Requires live Ceph/S3 endpoint**
  - Basic Ceph S3 connectivity check
  - Verifies bucket access with configured profile/endpoint
  - **Run manually** with S3 credentials

- **test_iouring_live.py** - ⚠️ **Requires live S3/Ceph endpoint + Linux**
  - End-to-end io_uring zero-copy transfer test
  - Verifies pinned buffer pool, io_uring operations, and S3 auth
  - Requires Linux kernel 5.1+ with io_uring support
  - **Run manually** with S3 credentials

- **test_test_presence_cache_live.py** - ⚠️ **Requires live S3/Ceph endpoint**
  - Manifest creation and loading
  - Conditional PUT for pointer updates
  - Multi-instance simulation
  - DELETE operations
  - Concurrent ADD/DELETE operations
  - ETag conflict handling
  - Avro serialization
  - **Run manually** with S3 credentials

- **validate_s3_cache.sh** - ⚠️ **Requires live S3/Ceph endpoint + vLLM**
  - End-to-end validation script
  - Tests vLLM with S3 backend
  - Verifies cache blocks written to S3
  - Can auto-start vLLM server
  - **Run manually** for full system validation

### Logic Tests (No External Dependencies)

- **test_presence_cache_removal.py** - ✅ **No S3 required, runs with pytest**
  - Cache removal operations
  - Manager API verification (`invalidate_cache_entry`)
  - Manifest DELETE operation processing
  - Cache synchronization with deletions
  - Full lifecycle testing (add → verify → delete → verify removed)
  - **Can run in CI/CD** - uses only mocks and logic tests

## Running Tests

### Live S3 Integration Tests

The `test_presence_cache_live.py` script validates the presence cache functionality against a real Ceph S3 endpoint or AWS S3.

#### Basic Usage (Ceph)

```bash
cd kv_connectors/llmd_s3_backend
python tests/integration/test_presence_cache_live.py \
    --bucket vllm \
    --profile zgw \
    --endpoint http://your-ceph-endpoint:8080
```

#### AWS S3

```bash
cd kv_connectors/llmd_s3_backend
python tests/integration/test_presence_cache_live.py \
    --bucket your-bucket \
    --profile your-profile \
    --region us-east-1
```

#### Options

- `--bucket` (required): S3 bucket name
- `--endpoint`: S3 endpoint URL (for Ceph or S3-compatible storage)
- `--profile`: AWS profile name from `~/.aws/credentials`
- `--region`: AWS region (default: us-east-1)
- `--no-cleanup`: Skip cleanup of test data (useful for debugging)

### Logic Tests (No S3 Required)

The `test_presence_cache_removal.py` tests use NO external dependencies and can be run anywhere:

```bash
cd kv_connectors/llmd_s3_backend
source venv/bin/activate
python -m pytest tests/integration/test_presence_cache_removal.py -v
```

### End-to-End Validation Script

The `validate_s3_cache.sh` script provides comprehensive end-to-end validation of the S3 cache backend with a live vLLM server.

#### Basic Usage

```bash
cd kv_connectors/llmd_s3_backend
source venv/bin/activate
./tests/integration/validate_s3_cache.sh
```

The script will:
1. Check if vLLM is running (or auto-start it)
2. Send test requests to vLLM
3. Verify cache blocks are written to S3
4. Check for errors in logs

#### Configuration

Environment variables:
- `VLLM_PORT` - vLLM server port (default: 8000)
- `VLLM_URL` - vLLM server URL (default: http://localhost:8000)
- `MODEL` - Model to use (default: ibm-granite/granite-3b-code-instruct)
- `S3_BUCKET` - S3 bucket name (default: vllm)
- `S3_PROFILE` - AWS profile name (default: zgw)
- `S3_PREFIX` - S3 key prefix (default: kv-cache)
- `AUTO_START` - Auto-start vLLM if not running (default: true)

#### Example with Custom Settings

```bash
cd kv_connectors/llmd_s3_backend
source venv/bin/activate
S3_BUCKET=my-bucket S3_PROFILE=my-profile ./tests/integration/validate_s3_cache.sh
```

#### What It Validates

✅ vLLM server is running and responding
✅ API requests complete successfully
✅ Cache blocks are written to S3
✅ S3 credentials and configuration are correct
✅ End-to-end integration works

## Test Coverage

### Live Tests (test_presence_cache_live.py)

1. **Basic Operations**
   - Creates ManifestManager
   - Writes delta files
   - Loads manifest from S3
   - Verifies block count

2. **Conditional PUT (Single Instance)**
   - Sequential delta writes
   - Verifies pointer updates
   - Checks version increments

3. **Conditional PUT (Concurrent)**
   - Simulates 3 vLLM instances writing simultaneously
   - Each instance writes 2 batches of 2 blocks
   - Tests for race conditions and lost updates
   - Validates optimistic concurrency control

4. **DELETE Operations**
   - Writes blocks then deletes some
   - Verifies deletions are persisted
   - Checks remaining blocks are intact

5. **Concurrent ADD/DELETE**
   - Multiple instances performing mixed operations
   - Validates data integrity
   - Ensures no duplicates

6. **ETag Conflict Handling**
   - Reads pointer with ETag
   - Updates pointer (changes ETag)
   - Attempts conditional PUT with old ETag
   - Verifies 412 Precondition Failed response

7. **Avro Serialization**
   - Writes 100 blocks
   - Verifies `.avro` file extension
   - Validates deserialization

### Removal Tests (test_presence_cache_removal.py)

1. **Cache Removal Operations**
   - Single and multiple block removal
   - Removal during sync operations

2. **Manager Integration**
   - invalidate_block() API
   - Cache sync with manifest deletions

3. **Manifest DELETE Processing**
   - DELETE operation structure
   - Processing deletes during manifest load

4. **Full Lifecycle**
   - Add -> verify -> delete -> verify removed
   - End-to-end integration

## Expected Output (Live Tests)

```
Presence Cache Live Integration Test
Bucket: vllm
Endpoint: http://your-ceph:8080
Profile: zgw
✓ Connected to S3

[TEST] Basic Manifest Operations
✓ Created ManifestManager
✓ Loaded empty manifest
✓ Wrote delta batch with 5 blocks
✓ Loaded manifest with 5 blocks

[TEST] DELETE Operations
✓ Added 10 blocks
✓ Verified 10 blocks in manifest
✓ Wrote DELETE operations for 5 blocks
✓ Verified deletions: 5 blocks remaining

[TEST] Concurrent ADD/DELETE Operations
✓ Created 3 ManifestManagers
✓ Pre-populated with 20 blocks
ℹ Starting concurrent ADD/DELETE operations...
✓ Manager 0 completed operations
✓ Manager 1 completed operations
✓ Manager 2 completed operations
ℹ Final manifest has 20 blocks
✓ No duplicates - data integrity maintained
✓ Concurrent ADD/DELETE operations completed successfully

Test Summary
  PASS Basic Operations
  PASS Conditional PUT (Single)
  PASS Conditional PUT (Concurrent)
  PASS ETag Conflict Handling
  PASS Avro Serialization
  PASS DELETE Operations
  PASS Concurrent ADD/DELETE

Result: 7/7 tests passed
```

## Troubleshooting

### Connection Issues

```
✗ Failed to connect to S3: ...
```

**Solutions:**
- Verify endpoint URL is correct
- Check AWS credentials in `~/.aws/credentials`
- Ensure bucket exists and you have permissions
- Test with `aws s3 ls s3://your-bucket --profile your-profile --endpoint-url http://...`

### Conditional PUT Failures

```
✗ Conditional PUT with old ETag should have failed
```

**Possible causes:**
- Ceph version doesn't support conditional PUT (requires Ceph Nautilus 14.2+ or later)
- S3 compatibility mode disabled
- Check Ceph logs for errors

### Lost Updates in Concurrent Test

```
✗ Lost updates detected: blocks missing
```

**This indicates a problem with conditional PUT!**
- Verify ETag support in your S3 implementation
- Check if `IfMatch` header is being honored
- Review Ceph configuration for S3 compatibility

## Debugging

### Keep Test Data

```bash
python integration/test_presence_cache_live.py --bucket vllm --profile zgw --no-cleanup
```

Then inspect files manually:
```bash
aws s3 ls s3://vllm/test-manifests/ --profile zgw --endpoint-url http://...
aws s3 cp s3://vllm/test-manifests/current-snapshot.avro . --profile zgw --endpoint-url http://...
```

### Verbose Logging

Add to script:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## What This Validates

✅ **Avro serialization** works with your S3 implementation  
✅ **Conditional PUT** prevents race conditions  
✅ **ETag support** is functional  
✅ **Multi-instance coordination** works correctly  
✅ **DELETE operations** are properly handled  
✅ **Cache removal** maintains data integrity  
✅ **Manifest system** is production-ready  

## Prerequisites

1. **Ceph S3 endpoint** with credentials configured (for live tests)
2. **Python environment** with dependencies installed:
   ```bash
   cd kv_connectors/llmd_s3_backend
   source venv/bin/activate
   pip install -e .
   ```

## Notes

- These tests are **not** run as part of the standard unit test suite
- Live tests require manual execution with proper S3 credentials
- Removal tests use mocks and can be run with pytest
- All test files follow the `test_` naming convention for pytest discovery
- Live S3 tests should be excluded from CI (they require real infrastructure)

## Next Steps

After successful testing:
1. Enable presence cache in production: `enable_presence_cache=True`
2. Monitor manifest file sizes and compaction
3. Adjust `compaction_threshold` based on workload
4. Set up monitoring for pointer update conflicts
5. Monitor cache removal operations in production logs