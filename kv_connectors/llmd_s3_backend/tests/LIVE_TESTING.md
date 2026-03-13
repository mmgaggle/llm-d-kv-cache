# Live Integration Testing with Ceph S3

This directory contains a live integration test script for validating the presence cache functionality against a real Ceph S3 endpoint.

## Test Script

**`test_presence_cache_live.py`** - Comprehensive integration tests for:
- Basic manifest operations (create, load, update)
- Conditional PUT with ETag-based concurrency control
- Multi-instance simulation with concurrent updates
- ETag conflict detection and retry logic
- Avro serialization and file format validation

## Prerequisites

1. **Ceph S3 endpoint** with credentials configured
2. **Python environment** with dependencies installed:
   ```bash
   cd kv_connectors/llmd_s3_backend
   source venv/bin/activate
   pip install -e .
   ```

## Running the Tests

### Basic Usage (Ceph)

```bash
cd kv_connectors/llmd_s3_backend/tests
python test_presence_cache_live.py \
    --bucket vllm \
    --profile zgw \
    --endpoint http://your-ceph-endpoint:8080
```

### AWS S3

```bash
python test_presence_cache_live.py \
    --bucket your-bucket \
    --profile your-profile \
    --region us-east-1
```

### Options

- `--bucket` (required): S3 bucket name
- `--endpoint`: S3 endpoint URL (for Ceph or S3-compatible storage)
- `--profile`: AWS profile name from `~/.aws/credentials`
- `--region`: AWS region (default: us-east-1)
- `--no-cleanup`: Skip cleanup of test data (useful for debugging)

## Test Coverage

### 1. Basic Operations
- Creates ManifestManager
- Writes delta files
- Loads manifest from S3
- Verifies block count

### 2. Conditional PUT (Single Instance)
- Sequential delta writes
- Verifies pointer updates
- Checks version increments

### 3. Conditional PUT (Concurrent)
- **Simulates 3 vLLM instances** writing simultaneously
- Each instance writes 2 batches of 2 blocks
- Tests for race conditions and lost updates
- **Key validation**: All 12 blocks must be present (no lost updates)

### 4. ETag Conflict Handling
- Reads pointer with ETag
- Updates pointer (changes ETag)
- Attempts conditional PUT with old ETag
- **Verifies 412 Precondition Failed** response

### 5. Avro Serialization
- Writes 100 blocks
- Verifies `.avro` file extension
- Validates deserialization

## Expected Output

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

[TEST] Conditional PUT - Single Instance
✓ Wrote delta batch 1
✓ Wrote delta batch 2
✓ Wrote delta batch 3
✓ All 6 blocks present after sequential writes

[TEST] Conditional PUT - Concurrent Updates
✓ Created 3 ManifestManagers (simulating 3 instances)
ℹ Starting concurrent writes...
✓ Manager 0 completed writes
✓ Manager 1 completed writes
✓ Manager 2 completed writes
ℹ Expected 12 blocks, found 12
✓ All 12 blocks present - no lost updates!

[TEST] ETag Conflict Handling
✓ Wrote initial delta
ℹ Current ETag: abc123...
ℹ New ETag: def456...
✓ ETag changed after update (as expected)
✓ Conditional PUT correctly rejected old ETag (412 Precondition Failed)

[TEST] Avro Serialization
✓ Wrote 100 blocks
✓ Delta file uses .avro extension: test-manifests-avro/delta-1234567890.avro
✓ Successfully loaded 100 blocks from Avro format

[TEST] Cleanup
ℹ Cleaned up 15 test files

Test Summary
  PASS Basic Operations
  PASS Conditional PUT (Single)
  PASS Conditional PUT (Concurrent)
  PASS ETag Conflict Handling
  PASS Avro Serialization

Result: 5/5 tests passed
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
✗ Lost updates detected: 2 blocks missing
```

**This indicates a problem with conditional PUT!**
- Verify ETag support in your S3 implementation
- Check if `IfMatch` header is being honored
- Review Ceph configuration for S3 compatibility

### Avro Deserialization Errors

```
✗ Avro test failed: Expected -62 bytes...
```

**Solutions:**
- Ensure `fastavro` is installed: `pip install fastavro>=1.9.0`
- Check Python version (requires 3.9+)

## Debugging

### Keep Test Data

```bash
python test_presence_cache_live.py --bucket vllm --profile zgw --no-cleanup
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
✅ **Manifest system** is production-ready  

## Next Steps

After successful testing:
1. Enable presence cache in production: `enable_presence_cache=True`
2. Monitor manifest file sizes and compaction
3. Adjust `compaction_threshold` based on workload
4. Set up monitoring for pointer update conflicts

## Support

If tests fail, please report:
- Ceph version
- Test output (full)
- Ceph S3 configuration
- Any relevant Ceph logs