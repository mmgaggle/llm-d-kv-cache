#!/usr/bin/env python3
"""Test script for Ceph S3 connectivity using zgw profile."""

from llmd_s3_backend.s3_client import S3ClientWrapper

print("Testing Ceph S3 connection with zgw profile...")
print("=" * 60)

try:
    # Initialize client with zgw profile and vllm bucket
    client = S3ClientWrapper(
        bucket="vllm",
        profile_name="zgw"
    )
    
    print(f"✓ S3 client initialized")
    print(f"  Bucket: {client.bucket}")
    print(f"  Region: {client.region}")
    print(f"  Endpoint: {client.endpoint_url}")
    print(f"  Addressing style: {client.addressing_style}")
    print()
    
    # Test connectivity by checking if a test key exists
    test_key = "test/connectivity-check.txt"
    print(f"Testing object existence: {test_key}")
    exists = client.object_exists(test_key)
    print(f"  Object exists: {exists}")
    print()
    
    print("✓ Ceph S3 connection successful!")
    print("=" * 60)
    
except Exception as e:
    print(f"✗ Error connecting to Ceph S3:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Made with Bob
