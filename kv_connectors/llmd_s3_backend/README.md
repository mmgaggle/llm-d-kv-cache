# llmd-s3-backend README

## Overview
The llmd-s3-backend extends the native [vLLM Offloading Connector](https://docs.vllm.ai/en/stable/features/disagg_prefill/) to support an S3 object storage backend.
This backend provides a cloud-native offloading layer for vLLM, moving KV-cache blocks between GPU and S3 efficiently using:

- Async S3 operations with boto3
- Multi-threaded I/O workers
- Staging memory pools for GPU↔CPU transfers
- Support for AWS S3 and S3-compatible services (Ceph, etc.)
- Flexible credential management (env vars, profiles, IAM roles)

The S3 connector stores each KV cache block as a separate S3 object, organized hierarchically for efficient lookup and retrieval.

## System Requirements
- vLLM version 0.11.0 or above, which includes the Offloading Connector
- Python 3.9+
- boto3 and botocore

## Installation

### 1. Install from source

```bash
pip install git+https://github.com/llm-d/llm-d-kv-cache-manager.git#subdirectory=kv_connectors/llmd_s3_backend
```

### 2. Developer mode (clone and editable install)

Clone the source and install in editable mode:

```bash
git clone https://github.com/llm-d/llm-d-kv-cache-manager.git
cd llm-d-kv-cache-manager/kv_connectors/llmd_s3_backend
pip install -e .
```

## Configuration

### Connector Parameters

- `s3_bucket`: S3 bucket name (required)
- `s3_prefix`: S3 key prefix (default: "kv-cache")
- `s3_region`: AWS region (default: from env or "us-east-1")
- `s3_endpoint_url`: Custom S3 endpoint URL for S3-compatible services
- `s3_addressing_style`: S3 addressing style - "auto", "path", or "virtual" (default: "auto")
- `s3_profile_name`: AWS profile name from credentials file
- `block_size`: Number of GPU blocks grouped into each S3 object
- `threads_per_gpu`: Number of I/O threads per GPU
- `max_staging_memory_gb`: Total staging memory limit in GB

### AWS Credentials

The connector supports multiple authentication methods (in order of precedence):

1. **Environment Variables**:
   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_REGION=us-west-2
   ```

2. **AWS Credentials File** (`~/.aws/credentials`):
   ```ini
   [default]
   aws_access_key_id = your_access_key
   aws_secret_access_key = your_secret_key
   region = us-west-2
   ```

3. **IAM Roles**: Automatic for EKS/EC2 instances with appropriate IAM roles

### Environment Variables

- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `AWS_REGION`: AWS region
- `STORAGE_CONNECTOR_DEBUG`: Enable debug logs (optional)

## Usage Example

### vLLM Configuration

```yaml
--kv-transfer-config '{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "S3OffloadingSpec",
    "spec_module_path": "llmd_s3_backend.spec",
    "s3_bucket": "my-kv-cache-bucket",
    "s3_prefix": "kv-cache",
    "s3_region": "us-west-2",
    "block_size": 256,
    "threads_per_gpu": 64
  }
}'
--distributed_executor_backend "mp"
```

It is recommended to use multiprocess mode by setting:
`--distributed_executor_backend "mp"`

### Using S3-Compatible Services (Ceph, etc.)

```yaml
--kv-transfer-config '{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "S3OffloadingSpec",
    "spec_module_path": "llmd_s3_backend.spec",
    "s3_bucket": "my-kv-cache-bucket",
    "s3_endpoint_url": "http://ceph.example.com:9000",
    "s3_addressing_style": "path",
    "block_size": 256,
    "threads_per_gpu": 64
  }
}'
```

## S3 Object Structure

KV cache blocks are stored in S3 with the following hierarchical structure:

```
s3://bucket/prefix/model_name/tp_size/rank_X/dtype/abc/de/abcdef0123456789.bin
                   └─────────┘ └─────┘ └────┘ └───┘ └─┘└─┘└──────────────────┘
                   model info   TP info  rank  dtype  hash-based hierarchy
```

Example:
```
s3://my-bucket/kv-cache/llama3-70b/tp_8/rank_0/float16/a1b/2c/a1b2c3d4e5f67890.bin
```

This structure:
- Organizes blocks by model, tensor parallelism, and data type
- Uses hash-based subdirectories to avoid S3 prefix hotspots
- Maintains compatibility with the filesystem connector's format

## K8s Deployment Example

A full Kubernetes deployment example can be found in the [`docs/deployment`](./docs/deployment) folder.

### Prerequisites

1. Create S3 bucket:
   ```bash
   aws s3 mb s3://my-kv-cache-bucket --region us-west-2
   ```

2. Create AWS credentials secret (if not using IAM roles):
   ```bash
   kubectl create secret generic aws-credentials \
     --from-literal=AWS_ACCESS_KEY_ID=your_access_key \
     --from-literal=AWS_SECRET_ACCESS_KEY=your_secret_key \
     --from-literal=AWS_REGION=us-west-2
   ```

3. Create HuggingFace token secret:
   ```bash
   export HF_TOKEN=<HF_TOKEN>
   kubectl create secret generic hf-token --from-literal=HF_TOKEN="$HF_TOKEN"
   ```

4. Apply the vLLM deployment:
   ```bash
   kubectl apply -f ./docs/deployment/vllm-s3.yaml
   ```

## Performance Considerations

- **Block Size**: Larger block sizes reduce S3 API calls but increase memory usage
- **Threads per GPU**: More threads improve throughput but consume more memory
- **S3 Region**: Use the same region as your compute for lower latency
- **S3 Transfer Acceleration**: Enable for cross-region deployments
- **Staging Memory**: Adjust based on available system memory

## Troubleshooting

### S3 Connection Issues

Check AWS credentials and region:
```bash
aws s3 ls s3://my-kv-cache-bucket --region us-west-2
```

### Permission Errors

Ensure the IAM role or credentials have these permissions:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-kv-cache-bucket",
        "arn:aws:s3:::my-kv-cache-bucket/*"
      ]
    }
  ]
}
```

### Debug Logging

Enable debug logging:
```bash
export STORAGE_CONNECTOR_DEBUG=1
```

## Comparison with Filesystem Connector

| Feature | Filesystem Connector | S3 Connector |
|---------|---------------------|--------------|
| Storage | Local/NFS filesystem | S3 object storage |
| Scalability | Limited by filesystem | Virtually unlimited |
| Multi-region | Requires replication | Native S3 replication |
| Cost | Storage + compute | Storage + API calls |
| Performance | Lower latency | Higher latency, better throughput |
| Setup | Requires PVC/NFS | Requires S3 bucket |

## Storage Cleanup

S3 lifecycle policies can automatically clean up old KV cache objects:

```json
{
  "Rules": [
    {
      "Id": "DeleteOldKVCache",
      "Status": "Enabled",
      "Prefix": "kv-cache/",
      "Expiration": {
        "Days": 7
      }
    }
  ]
}
```

Apply with:
```bash
aws s3api put-bucket-lifecycle-configuration \
  --bucket my-kv-cache-bucket \
  --lifecycle-configuration file://lifecycle.json
```

## Documentation

Comprehensive documentation is available in the [`docs/`](./docs) directory:

### Core Documentation
- **[Testing Guide](./docs/TESTING.md)** - How to run tests, test structure, and CI/CD integration
- **[Presence Cache](./docs/PRESENCE_CACHE.md)** - Distributed cache for tracking KV cache block locations

### io_uring Zero-Copy I/O (Experimental)
- **[io_uring Design](./docs/IOURING_DESIGN.md)** - Architecture and design for zero-copy S3 transfers
- **[io_uring Development](./docs/IOURING_DEVELOPMENT.md)** - Development environment setup with Podman and Ceph
- **[Multipathing Analysis](./docs/MULTIPATHING_ANALYSIS.md)** - S3 multipathing for improved throughput

### Platform-Specific Notes
- **[macOS Testing Limitations](./docs/MACOS_TESTING_LIMITATIONS.md)** - Known limitations when testing on macOS with Podman

### Deployment
- **[Kubernetes Deployment](./docs/deployment/)** - Example Kubernetes manifests for production deployment

## License

Apache License 2.0 - See LICENSE file for details.
