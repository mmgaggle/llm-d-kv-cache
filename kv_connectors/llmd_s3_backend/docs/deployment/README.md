# Kubernetes Deployment

Example manifests for deploying vLLM with the S3 KV cache backend on Kubernetes.

## Files

| File | Description |
|------|-------------|
| `vllm-s3.yaml` | Service + Deployment for vLLM with the S3 connector. Installs the connector at container start, configures 8×GPU with Llama 3.1 70B. |
| `aws-credentials-secret.yaml` | Template Secret for AWS credentials. Not needed if you use IAM Roles for Service Accounts (IRSA). |
| `iam-policy.json` | Minimum IAM policy granting `GetObject`, `PutObject`, `DeleteObject`, `ListBucket`, and `HeadObject` on the cache bucket. |

## Usage

1. **Create the S3 bucket** (if it doesn't exist):
   ```bash
   aws s3 mb s3://my-kv-cache-bucket --region us-west-2
   ```

2. **Attach the IAM policy.** Either:
   - Apply `iam-policy.json` to an IAM role and bind it to a Kubernetes
     service account via IRSA (recommended for EKS), or
   - Create the credentials secret:
     ```bash
     kubectl create secret generic aws-credentials \
       --from-literal=AWS_ACCESS_KEY_ID=AKIA... \
       --from-literal=AWS_SECRET_ACCESS_KEY=... \
       --from-literal=AWS_REGION=us-west-2
     ```

3. **Create the HuggingFace token secret** (needed to pull gated models):
   ```bash
   kubectl create secret generic hf-token \
     --from-literal=HF_TOKEN="$HF_TOKEN"
   ```

4. **Edit `vllm-s3.yaml`** to match your environment:
   - Set the model name, tensor-parallel size, and GPU count.
   - Update `s3_bucket`, `s3_region`, and other connector parameters.
   - Adjust resource requests/limits for your node type.

5. **Deploy:**
   ```bash
   kubectl apply -f vllm-s3.yaml
   ```

6. **Verify:**
   ```bash
   kubectl logs -f deployment/vllm-s3-deployment
   ```

## Customization

- **Ceph / S3-compatible storage:** Add `"s3_endpoint_url"` and
  `"s3_addressing_style": "path"` to `kv_connector_extra_config` in the
  deployment YAML.
- **Presence cache:** Add `"enable_presence_cache": true` for multi-instance
  deployments that benefit from cross-instance cache awareness.
- **IAM Roles for Service Accounts:** Uncomment the `serviceAccountName`
  line in `vllm-s3.yaml` and remove the `aws-credentials` secret references.
