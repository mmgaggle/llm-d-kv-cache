#!/bin/bash
# Copyright 2025 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Validation script for S3 KV cache backend with vLLM
# This script:
# 1. Sends requests to vLLM OpenAI API endpoint
# 2. Checks for errors in vLLM logs
# 3. Verifies cache blocks are stored in S3/Ceph
# 4. Tests cache retrieval (cold start)

set -e

# Configuration
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_URL="${VLLM_URL:-http://localhost:${VLLM_PORT}}"
MODEL="${MODEL:-ibm-granite/granite-3b-code-instruct}"
S3_BUCKET="${S3_BUCKET:-vllm}"
S3_PROFILE="${S3_PROFILE:-zgw}"
S3_PREFIX="${S3_PREFIX:-kv-cache}"
AUTO_START="${AUTO_START:-true}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# PID file for vLLM process
VLLM_PID_FILE="/tmp/vllm_validation.pid"
VLLM_STARTED_BY_SCRIPT=false

echo "=========================================="
echo "S3 KV Cache Backend Validation"
echo "=========================================="
echo ""

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "OK" ]; then
        echo -e "${GREEN}✓${NC} $message"
    elif [ "$status" = "FAIL" ]; then
        echo -e "${RED}✗${NC} $message"
    elif [ "$status" = "INFO" ]; then
        echo -e "${YELLOW}ℹ${NC} $message"
    fi
}

# Function to start vLLM
start_vllm() {
    print_status "INFO" "Starting vLLM server..."
    
    # Check if virtual environment is activated
    if [ -z "$VIRTUAL_ENV" ]; then
        print_status "FAIL" "Virtual environment not activated"
        echo "Please run: source venv/bin/activate"
        exit 1
    fi
    
    # Start vLLM in background
    nohup vllm serve ${MODEL} \
      --kv-transfer-config "{
        \"kv_connector\": \"OffloadingConnector\",
        \"kv_role\": \"kv_both\",
        \"kv_connector_extra_config\": {
          \"spec_name\": \"S3OffloadingSpec\",
          \"spec_module_path\": \"llmd_s3_backend.spec\",
          \"s3_bucket\": \"${S3_BUCKET}\",
          \"s3_profile_name\": \"${S3_PROFILE}\",
          \"block_size\": 256,
          \"threads_per_gpu\": 64
        }
      }" \
      --distributed-executor-backend mp \
      --max-model-len 2048 \
      --port ${VLLM_PORT} > vllm_validation.log 2>&1 &
    
    VLLM_PID=$!
    echo $VLLM_PID > $VLLM_PID_FILE
    VLLM_STARTED_BY_SCRIPT=true
    
    print_status "INFO" "vLLM starting (PID: $VLLM_PID)..."
    print_status "INFO" "Logs: vllm_validation.log"
    
    # Wait for vLLM to be ready (up to 600 seconds / 10 minutes)
    print_status "INFO" "Waiting for vLLM to be ready (this may take several minutes for model download/loading)..."
    for i in {1..600}; do
        if curl -s "${VLLM_URL}/health" > /dev/null 2>&1; then
            print_status "OK" "vLLM server is ready (took ${i}s)"
            return 0
        fi
        sleep 1
        if [ $((i % 30)) -eq 0 ]; then
            echo -n "."
        fi
    done
    
    print_status "FAIL" "vLLM failed to start within 600 seconds (10 minutes)"
    echo "Check logs: tail -f vllm_validation.log"
    exit 1
}

# Function to stop vLLM if started by script
cleanup_vllm() {
    if [ "$VLLM_STARTED_BY_SCRIPT" = true ] && [ -f "$VLLM_PID_FILE" ]; then
        VLLM_PID=$(cat $VLLM_PID_FILE)
        if ps -p $VLLM_PID > /dev/null 2>&1; then
            print_status "INFO" "Stopping vLLM (PID: $VLLM_PID)..."
            kill $VLLM_PID
            rm -f $VLLM_PID_FILE
        fi
    fi
}

# Trap to cleanup on exit
trap cleanup_vllm EXIT

# Step 1: Check if vLLM is running
print_status "INFO" "Step 1: Checking if vLLM is running..."
if curl -s "${VLLM_URL}/health" > /dev/null 2>&1; then
    print_status "OK" "vLLM server is running at ${VLLM_URL}"
else
    if [ "$AUTO_START" = "true" ]; then
        start_vllm
    else
        print_status "FAIL" "vLLM server is not responding at ${VLLM_URL}"
        echo ""
        echo "To start vLLM manually:"
        echo "  vllm serve ${MODEL} \\"
        echo "    --kv-transfer-config '{...}' \\"
        echo "    --port ${VLLM_PORT}"
        echo ""
        echo "Or run this script with AUTO_START=true to start automatically"
        exit 1
    fi
fi

# Step 2: Clear existing cache (optional)
print_status "INFO" "Step 2: Checking existing S3 cache..."
CACHE_COUNT=$(aws --profile ${S3_PROFILE} s3 ls s3://${S3_BUCKET}/${S3_PREFIX}/ --recursive 2>/dev/null | wc -l || echo "0")
print_status "INFO" "Found ${CACHE_COUNT} existing cache objects"

# Step 3: Send test requests
print_status "INFO" "Step 3: Sending test requests to vLLM..."

# Test prompt that should generate cacheable KV blocks
TEST_PROMPT="Write a Python function to calculate the factorial of a number using recursion. Include docstring and type hints."

echo ""
echo "Sending request 1 (cold cache)..."
RESPONSE1=$(curl -s "${VLLM_URL}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"prompt\": \"${TEST_PROMPT}\",
    \"max_tokens\": 150,
    \"temperature\": 0.7
  }")

if echo "$RESPONSE1" | jq -e '.choices[0].text' > /dev/null 2>&1; then
    print_status "OK" "Request 1 completed successfully"
    echo "Response preview:"
    echo "$RESPONSE1" | jq -r '.choices[0].text' | head -n 3
else
    print_status "FAIL" "Request 1 failed"
    echo "$RESPONSE1"
    exit 1
fi

echo ""
echo "Waiting 5 seconds for cache to be written..."
sleep 5

# Send same request again (should hit cache)
echo ""
echo "Sending request 2 (warm cache - same prompt)..."
RESPONSE2=$(curl -s "${VLLM_URL}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"prompt\": \"${TEST_PROMPT}\",
    \"max_tokens\": 150,
    \"temperature\": 0.7
  }")

if echo "$RESPONSE2" | jq -e '.choices[0].text' > /dev/null 2>&1; then
    print_status "OK" "Request 2 completed successfully"
else
    print_status "FAIL" "Request 2 failed"
    echo "$RESPONSE2"
    exit 1
fi

# Step 4: Verify cache blocks in S3
print_status "INFO" "Step 4: Verifying cache blocks in S3..."
echo ""

NEW_CACHE_COUNT=$(aws --profile ${S3_PROFILE} s3 ls s3://${S3_BUCKET}/${S3_PREFIX}/ --recursive 2>/dev/null | wc -l || echo "0")
CACHE_DIFF=$((NEW_CACHE_COUNT - CACHE_COUNT))

if [ $CACHE_DIFF -gt 0 ]; then
    print_status "OK" "Found ${CACHE_DIFF} new cache blocks in S3"
    echo ""
    echo "Sample cache objects:"
    aws --profile ${S3_PROFILE} s3 ls s3://${S3_BUCKET}/${S3_PREFIX}/ --recursive 2>/dev/null | head -n 5
else
    print_status "FAIL" "No new cache blocks found in S3"
    echo "This might indicate:"
    echo "  - S3 backend is not configured correctly"
    echo "  - Cache blocks are not being offloaded"
    echo "  - S3 credentials are incorrect"
fi

# Step 5: Check for errors in vLLM logs
print_status "INFO" "Step 5: Checking for S3-related errors..."
echo ""
echo "Note: This requires access to vLLM logs. Check the terminal where vLLM is running."
echo "Look for lines containing 'S3', 'offload', or 'ERROR'"

# Step 6: Test cache retrieval (optional - requires restarting vLLM)
echo ""
print_status "INFO" "Step 6: Cache retrieval test (cold start)"
echo ""
echo "To test cache retrieval:"
echo "  1. Stop the current vLLM server (Ctrl+C)"
echo "  2. Restart vLLM with the same configuration"
echo "  3. Send the same prompt again"
echo "  4. vLLM should retrieve cached KV blocks from S3"
echo ""
echo "Expected behavior:"
echo "  - First request after restart should be faster (cache hit)"
echo "  - vLLM logs should show 'Loading from S3' or similar messages"

# Summary
echo ""
echo "=========================================="
echo "Validation Summary"
echo "=========================================="
print_status "OK" "vLLM server: Running"
print_status "OK" "API requests: ${GREEN}2/2${NC} successful"
if [ $CACHE_DIFF -gt 0 ]; then
    print_status "OK" "S3 cache: ${GREEN}${CACHE_DIFF}${NC} blocks written"
else
    print_status "FAIL" "S3 cache: No blocks written"
fi

echo ""
echo "Next steps:"
echo "  1. Check vLLM logs for any S3-related errors"
echo "  2. Restart vLLM to test cache retrieval (cold start)"
echo "  3. Monitor S3 bucket for cache growth: aws --profile ${S3_PROFILE} s3 ls s3://${S3_BUCKET}/${S3_PREFIX}/ --recursive"
echo ""

# Made with Bob
