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

# Download Granite model locally with progress indicator

set -e

MODEL="${MODEL:-ibm-granite/granite-3b-code-instruct}"
CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface/hub}"

echo "=========================================="
echo "Downloading Model: ${MODEL}"
echo "=========================================="
echo ""
echo "Cache directory: ${CACHE_DIR}"
echo ""

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface-cli..."
    pip install -U "huggingface_hub[cli]"
fi

# Download the model with progress
echo "Downloading model files..."
echo ""
huggingface-cli download "${MODEL}" \
    --cache-dir "${CACHE_DIR}" \
    --resume-download

echo ""
echo "=========================================="
echo "✓ Model downloaded successfully!"
echo "=========================================="
echo ""
echo "Model location: ${CACHE_DIR}"
echo ""
echo "To use with vLLM, you can now reference it by name:"
echo "  vllm serve ${MODEL}"
echo ""
echo "Or set HF_HOME to use a custom cache location:"
echo "  export HF_HOME=/path/to/cache"
echo "  vllm serve ${MODEL}"

# Made with Bob
