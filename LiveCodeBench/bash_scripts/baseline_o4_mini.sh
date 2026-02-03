#!/bin/bash
model_name=o4-mini__high
# BACKEND=FLASH_ATTN

# Set these environment variables before running:
# export OPENAI_API_KEY="your-azure-openai-api-key"
# export AZURE_OPENAI_ENDPOINT="your-azure-openai-endpoint"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable not set"
    echo "Please set it with: export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

if [ -z "$AZURE_OPENAI_ENDPOINT" ]; then
    echo "Error: AZURE_OPENAI_ENDPOINT environment variable not set"
    echo "Please set it with: export AZURE_OPENAI_ENDPOINT='your-endpoint'"
    exit 1
fi

export STREAM_DEBUG=true

cd ../
VLLM_ATTENTION_BACKEND=$BACKEND python -m lcb_runner.runner.main --model $model_name --scenario codegeneration --evaluate --release_version release_v6 --max_tokens 131072 --n 32 --multiprocess_oai 32 --openai_timeout 3000
