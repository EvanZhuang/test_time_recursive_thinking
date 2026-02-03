# !/bin/bash
model_name=o4-miniagent__high
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
export MCP_SERVER_URL="http://localhost:8000/sse"

python -m lcb_runner.runner.trt --model $model_name --scenario codegeneration --evaluate --release_version release_v6 --max_tokens 200000 --n 1 --multiprocess_oai 16 --trt_rounds 8 --openai_timeout 1800 --logging_trace --roll_out_n 2 --start_date 2024-08-01 --difficulty hard --enable_strategy --eval_all_rollouts  --enable_test_gen

#--reference_sol_in_solver --enable_strategy
