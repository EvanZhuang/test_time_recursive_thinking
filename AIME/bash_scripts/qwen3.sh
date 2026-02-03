#!/bin/bash

model_name=Qwen/Qwen3-235B-A22B-Thinking-2507
BACKEND=FLASH_ATTN
export HF_HUB_ENABLE_HF_TRANSFER=1

VLLM_ATTENTION_BACKEND=$BACKEND TOKENIZERS_PARALLELISM=false PYTHONPATH="../":"$PYTHONPATH" python ../scripts/vllm_kflow_qwen.py \
   --model_name "$model_name" \
   --max_new_tokens 262144 \
   --temperature 0.6 \
   --reflex_size 64
