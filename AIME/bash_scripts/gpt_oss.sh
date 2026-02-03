#!/bin/bash

model_name=openai/gpt-oss-120b
BACKEND=TRITON_ATTN_VLLM_V1
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB='{"2":32,"4":32,"8":8}'
export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1

VLLM_ATTENTION_BACKEND=$BACKEND TOKENIZERS_PARALLELISM=false PYTHONPATH="../":"$PYTHONPATH" python3 ../scripts/vllm_kflow_oss.py \
   --model_name "$model_name" \
   --max_new_tokens 131072 \
   --temperature 0.6 \
   --reflex_size 64 