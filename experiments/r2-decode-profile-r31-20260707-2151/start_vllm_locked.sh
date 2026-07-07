#!/usr/bin/env bash
set -u
set -o pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy=127.0.0.1,localhost
export MODEL_DIR=/public/home/xdzs2026_c166/Qwen3.5-27B
export VLLM_CACHE_ROOT=/public/home/xdzs2026_c166/vllm_cache/vllm_cache
export TRITON_CACHE_DIR=/public/home/xdzs2026_c166/vllm_cache/triton_cache
export XDB_R31_FLASH_ATTN_PREFILL=1
export XDB_R31_FLASH_ATTN_MIN_Q=2
vllm serve "$MODEL_DIR" \
  --served-model-name Qwen3.5-27B \
  --port 8001 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.95 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --load-format runai_streamer
