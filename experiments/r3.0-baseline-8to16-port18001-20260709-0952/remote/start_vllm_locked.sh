#!/usr/bin/env bash
set -u
set -o pipefail

load_format_args=()
if [ -n "${GUARD_LOAD_FORMAT:-}" ]; then
  load_format_args=(--load-format "$GUARD_LOAD_FORMAT")
fi
eager_args=()
if [ "${GUARD_ENFORCE_EAGER:-0}" = "1" ]; then
  eager_args=(--enforce-eager)
fi

vllm serve "$MODEL_DIR"             --served-model-name Qwen3.5-27B             --port "${GUARD_SERVER_PORT:-8001}"             --trust-remote-code             --dtype bfloat16             --tensor-parallel-size 1             --max-model-len 32768             --max-num-seqs 128             --max-num-batched-tokens 4096             --gpu-memory-utilization 0.95             --default-chat-template-kwargs '{"enable_thinking": false}'             --reasoning-parser qwen3             --enable-auto-tool-choice             --tool-call-parser qwen3_coder             "${eager_args[@]}"             "${load_format_args[@]}"
