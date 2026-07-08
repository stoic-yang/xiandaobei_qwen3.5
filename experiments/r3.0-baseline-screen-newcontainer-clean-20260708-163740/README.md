# R3.0 baseline screen newcontainer clean 20260708-163740

Status: invalid.

Intent: third attempt to get a clean same-container 8-16K baseline for the R3.0 Inductor GEMM autotune A/B after earlier attempts collided with 0.8B smoke runs.

Method: `scripts/guard_bench.py --locked-start-script --load-format runai_streamer --num-prompts 3 --repetitions 3 --buckets 8-16K --accuracy none --stop-existing --keep-server`.

Verdict: discard. This run did not produce `summary.json`. The 27B service was terminated during RunAI streaming (`97% Completed | 1160/1199`), and a separate external `Qwen3.5-0.8B` vLLM service was later observed on port `8001`:

`/usr/local/bin/vllm serve /public/home/xdzs2026_c166/Qwen3.5-0.8B --served-model-name Qwen3.5-27B --port 8001 ...`

This container is not currently clean enough for the R3.0 same-container baseline/candidate pair.

Anchors:

- Remote run dir: `/public/home/xdzs2026_c166/codex_runs/r3.0-baseline-screen-newcontainer-clean-20260708-163740`
- Local pulled logs: `remote/driver.log`, `remote/vllm_server.log`, `remote/runtime_fingerprints.json`
- Failure log: `remote/vllm_server.log` ends with `KeyboardInterrupt: terminated` during startup.

Next action: wait for the external 0.8B service sequence to release the container, then start a fresh baseline/candidate pair with new run IDs; or move to another clean container.
