# r0-probe-20260706-1944

**Verdict: PASS for R0.1/R0.2 + R0.3 harness creation.**

## Intent

Establish Round 0 ground truth before any performance change:

- confirm Qwen3.5-27B text architecture from `config.json`;
- probe DCU identity and FP8 evidence from live container tools;
- add a reusable warm-container guard benchmark script for all later A/B runs.

## Method

Live container: SCNet job `655597`, node `e03r2n07`, container IP `173.0.148.8`.

Commands were run through `xiandaobei-worker-auto` after unsetting proxy
variables. No vLLM source was edited.

## Metrics

- Text layers: 64 total = 48 `linear_attention` + 16 `full_attention`.
- MoE: no `num_experts` / `num_experts_per_tok` keys in `text_config`.
- Full-attention interval: 4.
- `head_dim`: 256.
- attention heads / KV heads: 24 / 4.
- `hidden_size`: 5120.
- DCU: `BW`, `gfx936:sramecc+:xnack-`, 80 CU, wavefront 64.
- VRAM: `torch.cuda.get_device_properties(0).total_memory = 68702699520` bytes; `hy-smi --showmemavailable = 65454 MiB`.
- FP8 evidence: DTK target list includes `gfx936`; `/opt/dtk/include/du_mma.h` exposes `__hip_fp8_e4m3` / `__hip_fp8_e5m2` fragments and `du_mma_sync` overloads. Peak FP8 TFLOPS and peak memory bandwidth were not printed by `hy-smi`/`rocminfo`.

## Artifacts

- `raw/model_config.json` - direct copy of `$MODEL_DIR/config.json`.
- `raw/arch_parse.log` - structured parse of architecture fields.
- `raw/torch_device.log` - `torch.cuda.get_device_properties(0)` output.
- `raw/hy_smi.log` - `hy-smi -a`, hardware/product/CU/profile/memory output.
- `raw/rocminfo_fp8.log` - `rocminfo` gfx936 agent plus DTK target/DU-MMA FP8 snippets.
- `scripts/guard_bench.py` - guard benchmark harness added in this commit.

## Guard Script

Path: `scripts/guard_bench.py`.

Default protocol:

```bash
python3 scripts/guard_bench.py \
  --repo competition \
  --num-prompts 10 \
  --repetitions 3 \
  --accuracy full
```

The script installs the selected wheel, starts or reuses one warm vLLM server,
runs `run_throughput.sh 4-8K 1` warmup, then runs all three throughput buckets
three times and reports medians for `output_throughput`, `p99_ttft_ms`, and
`p99_tpot_ms`. It then runs all four accuracy tasks once and writes
`summary.json`.

## Changelog

- 2026-07-06 seed by Codex.
