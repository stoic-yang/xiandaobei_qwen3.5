# r3.0-gemm-autotune-survey-20260707-2249

## Intent

R3.0/R2.4 first probe after the post-R3.1 prefill profile showed GEMM/Tensile kernels dominating prefill. This run only surveys the current container toolchain and safe autotune entry points. It does not start vLLM, run throughput, or modify competition source.

## Method

- Local meta: `main` at `7258cf5`.
- Remote container: job `658058`, node `e03r1n03`, worker reachable through `xiandaobei-worker-auto`.
- Remote source repo: `/public/home/xdzs2026_c166/vllm_cscc_competition` at `847d1bef1 perf(qwen35): enable flash attention prefill path`.
- Commands: one SSH heredoc survey plus a narrow PyTorch TunableOp probe; proxies unset.
- Raw logs:
  - `raw/survey.txt`
  - `raw/dtk_gemm.txt`
  - `raw/torch_tunable.txt`

## Results

Device/runtime:

| item | value |
|---|---|
| device | `BW`, `gfx936:sramecc+:xnack-` |
| VRAM | 65520 MB |
| CUs | 80 |
| torch | 2.10.0 |
| HIP | 6.3.26093 |
| vLLM | 0.18.1 |

Available tools:

| tool | status |
|---|---|
| `hipprof` | available at `/opt/dtk/bin/hipprof` |
| `hipblaslt-bench` | not found |
| `rocblas-bench` | not found |
| `rocblas-gemm-tune` | not found |
| `tensilelite-client` / `tensile_client` | not found |
| `omniperf` | not found |

PyTorch knobs:

| item | value |
|---|---|
| `torch.cuda.tunable` | present |
| key APIs | `enable`, `tuning_enable`, `set_filename`, `read_file`, `tune_gemm_in_file` |
| `torch._inductor.config.max_autotune` | `False` |
| `torch._inductor.config.max_autotune_gemm` | `False` |
| `torch._inductor.config.search_autotune_cache` | `False` |

## Verdict

Do not start R3.0 by looking for standalone rocBLAS/hipBLASLt benchmark CLIs in this container: they are not exposed. The low-risk path is a vLLM/PyTorch-level A/B:

1. First attribute the hot `Cijk_*` rows to exact model ops and shapes.
2. Then test PyTorch TunableOp and/or Inductor GEMM autotune as environment/config-gated candidates.
3. Measure with locked `guard_bench.py --locked-start-script --load-format runai_streamer`; verify `max_seq_len=32768`.

This remains a configuration/library-selection task, not a kernel-writing task.

## Caveats

- This survey does not prove TunableOp improves the vLLM path; it only proves the API exists in the installed PyTorch.
- R3.0 needs same-container A/B because compile/autotune caches can create cold-start artifacts.
- The remote meta checkout still reports `dubious ownership` for root; use the local `meta-main` or fix safe.directory before relying on remote meta git commands.
