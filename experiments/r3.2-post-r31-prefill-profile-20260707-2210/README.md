# r3.2-post-r31-prefill-profile-20260707-2210

## Intent

R3.2 Step 0: after the R3.1 flash-attention prefill candidate, re-profile one 8-16K prefill request and decide whether GDN chunked-prefill is still the next high-ROI target.

## Method

- Source repo: `/public/home/xdzs2026_c166/vllm_cscc_competition` at `847d1bef10b0b5bb71b7e427535b610a20a4d263`.
- Service: `hipprof --trace-off --session xdb_r32_prefill_2210` wrapping locked `vllm serve`.
- Locked evidence:
  - `vllm_server.log`: `Using max model len 32768`
  - `vllm_server.log`: `max_seq_len=32768`, `load_format=runai_streamer`
  - `vllm_server.log`: `XDB_R31_FLASH_ATTN_PREFILL enabled`
- Warmup: `run_throughput.sh 4-8K 1`.
- Trace protocol: start hipprof trace before one non-streaming 8-16K request with `max_tokens=1`, stop/flush after the response.
- Large hipprof JSON/DB files were intentionally not copied into meta. The committed artifacts are compact CSV summaries, logs, and the request summary.

## Results

Request:

| item | value |
|---|---:|
| prompt tokens | 13964 |
| completion tokens | 1 |
| request wall | 3299.468 ms |
| hipprof kernel busy | 3163.713 ms |
| wall - kernel busy | 135.755 ms |
| kernel / wall | 95.886% |

Kernel groups:

| group | share of kernel time | total |
|---|---:|---:|
| Tensile GEMM (`Cijk_*`) | 67.101% | 2122.873 ms |
| flash-attn prefill | 8.994% | 284.532 ms |
| GDN core chunk kernels | 9.303% | 294.319 ms |
| GDN core + helpers | 13.277% | 420.044 ms |
| Triton misc fused kernels | 7.538% | 238.469 ms |
| old unified full-attn path | 0.000% | 0.000 ms |

Top kernels:

| rank | kernel | share |
|---:|---|---:|
| 1 | top Tensile GEMM WGM4 | 26.153% |
| 2 | top Tensile GEMM WGM8 | 20.804% |
| 3 | top Tensile GEMM WGM1 | 19.495% |
| 4 | `flash_fwd_kernel_16x64_prefetch...dim256...` | 8.994% |
| 5 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 5.050% |
| 6 | `chunk_fwd_kernel_o` | 2.231% |
| 7 | `_causal_conv1d_fwd_kernel` | 2.050% |
| 8 | `recompute_w_u_fwd_kernel` | 1.540% |
| 17 | `chunk_scaled_dot_kkt_fwd_kernel` | 0.482% |

## Verdict

Post-R3.1, the prefill hotspot has shifted. The old R0/R3.1-before profile said full-attn plus GDN chunk were the large blocks; this trace shows the old unified full-attn path is gone and Tensile GEMM now dominates at `67.1%` of kernel time.

R3.2 GDN is no longer the obvious next mainline. GDN core is `9.30%` of kernel time, or `13.28%` including helper kernels, so the upside is much smaller than the old `~30%` assumption. This satisfies the task-card rule to downgrade R3.2 unless a broader or 16-32K profile contradicts it.

Next priority should be R3.0/R2.4 GEMM library/autotune before a full GDN implementation cycle. A narrow GDN config/autotune spike remains reasonable only if it is cheap and env-gated.

## Caveats

- This is one 8-16K request with `max_tokens=1`, not a full multi-prompt guard.
- HIP API time includes blocking waits and async calls, so the kernel CSV is the primary hotspot evidence.
- A 16-32K post-R3.1 prefill profile can still be useful before permanently killing R3.2.
