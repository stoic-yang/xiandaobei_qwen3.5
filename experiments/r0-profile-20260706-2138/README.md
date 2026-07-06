# r0-profile-20260706-2138

- Intent: Round 0.4 reuse existing kernel profile before spending another warm-container run.
- Source: `/public/home/xdzs2026_c166/codex_logs/profile_runs/rocprofv2_8_16K_20260622_161546`
- Local raw anchor: `raw/official_kernel_summary.md`
- Profiler: `rocprofv2`, 8-16K long-context benchmark trace, official summary generated from `results_vllm_rocprofv2.csv`.
- Limitation: the source run is a mixed long-context trace and notes the benchmark main run failed after EngineCore died; it is still useful for kernel ranking, but it is not a clean split prefill-vs-decode profile.

## Bottleneck Table

| Window | Kernel / group | Share | Interpretation |
|---|---:|---:|---|
| last_60s | `kernel_unified_attention_2d.kd` | 38.82% | full-attention path dominates the long-context hot window |
| last_60s | `chunk_fwd_kernel_o.kd` | 22.65% | GDN/chunk prefill is second bottleneck |
| last_60s | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64.kd` | 7.66% | GDN chunk work |
| last_60s | top Tensile GEMM rows | 16.82% | dense matmul/MLP projection cost |
| last_60s | `FillFunctor<int>` | 7.53% | allocation/fill overhead, likely mixed setup/bookkeeping |
| full trace | flash full-attention prefill kernel | 19.49% | full-attention prefill remains a top full-trace contributor |
| full trace | `fused_recurrent_gated_delta_rule_packed_decode_kernel.kd` | 0.67% | decode GDN kernel itself is not the main full-trace time sink |

## Verdict

- Reused profile confirms the current weight of work is long-context prefill: full attention first, then GDN chunk kernels, then GEMM.
- Decode is not proven bandwidth-vs-launch-limited by this artifact alone. The low full-trace share of decode GDN plus stable TPOT in guard runs suggest decode is not the score-limiting path, but a clean decode-only trace is still a pending R0.4 item if we need a hard launch/bandwidth label.

