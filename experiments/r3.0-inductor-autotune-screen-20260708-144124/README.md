# R3.0 Inductor GEMM autotune screen 20260708-144124

Status: pass as a candidate-only screen; not valid as an A/B verdict.

Intent: run the first stable 8-16K throughput-only screen after the clean Inductor max-autotune smoke succeeded.

Candidate switch:

- `TORCHINDUCTOR_MAX_AUTOTUNE=1`
- `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1`
- TunableOp disabled.

Method: reused the warm Inductor service from `r3.0-inductor-autotune-smoke-clean-20260708-141641`, then ran `scripts/guard_bench.py --reuse-server --locked-start-script --load-format runai_streamer --num-prompts 3 --repetitions 3 --buckets 8-16K --accuracy none`.

Result:

- 8-16K median output throughput: `7.865904903201621 tok/s`
- 8-16K median TTFT-P99: `13231.200310122222 ms`
- 8-16K median TPOT-P99: `70.15169444930788 ms`

Verdict: no same-container sign yet. This run was completed in the old container, and the matching baseline run was lost when the container disappeared. Per project protocol, do not compare this absolute number against another container. A same-container baseline/candidate pair must decide whether Inductor is useful.

Remote anchor: `/public/home/xdzs2026_c166/codex_runs/r3.0-inductor-autotune-screen-20260708-144124`.
