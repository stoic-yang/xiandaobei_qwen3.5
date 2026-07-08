# R3.0 Inductor GEMM autotune smoke clean rerun

## Intent

Rerun Step2 after clearing the stale `VLLM::EngineCore` process that invalidated the prior smoke.

## Candidate switch

- `TORCHINDUCTOR_MAX_AUTOTUNE=1`
- `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1`

The runtime default GEMM backends are left unchanged (`ATEN,TRITON,CPP` unless the config dump says otherwise). TunableOp is not enabled.

## Method

- `scripts/guard_bench.py`
- `--locked-start-script`
- `--load-format runai_streamer`
- `--buckets 8-16K`
- `--num-prompts 1`
- `--repetitions 1`
- `--accuracy none`
- `--keep-server`

The local poller is interrupted after remote `guard_remote.sh` is launched; remote artifacts remain under `/public/home/xdzs2026_c166/codex_runs/r3.0-inductor-autotune-smoke-clean-20260708-141641`.
