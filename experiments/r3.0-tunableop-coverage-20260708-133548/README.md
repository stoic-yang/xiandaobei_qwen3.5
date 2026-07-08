# R3.0 TunableOp coverage probe

## Intent

Check whether PyTorch `torch.cuda.tunable` reaches the post-R3.1 prefill GEMM hotspot shapes before spending a full same-container A/B run.

## Method

- Use the competition wheel through `scripts/guard_bench.py`.
- Keep the locked serve command and `--load-format runai_streamer`.
- Enable TunableOp only through experiment-scoped `PYTHONPATH` + `XDB_TUNABLE_*` env vars.
- Run one 8-16K throughput request with `--accuracy none`, `--num-prompts 1`, `--repetitions 1`.
- Dump `torch.cuda.tunable.get_results()` periodically from the vLLM process because PyTorch 2.10 did not auto-write the CSV in smoke tests.

## Target GEMM shapes from Step 0

- `mlp.gate_up_proj`: weight `[34816, 5120]`
- `mlp.down_proj`: weight `[5120, 17408]`
- `linear_attn.in_proj_qkvz`: weight `[16384, 5120]`
- `linear_attn/self_attn.out_proj`: weight `[5120, 6144]`
- `self_attn.qkv_proj`: weight `[14336, 5120]`

## Runtime command

Started locally with `nohup`; see `local_guard_coverage.log` and remote run directory:

`/public/home/xdzs2026_c166/codex_runs/r3.0-tunableop-coverage-20260708-133548`

## Step2 prep observed while coverage was running

Current remote PyTorch/Inductor source:

- `torch 2.10.0`, config file `/usr/local/lib/python3.10/dist-packages/torch/_inductor/config.py`
- `max_autotune=False`
- `max_autotune_gemm=False`
- `search_autotune_cache=False`
- `max_autotune_gemm_backends=ATEN,TRITON,CPP`

Source-confirmed candidate env names:

- `TORCHINDUCTOR_MAX_AUTOTUNE=1`
- `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1`
- `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=<ATEN,TRITON,CPP|ATEN,TRITON,CK|...>`
- `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE=<DEFAULT|EXHAUSTIVE>`

Do not use `search_autotune_cache` or `autotune_fallback_to_aten` as candidate toggles in this environment; the current source marks both as deprecated/ignored.

## Verdict

TunableOp online candidate is not worth continuing in this environment.

- The hook loaded in vLLM and confirmed `max_seq_len=32768`, locked CLI, and `load_format=runai_streamer`.
- The service spent about 404.7s streaming weights and then remained pre-health during encoder profiling with TunableOp enabled.
- Periodic `get_results()` dumps produced only two visual-encoder bf16 GEMM shapes:
  - `tn_3456_65536_1152`
  - `tn_1152_65536_1536`
- Target text GEMM hits were all zero; see `tunable_coverage_summary.json`.

Decision: stop TunableOp before full 8-16K A/B. It fails the task-card coverage gate and adds startup/tuning risk before reaching the target prefill GEMMs.
