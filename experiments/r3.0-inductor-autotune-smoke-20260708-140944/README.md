# R3.0 Inductor GEMM autotune smoke

## Intent

Test the task-card Step2 candidate after TunableOp failed the target coverage gate.

## Candidate switch

Environment-gated only:

- `TORCHINDUCTOR_MAX_AUTOTUNE=1`
- `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1`

No TunableOp is enabled in this run. `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS` is left at the runtime default unless explicitly noted by the dump.

## Method

- Start with `scripts/guard_bench.py`.
- Keep `--locked-start-script` and `--load-format runai_streamer`.
- First run is smoke only: 8-16K, one prompt, one repetition, `--accuracy none`.
- A small `sitecustomize.py` records the actual `torch._inductor.config` values seen by each Python process.

## Decision rule

If the service starts and the smoke request completes, keep the server warm and run a same-container 8-16K stable A/B. If startup fails or max-autotune does not apply, stop and record Step2 as unavailable.

## Verdict

Invalid run, not a candidate verdict.

The Inductor env did apply in all dumped Python processes, but the service failed before health because an old TunableOp `VLLM::EngineCore` process still held roughly 51 GiB VRAM. The logged root cause was:

`Free memory on device cuda:0 (9.2/63.98 GiB) on startup is less than desired GPU memory utilization (0.95, 60.79 GiB).`

After killing the stale engine process, rerun Step2 in a fresh experiment directory.
