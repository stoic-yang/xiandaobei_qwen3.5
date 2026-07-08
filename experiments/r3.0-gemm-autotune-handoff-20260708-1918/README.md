# R3.0 GEMM autotune handoff 20260708-1918

Status: handoff; A/B incomplete.

Objective: finish `plans/task-r3.0-gemm-autotune.md` by deciding whether Inductor GEMM autotune gives a positive same-container R3.1-baseline delta.

Current facts:

- R3.1 official baseline is AC `74.6924`, official throughput `13.78 / 12.89 / 11.18`.
- Post-R3.1 prefill profile is GEMM dominated: `Cijk_*` `67.101%`.
- Shape attribution is done at `experiments/r3.0-gemm-shape-attribution-20260708-112017-wheel/`.
- TunableOp failed the coverage gate at `experiments/r3.0-tunableop-coverage-20260708-133548/`; do not reopen unless PyTorch/runtime changes.
- Inductor max-autotune smoke succeeded at `experiments/r3.0-inductor-autotune-smoke-clean-20260708-141641/`; config dumps prove the env switches apply.
- Old-container Inductor candidate-only screen `experiments/r3.0-inductor-autotune-screen-20260708-144124/` produced 8-16K median `7.865905 tok/s`, TTFT-P99 `13231.200ms`, TPOT-P99 `70.1517ms`, but it has no same-container baseline and therefore no sign.
- New-container baseline attempts under job `659779` are invalid: `r3.0-baseline-screen-newcontainer-20260708-155959`, `...-retry-20260708-162744`, and `...-clean-20260708-163740`.
- Last live check: `python3 scripts/scnetctl.py status` at 2026-07-08 19:18 returned `job: none`, `worker: unreachable`.

Next clean-container recipe:

1. `cd /public/home/xdzs2026_c166/meta && git pull`; locally `python3 scripts/scnetctl.py attach` to regenerate `~/.ssh/xiandaobei.generated.conf`.
2. Verify no concurrent smoke service: remote `ps -eo cmd | grep -E 'Qwen3.5-0.8B|modelscope_smoke|testdata/start_vllm|0p8b_' | grep -v grep` must be empty.
3. Run fresh baseline:

   ```bash
   python3 scripts/guard_bench.py \
     --run-id r3.0-baseline-screen-clean-<date> \
     --repo competition \
     --num-prompts 3 \
     --repetitions 3 \
     --buckets 8-16K \
     --accuracy none \
     --locked-start-script \
     --load-format runai_streamer \
     --stop-existing \
     --keep-server \
     --server-start-timeout 1800 \
     --poll-interval 300 \
     --remote-timeout 7200 \
     --env VLLM_CACHE_ROOT=/public/home/xdzs2026_c166/codex_runs/<run>/vllm_cache \
     --env TRITON_CACHE_DIR=/public/home/xdzs2026_c166/codex_runs/<run>/triton_cache
   ```

4. If baseline passes, stop/restart into candidate in the same container with the Inductor hook and env:

   ```bash
   --env PYTHONPATH=/public/home/xdzs2026_c166/codex_runs/<candidate>/sitecustomize:/usr/local
   --env XDB_INDUCTOR_DUMP_CONFIG=1
   --env XDB_INDUCTOR_CONFIG_JSON=/public/home/xdzs2026_c166/codex_runs/<candidate>/config/config.json
   --env TORCHINDUCTOR_MAX_AUTOTUNE=1
   --env TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1
   ```

Decision rule:

- Positive only if same-container 8-16K median output throughput improves or TTFT-P99 drops without TPOT regression; then expand to three buckets and accuracy smoke.
- Negative/stop if candidate is flat or worse; record Inductor as unavailable/no gain and move to the next task card.
