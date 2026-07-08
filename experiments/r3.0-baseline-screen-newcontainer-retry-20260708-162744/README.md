# R3.0 baseline screen newcontainer retry 20260708-162744

Status: invalid.

Intent: clean retry of the same-container 8-16K baseline after `r3.0-baseline-screen-newcontainer-20260708-155959` was contaminated by a 0.8B smoke service.

Method: `scripts/guard_bench.py --locked-start-script --load-format runai_streamer --num-prompts 3 --repetitions 3 --buckets 8-16K --accuracy none --stop-existing --keep-server`.

Verdict: discard. The guard started, but `driver.log` ended with `server exited before health check`. Process inspection showed the same container was concurrently running `/public/home/xdzs2026_c166/testdata/start_vllm.sh`, serving `/public/home/xdzs2026_c166/modelscope_smoke_qwen35_08b/model` on port `8001`. That external smoke sequence kept respawning under `codex_logs/0p8b_linear_only_*`, so this was not a clean R3.0 baseline attempt.

Root cause anchor:

- Remote run dir: `/public/home/xdzs2026_c166/codex_runs/r3.0-baseline-screen-newcontainer-retry-20260708-162744`
- Concurrent smoke logs observed:
  - `/public/home/xdzs2026_c166/codex_logs/0p8b_linear_only_baseline_20260708_161424`
  - `/public/home/xdzs2026_c166/codex_logs/0p8b_linear_only_patched_20260708_161424`
  - `/public/home/xdzs2026_c166/codex_logs/0p8b_linear_only_baseline_8_16k_20260708_162543`
  - `/public/home/xdzs2026_c166/codex_logs/0p8b_linear_only_patched_8_16k_20260708_162543`
  - `/public/home/xdzs2026_c166/codex_logs/0p8b_linear_only_baseline_8_16k_retry_20260708_163349`

Next action: wait for the external 0.8B smoke sequence to finish or move to another clean container before restarting the R3.0 same-container baseline/candidate pair.
