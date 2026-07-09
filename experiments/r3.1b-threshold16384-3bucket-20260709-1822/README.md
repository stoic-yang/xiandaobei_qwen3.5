# r3.1b-threshold16384-3bucket-20260709-1822

Guard benchmark run.

- Intent: fixed warm-container guard protocol for Round 0 / Round 1 comparisons.
- Method: warmup once, then three throughput buckets x 3 repetitions, median summary, plus accuracy mode `none`.
- Buckets: `4-8K,8-16K,16-32K`
- Overlay rev: ``
- Locked start script: `True`
- Load format: `runai_streamer`
- Enforce eager: `False`
- Remote run dir: `/public/home/xdzs2026_c166/codex_runs/r3.1b-threshold16384-3bucket-20260709-1822`
- Local summary: `summary.json`
- Raw logs: `raw/`
- Throughput result JSONs: `throughput/`
