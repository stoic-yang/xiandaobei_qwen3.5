# Xiandaobei Project Map

Use this reference when exact paths or ownership boundaries matter.

- Meta repo: `/Users/keynary/Code/xiandaobei/meta`
- R0/R1 worktree: `/Users/keynary/Code/xiandaobei/meta-r0-r1`
- Official PDFs: `/Users/keynary/Code/xiandaobei/source`
- SCNet home: `/public/home/xdzs2026_c166`
- SCNet meta clone: `/public/home/xdzs2026_c166/meta`
- Competition checkout: `/public/home/xdzs2026_c166/vllm_cscc_competition`
- Competition wheel: `/public/home/xdzs2026_c166/vllm_cscc_competition/dist/*.whl`
- Model directory: `/public/home/xdzs2026_c166/Qwen3.5-27B`
- Shared startup cache root: `/public/home/xdzs2026_c166/vllm_cache`
- Run output root: `/public/home/xdzs2026_c166/codex_runs`
- Local run summaries: `experiments/<run_id>/`
- Submission manifests and browser evidence: `snapshots/`
- Generated SSH config: `~/.ssh/xiandaobei.generated.conf`
- Generated worker alias: `xiandaobei-worker-auto`
- GitLab URL: `https://gitlab.eduxiji.net/T2026102809912095/2026pra-ohhu.git`

Primary scripts:

- `scripts/scnetctl.py`: attach to a running SCNet container and run benchmark
  tasks. It regenerates the worker SSH config and owns SSH multiplexing
  defaults.
- `scripts/guard_bench.py`: single-entry guard protocol for same-container
  warmup, 3x throughput medians, accuracy, overlays, and locked
  `runai_streamer` startup.
- `scripts/submit_job.py`: repository checks, performance gates, GitLab pushes,
  snapshot pushes, and manifests when present in the active checkout.
- `scripts/chrome_submit_adapter.mjs`: Xi platform GitLab URL dry-run and final
  submit through the logged-in Chrome session when present in the active
  checkout.
- `scripts/sync-meta.sh`: sync this meta repo between laptop and SCNet.
- `scripts/pool_manager.py`: maintained K running + B pending container pool.

Safety defaults:

- GitLab must stay private/internal; anonymous public visibility is a hard stop.
- GitLab visibility `unknown` is also a hard stop.
- Browser submission defaults to dry-run.
- Source staging is whitelisted to competition source paths.
- Wheels, weights, large logs, passwords, and tokens do not go into git.
