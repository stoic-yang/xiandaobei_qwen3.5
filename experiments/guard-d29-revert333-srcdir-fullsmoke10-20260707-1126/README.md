# guard-d29-revert333-srcdir-fullsmoke10-20260707-1126

## Verdict

**FAIL as a stop-loss candidate.** The candidate commit
`51cb6f325ab53854e94f5d4b5018712f4f662d7f` cleanly starts and preserves
accuracy, but it is slightly slower than the same-container d29 baseline in all
three throughput buckets.

Comparison baseline:
`experiments/guard-d29e9db3-hotserver-nooverlay-fullsmoke10-20260707-1011`.

## Method

- Repo head: `51cb6f325ab53854e94f5d4b5018712f4f662d7f`
- Source repo:
  `/public/home/xdzs2026_c166/vllm_cscc_r1_d29_revert333_20260707`
- Overlay source dir: `/root/overlay-d29-revert333-51cb6f325ab5`
- Protocol: warmup once, then 3 buckets x 3 repetitions, `num_prompts=10`,
  followed by `run_accuracy.sh all 10`.
- The overlay came from `/root`, not `git show`, to avoid NFS git-object stalls.

## Metrics

| bucket | candidate tok/s | d29 baseline tok/s | delta |
| --- | ---: | ---: | ---: |
| 4-8K | 12.209453 | 12.211258 | -0.0148% |
| 8-16K | 7.219372 | 7.223185 | -0.0528% |
| 16-32K | 4.644872 | 4.652457 | -0.1630% |

Weighted output throughput: `7.445038`, delta `-0.0610%`.

Smoke10 accuracy:

- hotpotqa: `67.71` (same as baseline)
- gov_report: `35.30` (+0.30 absolute vs baseline)
- retrieval_multi_point: `100.00 (10/10)` (same)
- aggregation_keyword_aggregation: `100.00 (10/10)` (same)

## Repro Anchors

- Summary: `summary.json`
- Runtime fingerprint: `runtime_fingerprints.json`
- Overlay manifest: `overlay_manifest.txt`
- Raw logs: `raw/`
- Throughput JSONs: `throughput/`
- Remote run dir:
  `/public/home/xdzs2026_c166/codex_runs/guard-d29-revert333-srcdir-fullsmoke10-20260707-1126`
