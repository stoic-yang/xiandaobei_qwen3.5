# guard-d29e9db3-hotserver-nooverlay-fullsmoke10-20260707-1011

## Verdict

**PASS — same-container d29 hot-server baseline for the 333 rollback candidate.**

This run reused the already-hot vLLM server started by
`guard-d29e9db3-overlay-fullsmoke10-20260707-0900`, after that startup had
installed the competition wheel and overlaid d29's ROCm attention path. The
guard itself ran with no new overlay to avoid NFS git-object stalls; the runtime
fingerprint confirms the active site package:

- `vllm/v1/attention/ops/triton_unified_attention.py` site SHA starts
  `acf4b51ba925`, differing from wheel SHA `8e8d393c4d55`.
- `vllm/version.py` site SHA starts `984b8316ba5d`.
- `qwen3_next.py` stayed on the installed wheel/site SHA `b33617ce76f0`.

## Metrics

| bucket | median output tok/s | TTFT P99 ms | TPOT P99 ms |
| --- | ---: | ---: | ---: |
| 4-8K | 12.211258 | 4539.247 | 69.366 |
| 8-16K | 7.223185 | 15631.965 | 70.707 |
| 16-32K | 4.652457 | 28691.588 | 72.199 |

Weighted output throughput: `7.449581`.

Smoke10 accuracy:

- hotpotqa: `67.71`
- gov_report: `35.00`
- retrieval_multi_point: `100.00 (10/10)`
- aggregation_keyword_aggregation: `100.00 (10/10)`

## Repro Anchors

- Summary: `summary.json`
- Runtime fingerprint: `runtime_fingerprints.json`
- Raw logs: `raw/`
- Throughput JSONs: `throughput/`
- Remote run dir:
  `/public/home/xdzs2026_c166/codex_runs/guard-d29e9db3-hotserver-nooverlay-fullsmoke10-20260707-1011`
