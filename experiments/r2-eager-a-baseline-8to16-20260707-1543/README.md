# r2-eager-a-baseline-8to16-20260707-1543

R2.1 CUDA graph diagnostic, arm A: current graph/compile path.

- Intent: measure the current locked-start decode path before disabling CUDAGraphs.
- Method: warmup once, then `8-16K` x 3 repetitions, 3 prompts per repetition; median summary; accuracy mode `none`.
- Buckets: `8-16K`
- Overlay rev: ``
- Locked start script: `True`
- Load format: `runai_streamer`
- Enforce eager: `False`
- Remote run dir: `/public/home/xdzs2026_c166/codex_runs/r2-eager-a-baseline-8to16-20260707-1543`
- Local summary: `summary.json`
- Raw logs: `raw/`
- Throughput result JSONs: `throughput/`

## Median result

| bucket | output tok/s | TTFT-P99 ms | TPOT-P99 ms | duration s |
| --- | ---: | ---: | ---: | ---: |
| 8-16K | 7.882584 | 13225.199 | 69.910539 | 73.453 |

## Evidence anchors

- `vllm_server.log:20`: engine config verified `max_seq_len=32768`, `load_format=runai_streamer`, `enforce_eager=False`, and `cudagraph_mode FULL_AND_PIECEWISE`.
- `vllm_server.log:129`: captured CUDA graphs for mixed prefill-decode `PIECEWISE`.
- `vllm_server.log:130`: captured CUDA graphs for decode `FULL`.
- `vllm_server.log:131`: graph capture finished in 121s.
- `vllm_server.log:110-111`: RunAI streamer loaded 51.7 GiB in 293.37s; model loading took 294.93s.

Verdict: current locked path is not graph-missing; decode FULL graph capture is present before the A/B toggle.
