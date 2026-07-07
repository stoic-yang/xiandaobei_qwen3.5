# r2-eager-b-enforce-8to16-20260707-1609

R2.1 CUDA graph diagnostic, arm B: `--enforce-eager`.

- Intent: disable CUDAGraphs and torch.compile through vLLM `--enforce-eager`, then compare against arm A in the same warm container protocol.
- Method: warmup once, then `8-16K` x 3 repetitions, 3 prompts per repetition; median summary; accuracy mode `none`.
- Buckets: `8-16K`
- Overlay rev: ``
- Locked start script: `True`
- Load format: `runai_streamer`
- Enforce eager: `True`
- Remote run dir: `/public/home/xdzs2026_c166/codex_runs/r2-eager-b-enforce-8to16-20260707-1609`
- Local summary: `summary.json`
- Raw logs: `raw/`
- Throughput result JSONs: `throughput/`

## Median result

| bucket | output tok/s | TTFT-P99 ms | TPOT-P99 ms | duration s |
| --- | ---: | ---: | ---: | ---: |
| 8-16K | 5.737616 | 13374.594 | 118.195300 | 103.876 |

## Evidence anchors

- `vllm_server.log:19`: vLLM reports `--enforce-eager` disables torch.compile and CUDAGraphs.
- `vllm_server.log:21`: CUDAGraph is disabled under eager mode.
- `vllm_server.log:24`: engine config verified `max_seq_len=32768`, `load_format=runai_streamer`, `enforce_eager=True`, `compilation_config mode NONE`, and `cudagraph_mode NONE`.
- `vllm_server.log:99-100`: RunAI streamer loaded 51.7 GiB in 230.78s; model loading took 231.32s.

Verdict: disabling the graph/compile stack makes 8-16K decode materially slower.
