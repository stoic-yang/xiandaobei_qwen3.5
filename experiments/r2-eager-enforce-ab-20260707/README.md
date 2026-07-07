# R2.1 enforce_eager A/B

Purpose: execute Step 2 of `plans/task-r2.1-cudagraph.md` and quantify whether the current vLLM path is already using CUDA graph coverage during decode.

Scope: diagnostic only. This is not a full guard run: it uses only the `8-16K` bucket, 3 prompts, 3 repetitions, and `--accuracy none`.

## Protocol

- Container/job: SCNet job `656918`, worker IP `173.0.253.4`.
- Guard: `scripts/guard_bench.py --locked-start-script --load-format runai_streamer`.
- Locked CLI verified in logs: `max_seq_len=32768`, `max-num-seqs=128`, `max-num-batched-tokens=4096`.
- Shared startup caches: `VLLM_CACHE_ROOT=/public/home/xdzs2026_c166/vllm_cache/vllm_cache`, `TRITON_CACHE_DIR=/public/home/xdzs2026_c166/vllm_cache/triton_cache`.
- A: `experiments/r2-eager-a-baseline-8to16-20260707-1543/`, current graph/compile path.
- B: `experiments/r2-eager-b-enforce-8to16-20260707-1609/`, same protocol with `--enforce-eager`.

## Median comparison

| arm | output tok/s | TTFT-P99 ms | TPOT-P99 ms | duration s |
| --- | ---: | ---: | ---: | ---: |
| A current graph/compile | 7.882584 | 13225.199 | 69.910539 | 73.453 |
| B enforce eager | 5.737616 | 13374.594 | 118.195300 | 103.876 |
| B - A | -2.144968 (-27.21%) | +149.394 (+1.13%) | +48.285 (+69.07%) | +30.423 (+41.42%) |

## Log anchors

- A `vllm_server.log:20`: `enforce_eager=False`, `cudagraph_mode FULL_AND_PIECEWISE`, `max_seq_len=32768`, `load_format=runai_streamer`.
- A `vllm_server.log:129-131`: mixed prefill-decode `PIECEWISE` graph capture and decode `FULL` graph capture completed.
- B `vllm_server.log:19`: `--enforce-eager` disables torch.compile and CUDAGraphs.
- B `vllm_server.log:24`: `enforce_eager=True`, compile mode `NONE`, `cudagraph_mode NONE`, `max_seq_len=32768`, `load_format=runai_streamer`.

## Verdict

Current decode is not missing CUDA graph coverage. Turning on `--enforce-eager` makes 8-16K TPOT-P99 worse by 48.285ms and output throughput worse by 27.21%, while the baseline log explicitly captures decode `FULL` CUDA graphs.

Interpretation caveat: this is a graph-plus-compile-stack diagnostic, not a pure CUDAGraph-only toggle. vLLM reports that `--enforce-eager` disables both torch.compile and CUDAGraphs. The safe conclusion is that the current compile/graph path is already doing substantial work; R2.1 should shift from "make graph work from zero" to decode-only launch-count confirmation and narrow residual overhead tuning.

## Cleanup

After the B run, the eager service was stopped and `http://127.0.0.1:8001/health` returned `000`, so later experiments should not accidentally reuse this server.
