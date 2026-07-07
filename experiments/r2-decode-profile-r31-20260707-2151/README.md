# r2-decode-profile-r31-20260707-2151

## Intent

R2.0 decode-only profile. GitLab official scoring was unavailable, so this run closes the local R2 diagnosis: split steady decode TPOT into GPU kernel busy time vs residual host/gap, and measure full-attn decode attention share.

## Method

- Source repo: `/public/home/xdzs2026_c166/vllm_cscc_competition` at `847d1bef10b0b5bb71b7e427535b610a20a4d263`.
- Service: `hipprof --trace-off --session xdb_r20_2151` wrapping locked `vllm serve`.
- Locked evidence:
  - `vllm_server.log`: `Using max model len 32768`
  - `vllm_server.log`: `max_seq_len=32768`, `load_format=runai_streamer`
  - `vllm_server.log`: `XDB_R31_FLASH_ATTN_PREFILL enabled`
  - `vllm_server.log`: `Capturing CUDA graphs (decode, FULL)`
- Trace protocol: send one streaming 8-16K prompt, wait for first non-empty streamed token, then `hipprof --session-client --start`; stop/flush after 64 content chunks.
- Large hipprof JSON files were intentionally not copied into meta. The committed artifacts are the compact CSV summaries plus `decode_stream.json`.

## Results

Streaming window:

| item | value |
|---|---:|
| traced chunks | 64 |
| TTFT | 3273.567 ms |
| trace wall | 4553.406 ms |
| wall per chunk | 71.147 ms |
| median chunk interval | 71.315 ms |
| P99 chunk interval | 72.370 ms |

GPU kernel split:

| item | value |
|---|---:|
| kernel busy sum | 4453.442 ms |
| kernel busy per chunk | 69.585 ms |
| wall - kernel sum | 99.963 ms |
| residual per chunk | 1.562 ms |

Top decode kernels:

| group | share of kernel time | per chunk |
|---|---:|---:|
| top Tensile GEMM row 1 | 62.878% | 43.754 ms |
| top Tensile GEMM row 2 | 20.985% | 14.602 ms |
| top Tensile GEMM row 3 | 5.199% | 3.618 ms |
| full-attn `kernel_unified_attention_3d` | 4.158% | 2.893 ms |
| GDN recurrent decode kernel | 0.910% | 0.633 ms |
| causal conv update | 0.347% | 0.242 ms |

Host/API signals:

- `hipGraphLaunch`: 64 calls for 64 traced chunks, one replay per chunk/token.
- `hipLaunchKernel`: 970 calls total, about 15.2 explicit launches/chunk outside graph replay accounting.
- HIP API durations include blocking waits and async copies, so they are not additive with kernel time. The decisive additive bound is `trace wall - kernel busy sum`, about `1.56ms/chunk`.

## Verdict

R2.3 host overlap is not a high-ROI mainline item. The measured decode wall is already almost all GPU kernel busy time (`4453ms` of `4553ms`), so the remaining host/gap slack is at most about `1.6ms/token` in this trace.

Full-attn decode attention is small: `4.158%` of decode kernel time, about `2.89ms/token`. R3 flash-attention work is still primarily a prefill/TTFT win; it should not be expected to materially move decode TPOT.

Decode graph coverage remains healthy: one `hipGraphLaunch` per traced chunk and service log shows decode `FULL` graph capture.
