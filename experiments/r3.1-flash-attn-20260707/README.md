# R3.1 flash-attention prefill candidate

Date: 2026-07-07

## Intent

Replace only eligible decoder prefill full-attention calls on gfx936 with a bf16 flash-attention backend for head_dim=256 / GQA 24q:4kv / chunked prefill. No model weights, model structure, scheduler code, locked CLI parameters, or decode path were changed.

## Implementation

- Overlay file: `overlay/vllm/v1/attention/backends/triton_attn.py`
- Gate: `XDB_R31_FLASH_ATTN_PREFILL=1`
- Min query length: `XDB_R31_FLASH_ATTN_MIN_Q=2`
- Default off: when the env gate is absent, the code falls through to the original `unified_attention` path.
- Candidate call: `flash_attn.flash_attn_interface.vllm_flash_attn_varlen_func`.
- Why not `varlen_fwd_unified`: the installed vLLM attention page/block size is 784, and gfx936/gfx938 unified flash attention rejects block sizes not divisible by 64. The candidate therefore packs paged KV cache into contiguous K/V for eligible prefill only.

Code anchors:
- Gate and log: `overlay/vllm/v1/attention/backends/triton_attn.py:415`
- Paged-KV pack: `overlay/vllm/v1/attention/backends/triton_attn.py:426`
- Eligibility guard: `overlay/vllm/v1/attention/backends/triton_attn.py:456`
- Flash call: `overlay/vllm/v1/attention/backends/triton_attn.py:495`
- Original fallback: `overlay/vllm/v1/attention/backends/triton_attn.py:628`

## Correctness smoke

Manual remote checks on gfx936 before guard runs:

- `flash_attn==2.8.3`, `flash_attn_2_cuda`, AITER, flash-mla imports succeeded under the DCU runtime env.
- Tiny varlen checks passed for `head_dim=256`, Q heads 24, KV heads 4, causal, and bottom-right chunk boundary:
  - `tail_chunk_small`
  - `full_prefill_small`
  - `batch_varlen`
- Module-level patched `TritonAttentionImpl.forward` check passed:
  - prefill `q=512,k=2048`: flash path enabled, max diff `0.000244`, bf16 allclose
  - decode `q=1,k=2048`: flash path disabled, exact fallback diff `0.0`

Standalone microbench before integration:

| q | k | original unified | pack+flash | speedup |
|---:|---:|---:|---:|---:|
| 128 | 512 | 0.649 ms | 0.706 ms | 0.92x |
| 512 | 2048 | 5.618 ms | 1.049 ms | 5.35x |
| 1024 | 4096 | 22.325 ms | 1.710 ms | 13.1x |
| 2048 | 8192 | 89.365 ms | 3.800 ms | 23.5x |

All standalone cases matched within bf16 tolerance.

## Guard runs

Locked/gate evidence:

- `../r3-fa-candidate-8to16-20260707-1905/vllm_server.log:13`: `Using max model len 32768`
- `../r3-fa-candidate-8to16-20260707-1905/vllm_server.log:20`: `max_seq_len=32768`, `load_format=runai_streamer`
- `../r3-fa-candidate-8to16-20260707-1905/vllm_server.log:33`: `XDB_R31_FLASH_ATTN_PREFILL enabled`
- `../r3-fa-candidate-accuracy-smoke-20260707-1948/vllm_server.log:33`: gate enabled again after fresh container attach
- `../r3-fa-candidate-extra-buckets-20260707-2010/poll.log:54`: `reuse_server=1 health=ok` for 4-8K and 16-32K补档

Primary 8-16K A/B:

| run | output tok/s | TTFT-P99 | TPOT-P99 | duration |
|---|---:|---:|---:|---:|
| baseline, `../r3-fa-baseline-8to16-20260707-1817/summary.json` | 7.233721 | 15603.455 ms | 70.666 ms | 236.393 s |
| candidate, `../r3-fa-candidate-8to16-20260707-1905/throughput/` median | 11.455965 | 3830.262 ms | 70.338 ms | 155.727 s |
| delta | +58.37% | -11773.193 ms (-75.45%) | -0.328 ms | -34.12% |

Three-bucket candidate补档:

| bucket | candidate output tok/s | candidate TTFT-P99 | candidate TPOT-P99 | reference baseline |
|---|---:|---:|---:|---|
| 4-8K | 13.416425 | 1780.959 ms | 69.210 ms | `../guard-a55f3c3-overlay-fullsmoke-20260707-0010/summary.json`: 12.156717 / 4536.688 ms / 69.731 ms |
| 8-16K | 11.455965 | 3830.262 ms | 70.338 ms | `../r3-fa-baseline-8to16-20260707-1817/summary.json`: 7.233721 / 15603.455 ms / 70.666 ms |
| 16-32K | 8.454834 | 5451.579 ms | 71.988 ms | `../guard-a55f3c3-overlay-fullsmoke-20260707-0010/summary.json`: 4.655501 / 28667.305 ms / 72.115 ms |

Accuracy smoke:

| metric | baseline | candidate | delta |
|---|---:|---:|---:|
| hotpotqa | 67.71 | 67.71 | 0.00 |
| gov_report | 35.00 | 34.45 | -0.55 |
| retrieval_multi_point | 100.00 | 100.00 | 0.00 |
| aggregation_keyword_aggregation | 100.00 | 100.00 | 0.00 |

Anchors:
- Baseline accuracy: `../r3-fa-baseline-8to16-20260707-1817/raw/accuracy.log:5`
- Candidate accuracy: `../r3-fa-candidate-accuracy-smoke-20260707-1948/raw/accuracy.log:5`

## Verdict

R3.1 candidate is positive on all measured throughput buckets and keeps smoke accuracy within the <1% delta gate. The biggest effect is TTFT: 8-16K TTFT-P99 drops from 15.603s to 3.830s, and 16-32K drops from the fullsmoke reference 28.667s to 5.452s.

Do not make this default without keeping the env gate. The implementation is safe as an overlay candidate because `XDB_R31_FLASH_ATTN_PREFILL` off is the original `unified_attention` path.
