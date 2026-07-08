# r3.0-gemm-shape-attribution-20260708-112017-wheel

## Intent

R3.0 Step 0: attribute the post-R3.1 prefill `Cijk_*` Tensile GEMM hotspot before running TunableOp or Inductor autotune.

## Method

- Remote run dir: `/public/home/xdzs2026_c166/codex_runs/r3.0-gemm-shape-attribution-20260708-112017-wheel`
- Locked service: `--max-model-len 32768`, `--max-num-seqs 128`, `--max-num-batched-tokens 4096`, `--load-format runai_streamer`
- Request: first `8-16K_throughput.jsonl` prompt, `max_tokens=1`, status `200`, wall `127.414s`
- Source profile anchor: `experiments/r3.2-post-r31-prefill-profile-20260707-2210/profile/vllm_prefill.hipkernel.csv`
- Shape sources:
  - Python hook on `UnquantizedLinearMethod.apply` as a diagnostic.
  - Compiled graph signature in remote torch compile cache: `/public/home/xdzs2026_c166/vllm_cache/vllm_cache/torch_compile_cache/b679024598/rank_0_0/backbone/computation_graph.py`
  - Model config `/public/home/xdzs2026_c166/Qwen3.5-27B/config.json`

## Results

The Python hook is not usable for language-model Cijk attribution under torch.compile. It captured only visual tower warmup/profile shapes (`visual.blocks.*`, 110 linear events), not the compiled text-request projection GEMMs.

The compiled graph gives the useful attribution:

| family | weight shape | layers | calls if 4 chunks |
|---|---:|---:|---:|
| `mlp.gate_up_proj` | `[34816,5120]` | 64 | 256 |
| `mlp.down_proj` | `[5120,17408]` | 64 | 256 |
| `linear_attn.in_proj_qkvz` | `[16384,5120]` | 48 | 192 |
| `linear_attn.in_proj_ba` | `[96,5120]` | 48 | 192 |
| `linear_attn.out_proj` | `[5120,6144]` | 48 | 192 |
| `self_attn.qkv_proj` | `[14336,5120]` | 16 | 64 |
| `self_attn.o_proj` | `[5120,6144]` | 16 | 64 |
| `lm_head/logits` | `[248320,5120]` | 1 | 4 |

Call-count reconciliation: language projection calls `1216` + logits calls `4` = observed Cijk calls `1220`.

## Verdict

Step0 is complete enough to start A/B. The exact WGM row to shape mapping remains ambiguous because the committed hipkernel CSV is aggregated by kernel name and does not include M/N/K per call. Target A/B coverage should focus on the compiled graph shapes above, especially MLP `[34816,5120]`, `[5120,17408]`, GDN `[16384,5120]`, and output projection `[5120,6144]`.

Next: run TunableOp 8-16K throughput-only A/B and require a tunable result file or equivalent shape record covering those target shapes. If it is empty or misses the targets, close TunableOp and try Inductor GEMM autotune.
