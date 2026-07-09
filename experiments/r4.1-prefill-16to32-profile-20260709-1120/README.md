# R4.1 16-32K Prefill Roofline Profile

## Intent

Run `plans/task-r4.1-prefill-gemm-roofline.md` Step 1: capture one locked `16-32K` prefill-only profile under R3.1 current source, then compute the `Cijk_*` GEMM roofline.

## Status

Completed on SCNet container `Instances_2607090940238205_0_0` / Slurm job `661607`.

- launched at: `2026-07-09T11:23:19+08:00`
- remote PID: `7677`
- exit: `0`
- post-run check: no `vllm`/`hipprof`/run processes left; ports `18001` and `8001` down; `hy-smi` HCU use `0%`, HCU memory use `0%`

## Remote Plan

- remote run dir: `/public/home/xdzs2026_c166/codex_runs/r4.1-prefill-16to32-profile-20260709-1120`
- port: `18001`
- request: first `16-32K_throughput.jsonl` prompt, `max_tokens=1`
- locked serving: `--max-model-len 32768`, `--max-num-seqs 128`, `--max-num-batched-tokens 4096`, `--load-format runai_streamer`
- source: install competition wheel, then overlay current `/public/home/xdzs2026_c166/vllm_cscc_competition` Python files into site-packages

## Expected Outputs

- `driver.log`
- `exit`
- `vllm_server.log`
- `profile/prefill_request.json`
- `profile/vllm_prefill16*.hipkernel.csv`
- `summary.json`

## Result

Locked serving evidence:

- `vllm_server.log`: `max_seq_len=32768`, `load_format=runai_streamer`
- R3.1 path enabled: `XDB_R31_FLASH_ATTN_PREFILL enabled`
- source head: `4653a56d52719dc8617d31e556e62a670ed5e6ec`; see `raw/repo_log.txt`
- request: first `16-32K_throughput.jsonl` prompt, API usage `prompt_tokens=20576`, `max_tokens=1`
- request wall: `7169.049ms`

Kernel profile:

| group | total ms | pct |
|---|---:|---:|
| `Cijk_*` GEMM | `3154.114` | `44.715%` |
| flash-attn `_fwd_kernel` | `2862.586` | `40.582%` |
| GDN core + helpers | `610.613` | `8.657%` |

Roofline:

- included dense projection work: `1001.103 TFLOP`
- aggregate `Cijk_*`: `317.396 TFLOPS`
- fraction of external gfx936 `395 TFLOPS` peak: `80.353%`

## Verdict

`Cijk_*` remains yellow-zone, consistent with the 8-16K precheck (`~320 TFLOPS`). Direct GEMM kernel work is not justified from this alone, and the already-tested TunableOp/Inductor routes should stay closed.

The new score-growth signal is long-context full-attention: at 16-32K, `_fwd_kernel` rises to `40.582%`, nearly the same scale as GEMM. Follow-up Step 0 attributed this symbol to the local fused Triton prefill kernel (`vllm/v1/attention/ops/triton_flash_prefill.py`), not package `flash_attn` varlen. Next task should benchmark `VLLM_TRITON_FUSED_PREFILL` before spending effort on INT8 GEMM implementation.
