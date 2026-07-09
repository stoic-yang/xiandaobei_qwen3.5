# R3.1b fused prefill A/B

## Intent

Run `plans/task-r3.1b-long-context-flash-attn.md` Step 1 on the fresh `27B` container:

- A: current default fused Triton prefill (`VLLM_TRITON_FUSED_PREFILL=1`, `XDB_R31_FLASH_ATTN_PREFILL=1`)
- B: fused disabled, xdb/package varlen forced (`VLLM_TRITON_FUSED_PREFILL=0`, `XDB_R31_FLASH_ATTN_PREFILL=1`)

This is a direction-finding screen for the 16-32K bucket only. It is not a submission gate.

## Protocol

- Container: `Instances_2607090940238205_1_0` / Slurm job `662724`
- Port: isolated `18001`; do not touch `8001`
- Startup: `guard_bench.py --locked-start-script --load-format runai_streamer`
- Data: `16-32K`, `3` prompts, `3` repetitions, accuracy `none`
- Comparison: same container, same remote source HEAD `4653a56d5`

## Results

### Step 1 single-bucket A/B

`16-32K`, `3` prompts x `3` repetitions, locked `runai_streamer`, isolated port `18001`.

| arm | path | output tok/s | TTFT-P99 ms | TPOT-P99 ms | anchor |
| --- | --- | ---: | ---: | ---: | --- |
| A | fused default (`VLLM_TRITON_FUSED_PREFILL=1`) | `4.122288` | `21793.173` | `71.514` | `experiments/r3.1b-fused-A-default-16to32-20260709-1605/summary.json` |
| B | xdb/package varlen (`VLLM_TRITON_FUSED_PREFILL=0`) | `4.297041` | `19279.477` | `71.532` | `experiments/r3.1b-fused-B-xdbvarlen-16to32-20260709-1605/summary.json` |

B is `+4.239%` output throughput and `-11.535%` TTFT-P99 on this long bucket. TPOT is unchanged, so the gain is prefill/TTFT-side.

### B three-bucket check

Global B is not a safe default:

| bucket | output tok/s | TTFT-P99 ms | TPOT-P99 ms | anchor |
| --- | ---: | ---: | ---: | --- |
| 4-8K | `8.021708` | `4515.288` | `68.772` | `experiments/r3.1b-xdbvarlen-3bucket-20260709-1700/summary.json` |
| 8-16K | `8.175304` | `11727.982` | `69.952` | same |
| 16-32K | `4.294348` | `19278.517` | `71.685` | same |

The log repeatedly shows prompt throughput near `~758 tok/s`, matching the old slower attention behavior. Do not globally disable fused prefill.

### Threshold candidate

Prepared an env-gated hybrid patch:

- remote isolated worktree: `/public/home/xdzs2026_c166/vllm_cscc_codex_r31b_threshold_20260709_173633`
- remote candidate commit: `b82c6a4a6fef63c276b982194f8baee3bd91a7f9`
- patch anchor: `experiments/r3.1b-fused-ab-20260709/threshold-varlen.patch`
- intended env: `VLLM_TRITON_FUSED_PREFILL=1`, `VLLM_TRITON_FUSED_PREFILL_MAX_SEQ_LEN=16384`, `XDB_R31_FLASH_ATTN_PREFILL=1`
- semantics: default `0` keeps current fused behavior; threshold `16384` routes long-context prefill to xdb varlen while leaving shorter prompts on fused.

Validation was blocked, not failed: `experiments/r3.1b-threshold16384-3bucket-20260709-1739/vllm_server.log` shows vLLM startup failed because only `59.54/63.98 GiB` was free, below the locked `0.95` requirement (`60.79 GiB`). A separate `Qwen3.5-0.8B` service was running on port `8001`, so this agent did not clean it up.

### Threshold 16384 three-bucket validation

Follow-up run after forcing the guard to use the persistent 27B model path:

- launcher: `experiments/r3.1b-fused-ab-20260709/run_threshold16384_guard.sh`
- valid run: `experiments/r3.1b-threshold16384-3bucket-20260709-1822/summary.json`
- env: `VLLM_TRITON_FUSED_PREFILL=1`, `VLLM_TRITON_FUSED_PREFILL_MAX_SEQ_LEN=16384`, `XDB_R31_FLASH_ATTN_PREFILL=1`
- model guard: `preflight.txt` confirms persistent `/public/home/xdzs2026_c166/Qwen3.5-27B` is `layers=64 hidden=5120 heads=24 kv_heads=4 head_dim=256`; the launcher passes a non-existent `--local-model-dir` so `/root/Qwen3.5-27B` cannot shadow it.

| bucket | output tok/s | TTFT-P99 ms | TPOT-P99 ms | anchor |
| --- | ---: | ---: | ---: | --- |
| 4-8K | `9.758505` | `2009.492` | `68.648` | `experiments/r3.1b-threshold16384-3bucket-20260709-1822/summary.json` |
| 8-16K | `11.458883` | `4482.659` | `69.782` | same |
| 16-32K | `8.176126` | `6191.184` | `71.481` | same |

This validates that threshold routing is far better than global varlen (`8.021708 / 8.175304 / 4.294348`), but it is still not a submission-ready growth point because the required same-container default A was not completed.

### Default A retry was invalidated

Attempted same-container default A with `VLLM_TRITON_FUSED_PREFILL_MAX_SEQ_LEN=0` in `experiments/r3.1b-fuseddefault-3bucket-20260709-1906/`. It is invalid and stopped:

- `4-8K_rep*.json` all have `completed=0 failed=3` with `ConnectionRefusedError` to `127.0.0.1:18001`.
- During the run, a separate `/public/home/xdzs2026_c166/Qwen3.5-0.8B` service appeared on port `8001`.
- Per shared-container discipline, this agent stopped only the `18001` guard/benchmark and preserved the `8001` 0.8B service. See `experiments/r3.1b-fuseddefault-3bucket-20260709-1906/stopped.txt`.

### Default A next command

Run this only on a clean 27B-only container with enough lifetime for a cold `runai_streamer` start plus three buckets:

```bash
cd /Users/keynary/Code/xiandaobei/meta-main
RUN_ID="r3.1b-fuseddefault-3bucket-$(date '+%Y%m%d-%H%M')"
SESSION="xdb_${RUN_ID//./_}"
tmux new-session -d -s "$SESSION" \
  "cd /Users/keynary/Code/xiandaobei/meta-main && RUN_ID='$RUN_ID' bash experiments/r3.1b-fused-ab-20260709/run_default_fused_guard.sh"
```

Equivalent direct form inside any long-running shell:

```bash
RUN_ID=r3.1b-fuseddefault-3bucket-$(date '+%Y%m%d-%H%M') \
  bash experiments/r3.1b-fused-ab-20260709/run_default_fused_guard.sh
```

Compare its `summary.json` against `experiments/r3.1b-threshold16384-3bucket-20260709-1822/summary.json`.

## Verdict

- Confirmed a real long-bucket growth signal: xdb/package varlen beats fused default on `16-32K` by `+4.239%` in same-container A/B.
- Rejected global `VLLM_TRITON_FUSED_PREFILL=0` as a default candidate because three-bucket B is weak.
- Threshold `16384` is promising versus global varlen, but not yet proven versus current default across all three buckets.
- Next GPU action on a clean 27B-only container: run the same three-bucket default A (`VLLM_TRITON_FUSED_PREFILL_MAX_SEQ_LEN=0`) with the fixed launcher, then compare against `r3.1b-threshold16384-3bucket-20260709-1822`. Run accuracy smoke only if all three buckets are non-regressing.
