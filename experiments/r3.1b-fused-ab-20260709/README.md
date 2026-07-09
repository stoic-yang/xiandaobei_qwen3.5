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

## Status

Detached A/B launcher started from local meta-main.

- launched at: `2026-07-09T16:03:54+08:00`
- local launcher pid: `63192`
- A remote guard pid: `119`
- first observed state: A arm running, summary pending
