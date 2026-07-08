# r3.0-shape-attribution-blocked-20260708-1051

## Status

`blocked-no-container`

## Intent

Continue the remaining task cards after the R3.1 official AC result, starting with `plans/task-r3.0-gemm-autotune.md` Step 0 shape attribution.

## Decision

It is valid to continue, with these adjustments:

1. Treat R3.1 official AC (`74.6924`) as the current safe baseline, not the old d29 `59.0018` row.
2. Keep R3.0/R2.4 GEMM autotune as the next mainline.
3. Do not start TunableOp/Inductor A/B blindly. First attribute the hot `Cijk_*` kernel rows to exact model ops/shapes.
4. Keep R3.2 GDN downgraded unless a later 16-32K post-R3.1 profile contradicts the current `9.303%` GDN-core result.
5. Do not start R4 INT8 before R3.0 Step0; official accuracy penalty is small (`0.5644`) but still leaves no reason to risk a precision branch before the lower-risk GEMM/config path.

## Attempted Action

Checked SCNet state before launching any long job:

```text
checked_at: 2026-07-08T10:51:06
job: none
worker: unreachable
Connection closed by UNKNOWN port 65535
scnetctl: no running SCNet container job; Chrome startup is required
```

No vLLM/profiler/guard job was started.

## Next Step

After a container is started through the SCNet UI/Chrome flow:

1. `cd /Users/keynary/Code/xiandaobei/meta-main && git pull --ff-only`
2. `python3 scripts/scnetctl.py attach`
3. Confirm SSH reuse:
   `ssh -F ~/.ssh/xiandaobei.generated.conf -O check xiandaobei-worker-auto`
4. Run `plans/task-r3.0-gemm-autotune.md` Step 0:
   - launch a locked R3.1 service with `--max-model-len=32768` and `--load-format runai_streamer`
   - capture one 8-16K `max_tokens=1` request
   - attribute top `Cijk_*` rows to model ops/shapes
   - write `experiments/r3.0-gemm-shape-attribution-<date>/summary.json`

Keep the run backgrounded/nohup if startup or profiling exceeds two minutes.
