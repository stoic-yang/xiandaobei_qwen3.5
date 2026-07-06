# r1-guard-overlay-runner-20260706-2350

- Intent: make the R1 rollback matrix runnable without repeating the wheel/source mismatch mistake.
- Change: `scripts/guard_bench.py` now supports `--overlay-rev` to overlay selected Python files from a remote git revision after installing the wheel.
- Change: `scripts/guard_bench.py` now supports `--buckets`, so screening runs can target one bucket while full guard runs still use all three buckets.
- Change: future guard runs collect `runtime_fingerprints.json`; overlay runs also collect `overlay_manifest.txt`.

## Verification

- `python3 -m py_compile scripts/guard_bench.py` passed.
- Dry-run command:

```bash
python3 scripts/guard_bench.py \
  --dry-run \
  --run-id dryrun-r1-overlay \
  --overlay-rev a55f3c316 \
  --buckets 8-16K \
  --accuracy none \
  --repetitions 1 \
  --num-prompts 1
```

- Dry-run output saved at `raw/dryrun_overlay_8to16.sh`.
- `bash -n raw/dryrun_overlay_8to16.sh` passed.

## Next Use

In a fresh container window, run one guard per commit, for example:

```bash
python3 scripts/guard_bench.py \
  --run-id guard-a55f3c3-full-YYYYMMDD-HHMM \
  --overlay-rev a55f3c316 \
  --buckets 4-8K,8-16K,16-32K \
  --repetitions 3 \
  --num-prompts 10 \
  --accuracy smoke \
  --copy-model-local \
  --stop-existing
```

Use `--accuracy full` only when the container window is long enough; the prior full run exceeded the stop-loss window.

