# r1-container-start-20260707-0854

## Verdict

**Chrome start required for this window.** Directly replaying a previous
`sacct SubmitLine` is not currently a valid automatic container-pool primitive.

Evidence:

- `656380`: direct `sbatch --parsable -p hx1hdexclu08 .../Instances_2607070059422056_0_0/job_xdzs2026_c166_20260707_010007`
  completed after 15s and did not create container `656380_e03r1n11`.
- Its log reports `_dockerlist_656380 does not exist or has wrong format` and
  `No such container: 656380_e03r1n11`.
- `656384`: Chrome UI "confirm and start" on stopped instance
  `Instances_2607051832494555` created a real RUNNING job on `e03r1n11`, with
  `scnetctl.py attach` resolving container IP `173.0.195.9`.

Implication: `sacct SubmitLine` remains a useful clue, but do not wire stale
job scripts as the default `pool_manager.py` submit command unless a fresh
direct-sbatch test proves `RUNNING` plus successful `scnetctl.py attach`.

Raw evidence: `raw/startup_evidence.log`.
