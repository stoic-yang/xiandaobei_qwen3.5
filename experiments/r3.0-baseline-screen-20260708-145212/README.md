# R3.0 baseline screen 20260708-145212

Status: invalid.

Intent: collect the same-container baseline for `r3.0-inductor-autotune-screen-20260708-144124`.

Verdict: discard. The old container disappeared before the baseline produced artifacts; local `start.stdout.log` only contains the detached local PID and no remote `summary.json` was collected. This means the preceding Inductor screen remains candidate-only background evidence, not an A/B result.

Replacement: rerun in the new container as `r3.0-baseline-screen-newcontainer-retry-*`, then compare against a new same-container Inductor screen.
