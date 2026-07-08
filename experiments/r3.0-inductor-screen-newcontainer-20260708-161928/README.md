# R3.0 Inductor screen newcontainer 20260708-161928

Status: prepared only; not run.

Intent: fresh same-container Inductor max-autotune candidate screen after collecting a clean new-container baseline.

Prepared artifact: `sitecustomize/sitecustomize.py`, copied from the clean Inductor smoke experiment. It writes `torch._inductor.config` dumps when `XDB_INDUCTOR_DUMP_CONFIG=1` and `XDB_INDUCTOR_CONFIG_JSON=<path>` are set.

Candidate env intended for the run:

- `TORCHINDUCTOR_MAX_AUTOTUNE=1`
- `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1`
- `PYTHONPATH=<remote-run>/sitecustomize:/usr/local`
- `XDB_INDUCTOR_DUMP_CONFIG=1`
- `XDB_INDUCTOR_CONFIG_JSON=<remote-run>/config/config.json`

Verdict: no measurement. The matching baseline never completed because the new container was shared with external 0.8B smoke runs and then disappeared. Reuse the hook content in the next clean container, but create a fresh run ID.
