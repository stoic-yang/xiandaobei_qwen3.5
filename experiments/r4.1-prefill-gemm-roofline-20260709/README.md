# R4.1 Prefill GEMM Roofline Precheck

## Intent

Use existing post-R3.1 profile artifacts to estimate whether the current `Cijk_*` prefill GEMM hotspot has enough headroom for another score-growth task before spending a fresh container.

## Inputs

- `experiments/r3.2-post-r31-prefill-profile-20260707-2210/summary.json`
- `experiments/r3.0-gemm-shape-attribution-20260708-112017-wheel/summary.json`
- External gfx936/DCU lecture anchor already reflected in `memory/50-arch-bottleneck.md`: sustained compute peak about `395 TFLOPS`.

## Method

For the R3.2 prefill-only request:

- prompt tokens: `13964`
- `Cijk_*` time: `2122.873132 ms`
- dense projection families included: MLP gate/up, MLP down, GDN in/out projection, full-attn qkv/o projection
- excluded from the main roofline estimate: tiny `linear_attn.in_proj_ba` and low-share `lm_head/logits`

FLOPs are counted as `2 * M * N * K * layers`.

## Result

Approximate dense-projection work: `679.403 TFLOP`.

Aggregate achieved GEMM throughput using `Cijk_*` time: `320.040 TFLOPS`, about `81.0%` of the `395 TFLOPS` external gfx936 peak anchor.

This is a yellow-zone result. It is not close enough to peak to dismiss all work, but it is also not a 2-3x gap. Since TunableOp did not cover target language GEMMs and Inductor same-container A/B was neutral/negative, the next step should be a 16-32K roofline confirmation plus low-precision compute microbench, not direct GEMM kernel work.

## Verdict

Proceed to `plans/task-r4.1-prefill-gemm-roofline.md` Step 1 if a clean container is available. Do not change defaults from this precheck alone.
