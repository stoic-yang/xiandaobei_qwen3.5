# R3.1b Step0 · `_fwd_kernel` source attribution

## Intent

Execute `plans/task-r3.1b-long-context-flash-attn.md` Step 0 without starting a new GPU run: use the R4.1 profile artifact plus current persistent source to identify what the `16-32K` `_fwd_kernel` actually is.

## Inputs

- R4.1 profile: `experiments/r4.1-prefill-16to32-profile-20260709-1120/profile/vllm_prefill16.hipkernel.csv`
- R4.1 remote source head: `4653a56d52719dc8617d31e556e62a670ed5e6ec`
- R4.1 remote repo log: `../r4.1-prefill-16to32-profile-20260709-1120/raw/repo_log.txt`
- Current SCNet worker status at `2026-07-09T15:40:08+08:00`: no running container job; GPU-side A/B could not be started in this pass.

## Finding

The profile row:

```text
"_fwd_kernel","96","2862586283","29818607","40.582"
```

matches the Triton kernel defined in:

```text
vllm/v1/attention/ops/triton_flash_prefill.py
```

Current `triton_attn.py` calls this kernel through `flash_layout_chunked_prefill_attention()` before the older xdb varlen branch:

1. `_can_use_rocm_flash_prefill_fused(...)`
2. `flash_layout_chunked_prefill_attention(...)`
3. fallback to `_can_use_xdb_flash_prefill(...)`
4. fallback to `unified_attention(...)`

So the R4.1 `16-32K` hotspot is **not** proven to be `flash_attn.flash_attn_interface.vllm_flash_attn_varlen_func`. It is the newer fused Triton prefill path introduced by `f014885bc perf(attention): add fused triton flash prefill kernel` and active at `4653a56d5`.

The `96` calls still fit the structural expectation: `16` full-attention layers x `6` prefill chunks.

## Consequence

R3.1b should not start by tuning `FLASH_ATTENTION_TRITON_AMD_AUTOTUNE`; that knob applies to the package flash-attn path, while the current default hot kernel is in local Triton source.

Next same-container matrix:

| arm | env | purpose |
|---|---|---|
| A | current defaults: `VLLM_TRITON_FUSED_PREFILL=1`, `XDB_R31_FLASH_ATTN_PREFILL=1` | baseline; fused Triton path should win |
| B | `VLLM_TRITON_FUSED_PREFILL=0`, `XDB_R31_FLASH_ATTN_PREFILL=1` | force xdb/package varlen path |
| C | `VLLM_TRITON_FUSED_PREFILL=0`, `XDB_R31_FLASH_ATTN_PREFILL=0` | unified-attention fallback attribution only |
| D | B plus `FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=1` if supported | only if B is competitive |

Run `16-32K` throughput-only first, 3 prompts x 3 reps, on isolated port `18001`. If A is clearly best, proceed to an env-gated tile-shape probe for `triton_flash_prefill.py` rather than package flash-attn autotune.

## Verdict

Step 0 partially complete from existing evidence. Full GPU-side confirmation still needs one same-container short profile with `VLLM_TRITON_FUSED_PREFILL=0`, but the source-level attribution is strong enough to reprioritize Step 1.
