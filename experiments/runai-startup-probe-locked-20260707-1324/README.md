# runai-startup-probe-locked-20260707-1324

## Verdict

**PASS as a fast-iteration startup method, not a model-performance A/B.**

With the competition locked `--max-model-len 32768` restored, `runai_streamer`
started the service from the persistent home-directory model without copying
`Qwen3.5-27B` to `/root`. It reached `/health` in `725s` from `vllm serve`
start. This does not reproduce a 300s full-ready startup, but it is still much
faster than the previously observed 40-60 minute explicit model-copy workflow.

This experiment should be treated as startup infrastructure evidence only. It
does not alter model weights, model structure, scheduler code, or R1 benchmark
logic.

## Method

- SCNet job: `656918`, IP `173.0.253.4`.
- Competition wheel:
  `/public/home/xdzs2026_c166/vllm_cscc_competition/dist/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl`.
- Model path: `/public/home/xdzs2026_c166/Qwen3.5-27B`.
- Load format: `runai_streamer`.
- Locked args kept: `--max-model-len 32768`,
  `--max-num-seqs 128`, `--max-num-batched-tokens 4096`.
- Cache dirs:
  `VLLM_CACHE_ROOT=/public/home/xdzs2026_c166/vllm_cache/vllm_cache`,
  `TRITON_CACHE_DIR=/public/home/xdzs2026_c166/vllm_cache/triton_cache`.
- Cache state: warm after the invalid 1259 probe had already populated shared
  vLLM compile cache.

## Metrics

| Stage | Time |
| --- | ---: |
| pip install same wheel | `1s` |
| RunAI stream to CPU | `341.16s` |
| model loading | `342.118507s` |
| torch.compile total | `44.08s` |
| initial profiling / warmup | `171.88s` |
| graph capture | `42s` |
| engine init after load | `279.43s` |
| vLLM start to `/health` | `725s` |

Streamer throughput: `51.7 GiB` at `155.3 MiB/s`.

The log confirms the compliant context:

- `non-default args` includes `max_model_len: 32768`.
- vLLM resolved `Using max model len 32768`.
- engine config shows `max_seq_len=32768`.

## Interpretation

The useful win is avoiding a full persistent copy to `/root` and reading the
safetensors in the loader through Run:ai Model Streamer. The remaining startup
time is dominated by weight streaming plus vLLM engine warmup/profiling, not by
Python package install.

For future container rebuilds, this is a good default startup path for
measurement iterations when a local `/root/Qwen3.5-27B` copy is absent. It
should not be mixed into R1 commit sign comparisons; R1 throughput A/B still
needs the fixed guard protocol.

## Anchors

- Summary: `raw/summary.json`
- Raw driver log: `raw/driver.log`
- Raw vLLM log: `raw/vllm_server.log`
- Startup script: `raw/start_vllm_runai_locked.sh`
- Remote run dir:
  `/public/home/xdzs2026_c166/codex_runs/runai-startup-probe-locked-20260707-1324`
