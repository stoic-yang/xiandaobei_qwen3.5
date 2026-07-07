# runai-startup-probe-fresh-20260707-1259

## Verdict

**INVALID for competition timing, but useful as a diagnostic.** This run used
the user-provided `runai_streamer` startup shape without the locked
`--max-model-len 32768` argument. vLLM therefore resolved `max_seq_len=262144`
and spent much more time in compile/profiling than a competition-compliant
startup should.

Do not compare this run against R1 throughput or startup criteria. Use it only
as evidence for the mechanism and for why the script must keep locked CLI
parameters.

## Observed Facts

- SCNet job: `656918`, IP `173.0.253.4`.
- Started: `2026-07-07T13:00:00+08:00`.
- Model path: `/public/home/xdzs2026_c166/Qwen3.5-27B`.
- Load format: `runai_streamer`.
- Missing locked arg: `--max-model-len 32768`.
- vLLM log resolved `Using max model len 262144` / `max_seq_len=262144`.
- RunAI streamer loaded `51.7 GiB` in `434.77s` at `121.9 MiB/s`.
- Model loading reported `436.038423s`.
- `torch.compile` reported `429.97s`.
- `/health` was still not ready at `2026-07-07T13:21:33+08:00`.
- The process was killed at `2026-07-07T13:22:41+08:00` to free the DCU.

## Interpretation

`runai_streamer` is not merely "avoid copying the model"; it streams safetensors
through Run:ai Model Streamer. In this run it avoided the explicit `/root`
model copy, but the missing max-model-len made the startup path non-comparable.

## Anchors

- Raw driver log: `raw/driver.log`
- Raw vLLM log: `raw/vllm_server.log`
- Startup script: `raw/start_vllm_runai.sh`
- Remote run dir:
  `/public/home/xdzs2026_c166/codex_runs/runai-startup-probe-fresh-20260707-1259`
