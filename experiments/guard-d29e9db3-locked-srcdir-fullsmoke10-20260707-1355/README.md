# guard-d29e9db3-locked-srcdir-fullsmoke10-20260707-1355

## Verdict

**PARTIAL / intentionally stopped after user reprioritized.** This was a valid
locked-start d29 baseline attempt using `/root/overlay-d29e9db3-locked`,
`--max-model-len 32768`, and `--load-format runai_streamer`. The service
started correctly and 4-8K completed all three repetitions, but the run was
stopped during 8-16K rep1 after the user asked to pause remaining R1 work and
move to higher-ROI tasks.

Do not use this as an R1 sign-table row because 8-16K, 16-32K, and smoke
accuracy did not complete.

## Startup Evidence

- vLLM log confirms `max_model_len: 32768`.
- Engine config confirms `max_seq_len=32768`.
- vLLM log confirms `load_format=runai_streamer`.
- `server_ready_at=2026-07-07T14:05:52+08:00`.
- RunAI streamed `51.7 GiB` in `303.38s` at `174.7 MiB/s`.
- Model loading took `304.596451s`.
- `torch.compile` took `35.78s`.
- Initial profiling/warmup took `148.96s`.
- Engine init after load took `239.24s`.

## Partial 4-8K Metrics

| rep | output tok/s | completed | failed |
| --- | ---: | ---: | ---: |
| 1 | 12.032690 | 10 | 0 |
| 2 | 12.258413 | 10 | 0 |
| 3 | 12.256351 | 10 | 0 |

4-8K median output throughput: `12.256351`.

The initial single-prompt warmup before rep1 had a large first-request cost:
two warmup requests took `6m51s`, with the first at about `400.90s`. Subsequent
per-rep warmups were back around 10s/request.

## Anchors

- Runtime fingerprints: `runtime_fingerprints.json`
- Overlay manifest: `overlay_manifest.txt`
- Raw logs: `raw/`
- Partial throughput JSONs: `throughput/`
- Remote run dir:
  `/public/home/xdzs2026_c166/codex_runs/guard-d29e9db3-locked-srcdir-fullsmoke10-20260707-1355`
