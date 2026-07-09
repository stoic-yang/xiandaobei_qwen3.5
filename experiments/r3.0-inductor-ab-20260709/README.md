# R3.0 Inductor GEMM autotune A/B 20260709

## Intent

Complete the pending same-container A/B for R3.0 Step2 after a fresh user-started 27B container became available.

## Container

- SCNet instance: `Instances_2607090940238205_0_0`
- Slurm job: `661607`
- Node: `e03r1n10`
- Worker alias: `xiandaobei-worker-auto`
- Isolated benchmark port: `18001`

Port `8001` was not used or stopped.

## Runs

| role | run id | status |
|---|---|---|
| baseline | `r3.0-baseline-8to16-port18001-20260709-0952` | pass |
| Inductor candidate | `r3.0-inductor-8to16-port18001-20260709-1024` | pass |

Both runs used `guard_bench.py --locked-start-script --load-format runai_streamer --buckets 8-16K --num-prompts 3 --repetitions 3 --accuracy none --server-port 18001`.

## A/B Result

| metric | baseline | Inductor | delta | delta % |
|---|---:|---:|---:|---:|
| output throughput | `8.096026242866404` | `8.091542528776012` | `-0.00448371409039261` | `-0.055381664484388526%` |
| TTFT-P99 ms | `12779.450334100984` | `12787.138073854148` | `+7.687739753164351` | `+0.060157045508058005%` |
| TPOT-P99 ms | `69.74396645218654` | `70.01528968467291` | `+0.2713232324863668` | `+0.3890275335463933%` |
| duration s | `78.31002284097485` | `78.35341626708396` | `+0.04339342610910535` | `+0.055412352767691964%` |

## Verdict

Inductor GEMM autotune is a stop-loss for this target. The config switch applies, but stable 8-16K throughput is flat to slightly negative and TPOT-P99 moves the wrong way. Do not expand to three buckets or smoke accuracy; do not put this switch into defaults.

The next score-growth work should move away from generic Inductor GEMM autotune. Remaining options are a more targeted library/backend investigation with per-shape proof, or another task card outside R3.0 Inductor.
