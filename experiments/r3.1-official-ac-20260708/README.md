# r3.1-official-ac-20260708

## Intent

Record the official platform result for the R3.1 flash-attention prefill candidate after GitLab/platform evaluation recovered. This is an official-score anchor, not a local proxy benchmark.

## Source

User-reported official platform row on 2026-07-08:

- status: `AC`
- final score: `74.6924`
- 4K-8K actual throughput: `13.78`
- 8K-16K actual throughput: `12.89`
- 16K-32K actual throughput: `11.18`
- SLA penalty: `0.0`
- accuracy penalty: `0.5644`

The submitted code corresponds to the R3.1 flash-attention prefill candidate recorded locally at source commit `847d1bef10b0b5bb71b7e427535b610a20a4d263`.

## Derived Notes

- Weighted official throughput proxy: `12.555` tok/s using weights `0.2/0.5/0.3`.
- Score before the reported accuracy penalty, assuming no other hidden penalty: `75.2568`.
- Compared with the local R3.1 candidate guard:
  - local 4K-8K `13.416425` vs official `13.78`
  - local 8K-16K `11.455965` vs official `12.89`
  - local 16K-32K `8.454834` vs official `11.18`
- Local proxy was conservative for throughput, especially at 16K-32K.
- Accuracy did not collapse; the official accuracy penalty is small enough that R3.1 is confirmed as the current safe positive baseline.

## Verdict

R3.1 is officially positive and accepted. Keep `847d1bef` / R3.1 as the current performance baseline for subsequent local A/B and official-submission planning.

The next optimization line remains R3.0/R2.4 GEMM autotune, because post-R3.1 profiling shows `Cijk_*` GEMM is now the dominant prefill hotspot.
