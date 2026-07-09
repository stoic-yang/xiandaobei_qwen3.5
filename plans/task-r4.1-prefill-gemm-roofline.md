# 任务卡 · R4.1 Prefill GEMM roofline 与 work-elimination 探针

> 上位：[`roadmap.md`](roadmap.md) Round 4、[`task-r3.0-gemm-autotune.md`](task-r3.0-gemm-autotune.md)、[`../memory/50-arch-bottleneck.md`](../memory/50-arch-bottleneck.md)。
> 本卡接在 R3.1 官方 AC 与 R3.0 Inductor/TunableOp stop-loss 之后。目的不是直接写 GEMM kernel，而是用 gfx936 实测 roofline 方法判断 prefill GEMM 还有没有可兑现空间，再决定是否进入低精度 compute 或 work-elimination 候选。

## 一句话目标

把 R3.1 后 8-16K/16-32K prefill 的 `Cijk_*` GEMM 拆成：

1. 已达成多少 TFLOPS，占 gfx936 外部实测 `~395 TFLOPS` 峰值多少；
2. 是否存在无用计算或低价值重复计算可以删掉；
3. INT8/FP8 只作为 prefill compute-side microbench 候选，确认硬件吞吐与精度风险后再决定是否实现。

## 现状锚点

- R3.1 官方结果：`experiments/r3.1-official-ac-20260708/`，AC，score `74.6924`，三档吞吐 `13.78 / 12.89 / 11.18`，SLA 扣分 `0.0`。
- R3.2 post-R3.1 prefill profile：`experiments/r3.2-post-r31-prefill-profile-20260707-2210/`
  - request `13964` prompt tokens，`max_tokens=1`
  - wall `3299.468ms`，kernel busy `3163.713ms`
  - `Cijk_*` GEMM `2122.873ms`，占 kernel `67.101%`
  - flash-attn prefill `8.994%`
  - GDN core `9.303%`，含 helper `13.277%`
- R3.0 shape attribution：`experiments/r3.0-gemm-shape-attribution-20260708-112017-wheel/`
  - 4 个 prefill chunks
  - language projection calls `1216` + logits calls `4` = observed Cijk `1220`
  - hot families：MLP gate/up、MLP down、GDN in/out projection、full-attn qkv/o projection
  - `lm_head/logits` only `0.24%` kernel share in the committed profile，不能预设成大肉
- R3.0 TunableOp/Inductor：
  - TunableOp 没覆盖目标 language GEMM
  - Inductor same-container A/B `-0.055%` output，TTFT `+0.060%`，TPOT `+0.389%`
  - 结论：通用 autotune 停止，不进默认

## 零成本 roofline 预判

基于 R3.2 8-16K profile 的 `13964` prompt tokens 和 R3.0 shape set，排除 tiny `in_proj_ba` 与低占比 logits，仅语言 dense projection 的理论工作量约：

| family | layers | weight shape | approx TFLOP |
|---|---:|---:|---:|
| `linear_attn.in_proj_qkvz` | 48 | `[16384,5120]` | `112.45` |
| `linear_attn.out_proj` | 48 | `[5120,6144]` | `42.17` |
| `mlp.gate_up_proj` | 64 | `[34816,5120]` | `318.62` |
| `mlp.down_proj` | 64 | `[5120,17408]` | `159.31` |
| `self_attn.qkv_proj` | 16 | `[14336,5120]` | `32.80` |
| `self_attn.o_proj` | 16 | `[5120,6144]` | `14.06` |
| total | - | - | `679.40` |

用 `Cijk_*` total `2122.873ms` 计，aggregate achieved ≈ `320.0 TFLOPS`，约为外部 gfx936 实测 `395 TFLOPS` 的 `81%`。这说明：

- GEMM 不是 2-3x 的大肥肉；
- 仍可能有 `10-20%` kernel-local gap，但 R3.0 通用 autotune 已经没有正向；
- 在没有更强证据前，不写 GEMM kernel，先查 16-32K 是否同样约 `320 TFLOPS`，再做低精度 compute microbench。

## 硬约束

- 不改 decode GEMV、双缓冲、占用率；外部 DCU 实测和本仓 R2.0 已判定 decode 带宽侧到顶。
- 不做持久化权重量化、剪枝、跳层、投机解码，不改模型结构/权重。
- 不改 batch scheduler；不动锁定 CLI：`--max-model-len=32768`、`--max-num-seqs=128`、`--max-num-batched-tokens=4096`、temperature/tokenizer/chat template/API。
- 任何实现候选必须 env-gated；关闭后数值等价回到原路径。
- 任何 remote guard 用 `scripts/guard_bench.py --locked-start-script --load-format runai_streamer`，并验证 `max_seq_len=32768`。
- 不清理未知 `8001` 服务；新实验默认用隔离端口，例如 `18001`。

## 执行步骤

### Step 0 — 固化现有 profile 的 roofline 表

用已有 `r3.2-post-r31-prefill-profile` 和 `r3.0-gemm-shape-attribution` 生成 `experiments/r4.1-prefill-gemm-roofline-<date>/summary.json`：

- total projection FLOPs
- `Cijk_*` total ms
- achieved TFLOPS
- peak fraction vs `395 TFLOPS`
- 结论：green/yellow/red

判定：

- `>=360 TFLOPS`：接近算力墙，GEMM kernel 路线关闭，转 R5 或只做 work-elimination。
- `300-360 TFLOPS`：黄区，只允许 microbench/规则确认，不直接写 GEMM kernel。
- `<300 TFLOPS`：有库选择或 shape 路径问题，优先查 hipBLASLt/rocBLAS dispatch 与 target-shape microbench。

### Step 1 — 16-32K prefill roofline 复核

用 R3.1 当前默认候选，在可用容器上抓一个 16-32K prefill-only profile：

- locked start，`max_tokens=1`
- 优先复用 `r3.2-post-r31-prefill-profile` 的 hipprof session-control 脚本结构
- 只抓一个 request，不跑 throughput guard
- 输出 `Cijk_*` total ms、prompt tokens、chunks、flash-attn/GDN/GEMM share

目的：确认 8-16K 的 `~320 TFLOPS` 不是单请求偶然值，也确认 16-32K 是否出现新的非 GEMM 大头。

### Step 2 — work-elimination 快筛

只找“删掉不会改输出”的工作，不做近似：

- `lm_head/logits`：若新 profile 仍 `<1%`，直接关闭该方向；若异常升高，再查是否非最后 chunk 计算了无用 logits。
- fill/index/copy/reshape 类小 kernel：若单类 `>3%`，查是否来自重复 mask/page/index 构造；只允许 cache 或复用不会跨请求污染的中间量。
- GDN helper：若 16-32K helper 占比升高到 `>15%`，回到 `task-r3.2-gdn-prefill.md` 的 cheap config spike；否则不做。

### Step 3 — INT8/FP8 prefill compute microbench

仅在 Step 0/1 仍显示 GEMM 是主热点且未贴近 `360 TFLOPS` 时做。先不接 vLLM，只测 target shapes：

- bf16 baseline：`M in {1024,2048,4096}`，target weight shapes 覆盖 MLP gate/up、down、GDN in/out
- INT8/FP8 candidate：现场核实 gfx936/DTK/PyTorch/hipBLASLt 是否真的有快速路径
- 记录转换成本：权重仍 bf16 驻留 HBM，不允许持久化低精度权重缓存
- 输出：effective TFLOPS、误差分布、是否存在 `>1.3x` 净收益

判定：

- 无快速硬件路径或净收益 `<1.3x`：关闭低精度 compute。
- 有 `>1.3x`，但误差触发输出等价/accuracy smoke 风险：关闭或降级。
- 有 `>1.3x` 且误差可控：再开单独实现任务卡，不在本卡直接改默认。

### Step 4 — 守门员

本卡默认只产出诊断，不进入默认。若后续派生实现候选，验收顺序必须是：

1. target-shape microbench 正向；
2. 单 projection family env-gated 实现；
3. eager/原路径 output-equivalence prompt set；
4. 8-16K same-container guard；
5. 三档 guard；
6. accuracy smoke，round close 前 full accuracy。

## 出口标准

交回一张决策表：

| item | result | verdict |
|---|---|---|
| 8-16K GEMM achieved TFLOPS | value | red/yellow/green |
| 16-32K GEMM achieved TFLOPS | value | red/yellow/green |
| logits/work-elimination share | value | keep/kill |
| INT8/FP8 microbench | value | keep/kill |
| next score-growth task | path | reason |

## Changelog

- 2026-07-09 create（Codex；用户提供两篇 gfx936/DCU 带宽讲义后，将方法论落到 R3.1 后瓶颈：decode 带宽侧继续关闭，prefill GEMM 先做 roofline 与 work-elimination 诊断，再决定是否进入低精度 compute）。
- 2026-07-09 Step 1 result（Codex；`experiments/r4.1-prefill-16to32-profile-20260709-1120/`）：16-32K `Cijk_*` aggregate `317.396 TFLOPS` / `80.353%` peak，仍是黄区；但 flash-attn `_fwd_kernel` 占 `40.582%`，与 GEMM `44.715%` 同量级。直接 GEMM/INT8 实现降级，下一入口转 `task-r3.1b-long-context-flash-attn.md`。
