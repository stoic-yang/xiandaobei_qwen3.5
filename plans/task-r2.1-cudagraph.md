# 任务卡 · R2.1 CUDA graph 覆盖度（decode host/launch gap）

> 单页可执行任务卡，供 Codex/GLM 直接照做。上位文档：[`r2-r5-detail.md`](r2-r5-detail.md) §2.1、[`../memory/50-arch-bottleneck.md`](../memory/50-arch-bottleneck.md)。
> 定位：**不写 kernel、不碰带宽、不碰量化**。只查/调执行路径，抢 decode 的 host/launch 空隙。

## 一句话目标
查清 **bs=1 decode 是否全程走 full CUDA graph**、消除层间 kernel launch，把端到端 TPOT 从 ~69ms 往纯前向 ~49ms 方向压（可抢空间 ~20ms/token，见 memory/50 TPOT 三段分解）。

## 背景事实（已知，别重新发现）
- `cudagraph_mode=FULL_AND_PIECEWISE`；`splitting_ops` 把 `unified_attention`/`linear_attention`/`mamba_mixer2`/`gdn_attention_core` 等列为**图分割点** → 每层 attention/GDN 都可能打断 graph。
- 混合架构 64 层交替 GDN/full-attn；若每层被切断，bs=1 decode 每 token = 大量独立 launch。
- `Capping cudagraph capture sizes ... to fit Mamba cache blocks (141 blocks)` 把 max capture size 压到 136；**但 bs=1 只需 capture size=1，理论不受此限** —— 需实证确认 size=1 是否稳定命中。
- 并发固定=1，所以只关心 batch-size=1 的 decode 图。

## 测量口径（硬性）
- `scripts/guard_bench.py --locked-start-script --load-format runai_streamer`，并在 `vllm_server.log` 验证 `max_seq_len=32768`（否则作废）。
- 同容器比相对 Δ，不跨容器比绝对值。先 unset proxy + 导 DTK env（见 memory/20 + `automation/config.json` dcu_env）。

## 执行步骤

**Step 1 — decode-only launch 计数（= R2.0，本卡前置）**
- rocprofv2/hipprof 抓一段**稳态 decode**（长 prompt、纯生成若干十 token），统计：每 token 的 kernel launch 次数、graph replay 次数、kernel 忙时 vs 空隙占比。
- 判定：每 token 若只有 ≈1 次 graph replay → 已 full graph；若几十次独立 launch → 有空间。

**Step 2 — enforce_eager 对照（关键诊断，最先跑）**
- 起一个 `--enforce-eager`（关全部 graph）的服务，跑守门员，与现状对照。这一步直接量化"graph 现在到底帮了多少"：
  - **决策树**：
    - 关 graph 后 TPOT 明显变慢（如 69→85ms）→ graph 已生效、省了 ~16ms，**剩余空间小**，R2.1 转为微调；
    - 关 graph 后 TPOT 几乎不变（还 ~69ms）→ **graph 在 decode 没真正生效，那 ~20ms 几乎全可抢 = 大机会**，重点做 Step 4。
- 纯诊断，不进默认。

**Step 3 — 读源码确认 graph 语义 [需现场核实 0.18.1/das-fork]**
- 定位 vLLM `compilation` / cudagraph 逻辑，确认 `FULL_AND_PIECEWISE` 对 attention/mamba 层的实际处理、bs=1 decode 走哪条路径、`splitting_ops` 如何把一步切成多段。
- 产出：decode 一步实际被切成几个 graph 段 + 每段边界。

**Step 4 — 尝试提升覆盖度（每个开关单独 A/B，config/env 门控）**
- 候选（可行性由 Step 3 判定，先验证数值等价再谈提速）：
  - 更激进的 `cudagraph_mode`（核实是否支持 `FULL_DECODE_ONLY` 或等价，让 decode 走满 graph）；
  - 在合规范围内减少 `splitting_ops` 断点，让 GDN/attention 进同一 graph；
  - 确认 capture size=1 命中且未被 Mamba cap 挤掉。

## 开关矩阵（每个跑守门员，记录三档吞吐 + 四类精度 + 是否数值等价）
| 开关 | 用途 |
|---|---|
| A 现状 FULL_AND_PIECEWISE | baseline |
| B `--enforce-eager` | 诊断 graph 贡献（不进默认） |
| C 提升覆盖度候选 | 目标改动 |

## 判定与验收门槛
- **命中 full graph**：decode 稳态每 token graph replay ≈1、独立 launch 骤降、TPOT 向 ~49ms 靠。
- **进默认的门槛**：8-16K bucket TPOT-P99 改善 **且**三档吞吐无一倒退；四类精度 Δ<1%；**关掉开关 = 与原路径数值等价**。不达标 → 留档负结果，不进默认。

## 红线自查（做前逐条过）
- 不改 `--max-num-seqs` / `--max-num-batched-tokens` / vLLM batch scheduler 代码（锁定）。
- cudagraph_mode / compilation config 属执行路径（绿区）；**改前核对官方评测脚本无锁定冲突**。
- `temperature=0`、输出口径、chat template 不变。

## 产出物
- `experiments/<id>/`：intent / 开关矩阵 / 守门员 `summary.json` / decode launch 计数 / verdict。
- 若拿到正向：回填校正 `memory/50` 的 TPOT 三段分解（49 vs 69 的 host 拆分实测值）。

## 交回一句话
当前 bs=1 decode 是否已 full graph；enforce_eager 对照量化的 graph 贡献（ms）；覆盖度候选有没有正向 Δ（及数值等价性）。

## Changelog
- 2026-07-07 create（Claude；R2.1 首张任务卡，enforce_eager 对照为核心诊断）。
