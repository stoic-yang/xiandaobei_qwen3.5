# 任务卡 · R3.1 Prefill Flash-Attention on gfx936（head_dim=256）

> **主攻卡（唯一大胜负手）**。上位：[`r2-r5-detail.md`](r2-r5-detail.md) §3.1、[`../memory/50-arch-bottleneck.md`](../memory/50-arch-bottleneck.md)。
> R2 已证明 decode 到顶（graph FULL、带宽满、量化=红线），prefill 是唯一几十分级的盘子。

## 一句话目标
把 16 层 full-attention 的 **prefill attention**（profile 里 `kernel_unified_attention_2d` 占 8-16K 热窗口 **38.82%**，最大单块）在 gfx936 上做成高效 flash-attention，拉高 prefill（现 ~758 tok/s，异常慢），直接压低 8-16K/16-32K 两档（**80% 权重**）的 TTFT-P99。

## 为什么押它（方向已定，别再论证）
- profile 靶心：`unified_attention` 38.82% 是最大单块（GDN chunk ~30% 次之，留 R3.2）。
- prefill 主导 80% 权重两档的时间（TTFT 占 56%/68%）。
- decode 侧已到物理顶，无大肉（见 R2.1 结论）。

## 硬约束与难点（先内化，别踩）
- 🔴 **head_dim=256**：多数 flash-attn（FA2、triton FA）主打 ≤128。256 让 Q/K/V tile 与 online-softmax 累加器翻倍占 LDS/寄存器 → occupancy 压力大，可能要 split-D。**这是最大技术风险，也是护城河。第一件事就是确认候选实现能在 head_dim=256 正确跑。**
- **GQA 6:1**：heads=24 / kv_heads=4。kernel 内 K/V 从 HBM 读一次供 6 个 query head 复用（有效 KV 带宽 ÷6）。别退化成逐头 MHA 读。
- **chunked prefill**：`--max-num-batched-tokens=4096` 锁定 → 长 prompt 按 4096-token chunk 喂。kernel 要在"当前 chunk 的 Q × 至今全部 K/V"结构下工作且 causal；确认正确处理 chunk 边界 + causal mask。
- 只有 16 层 full-attn（`full_attention_interval=4`），但 O(S²) 是长档主力。
- bf16 权重不可动（红线）；attention 内部低精度（FP8 QK/PV）属白名单但**留到 R4**——R3.1 只做 bf16 正确版。

## 执行步骤

**Step 1 — 现状 η 体检（先确认真有肉）**
- 定位 vLLM 当前 full-attn 走的 attention backend（`kernel_unified_attention_2d`，triton？）。
- 在真实 shape（head_dim256 / 4 kv / chunk4096 / bf16 / causal）下测它的**达成算力利用率 + HBM 带宽利用率 η**（omniperf/rocprof 对 roofline）。看它是否把 S×S 注意力矩阵物化到 HBM。
- 判定：η 低（算力 <0.5 或明显物化中间矩阵）→ 有肉，继续；η 已高 → 收益有限，回报设计层重估。**别在没测 η 前就动手写 kernel。**

**Step 2 — 选实现（优先复用，别从零手写）**
- 调研 DTK 26.04 是否自带、且**支持 head_dim=256** 的：CK(Composable Kernel) flash-attn、ROCm/AITER flash-attn、triton flash-attn。
- 候选按"能否 head_dim=256 正确跑 + 达成 η"排序。**能接入成熟实现就别手写**——手写 gfx936 flash-attn 成本极高、周期不够。

**Step 3 — 正确性优先（先正确后快，硬门槛）**
- tiny-shape 数值等价单测：候选 kernel vs 原 `unified_attention`，误差 <bf16 容差阈值。
- 必测组合：head_dim=256、GQA(4 kv / 24 q)、causal、chunk 边界（跨 chunk 的 K/V 拼接）。**任一不过，不进 Step 4。**

**Step 4 — 接入 + 单档 A/B（开关门控）**
- config/env 开关切换新旧 attention backend（关掉=原路径、数值等价）。
- 8-16K 单档守门员 A/B（locked 口径），看 TTFT-P99 + 吞吐 Δ。

**Step 5 — 精度 + 全档回归**
- 四类精度 Δ<1%；三档守门员全过、无一倒退（尤其 4-8K 别因 attention 改动倒退）。

## 纪律（硬）
- **config/env 开关门控**，关掉即回原 backend、数值等价。
- **先正确后快**：Step 3 不过禁止谈提速。
- 同容器比相对 Δ；locked 口径（`max_seq_len=32768`）。
- 不改 batch scheduler / 锁定 CLI；不改模型结构/权重；不投机解码。

## 判定 / 验收门槛
- Step 1 η 证明有肉；Step 3 数值等价过；Step 4 8-16K TTFT-P99 改善且吞吐不倒退；Step 5 四类精度 Δ<1% 且全档不倒退。
- 目标：prefill tok/s 明显上抬，报出三档 TTFT-P99 改善幅度。

## 产出 / 交回
- `experiments/<id>/`：现状 η + 候选实现对比 + 数值等价证明 + 单档/全档守门员 + verdict。
- 交回一句话：`unified_attention` 现状 η（有没有肉）；head_dim=256 下哪个实现能正确+更快；8-16K TTFT-P99 改善多少。

## 参考
- FlashAttention `github.com/Dao-AILab/flash-attention`；ROCm/CK flash-attn；AITER；Triton（HIP 后端）。
- **head_dim=256 支持是筛选实现的第一硬门槛。**
- 若全部候选都不支持 head_dim=256 → 回报设计层：考虑 split-head_dim 方案，或改先做 R3.2 GDN（~30%，无 head_dim=256 限制）。

## Changelog
- 2026-07-07 create（Claude；R3 主攻卡，η 体检先行 + head_dim=256 正确性硬门槛 + 先正确后快）。
