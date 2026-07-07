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

## 执行步骤（直接干，不设 η 前置闸门）

**Step 1 — 直接选实现并接入**
- 直接调研并接入**支持 head_dim=256** 的成熟 flash-attn：CK(Composable Kernel) / ROCm / AITER。能接就接，接不上再考虑手写/移植。head_dim=256 支持是第一硬门槛。
- 利用率 η **并行测**（用来解释收益来源、写消融表），**不阻塞动手**——先干起来。

**Step 2 — tiny-tensor 数值对拍（5 分钟，保命，唯一不可省）**
- 接完先在小张量上把新 attention 输出与原 `unified_attention` 对一遍：head_dim=256 / GQA(4 kv / 24 q) / causal / chunk 边界，误差 < bf16 容差。
- ⚠️ **为什么不能省**：attention 算错**照样能跑、还会显示吞吐变快**（算少了/算错了），但精度会崩 → 在"吞吐+40%"的假象下精度四类清零、总分暴跌。5 分钟，防自己骗自己。
- **对拍不过的吞吐数字一律不采信。**

**Step 3 — 单档 A/B + 精度回归**
- config/env 开关门控（关掉=原 backend、数值等价）；8-16K 守门员 A/B（locked 口径）看 TTFT-P99 + 吞吐 Δ；四类精度 Δ<1%；三档无一倒退（尤其 4-8K 别因 attention 改动倒退）。

## 纪律（硬）
- **直接干，但接完必对拍**（Step 2）—— 对拍不过的吞吐一律不采信。**这是唯一不可省的验证。**
- config/env 开关门控，关掉=原 backend、数值等价。
- 同容器比相对 Δ；locked 口径（`max_seq_len=32768`）。
- 不改 batch scheduler / 锁定 CLI；不改模型结构/权重；不投机解码。
- 先接现成实现，别一上来手写 gfx936 kernel（周期不够）。

## 判定 / 验收门槛
- Step 2 数值对拍过（否则吞吐不采信）；Step 3 8-16K TTFT-P99 改善、四类精度 Δ<1%、三档无一倒退。
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
- 2026-07-07 revise（用户定"直接做"）：去 η 前置闸门（改并行测）；5 步压成 3 步（直接接入 → tiny-tensor 数值对拍保命 → 单档A/B+精度）；数值对拍是**唯一不可省**验证（防 attention 算错→假性变快→精度清零）。
