# 任务卡 · R2.0 decode-only profile（R2 收尾诊断）

> 小卡，收尾 R2。上位：[`r2-r5-detail.md`](r2-r5-detail.md) §2.0、[`../memory/50-arch-bottleneck.md`](../memory/50-arch-bottleneck.md)。
> R2.1 已证明 graph 到位，此卡给剩余 gap 定性，决定 R2 是否收工、R2.3 要不要做。不占 GPU 主线。

## 目标
拆清 baseline 8-16K decode 端到端 TPOT-P99 **≈69.9ms** 的构成（对照：权重带宽下限 ≈45ms、外部讲义纯前向 ≈49ms），回答两个问题：① 剩余 ~20ms 是可抢的 host 开销还是不可消的固有开销；② decode 里 full-attn attention kernel 占多少（R3 能否顺带帮 decode）。

## 背景（已知，别重测）
- R2.1 `enforce_eager` A/B 已证明：graph 是 **FULL、已省 48ms**（关掉 69.9→118ms）。**graph 覆盖度没肉，不要再动 graph。** 见 `experiments/r2-eager-enforce-ab-20260707/`。
- 剩余 ~20ms（69.9 vs 讲义 49）是 graph 生效**之后**仍剩的，构成未知——这正是本卡要拆的。

## 步骤
1. locked 口径（`--locked-start-script --load-format runai_streamer`，验证 `max_seq_len=32768`）起服务，抓一段**稳态 decode** trace（长 prompt、纯生成若干十 token，避开 prefill 窗口），用 rocprofv2/omniperf。
2. 每 token 时间拆成四份：**kernel 忙时总和 / graph replay 开销 / 段间空隙 / host（采样+detokenize+python）**，并数每 token 的 launch/replay 次数。
3. 列 decode kernel top-N，**特别标出 full-attn(GQA) 的 decode attention kernel 占 decode 的百分比**。

## 产出（决策树，直接给结论）
- 剩余主要是 **host 未与 GPU 重叠**（采样/detokenize 同步阻塞）→ R2.3 有小肉，可一试（预期分值小、非大头）。
- 剩余主要是 **段间同步 / graph replay 固有开销** → decode 到顶，**R2 收工**，全部精力转 R3。
- **kernel 忙时本就接近 69.9ms**（说明讲义 49ms 与我们不可比）→ decode 到顶；但若 **full-attn decode attention kernel 占比高** → 记入 R3：flash-attention 可能**一箭双雕**（顺带压低 decode）。

## 验收 / 交回
- decode 每 token 时间构成（忙时 / 段间 / host 三分）+ launch 次数。
- 一句话：R2.3 值不值得做；full-attn decode attention kernel 占 decode 多少（R3 能否顺带帮 decode）。
- **纯诊断，不进默认**；结果回填校正 `memory/50` 的 TPOT 三段分解。

## Changelog
- 2026-07-07 create（Claude；R2 收尾诊断，紧接 R2.1 enforce_eager 结论）。
