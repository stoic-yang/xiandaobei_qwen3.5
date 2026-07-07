# 50 · 模型架构与瓶颈画像

> 本文件是"优化往哪使力"的事实地基。数字均来自 baseline 画像，非最终目标值。
> 来源：`experiments/baseline-full-reuse-20260706-0155/`（competition wheel @06-22 build，
> warm / `--reuse-server`，本地 10-prompt proxy）+ 同批 `codex-full10-*` 的 `vllm_server.log`。
> ⚠️ 本地 proxy ≠ 官方评测集；绝对值仅供内部对照，趋势与结构性结论才是重点。

## 架构：混合 Gated-DeltaNet 线性注意力（非普通 dense transformer）

`Qwen3.5-27B` = **GDN(Gated DeltaNet) 线性注意力层 + 少量标准 full-attention 层的混合体**
（Qwen3-Next 那一路的放大版）。**已由 codex R0.1 确认（见 changelog）**：64 文本层 = **48 GDN 线性层 + 16 full-attention 层**（`full_attention_interval=4`），**非 MoE**，head_dim=256 / num_kv_heads=4 / hidden=5120。

证据链（均出自 vLLM 启动日志）：
- `Resolved architecture: Qwen3_5ForConditionalGeneration`
- `Setting attention block size to 784 tokens to ensure that attention page size is >= mamba page size` + `Padding mamba page size` → 存在 mamba 式循环状态缓存。
- `splitting_ops` 含 `gdn_attention_core` / `linear_attention` / `mamba_mixer2` / `olmo_hybrid_gdn_full_forward`。
- `Capping cudagraph capture sizes from max 256 to 136 to fit Mamba cache blocks (141 blocks available)`。
- 权重里同时有 `linear_attn.out_proj.weight`（线性层）与 `self_attn.o_proj.weight`（标准层）。

**含义（决定优化重心）：**
- 线性注意力层把"回看全部历史"压成**固定大小的循环状态**，不随上下文增长。
- 只有少数 full-attention 层持有随长度线性增长的 KV Cache。

## 关键数字（competition wheel, warm）

| 档 | 权重 | output tok/s | TTFT-P99 | TPOT-P99 | 单请求 TTFT 占比* |
|---|---|---|---|---|---|
| 4-8K | 20% | 12.23 | 4.54 s | 69.2 ms | ~20% |
| 8-16K | 50% | 7.23 | 15.62 s | 70.5 ms | ~56% |
| 16-32K | 30% | 4.66 | 28.69 s | 72.1 ms | ~68% |

\* 单请求耗时 = TTFT + (输出tok−1)×TPOT，反推平均输出 ~170–260 tok/请求。

- 权重占显存 **51.2 GiB**；KV Cache 仅 **6.76 GiB / 27,440 tokens**；64GB 显存**仍有余量**。
- decode 稳定 **14.5 tok/s ≈ 69ms/tok**，**几乎不随上下文变化**（69→72ms，长度却 ×4）→ 线性注意力主导的铁证。
- prefill 峰值 **~758 tok/s**（TTFT 反推 750–1300 tok/s，长档更慢）→ **异常慢**，优化空间巨大。

## 瓶颈结论

1. **主战场 = Prefill / 首字时延(TTFT)。** 占 80% 权重的 8-16K + 16-32K 两档，时间大头（56%/68%）在 prefill，且 prefill 只有 ~758 tok/s。长上下文 prefill 算子（full-attn + GDN chunked-prefill + GEMM）是最大肥肉。
2. **KV Cache 量化不是重点**（推翻纯文档时的判断）。混合架构下 KV 仅 6.76GB、显存有余；`fp8kv_noscale` 实验已验证"不值得押"。仅当 R0.4 profile 证明 full-attn KV 读取确为 decode 瓶颈时才回炉。
3. **decode 瓶颈性质待定**：若非 MoE，14.5 tok/s ≈ 有效带宽 742 GB/s（读满 51.2GB 权重），有中等优化空间；若 MoE 激活参数少，则 decode 转为 launch/python 受限 → CUDA graph 覆盖度是关键。**R0.1（是否MoE）+ R0.4（decode profile）拍板。**
4. **当前提交 `d29e9db3` 官方 59.0018 = 净负优化**（baseline 保底 60）。`fast GQA`/`tile-64`/`GDN chunk` 已知让 8-16K 变慢。⚠️ "前9名反推的 baseline 数字"来自强队非基准，不可当锚点；唯一确定锚点是"自己 59 < 保底 60"。
5. **止损有坑**：baseline wheel 在本容器 model-loader strict-fail 装不起来，competition 分支某改动（`torch.empty+flatten`）才修好加载。回退时必须分清"功能必需"与"性能优化"，不能整体退回 baseline。见 `experiments/20260706-loader-fix/`。

## 硬件 roofline 与合规交互（外部 gfx936 实测 + 规则红线）
> 来源：选手/外部分享的 gfx936 微基准（expA / dbocc.hip）。架构与 codex R0.1 **完全吻合**
> （64层=48GDN+16full、gfx936、非MoE）→ 可信度高；但绝对 TTFT 与本仓不一致（作业 32K=11.4s
> vs 本仓 16-32K p99≈28.7s）→ **定性用它、绝对值用自家实测**。

- **硬件峰值（补 codex R0.2 未拿到的）**：HBM ≈ **1206 GB/s**、bf16 ≈ **395 TFLOPS**、roofline 拐点 ≈ **327 FLOP/byte**。
- **decode = 权重带宽受限，合规下已到物理顶**：实测 GEMV 无双缓冲已达 HBM 峰值 **92–101%**；
  双缓冲/提占用率/调 VGPR **全无效**（占用率已高、硬件自藏延迟）。权重 IO 54GB÷1.2TB/s ≈ **45ms/tok**
  = TPOT 下限。codex R0.4 印证：decode GDN kernel 全 trace 仅 0.67%。→ **decode 的 GEMV/双缓冲别再碰。**
- **🔴 量化红线交互（关键，别抄）**：减 decode 时间的唯一路径 = 权重低精度**驻留 HBM** = **持久化量化 = 红线**。
  白名单"kernel 内临时低精度矩阵乘"权重仍以 bf16 读入、不减 IO，对带宽受限 decode **无效**。
  **规则实际封死了 decode 的量化捷径**；外部作业靠权重量化把 TPOT 49→40，**我方不能抄（违规清零风险）**。
- **prefill = 唯一胜负手**：投影 GEMM O(S) 已贴算力峰值（无肉）；注意力 O(S²) 是长档主力
  （S 翻倍 attn ×3.2–4；8192 时单层 attn 23.7ms = 投影 3.6ms 的 6.5×）。混合架构仅 16 层 full-attn 救了长档。
  codex R0.4 印证：`unified_attention` 38.82% + `chunk_fwd`(GDN prefill) 22.65% = prefill 大头。
  → **R3 靶心 = flash-attention on gfx936；投影 GEMM 交 rocBLAS/hipBLASLt + Matrix Core。**
- decode 合规下唯一剩余空间 = **host 侧**（图捕获消 launch + 算子融合，Amdahl：每 tok 几十个小算子+launch），**不是带宽侧**。

## 外部 DCU 实测讲义的额外锚点（2 份 gfx936 微基准，非本仓测量）
> 来源：用户提供《大模型decode访存瓶颈与双缓冲_DCU实测》(expA.py/dbocc.hip 原文，即本文件已引用的外部 gfx936 作业**原始出处**——HBM 1206/395 TFLOPS/拐点 327 出自此) +《第三集·带宽利用与算子优化》(带宽利用率方法论)。

- **TPOT 三段分解（精确化 decode 可抢空间，重要）**：权重带宽下限 **~45ms** ＜ 讲义实测 bf16 纯前向 **~49ms** ＜ 本仓端到端 P99 **~69ms**。
  → kernel 级已贴物理顶（45→49 仅差 4ms），**剩 ~20ms 几乎全是 host/框架开销**（kernel 间 launch gap + sampling + detokenize + python 调度）。这才是 R2 能抢的量；decode 带宽侧一步不能动。⚠️ 49 与 69 不同源，R2.0 decode-only trace 需证实此拆分。
- **GQA 6:1 复用**：`heads=24 / kv_heads=4` → full-attn kernel 里 K/V 读一次供 6 个 query head 复用，有效 KV 带宽 ÷6。R3.1 flash-attn kernel 必须吃满这个复用。
- **GDN 参考库 = FLA (flash-linear-attention)** `github.com/fla-org/flash-linear-attention`：GDN(Gated DeltaNet) 属 FLA 家族，是 R3.2 GDN chunked-prefill kernel 的直接参考。full-attn 侧参考 FlashAttention（tiling + online-softmax）。
- **inductor combo_kernels 已开**（vllm_server.log 确认 `combo_kernels:True`）：vLLM 已自动融了一批小算子；R2/R3 手动融合前先查它融了什么，别做无用功。
- **🔴 红线重申（别抄药方）**：两份讲义（尤其第一份）最终药方是"权重 int8/int4 量化把 TPOT 49→40ms"——**在本赛题 = 持久化量化红线，违规清零，绝不可抄**。讲义的分析（roofline / 双缓冲实测无效 / prefill O(S²)）全对且可用，但那个药方是给"无规则约束"场景的。

## 待 R0 确认
- ~~是否 MoE / 层配比 / head_dim / KV heads~~ → codex R0.1 已确认（非 MoE，64=48+16）。
- ~~DCU gfx / 是否 FP8~~ → gfx936 确认、FP8 支持确认（`du_mma`）；峰值带宽/算力由外部作业补（1206 GB/s / 395 TFLOPS）。
- throughput 数据集 max_tokens（输出长度）→ 定 prefill/decode 投入比。
- 精度口径 EM vs F1（官方群）→ 定量化激进程度。
- **运行时非持久权重量化是否违规**（每前向临时量化、不落盘、复用显存缓存）→ 决定 decode 有无空间；
  我方判断更可能**不允许**（复用显存量化缓存 ≈ 红线"可复用量化权重缓存"）。这是 decode 唯一可能翻盘点，值得群里问。

## Changelog
- 2026-07-06 create（Claude 读 baseline 实验 + vllm_server.log 得出混合GDN架构与prefill瓶颈判断；推翻"KV量化是主战场"的纯文档预判）。
- 2026-07-06 Claude 沉淀外部 gfx936 实测作业：补硬件峰值（HBM 1206 GB/s / 395 TFLOPS / 拐点 327，与 codex R0.1 gfx936 吻合）；**钉死 decode 合规下到物理顶**（GEMV 已达带宽 92–101%、双缓冲无效；减字节=持久化量化=红线，规则封死 decode 量化捷径，外部作业的权重量化不可抄）；**prefill flash-attention 是唯一战场**；新增待问官方"运行时非持久权重量化是否违规"。
- 2026-07-06 R0.1/R0.2 correction（Codex live probe on SCNet job 655597）：`config.json` confirms 64 text layers = 48 `linear_attention` + 16 `full_attention`, `full_attention_interval=4`, **not MoE** (`num_experts`/`num_experts_per_tok` absent), `head_dim=256`, `num_attention_heads=24`, `num_key_value_heads=4`, `hidden_size=5120`; DCU is `BW`/`gfx936:sramecc+:xnack-`, 80 CU, 64 GiB class VRAM, wavefront 64, HIP 6.3 runtime; DTK exposes `du_mma` FP8 fragments/conversion builtins, but `hy-smi` did not print peak memory bandwidth or peak TFLOPS directly.
- 2026-07-06 R0.4 profile reuse（`experiments/r0-profile-20260706-2138/`, source `/public/home/xdzs2026_c166/codex_logs/profile_runs/rocprofv2_8_16K_20260622_161546`）：8-16K long-context hot window ranks `kernel_unified_attention_2d.kd` 38.82%, `chunk_fwd_kernel_o.kd` 22.65%, `chunk_gated_delta_rule_fwd_kernel_h_blockdim64.kd` 7.66%, top Tensile GEMM rows about 16.82%; decode GDN kernel is only 0.67% in full trace, so score-limiting path remains prefill/full-attn+GDN+GEMM. Caveat: reused trace is not a clean prefill/decode split; decode bandwidth-vs-launch label still needs a decode-only trace if required.
- 2026-07-07 R1 guard anchor（`experiments/guard-a55f3c3-overlay-fullsmoke-20260707-0010/`）：a55/runtime-wheel-equivalent warm guard medians are 12.156717 / 7.231679 / 4.655501 output tok/s, weighted 7.443833, TTFT-P99 4.537s / 15.616s / 28.667s, TPOT-P99 69.731ms / 70.654ms / 72.115ms. Compared to `experiments/guard-d29e9db3-20260706-2005/` installed-wheel guard, deltas are -0.58% / +0.02% / +0.09%; this is consistent with the same runtime path plus noise, not proof of a new speedup.
- 2026-07-07 Claude 沉淀 2 份 gfx936 实测讲义（新增「外部 DCU 实测讲义额外锚点」小节）：TPOT 三段分解 45/49/69ms → **host overhead ~20ms 是 R2 靶子**（decode 带宽侧仍到顶不可动）；GQA 6:1 复用；GDN 参考库 FLA；inductor combo_kernels 已开（手融前先查）；红线重申——讲义"权重量化降 TPOT"药方在本赛题违规不可抄。
