# R2–R5 细化方案（基于 R0/R1 硬结论重新校准）

> 前置阅读：[`roadmap.md`](roadmap.md)（总纲）、[`../memory/50-arch-bottleneck.md`](../memory/50-arch-bottleneck.md)（架构/瓶颈/roofline）。
> 本文件把 R2–R5 从"纯文档预设"更新为"R0/R1 实证驱动"。R1 已按用户决定暂停（各 commit 在 ±0.2% 噪声内，无大负优化可回退）。
> 测量口径统一：`scripts/guard_bench.py --locked-start-script --load-format runai_streamer`，且必须在 `vllm_server.log` 验证 `max_seq_len=32768`；同容器比相对 Δ，不跨容器比绝对值。

## 0. R0/R1 定死的边界（不要再挑战）
- 架构：64 层 = 48 GDN 线性 + 16 full-attn（`full_attention_interval=4`），**非 MoE**，dense；head_dim=**256**、heads=24、kv_heads=4(GQA)、hidden=5120；gfx936 / 80 CU / FP8(`du_mma`) 可用。
- roofline：HBM **1206 GB/s**、bf16 **395 TFLOPS**、拐点 327 FLOP/byte。
- **decode 带宽侧到物理顶**：GEMV 已达 HBM 峰值 92–101%；权重 IO ≈ 45 ms/tok 是 TPOT 硬下限。减字节=权重量化=持久化红线，规则封死。**别碰 decode 的 GEMV/双缓冲/占用率。**
- **瓶颈地图（8-16K 热窗口）**：`unified_attention` 38.82%（full-attn prefill）＞ GDN `chunk_fwd` 22.65% + `chunk_gated_delta` 7.66%（≈30%，GDN prefill）＞ Tensile GEMM ~16.82%；decode GDN 仅 0.67%。
- 结论一句话：**prefill flash-attention 是唯一大胜负手（R3）；decode 只剩 host 侧的 launch/gap 可抢（R2）。**

## 1. 战略校准：R0/R1 如何改变各轮价值

| 轮 | 原定位 | 校准后定位 | 价值变化 |
|---|---|---|---|
| R2 | 低风险普惠"顺手做" | **decode 侧唯一战场 = 消 host/launch gap**；+ 为 R3 铺路 | ⬆️ 升级 |
| R3 | 主攻 | **唯一大胜负手**，靶心已被 profile 锁死 | ⬆️ 最高 |
| R4 | 独立一轮量化 | **基本被规则+架构封死**，降为 R3 的可选精度增强 | ⬇️ 降级 |
| R5 | 合规收尾 | 不变；补 SLA 完成率专项 | = |

---

## 2. Round 2 细化 —— 抢 decode 的 host/launch gap（不写 kernel）

### 核心机会（本轮的数字锚点，已被外部讲义精确化）
decode TPOT 三段：权重带宽下限 **~45ms** ＜ 外部讲义实测 bf16 纯前向 **~49ms** ＜ 本仓端到端 P99 **~69ms**。
→ kernel 级已贴物理顶（45→49 只差 4ms），**剩 ~20ms（占 TPOT ~30%）几乎全是 host/框架开销**：kernel 间 launch gap + sampling + detokenize + python 调度。这就是 R2 能抢的（decode 带宽侧一步不能动）。
抢下其中一半 → TPOT ~69→59ms → decode ~15% 提速（4-8K 档 decode 主导，近似同幅吞吐；两长档也有份）。

> ⚠️ 49ms（外部讲义微基准）与 69ms（本仓端到端 P99）不同源，那 ~20ms 差**未必全是 host**，也可能含口径差异。**R2.0 decode-only trace 必须证实这个拆分**（kernel 忙时 vs launch/host 空隙）再动手。

### R2.0（先决）decode-only profile
- 用 rocprofv2/hipprof 抓一段**纯 decode**（长 prompt、只生成、稳态若干 token）的时间线，量化：kernel 忙时 vs 空隙、每 token 的 launch 次数、host python 占比。
- 产出：把 ~20ms 差拆成 launch-bound / host-bound / 口径差 三部分。→ 决定 R2.1 与 R2.3 谁先做，并回填校正 memory/50 的三段分解。

### R2.1 CUDA graph 覆盖度 — ✅ 已诊断关闭（2026-07-07）
> `enforce_eager` A/B 已证明 graph 是 **FULL、已省 48ms**（关掉后 8-16K TPOT-P99 69.9→118ms、吞吐 −27%；日志实锤 `decode, FULL` 已捕获）。**结论：graph 覆盖度无肉，本项关闭，不再重做 graph。** 见 `experiments/r2-eager-enforce-ab-20260707/` + memory/50 changelog。剩余 ~20ms 交 R2.0 decode-only profile 定性（`task-r2.0-decode-profile.md`）。
>
> —— 下方为原调查设计，留档（结论已被上面的实测取代）——
- 现状：`cudagraph_mode=FULL_AND_PIECEWISE`，但 `splitting_ops` 把 `unified_attention`/`linear_attention`/`mamba_mixer2`/`gdn_attention_core` 等列为**图分割点**——意味着每层 attention/GDN 都可能打断 cudagraph，bs=1 decode 每 token 有 64 层 × 数个断点 = 大量 launch。
- 调查：bs=1 decode 到底走 FULL 还是掉回 PIECEWISE？能否让整个 decode step（含 GDN/attention）进**一个** full graph？（vLLM 0.18.1 对 mamba/gdn 的 full-cudagraph 支持是关键，需读源码确认。）
- 已知约束：`Capping cudagraph capture sizes ... to fit Mamba cache blocks (141 blocks)` —— capture size 被 mamba 状态显存挤压。但 bs=1 只需 capture size=1，不受这个上限影响，值得单独确认 bs=1 是否稳定命中 graph。
- 纪律：任何 graph 相关开关（`cudagraph_mode`、capture sizes、`enforce_eager` 对照）单独 A/B，验证数值等价。

### R2.2 显存/KV 排查（**不是收益项，是 SLA 排雷**）
- decode 带宽到顶、KV 仅用 6.76GB、显存有余 → **显存余量本身换不来速度，别为"用满显存"而调参**。
- 真正要查的隐患：`GPU KV cache size: 27,440 tokens < max-model-len 32768`。确认 **16-32K 满长度请求**会不会因 KV/mamba block 不足触发 preempt/recompute → 伤 TTFT-P99 与完成率（SLA 熔断风险，接 R5）。若有风险，在合规范围内调 `--gpu-memory-utilization`/page size 平衡（注意 `--max-num-batched-tokens=4096` 等是锁定项，不能动）。

### R2.3 Host 路径开销（与 R2.1 同属抢 gap）
- 并发=1 时，采样、detokenize、HTTP 流式返回若不能和 GPU 计算重叠，直接加到 TPOT。
- 查：detokenize 是否逐 token 同步、采样是否有多余 D2H 拷贝/同步、logits 处理路径。目标是让 host 工作与 GPU decode 重叠。
- 红线自查：不改采样语义（`temperature=0` 锁定）、不改输出口径。

### R2.4 GEMM 库与 DTK 环境层（低风险，兼产合规文档）
- prefill 投影 GEMM（占 ~16.82%）：确认走的是 rocBLAS/hipBLASLt 最优 kernel + Matrix Core，试 autotune / `hipblaslt` 算法选择。（decode GEMV 已达峰值，此项主要利好 prefill。）
- DTK/HIP 环境变量调优，每个进**环境变量说明文档**（R5 合规硬性要求，顺手产出）。

**R2 出口**：decode-only trace 出 gap 构成表；R2.1/R2.3 各有独立 A/B 数字；TPOT 目标从 69ms 往 45ms 下限方向压，报出实际拿到多少 ms。

---

## 3. Round 3 细化 —— Prefill flash-attention（唯一大胜负手）

### 靶心与预算
按 8-16K profile 权重：full-attn `unified_attention` 38.82% 是最大单块，GDN prefill ~30% 次之。两者拿下 = prefill 时间大头。prefill 是 8-16K/16-32K 两档（80% 权重）的时间主导（TTFT 占 56%/68%）。

### R3.1 full-attn prefill flash-attention on gfx936（首要）
- 现状 `unified_attention` triton kernel 在 gfx936 上大概率没调好（占 38.82%）。
- 路线优先级：① 先试 DTK 26.04 是否自带 CK(Composable Kernel)/ROCm flash-attention，直接接入 vLLM 的 attention backend；② 不行再手写/移植 flash-attention。
- **关键难点 = head_dim=256**（多数 flash-attn 实现只优化到 128）。这既是难点也是别人容易翻车的地方，做对了就是护城河。先验证目标 kernel 在 head_dim=256 下的正确性+性能，再谈提速。
- 注意：只有 16 层 full-attn（`full_attention_interval=4`），但 O(S²) 使其在长档是主力。
- **吃满 GQA 6:1 复用**（heads=24 / kv_heads=4）：kernel 内 K/V 从 HBM 读一次供 6 个 query head 复用，有效 KV 带宽 ÷6（外部《第三集》讲义明确点名的手法）。
- 范本：FlashAttention（tiling + online-softmax，避免把 S×S 注意力矩阵落 HBM）；参考实现见文末资料。

### R3.2 GDN chunked-prefill kernel（次要，~30%）
- `chunk_fwd_kernel_o` 22.65% + `chunk_gated_delta_rule_fwd` 7.66%。GDN(Gated DeltaNet) 特有的 chunked 线性注意力 prefill。
- 先 profile 单 kernel 的 occupancy/带宽/tile，判断是 tile 尺寸、bank conflict 还是 launch 配置问题。注意历史上 `33323a1 GDN chunk` 是负优化——**别重蹈覆辙，任何 GDN prefill 改动严格 A/B**。
- **直接参考 FLA (flash-linear-attention)** `github.com/fla-org/flash-linear-attention`：GDN 属该家族，其 chunked 融合 kernel 是移植/对标对象（先看 vLLM 现用实现与 FLA 上游差多少）。

### R3.3 投影 GEMM —— 交给库，别自己写
- 已贴 bf16 算力峰值（memory/50 结论）。R3 不手写 GEMM，只确保 rocBLAS/hipBLASLt + Matrix Core 选到最优（与 R2.4 重叠）。

### R3.4 decode 侧（仅 host，重申）
- 不写 GEMV/双缓冲。若发现可把相邻小算子融合以减 launch（如 norm+proj、GDN 内部小算子），做，但归为"减 host/launch"，不碰带宽。

**R3 纪律**：每个 kernel 先小张量数值等价单测（关掉=原路径 bit-一致或误差 <阈值）→ 单档 A/B → 四类精度回归 → 全档回归。**config/env 开关门控，关掉即回退。**
**R3 出口**：目标 kernel 相对原实现明确正向且数值等价；prefill tok/s 从 ~758 往上抬，报出各档 TTFT-P99 改善。

---

## 4. Round 4 细化 —— 精度换速度（大幅降级，诚实说明）

规则+架构对量化的双重封锁：
- **decode 量化 = 封死**：减 decode 时间唯一路径是权重低精度驻留 HBM = 持久化量化红线。
- **prefill 计算侧低精度 = 唯一合法空间，但收益有限**：
  - FP8 投影 GEMM（gfx936 `du_mma` 支持）：投影只占 16.82% 且已贴 bf16 峰值，FP8 翻倍算力最多省这块的一半 → prefill ~+8%。有肉但非大头。
  - **FP8/低精度 attention（白名单"Attention 内核优化/核内临时类型转换"）**：attention 是 O(S²) 大头，若能在 flash-attention 内部用低精度做 QK^T/PV，收益可能显著——但 attention 对精度敏感，**精度风险高**。
- **定位**：R4 不再是独立一轮，而是 **R3 flash-attention kernel 的可选精度增强分支**，且必须过严格精度门。
- 精度门（硬）：目标每类 Δ<1%（系数 k=1）；**任何单类 Δ>10% → 该类系数归零 = 亏 25% 总分，一票否决**。先跑精度回归再谈提速。
- ★ 唯一可能翻盘 decode 的开放问题：**运行时非持久权重量化是否违规**（每前向临时量化、不落盘、复用显存缓存）。我方判断更可能违规（≈红线"可复用量化权重缓存"）。**群里问到官方明确许可前，decode 量化按封死处理。**

---

## 5. Round 5 细化 —— 合规与 SLA 收尾

- **SLA 完成率专项（新增重点）**：16-32K 满长度 × 官方评测集全量，压测完成率>99%、TTFT-P99 与 TPOT-P99 不超 baseline×1.5、P99 长尾无异常抖动。接 R2.2 的 KV 排查。任何 kernel 改动都要过这个稳定性门（防止提速却引入长尾/OOM）。
- **三份必交文档**：优化方案说明（含各优化贡献消融表，正好由 guard A/B 数字沉淀）、环境变量说明（由 R2.4 沉淀）、README 头部第三方代码 + **AI 辅助声明**（章程 7.4 硬性）。
- 平台离线编译验证（依赖全 vendored，无外网）+ 提交缓冲，防编译失败返修。

---

## 6. 两个战略动作（建议插在 R2 之前/并行，价值高于剩余 R1）

### 战略动作 A：拿第二个官方分，校准"本地 Δ ↔ 官方分"映射 🔑
- 现状：唯一官方数据点 = d29e9db3 得 **59.0018**。R1 已证明各 commit 在噪声内、**59<60 的根因不在 commit 回退**（更可能是本地 proxy≠官方评测集，或精度系数，或 baseline 非精确 60）。
- 问题：**不建立"本地提升率 → 官方分"的映射，R2/R3 拿到的本地 Δ 不知道值多少官方分，也不知道真实起点**。
- 动作：用锁定口径提交一个**可控对照版本**（如干净 baseline-safe 或单一明确改动），拿到第二个官方分。两点定标，之后本地 Δ 才能外推官方分。**这比继续跑剩余 R1 commit 更值钱。**

### 战略动作 B：口径统一重测本地 baseline
- memory/50 里的基线数字（12.23/7.23/4.66）来自早期 run，很可能是错误的 `max_seq_len=262144` 口径。
- 动作：用 `--locked-start-script`（`max_seq_len=32768`）重测一版干净本地 baseline，作为 R2/R3 所有 A/B 的**同口径对照锚**。否则 R2 的提升会被口径差污染。

---

## 7. 待确认（阻塞相应项）
| # | 问题 | 阻塞 | 渠道 |
|---|---|---|---|
| ★ | 运行时非持久权重量化是否违规 | decode 唯一翻盘点 | 选手 QQ 群 795757156 |
| 1 | 精度 EM vs F1 | R4 精度门松紧 | 官方群 |
| 2 | throughput 数据集 max_tokens | prefill/decode 投入比精确化 | 官方群/调试文档 |
| 4 | 第二个官方分（战略动作 A） | 本地↔官方映射 | 提交一版 |

## 建议执行顺序（2026-07-07 更新：R2.1 已诊断关闭，主力转 prefill）
- ✅ 战略动作 B 部分完成：locked 口径 baseline 8-16K = **7.88 tok/s**（旧口径 7.23 作废；memory/50 旧基线数字标注"旧口径仅参考"）。
- ✅ R2.1 已诊断：graph 已 FULL、省 48ms → **覆盖度无肉，关闭**。
- **现在并行推进两张卡**：
  1. `task-r2.0-decode-profile.md` —— 收尾 R2，定性剩余 ~20ms（+ 看 full-attn decode 占比，可能 R3 一箭双雕）。**不占 GPU 主线。**
  2. `task-r3.1-flash-attention.md` —— **主力精力**，唯一大胜负手（head_dim=256）。
- 之后：R2.3（仅当 R2.0 证明 host 有肉）/ R2.4（GEMM 库，利好 prefill）→ R3.2 GDN → R4（可选精度增强）→ R5 收尾。
- 战略动作 A（第二个官方分校准）仍值得，与上并行、不占算力主线。

## 参考资料（外部实测讲义 + 开源 kernel）
- 2 份 gfx936 实测讲义（用户提供）：《decode访存瓶颈与双缓冲_DCU实测》(expA.py/dbocc.hip)、《第三集·带宽利用与算子优化》。要点已提炼进 memory/50「外部 DCU 实测讲义额外锚点」。**⚠️ 讲义的"权重量化降 TPOT"药方在本赛题违规，不可抄。**
- R3.1 full-attn 参考：FlashAttention `github.com/Dao-AILab/flash-attention`；ROCm/CK flash-attn（gfx936 后端）。
- R3.2 GDN 参考：FLA `github.com/fla-org/flash-linear-attention`。
- 库/工具：rocBLAS、hipBLASLt、omniperf（Roofline/带宽利用率）、Triton（HIP 后端）。

## Changelog
- 2026-07-07 create（Claude 基于 R0/R1 实证细化 R2–R5：R2 重定位为抢 decode host/launch gap；量化 decode gap 数字锚点；R3 靶心 flash-attention on gfx936 + head_dim=256 难点；R4 降级为 R3 精度增强；新增战略动作 A/B 校准官方分与口径）。
- 2026-07-07 Claude 纳入 2 份 gfx936 实测讲义：decode gap 精确化为 45/49/69ms 三段（~20ms host 是 R2 靶子）；R3.1 加 GQA 6:1 复用 + FlashAttention 范本；R3.2 加 FLA 参考库；R2/R3 手融前先查 inductor combo_kernels；补参考资料小节 + 红线重申。
