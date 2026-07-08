# R2–R5 细化方案（基于 R0/R1 硬结论重新校准）

> 前置阅读：[`roadmap.md`](roadmap.md)（总纲）、[`../memory/50-arch-bottleneck.md`](../memory/50-arch-bottleneck.md)（架构/瓶颈/roofline）。
> 本文件把 R2–R5 从"纯文档预设"更新为"R0/R1 实证驱动"。R1 已按用户决定暂停（各 commit 在 ±0.2% 噪声内，无大负优化可回退）。
> 测量口径统一：`scripts/guard_bench.py --locked-start-script --load-format runai_streamer`，且必须在 `vllm_server.log` 验证 `max_seq_len=32768`；同容器比相对 Δ，不跨容器比绝对值。

## 0. R0/R1 定死的边界（不要再挑战）
- 架构：64 层 = 48 GDN 线性 + 16 full-attn（`full_attention_interval=4`），**非 MoE**，dense；head_dim=**256**、heads=24、kv_heads=4(GQA)、hidden=5120；gfx936 / 80 CU / FP8(`du_mma`) 可用。
- roofline：HBM **1206 GB/s**、bf16 **395 TFLOPS**、拐点 327 FLOP/byte。
- **decode 带宽侧到物理顶**：GEMV 已达 HBM 峰值 92–101%；权重 IO ≈ 45 ms/tok 是 TPOT 硬下限。减字节=权重量化=持久化红线，规则封死。**别碰 decode 的 GEMV/双缓冲/占用率。**
- **瓶颈地图（8-16K 热窗口）**：`unified_attention` 38.82%（full-attn prefill）＞ GDN `chunk_fwd` 22.65% + `chunk_gated_delta` 7.66%（≈30%，GDN prefill）＞ Tensile GEMM ~16.82%；decode GDN 仅 0.67%。
- 🔴 **MTP 违禁**（审计提醒，防队友"发现"）：本模型 vLLM 头号 decode 吞吐旋钮 `qwen3_next_mtp`（多 token 预测）= 投机解码红线；任何 MTP / speculative / draft 一律越界。
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
→ kernel 级已贴物理顶（45→49 只差 4ms），原假设**剩 ~20ms（占 TPOT ~30%）可能是 host/框架开销**：kernel 间 launch gap + sampling + detokenize + python 调度。
R2.0 已验证：本仓 local trace 里这个假设不成立，剩余 TPOT 主要仍是 GPU kernel 忙时。

> ⚠️ 49ms（外部讲义微基准）与 69ms（本仓端到端 P99）不同源。R2.0 证实这确实是口径差异/内核构成差异，不是一个可直接回收的 20ms host gap。

### R2.0（先决）decode-only profile — ✅ 已关闭（2026-07-07）
- `experiments/r2-decode-profile-r31-20260707-2151/`：hipprof session trace starts after the first streamed token and covers 64 steady decode chunks on an 8-16K prompt.
- Result: wall `4553.406ms` (`71.147ms/chunk`), kernel busy sum `4453.442ms` (`69.585ms/chunk`), residual wall-minus-kernel only `99.963ms` = `1.562ms/chunk`.
- `hipGraphLaunch=64` for 64 chunks → graph replay is one per token/chunk; decode graph coverage is healthy.
- full-attn decode `kernel_unified_attention_3d` is `4.158%` of decode kernel time (`2.893ms/chunk`); GDN recurrent decode kernel is `0.910%`.
- Decision: R2.3 host overlap has no large slack; R2 decode work closes unless a future trace contradicts this.

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

### R2.3 Host 路径开销（降级）
- 并发=1 时，采样、detokenize、HTTP 流式返回若不能和 GPU 计算重叠，直接加到 TPOT。
- 查：detokenize 是否逐 token 同步、采样是否有多余 D2H 拷贝/同步、logits 处理路径。目标是让 host 工作与 GPU decode 重叠。
- 红线自查：不改采样语义（`temperature=0` 锁定）、不改输出口径。
- R2.0 实测 residual 仅 `~1.56ms/token`，所以本项不作为主线优化；仅在 R3/R5 空档或发现明显同步 bug 时再做。

### R2.4 GEMM 库与 DTK 环境层（低风险，兼产合规文档）
- prefill 投影 GEMM（占 ~16.82%）：确认走的是 rocBLAS/hipBLASLt 最优 kernel + Matrix Core，试 autotune / `hipblaslt` 算法选择。（decode GEMV 已达峰值，此项主要利好 prefill。）
- DTK/HIP 环境变量调优，每个进**环境变量说明文档**（R5 合规硬性要求，顺手产出）。

**R2 出口**：R2.1/R2.0 均已关闭。Decode graph 到位，host/gap 大肉不存在；decode 不再是主线，转 R3 prefill。

---

## 3. Round 3 细化 —— Prefill flash-attention（唯一大胜负手）

### R3.0 config/autotune 前置（near-free 绿区，先做，审计增量）
- prefill 投影 GEMM 实测**仅 ~33% 峰值效率**（31% 非注意力份额 @758 tok/s ⇒ ~130 TFLOPS，本应 70%+）。→ 离线 **hipBLASLt/rocBLAS autotune** 针对确切 shape（hidden 5120、MLP intermediate、QKV/O proj）+ 打开 **FA autotune**，**不写 kernel 就能捡一部分 prefill gap**，绿区、近乎白拿。**排在 R3.1 kernel 前。**

### 靶心与预算
按 8-16K profile 权重：full-attn `unified_attention` 38.82% 是最大单块，GDN prefill ~30% 次之。两者拿下 = prefill 时间大头。prefill 是 8-16K/16-32K 两档（80% 权重）的时间主导（TTFT 占 56%/68%）。

### R3.1 full-attn prefill flash-attention on gfx936（首要）
- 现状 `unified_attention` 占 38.82% 却只 758 tok/s——**这是 reference/fallback kernel 的特征，不是调优过的 FA**（审计）。→ **R3 第一步 = dump vLLM 实际给 16 层 full-attn dispatch 的是哪个 attention backend**，确认是不是 fallback（大概率是）。
- **head_dim=256 不是"几乎不存在"，是"gfx936 build/enable 问题"（审计纠正，比原判乐观）**：CK 上游 fwd/bwd 支持到 256、Triton 后端功能完整可 autotune（`FLASH_ATTENTION_TRITON_AMD_AUTOTUNE`）。→ 路线是**移植/启用/autotune CK 或 Triton FA for gfx936**，手写 HIP 降为**最后**手段。gfx936 的 CK/Triton build 是否支持 head_dim=256 是 R3 第一批实测项（见 memory/50 待验证）。
- **量级预期 1.5–2.5×，不是 10×**：hd=256 → Q tile[128,256]bf16=64KB，LDS-bound、被迫小 tile，达不到 hd=128 的效率。别对外承诺 10×。
- 注意：只有 16 层 full-attn（`full_attention_interval=4`），但 O(S²) 使其在长档是主力。
- **吃满 GQA 6:1 复用**（heads=24 / kv_heads=4）：kernel 内 K/V 从 HBM 读一次供 6 个 query head 复用，有效 KV 带宽 ÷6（外部《第三集》讲义明确点名的手法）。
- 范本：FlashAttention（tiling + online-softmax，避免把 S×S 注意力矩阵落 HBM）；参考实现见文末资料。

### R3.2 GDN chunked-prefill kernel（已重估，降级）
- Pre-R3.1 旧 profile：`chunk_fwd_kernel_o` 22.65% + `chunk_gated_delta_rule_fwd` 7.66%。这是 R3.1 flash-attn 之前的热点图，只能解释为什么需要重测，不能再当收益承诺。
- Post-R3.1 实测（`experiments/r3.2-post-r31-prefill-profile-20260707-2210/`）：one 8-16K prefill request shows Tensile GEMM `67.101%`, flash-attn prefill `8.994%`, GDN core only `9.303%`（含 helper `13.277%`）。因此 R3.2 不再是默认主线；除非 16-32K profile 反转，否则先转 R3.0/R2.4 GEMM library/autotune。
- 先 profile 单 kernel 的 occupancy/带宽/tile，判断是 tile 尺寸、bank conflict 还是 launch 配置问题。注意历史上 `33323a1 GDN chunk` 是负优化——**别重蹈覆辙，任何 GDN prefill 改动严格 A/B**。
- **直接参考 FLA (flash-linear-attention)** `github.com/fla-org/flash-linear-attention`：GDN 属该家族，其 chunked 融合 kernel 是移植/对标对象（先看 vLLM 现用实现与 FLA 上游差多少）。
- **⚠️ 审计警告仍保留**：FLA GDN 不是库白拿。FLA chunked-scan 是 CUDA 假设很重的 Triton，在 gfx936 Triton 上的可移植性+性能是真未知数；现在 post-R3.1 占比已降到 `9.303%` core / `13.277%` with helpers，所以更不值得直接大移植，只适合 cheap spike。

### R3.3 投影 GEMM —— 交给库，别自己写
- 已贴 bf16 算力峰值（memory/50 结论）。R3 不手写 GEMM，只确保 rocBLAS/hipBLASLt + Matrix Core 选到最优（与 R2.4 重叠）。

### R3.4 decode 侧（仅 host，重申）
- 不写 GEMV/双缓冲。若发现可把相邻小算子融合以减 launch（如 norm+proj、GDN 内部小算子），做，但归为"减 host/launch"，不碰带宽。

**R3 纪律**：每个 kernel 先小张量数值等价单测（关掉=原路径 bit-一致或误差 <阈值）→ 单档 A/B → 四类精度回归 → 全档回归。**config/env 开关门控，关掉即回退。**
**R3 出口**：目标 kernel 相对原实现明确正向且数值等价；prefill tok/s 从 ~758 往上抬，报出各档 TTFT-P99 改善。

---

## 4. Round 4 细化 —— 精度换速度（大幅降级，诚实说明）

规则+架构对量化的双重封锁 + opus 审计修正（FP8→INT8）：
- **decode 量化 = 封死** ✓：减 decode 时间唯一路径是权重低精度驻留 HBM = 持久化量化红线。
- **FP8 死于硬件未确认（不是"被摊薄"）**：prefill attention 是 prefill 的 ~69%（**恰恰不是**被摊薄）；FP8 真死因是 **gfx936 无确认的快速 FP8 MFMA**（`du_mma` 有转换 builtin ≠ 有 FP8 矩阵吞吐，待实测）。无 FP8 吞吐 → downcast 白搭。
- **★ INT8 是被漏掉的合规上行（审计增量，别再说"没肉"）**：硬件有 INT8 MFMA（通常 2–4× bf16）。**核内动态 INT8** 用于 compute-bound 的 prefill GEMM 与 FA matmul、**权重仍 bf16 驻留 HBM = 非持久化 = 界内**。这是主导 prefill 成本上的真实上行。但三道硬门控：① **>10% 单类精度悬崖**一票否决；② `temperature=0` greedy 使数值误差**系统性**（确定性翻 argmax，不会平均掉）→ 比训练/采样更易崩精度；③ 任何 calibration 集**不能用测试数据**（红线）。
- **定位**：R4 = R3 flash-attention/GEMM 的**可选 INT8 精度增强分支**（不是 FP8），描述为"INT8、硬件支持、悬崖门控"。先过精度门再谈提速。
- 精度门（硬）：每类 Δ<1%（k=1）；**任何单类 Δ>10% → 该类归零 = 亏 25% 总分，一票否决**。
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
- 动作（审计更新）：**别花一次提交在合成对照上**——让第二次提交就是**第一个真正的正向优化**（R2.1 已关闭，故载体 = R3 首个 prefill 提升，或 R2.0/R2.4 小收益），既拿校准点又存真收益，严格优于烧一次提交在不会 ship 的东西上。
- **每次提交榨满信息**：若官方报告分档吞吐 + 分类精度，一次可读 ~4 吞吐点 + 4 精度点（非一个标量）；设计提交与日志以便读全。
- **留意噪声**：一次提交是可能有噪声的 scorer 的一个样本；两点给指数曲线的 **local slope**（够回答"下一个 kernel 值不值"），但别用两个噪声点拟合自信的线。

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
| 4 | ✅ 第二个官方分（战略动作 A） | 已由 R3.1 official AC `74.6924` 完成 | `experiments/r3.1-official-ac-20260708/` |

## 建议执行顺序（2026-07-07 更新：R2/R3.2 Step0 已诊断关闭，主力转 GEMM autotune）
- ✅ 战略动作 B 部分完成：locked 口径 baseline 8-16K = **7.88 tok/s**（旧口径 7.23 作废；memory/50 旧基线数字标注"旧口径仅参考"）。
- ✅ R2.1 已诊断：graph 已 FULL、省 48ms → **覆盖度无肉，关闭**。
- ✅ R2.0 已诊断：kernel busy `69.585ms/chunk` of wall `71.147ms/chunk`; residual only `1.562ms/chunk`; full-attn decode `4.158%`。**R2.3 降级，R2 收工。**
- ✅ R3.1 flash-attn candidate 已本地正向，且 full accuracy sanity 通过（见 memory/50）。
- ✅ R3.1 official AC：最终得分 `74.6924`，官方三档吞吐 `13.78 / 12.89 / 11.18`，SLA 扣分 `0.0`，精度扣分 `0.5644`。这完成了战略动作 A 的本地↔官方校准，且确认 R3.1 是当前安全正向 baseline。
- ✅ R3.2 Step 0 已重测：post-R3.1 prefill hotspot 变为 GEMM `67.101%`; GDN core `9.303%` / with helpers `13.277%`。**R3.2 降级**，只保留 cheap env-gated config/autotune spike，不进入大实现。
- **现在推进 [`task-r3.0-gemm-autotune.md`](task-r3.0-gemm-autotune.md)**：针对 post-R3.1 profile 中 `Cijk_*` Tensile GEMM 67% 热点，先做 shape attribution，再做 PyTorch TunableOp / Inductor autotune 这类可回退绿区。Survey 已确认当前容器没有 standalone `hipblaslt-bench`/`rocblas-bench` 工具，别在这个方向重复找。
- 之后：16-32K post-R3.1 profile（若需要确认长档是否同样 GEMM 主导）→ R3.2 cheap GDN config spike（可选）→ R4（可选 INT8 prefill 增强）→ R5 收尾。
- 战略动作 A 已完成：后续每个候选都应与 R3.1 official AC 的三档吞吐和精度扣分对齐，不再用 `59.0018` 作为唯一官方锚点。

## 参考资料（外部实测讲义 + 开源 kernel）
- 2 份 gfx936 实测讲义（用户提供）：《decode访存瓶颈与双缓冲_DCU实测》(expA.py/dbocc.hip)、《第三集·带宽利用与算子优化》。要点已提炼进 memory/50「外部 DCU 实测讲义额外锚点」。**⚠️ 讲义的"权重量化降 TPOT"药方在本赛题违规，不可抄。**
- R3.1 full-attn 参考：FlashAttention `github.com/Dao-AILab/flash-attention`；ROCm/CK flash-attn（gfx936 后端）。
- R3.2 GDN 参考：FLA `github.com/fla-org/flash-linear-attention`。
- 库/工具：rocBLAS、hipBLASLt、omniperf（Roofline/带宽利用率）、Triton（HIP 后端）。

## Changelog
- 2026-07-07 create（Claude 基于 R0/R1 实证细化 R2–R5：R2 重定位为抢 decode host/launch gap；量化 decode gap 数字锚点；R3 靶心 flash-attention on gfx936 + head_dim=256 难点；R4 降级为 R3 精度增强；新增战略动作 A/B 校准官方分与口径）。
- 2026-07-07 Claude 纳入 2 份 gfx936 实测讲义：decode gap 精确化为 45/49/69ms 三段（~20ms host 是 R2 靶子）；R3.1 加 GQA 6:1 复用 + FlashAttention 范本；R3.2 加 FLA 参考库；R2/R3 手融前先查 inductor combo_kernels；补参考资料小节 + 红线重申。
- 2026-07-07 Claude 整合 opus4.8 审计（存档 `audit-opus-20260707.md`）：§4 FP8→INT8（INT8 是合规 prefill 上行、FP8 硬件未确认）；§3 新增 R3.0 config/autotune 前置（GEMM 仅 ~33% 效率）；§3.1 加 dump backend + head_dim=256 改"enablement 非绿地手写"+ 量级 1.5–2.5×；§3.2 FLA 列一等风险；§0 加 MTP 违禁；§6 战略 A 改"第二次提交=真优化非合成对照"。审计对 R2.1 的 enforce_eager 担心已被本仓 decode-FULL 日志超越。
- 2026-07-07 Codex R2.0 close：`experiments/r2-decode-profile-r31-20260707-2151/` 证实 decode wall 几乎全是 kernel busy（residual `1.56ms/token`），`hipGraphLaunch=64/64`，full-attn decode `4.158%`；R2.3 降级，主线转 `task-r3.2-gdn-prefill.md`。
- 2026-07-07 Codex R3.2 Step0 close：`experiments/r3.2-post-r31-prefill-profile-20260707-2210/` 证实 R3.1 后 prefill 热点已转 GEMM（`Cijk_*` `67.101%`），GDN core 仅 `9.303%` / with helpers `13.277%`；R3.2 降级，下一主线改为 R3.0/R2.4 GEMM 库/autotune。
- 2026-07-07 Codex R3.0 survey：`experiments/r3.0-gemm-autotune-survey-20260707-2249/` 证实当前容器没有 standalone GEMM tuning CLIs，但 PyTorch `torch.cuda.tunable` 存在且 Inductor GEMM autotune 默认关闭；新增 `task-r3.0-gemm-autotune.md`，下一步是 shape attribution + TunableOp/Inductor A/B。
- 2026-07-08 Codex official R3.1 anchor：`experiments/r3.1-official-ac-20260708/` 记录 R3.1 平台 `AC`，final score `74.6924`，official throughput `13.78 / 12.89 / 11.18`，SLA penalty `0.0`，accuracy penalty `0.5644`；战略动作 A 完成，R3.1 成为当前官方安全 baseline。
