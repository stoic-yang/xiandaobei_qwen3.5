# 优化路线图 · roadmap

> 面向 Codex / GLM / OpenCode 的可执行计划书。Claude 任设计，其余 agent 任实现。
> 读本文件前先读 [`memory/50-arch-bottleneck.md`](../memory/50-arch-bottleneck.md)（架构与瓶颈事实）
> 与 [`memory/10-project.md`](../memory/10-project.md)（规则红线/绿区/评分）。
> 截止：初赛 2026-07-15 12:00。
> **R0/R1 已完成（R1 按用户决定暂停）；R2 起的实证细化见 [`r2-r5-detail.md`](r2-r5-detail.md)（含战略校准+decode gap 数字锚点+官方分校准动作）。**

## 一句话现状
当前提交 `d29e9db3` 官方 **59.0018 = 净负优化**（baseline 保底 60）。瓶颈是**长上下文 Prefill/TTFT**（占 80% 权重的两档时间大头 + prefill 仅 ~758 tok/s），**不是** decode、**不是** KV Cache。
**gfx936 实测 + 规则红线已钉死（见 memory/50）：decode 合规下到带宽物理顶——别碰；prefill flash-attention 是唯一战场。**

## 优化优先级（已被 gfx936 实测 + 规则红线钉死，见 memory/50）
1. **Prefill flash-attention on gfx936 —— 唯一胜负手**。注意力 O(S²) 是长档主力（codex R0.4：
   `unified_attention` 38.82% + GDN `chunk_fwd` 22.65%）；投影 GEMM 已贴算力峰值，交 rocBLAS/hipBLASLt + Matrix Core，别自己写。
2. **图捕获消 launch / 算子融合**——decode 合规下的唯一剩余空间在 **host 侧**（Amdahl：每 tok 几十个小算子+launch）。
3. ❌ **decode 的 GEMV / 双缓冲 / 提占用率——别碰**：实测已达 HBM 带宽 92–101%，到物理顶；减字节（权重量化）
   = 持久化量化红线，规则封死，外部作业那招不可抄。
4. **KV Cache 量化最后甚至跳过**——KV 仅 6.76GB。低精度只可能在 prefill 计算侧（收益小、多一步反量化）。

---

## 全局纪律（每一轮都适用）

- **守门员协议（R0.3 产出）是唯一验收口径**：固定同一热容器 → warm-up → 同数据 → 每档跑 3 次取中位数 → 记录【三档 output_throughput + TTFT-P99 + TPOT-P99 + 四类精度】。任何改动合入前必须过这个门。
- **一个改动 = 一个开关**（config-gated 或 env-gated）：关掉即回退到原路径，且与原路径**数值等价**。这是止损底线，也直接产出参赛要交的消融表。
- **受控对比**：A/B 必须同容器、同 warm 状态、同数据。禁止拿冷启动数字和热启动数字比较（历史教训：正负号被冷热差异淹没）。
- **红线自查**（详见 10-project.md）：不改模型结构/权重/持久化量化/剪枝/跳层/投机解码；不改 batch scheduler 代码；不动锁定 CLI 参数（`--max-model-len=32768`、`--max-num-seqs`、`--max-num-batched-tokens=4096`、`temperature=0` 等）。
- **每个实验 = 一个 `experiments/<id>/` 目录 + 一次 commit**（intent/method/diff/logs/metrics/verdict），见 AGENTS.md。
- **不动队友未提交的工作副本**（`vllm_cscc_competition` 有未 commit 改动）；需要隔离时用 worktree。

---

## Round 0 — 探针与诊断（不改默认行为，只测量）

> 目的：建"瓶颈地图"+"守门员协议"。所有后续决策的地基。**先做，不写 kernel。**

- **R0.1 架构体检**（~5 min）：`cat $MODEL_DIR/config.json`。报出：总层数、GDN/线性层与 full-attn 层各几层、**是否 MoE**（`num_experts`/`num_experts_per_tok`/激活参数量）、`head_dim`、KV heads、`hidden_size`。→ 写入 `memory/50` 待确认项。**决定 decode 是带宽还是 launch 受限。第一个做。**
- **R0.2 设备体检**（~2 min）：`python -c "import torch; print(torch.cuda.get_device_properties(0))"` + `hy-smi -a`。报出显存带宽、算力、gfx 架构名、是否支持 FP8。→ 定 roofline 上限 + R4 可行性。
- **R0.3 守门员脚本**（~半天）：把上面的测量协议固化成一个脚本，一次跑出三档吞吐/TTFT-P99/TPOT-P99 + 四类精度的中位数报告。**本轮最重要产出。** 之后每个改动都用它验收。
- **R0.4 时间分解 profile**（~半天）：hipprof 或 torch profiler 分别抓**一次 prefill** 与**一段 decode** 的 kernel 时间线，量化 full-attn / GDN / GEMM /(MoE)/ launch-gap 各占百分比。→ 直接点名 R3 先重写哪个 kernel。**先找队友 `maoym`（已用 hipprof 跑过）要现成结果，避免重复。**

**R0 出口标准**：产出一张表——"prefill 里最慢 kernel = X 占 Y%；decode = 带宽/launch 受限"。没这张表不进 R3。

---

## Round 1 — 止损回退（可与 R0 并行，目标：站稳 ≥60）

> 目的：把 59 分的负优化找出来回退，拿到"净效应≥0"的安全提交垫底。

- **R1.1 逐 commit 体检**：对 competition 分支 6 个 commit（`a55f3c3`/`fde463d`/`293566c`/`0ba4953`/`993a944`/`33323a1`）+ 已提交的 `d29e9db3`，逐个用 R0.3 协议做 A/B，标真实符号。已知负嫌疑：`33323a1 GDN chunk`、`d29e9db3 rocm gqa path`。
- **R1.2 分离"功能必需"vs"性能优化"**：保留让模型能加载的必需修复（`torch.empty+flatten`，见 `experiments/20260706-loader-fix/`）；回退确认为负的性能改动。
- **R1.3 产出 `baseline-safe` 提交候选**：净效应相对"能跑起来的最干净版本"≥0，先提上去锁定基本盘。

**R1 出口标准**：三档吞吐全部 ≥ 当前 competition 分支且无一档倒退；四类精度 Δ 均 <1%。

---

## Round 2 — 低风险普惠优化（不写 kernel，执行路径+配置）

> 指数评分下这段性价比最高。每项单独开关、单独测，产出消融表 + 环境变量文档。

- **R2.1 CUDA graph 覆盖度**：日志显示 capture 被 Mamba blocks 限到 136。确认 **bs=1 的 decode 是否全程 full graph**、有无掉回逐 kernel launch。对 launch 受限的 decode 是直接收益。
- **R2.2 显存再分配**：KV 只用 6.76GB、有余量。研究把余量用于改善 prefill 分块/locality（`784-token attention block` 是被 mamba page size 逼出的妥协，值得查）。
- **R2.3 Host 开销**：detokenize / 采样 / 流式返回路径的 Python 开销削减。
- **R2.4 DTK/HIP 环境层**：GEMM autotune、库算法选择、环境变量（进环境变量说明文档）。

**R2 出口标准**：每项有独立 A/B 数字，正向的进默认、负向的关掉留档。

---

## Round 3 — 算子重写（主攻，最大收益）

> 针对 R0.4 profile 点名的最慢 kernel 写 DCU 定制实现。**做哪个由 profile 决定，不预设。**

- **R3.1 Prefill 侧（优先）**：长上下文 prefill 的 full-attn kernel（flash-attention 式 DCU 适配）+ GDN chunked-prefill kernel + 热点 GEMM。目标把 ~758 tok/s 拉上去。
- **R3.2 Decode 侧（仅 host，别碰带宽侧）**：decode 带宽侧已到物理顶（GEMV 达峰值 92–101%，见 memory/50），
  **不要写 GEMV/双缓冲**；只做图捕获消 launch + 算子融合（host 开销，Amdahl）。
- **R3.3 KV FP8 基本跳过**：KV 仅 6.76GB，收益有限；仅当 profile 证明 full-attn KV 读取是瓶颈才考虑。
- 纪律：每个 kernel 先小张量**数值等价单测**（关掉=原路径）→ 单档 A/B → 精度回归 → 全档回归。

**R3 出口标准**：目标 kernel 相对原实现有明确正向且数值等价；全档不倒退。

---

## Round 4 — 精度换速度（条件性，白名单内）

> 仅当 R0.2 确认 DCU 有对应低精度算力时才做。

- 激活值动态量化 / 低精度 GEMM（规则白名单允许，非持久化）。每步守精度：目标 Δ<1%（k=1）；**红线：任何单类不得掉过 10%**（该类系数归零 = 亏 25% 总分）。

---

## Round 5 — 收尾与合规

- 全量三档 + **32k 稳定性压测**（完成率>99%、P99 长尾、防 OOM）——SLA 熔断真正风险在这。
- 三份必交文档：优化方案说明（含各项贡献消融表）、环境变量说明、README 头部**第三方代码 + AI 辅助声明**（章程 7.4 硬性）。
- 平台离线编译验证 + 留返修缓冲。

---

## 待确认问题（阻塞相应轮次）
| # | 问题 | 阻塞 | 负责渠道 |
|---|---|---|---|
| ★ | **运行时非持久权重量化是否违规** | decode 有无空间（唯一翻盘点） | 选手 QQ 群 795757156 |
| 1 | 精度指标 EM vs F1 | R4 量化激进程度 | 选手 QQ 群 795757156 |
| 2 | throughput 数据集 max_tokens | prefill/decode 投入比 | 官方群/调试文档 |
| 3 | ✅ 是否 MoE + 层配比 | — | codex R0.1 已确认（非 MoE，64=48+16） |
| 4 | 干净 baseline 在本容器的官方分 | 确认 59 负多少 | 队友对齐 |
| 5 | ✅ maoym 的 hipprof profile | — | codex R0.4 已复用 |

## 建议执行顺序
**先只做 Round 0 + Round 1**（一天）：把守门员协议和瓶颈地图建起来、把 59 分的负优化止损掉、锁定基本盘。地基不牢，Round 3 写再多 kernel 也是流沙盖楼。R0.1 是最快见效的一步（`cat config.json`），第一个做。

## Changelog
- 2026-07-06 create（Claude 设计，基于 memory/50 瓶颈画像）。
- 2026-07-06 Claude 按 gfx936 实测+规则红线更新优先级：**prefill flash-attention 唯一战场**；decode 带宽侧到顶别碰（GEMV 92–101%、权重量化=红线）；decode 仅做 host 侧图融合；新增 ★ 待问"运行时非持久权重量化是否违规"。
