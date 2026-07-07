# 审计存档 · opus4.8 web-grounded review（2026-07-07）

> 外部技术审计,由另一 Claude 会话(opus4.8)用 web 搜索 grounding 了 vLLM/ROCm 具体事实后给出。
> 输入 = 用户委托书(R2–R5 方案精炼),**当时 R2.1 尚未实测关闭**——故审计对 R2.1 的部分判断已被本仓实测超越。
> 本文件记录：哪些已采纳(净增量)、哪些被实测超越、哪些审计自认需本地验证。审计原文见对话记录。

## 一、已被本仓实测超越（审计不知道，勿再据审计反向操作）
- **审计说"enforce_eager 单独探针会误导,PIECEWISE 可能比 eager 更慢,需三方对照"** → 本仓 R2.1 实测日志**直接实锤**：`vllm_server.log` 显示 decode 走 **FULL** graph capture(非 piecewise),`--enforce-eager` 关掉后 TPOT-P99 69.9→118ms(−27%)。审计担心的"piecewise 误判"场景**未发生**,graph 已省 48ms,**R2.1 覆盖度无肉、已关闭**。见 `experiments/r2-eager-enforce-ab-20260707/`。
- **审计说"第二次提交 = R2 的 cudagraph_mode 修复"** → R2.1 已证明无 cudagraph 修复可提。改为：**第二次提交 = 第一个真正的正向优化**(R3 首个 prefill 提升,或 R2.0/R2.4 小收益),精神(真优化而非合成对照)仍成立。

## 二、净增量（已整合，指向落点）
| # | 审计增量 | 落点 |
|---|---|---|
| 1 | **vLLM issue #35238**：Qwen3.5-27B DeltaNet torch.compile dtype mismatch + FP8-DeltaNet **出乱码而非崩溃** → 必须 output-equivalence gate | memory/50 + task-r3.1 + task-r2.0 |
| 2 | **INT8 prefill 路径**：硬件有 INT8 MFMA(2–4× bf16),用于 compute-bound prefill GEMM/FA matmul,权重仍 bf16 驻留 HBM = **非持久化 = 合规**;被 >10% 单类精度悬崖门控 | r2-r5 §4 |
| 3 | **FP8 未必可用**：`du_mma` 有 FP8 转换 builtin ≠ 有快速 FP8 矩阵吞吐;gfx936 FP8 MFMA 未确认,需实测 | memory/50 待验证 |
| 4 | **R3 先 dump 当前 attention backend dispatch**:38.82%@758tok/s 是 reference fallback 特征;head_dim=256 **CK 上游支持到 256、Triton 功能完整**,瓶颈是"gfx936 build/enable"非"绿地手写" | task-r3.1 |
| 5 | **量级预期 1.5–2.5×,不是 10×**:hd=256 → Q tile[128,256]bf16=64KB,LDS-bound,被迫小 tile | task-r3.1 |
| 6 | **FLA GDN 是更 risky 的一半**:FLA chunked-scan 是 CUDA 假设很重的 Triton,gfx936 可移植性是真未知;~30% 占比,给独立 spike | r2-r5 §3.2 |
| 7 | **config/autotune 前置**:prefill GEMM 实测仅 ~33% 峰值效率(~130 TFLOPS);离线 hipBLASLt/rocBLAS autotune + `FLASH_ATTENTION_TRITON_AMD_AUTOTUNE` 是 near-free 绿区,应在写 kernel **前**做 | r2-r5 §2.4→前置 |
| 8 | **MTP 违禁**:`qwen3_next_mtp` speculative = 红线投机解码,写下来防队友"发现" | memory/50 红线 |
| 9 | **GDN decode fusion slack**:49ms 里非 GEMV 部分(48× GDN state/norm/gating)不一定在峰值,vLLM 有 GDN decode fusion PR,可能 3–6ms + 减 kernel 数帮 host gap | task-r2.0 |
| 10 | **R3 需自己的精度门**:FA online-softmax 非 bitwise,temp=0 greedy 下可能翻 token 级联发散→"无损"kernel 也能造精度回归 | task-r3.1（已有 tiny-tensor,升级为对 eager baseline 逐 token） |
| 11 | **detokenize V1 异步**:vLLM V1 通常异步 detokenize(独立进程);确认本栈是否同步,同步则是 20ms 里可回收的一块 | r2-r5 §2.3 |
| 12 | **战略校准**:精度是带零悬崖的乘数、吞吐指数衰减 → bf16-exact 收益(cudagraph/fusion/FA-bf16)不成比例地值钱;三档普涨 > 单档推平尾;每次提交若官方报告分档则可读 4 吞吐+4 精度点,但两点只够定 local slope 别拟合自信线 | r2-r5 §6 |

## 三、审计自认需本地验证（列为显式验证项，别当已知）
- **gfx936 是否有快速 FP8 MFMA throughput**(builtin 存在≠throughput 快)。
- **gfx936 的 CK/Triton flash-attention 是否 build 支持 head_dim=256**(需针对 gfx936 target 编译)。
- 两者本地 microbench 比猜快;R3 应把它们当第一批实测项。

## 四、审计原文来源链接
- vLLM CUDA Graphs design: https://docs.vllm.ai/en/stable/design/cuda_graphs/
- Hybrid Models as First-Class Citizens in vLLM (PyTorch blog): https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/
- PR #34571 cap FULL decode cudagraph sizes for Mamba/hybrid: https://github.com/vllm-project/vllm/pull/34571
- PR #22594 full CUDA graph by default for mamba2: https://github.com/vllm-project/vllm/pull/22594
- **Issue #35238 Qwen3.5-27B DeltaNet torch.compile dtype mismatch**: https://github.com/vllm-project/vllm/issues/35238
- ROCm/flash-attention support (CK + Triton, head_dim 256): https://deepwiki.com/ROCm/flash-attention/2.4-rocm-support
- ROCm model acceleration libraries: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/model-acceleration-libraries.html

## Changelog
- 2026-07-07 create（Claude 消化 opus4.8 web-grounded 审计,分类采纳/超越/待验证并整合进 memory/50 + r2-r5-detail + task-r3.1 + task-r2.0）。
