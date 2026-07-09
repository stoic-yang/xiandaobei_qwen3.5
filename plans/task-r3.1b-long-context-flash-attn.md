# 任务卡 · R3.1b Long-context flash-attn 诊断与调参

> 上位：[`task-r3.1-flash-attention.md`](task-r3.1-flash-attention.md)、[`task-r4.1-prefill-gemm-roofline.md`](task-r4.1-prefill-gemm-roofline.md)、[`../memory/50-arch-bottleneck.md`](../memory/50-arch-bottleneck.md)。
> 本卡由 R4.1 16-32K profile 触发：R3.1 后 8-16K 里 `Cijk_*` GEMM 是最大块，但 16-32K 里 flash-attn `_fwd_kernel` 升到同量级热点。

## 一句话目标

确认并优化 R3.1 flash-attn 在 16-32K 长上下文 prefill 下的 kernel 形态，目标降低 16-32K TTFT-P99，同时不牺牲 4-8K/8-16K 和精度。

## 现状锚点

- R3.1 官方 AC：`74.6924`，三档吞吐 `13.78 / 12.89 / 11.18`。
- R4.1 8-16K precheck：`Cijk_*` aggregate `~320.0 TFLOPS`，约 `81.0%` of `395 TFLOPS` peak。
- R4.1 16-32K profile：`experiments/r4.1-prefill-16to32-profile-20260709-1120/`
  - request API prompt tokens `20576`
  - hipkernel total `7053.787ms`
  - `Cijk_*` `3154.114ms` / `44.715%` / `317.396 TFLOPS`
  - flash-attn `_fwd_kernel` `2862.586ms` / `40.582%`
  - GDN core + helpers `610.613ms` / `8.657%`

## 硬约束

- 不改 decode GEMV/双缓冲/权重带宽侧。
- 不改模型结构/权重/持久化量化/剪枝/跳层/投机解码。
- 不改 batch scheduler；不动锁定 CLI：`--max-model-len=32768`、`--max-num-seqs=128`、`--max-num-batched-tokens=4096`、temperature/tokenizer/chat template/API。
- 候选必须 env-gated；关闭后回到 R3.1 当前路径。
- 每个 A/B 用同容器、同 warm 状态、同数据；默认隔离端口 `18001`，不碰未知 `8001` 服务。

## 执行步骤

### Step 0 — 确认 `_fwd_kernel` 来源

确认 16-32K profile 里的 `_fwd_kernel` 是 R3.1 flash-attn varlen 路径，而不是 hipprof 符号截断误归因：

- 对照 `hiptrace`/JSON trace 中的 code object 或周边调用；
- 在同服务上跑一个 `XDB_R31_FLASH_ATTN_PREFILL=0` 的短 profile 只作符号对照，不用于速度结论；
- 记录 calls 与 chunk 关系：当前 96 calls 应对应 16 full-attn layers × 6 prefill chunks。

### Step 1 — 无代码 env/config 探针

现场核实当前 `flash_attn==2.8.3` / ROCm 后端支持的 env：

- `FLASH_ATTENTION_TRITON_AMD_AUTOTUNE` 或等价 autotune 开关；
- backend/tiling/cache 相关开关；
- R3.1 现有 `XDB_R31_FLASH_ATTN_MIN_Q` 只调适用阈值，不应影响本 16-32K prefill 主路径。

每个开关单独 A/B：

- 先 16-32K throughput-only，3 prompts × 3 reps；
- 正向后再三档 guard；
- 精度 smoke 作为进入默认前置。

### Step 2 — shape microbench

不接 vLLM，直接测 flash-attn varlen target shapes：

- head_dim `256`
- query heads `24`，KV heads `4`
- causal
- q chunk sizes around `4096` plus final partial chunk
- cumulative K lengths covering `8K/16K/20K/32K`

对比：

- R3.1 当前 `vllm_flash_attn_varlen_func`
- vLLM 原 `context_attention_fwd` / `unified_attention` 可用形态，仅作 correctness/speed reference
- 若存在其他 ROCm flash-attn backend，先 microbench 后接服务

### Step 3 — 只在有明确 microbench 信号时实现候选

候选方向：

- backend/tile 选择；
- 减少 R3.1 packing/format 转换成本；
- 保持 GQA 6:1 K/V 复用，不退化成逐 query head 读 K/V；
- 不改 scheduler 和 chunk size。

验收：

1. 小张量 correctness；
2. output-equivalence prompt set；
3. 16-32K same-container guard 正向；
4. 三档 guard 无倒退；
5. accuracy smoke，round close 前 full accuracy。

## 出口标准

交回：

- `_fwd_kernel` 来源确认；
- 16-32K flash-attn 当前 ms/占比；
- env/config A/B 表；
- 若有实现候选：三档 throughput、TTFT-P99、TPOT-P99、精度 smoke。

## Changelog

- 2026-07-09 create（Codex；R4.1 16-32K profile 显示 flash-attn `_fwd_kernel` `40.582%`，与 `Cijk_*` `44.715%` 同量级；下一增长点从“直接 INT8/GEMM”调整为先做 long-context flash-attn 诊断/调参）。
