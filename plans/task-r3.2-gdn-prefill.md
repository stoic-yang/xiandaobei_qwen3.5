# 任务卡 · R3.2 GDN chunked-prefill on gfx936

> 上位：[`r2-r5-detail.md`](r2-r5-detail.md) §3.2、[`task-r3.1-flash-attention.md`](task-r3.1-flash-attention.md)、[`../memory/50-arch-bottleneck.md`](../memory/50-arch-bottleneck.md)。
> R3.1 已把 full-attn prefill 变成当前最大正向候选；R3.2 只打剩余 prefill 热点里的 GDN chunk，不碰 decode GEMV。

## 一句话目标

在 R3.1 flash-attn 开启后，重新确认 8-16K/16-32K prefill 热点，并优化 GDN chunked-prefill kernels：
`chunk_fwd_kernel_o.kd`、`chunk_gated_delta_rule_fwd_kernel_h_blockdim64.kd`、`recompute_w_u_fwd_kernel.kd`、`chunk_scaled_dot_kkt_fwd_kernel.kd`。目标是进一步降低长档 TTFT-P99，三档吞吐不倒退，四类精度 Δ<1%。

## 现状锚点

- R0 profile（R3.1 之前）：8-16K hot window 里 GDN chunk 约 32.91%：
  - `chunk_fwd_kernel_o.kd` 22.65%
  - `chunk_gated_delta_rule_fwd_kernel_h_blockdim64.kd` 7.66%
  - `recompute_w_u_fwd_kernel.kd` 2.60%
  - `chunk_scaled_dot_kkt_fwd_kernel.kd` 0.12%-3% 量级，随窗口变化
- R3.1 candidate 已显著压低 full-attn TTFT；因此 **R3.2 第一步必须重抓 R3.1 开启后的 prefill profile**，旧占比只能做方向锚点，不能当收益承诺。
- ✅ Step 0 已完成（`experiments/r3.2-post-r31-prefill-profile-20260707-2210/`）：post-R3.1 8-16K prefill one-request profile 显示 `Cijk_*` Tensile GEMM `67.101%`、flash-attn prefill `8.994%`、GDN core `9.303%`（含 helper `13.277%`）。R3.2 已按本卡规则降级；下一主线应转 R3.0/R2.4 GEMM library/autotune，GDN 只保留 cheap env-gated config/autotune spike 或等 16-32K profile 反证。
- 历史警告：`33323a1 perf(qwen3next): chunk long gdn prefills` 曾是负优化嫌疑，GDN 改动必须一改一开关、同容器 A/B。

## 硬约束

- 只改 GDN prefill 路径；decode 侧 `fused_recurrent_gated_delta_rule_packed_decode_kernel` 不在本卡默认范围。
- 不改模型结构/权重/持久化量化/剪枝/跳层/投机解码。
- 不改 batch scheduler；不动锁定 CLI：`--max-model-len=32768`、`--max-num-seqs=128`、`--max-num-batched-tokens=4096`、temperature/tokenizer/chat template/API。
- 一个候选 = 一个 env/config 开关；关闭必须走原 GDN 路径且数值等价。
- 所有吞吐用 `scripts/guard_bench.py --locked-start-script --load-format runai_streamer`，日志验证 `max_seq_len=32768`。

## 执行步骤

### Step 0 — R3.1 后重画热点图

用 R3.1 默认开启的源码提交跑一次 8-16K 稳态 prefill profile，最好复用 warm container：

- 记录 `vllm_server.log` 中 `max_seq_len=32768`、`load_format=runai_streamer`、`XDB_R31_FLASH_ATTN_PREFILL` 状态。
- 输出 post-R3.1 hot window top-N，单独汇总 GDN kernels / GEMM / fill / flash-attn。
- 若 GDN chunk 已低于 10%，本卡降级，先转 R2.0/R2.4 或 R5；若 GDN 仍是前二热点，继续。

### Step 1 — 固化 GDN shape 与正确性对拍（降级后仅在 cheap spike 时做）

在现有 `vllm/model_executor/layers/fla/ops/` 上加临时诊断或单测，记录实际形状：

- `q/k/v/g/beta/initial_state/out` shape、dtype、stride、contiguous。
- `cu_seqlens` 是否存在、chunk size、`BT/BK/BV/num_warps/num_stages` 实际 autotune 选择。
- 长档 chunk 边界：T=4096 chunk 与跨 chunk final_state。

必须先做小张量/模块级对拍：

- 旧路径 vs 新候选的 `o` 与 `final_state`。
- 覆盖 T=64/512/4096，含 `cu_seqlens` variable-length。
- bf16 容差内通过；不过则不采信吞吐。

### Step 2 — 低风险候选 A：Triton config/autotune

先调现有 FLA Triton kernel，不绿地手写 HIP：

- `chunk_o.py`: `BKV_LIST`、`NUM_WARPS`、`num_stages`、`BT`/`chunk_size`。
- `chunk_delta_h.py`: blockdim / num_warps / stages。
- `wy_fast.py`、`chunk_scaled_dot_kkt.py`、`solve_tril.py`: 只在 profile 指向后动。

建议开关：

- `XDB_R32_GDN_PREFILL=1`
- `XDB_R32_GDN_BT=<64|128>`
- `XDB_R32_GDN_BK=<32|64|128>`
- `XDB_R32_GDN_BV=<32|64|128>`
- `XDB_R32_GDN_WARPS=<2|4|8>`

每个实验只打开一个候选组合；先 microbench 单 kernel，再 8-16K guard。

### Step 3 — 候选 B：对齐/移植 FLA 上游

若 config/autotune 无明显收益，比较当前 fork 与 FLA `flash-linear-attention` 对应 Gated DeltaNet kernels：

- 只移植 prefill fwd 所需最小文件。
- 明确列第三方代码来源、license、改动点，给 R5 文档留证据。
- 仍需 env gate 和旧路径 fallback。

### Step 4 — 守门员 A/B

最小验收顺序：

1. 单 kernel/microbench：目标 kernel 明确快，数值对拍过。
2. 8-16K guard：3 reps，TTFT-P99 改善，TPOT 不明显倒退。
3. 三档 guard：4-8K / 8-16K / 16-32K 全部不倒退。
4. accuracy smoke；round close 前 full accuracy。

## 判定

- 正向：8-16K 或 16-32K TTFT-P99 改善，三档 output throughput 无倒退，accuracy smoke Δ<1%。
- 负向：任何一档 throughput 低于同容器 baseline 超过噪声，或 `final_state` 对拍不过，立即关掉并记录 fail。
- 若 GDN 优化低收益但 post-R3.1 profile 显示 GEMM/fill 变成主块，转 R2.4/R3.0 库 autotune，不继续死磕 GDN。2026-07-07 Step 0 已触发这一条。

## 产出

- `experiments/r3.2-gdn-prefill-<date>/`：post-R3.1 profile、shape dump、候选 diff、microbench、guard summary、verdict。
- `memory/50-arch-bottleneck.md` changelog：只追加 R3.2 的实测占比和结论，不改旧行。
- 交回一句话：R3.1 后 GDN 还占多少；哪个 GDN 候选正确且更快；8-16K/16-32K TTFT-P99 改善多少。

## Changelog

- 2026-07-07 create（Codex；GitLab 评测不可用时继续本地闭环，R3.1 后转向 GDN chunked-prefill，但先强制 post-R3.1 profile 与数值对拍）。
- 2026-07-07 Step0 result（Codex；`experiments/r3.2-post-r31-prefill-profile-20260707-2210/`）：post-R3.1 prefill 热点转为 GEMM `67.101%`，GDN core `9.303%` / with helpers `13.277%`；本卡降级，下一主线改 R3.0/R2.4 GEMM library/autotune。
