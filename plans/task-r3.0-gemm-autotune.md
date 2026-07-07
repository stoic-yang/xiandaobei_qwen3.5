# 任务卡 · R3.0/R2.4 GEMM library/autotune

> 上位：[`r2-r5-detail.md`](r2-r5-detail.md) §2.4 / §3.0 / §3.3、[`../memory/50-arch-bottleneck.md`](../memory/50-arch-bottleneck.md)、[`task-r3.2-gdn-prefill.md`](task-r3.2-gdn-prefill.md)。
> 本卡接在 R3.1 flash-attn 与 R3.2 Step0 之后：post-R3.1 prefill 热点已转为 GEMM，不写 GEMM kernel，只做库/环境/编译配置可回退优化。

## 一句话目标

针对 R3.1 后 8-16K prefill profile 中 `Cijk_*` Tensile GEMM `67.101%` 的热点，做可回退的 GEMM library/autotune 诊断与 A/B，目标降低长档 TTFT-P99，三档 throughput 不倒退，精度 smoke/full 不崩。

## 现状锚点

- R3.2 Step0：`experiments/r3.2-post-r31-prefill-profile-20260707-2210/`
  - request wall `3299.468ms`
  - hipkernel busy `3163.713ms`
  - `Cijk_*` Tensile GEMM `67.101%`
  - flash-attn prefill `8.994%`
  - GDN core `9.303%`, with helpers `13.277%`
- R3.0 survey：`experiments/r3.0-gemm-autotune-survey-20260707-2249/`
  - device `BW`, `gfx936:sramecc+:xnack-`, 80 CU, 65520 MB
  - PyTorch `2.10.0`, HIP `6.3.26093`, vLLM `0.18.1`
  - available standalone GEMM tools: only `hipprof`; no `hipblaslt-bench`, `rocblas-bench`, `rocblas-gemm-tune`, Tensile client, or `omniperf`
  - `torch.cuda.tunable` exists; `torch._inductor.config.max_autotune`, `max_autotune_gemm`, and `search_autotune_cache` are currently `False`

## 硬约束

- 不写 GEMM kernel；不改 decode GEMV/权重带宽侧。
- 不改模型结构/权重/持久化量化/剪枝/跳层/投机解码。
- 不改 batch scheduler；不动锁定 CLI：`--max-model-len=32768`、`--max-num-seqs=128`、`--max-num-batched-tokens=4096`、temperature/tokenizer/chat template/API。
- 每个候选必须 env/config gated；关闭后回到原路径。
- 任何 A/B 用 `scripts/guard_bench.py --locked-start-script --load-format runai_streamer`，并在 `vllm_server.log` 验证 `max_seq_len=32768`。
- 同容器、同 warm 状态、同数据比相对 delta；不要跨容器比绝对值。

## 执行步骤

### Step 0 — shape attribution（先做）

把 post-R3.1 profile 里的 top `Cijk_*` rows 归因到具体模型 op 和 shape：

- 目标：至少标出 top 3 GEMM 对应 QKV/O projection、MLP up/gate/down、GDN projection 还是其他路径。
- 可选手段：
  - 用 PyTorch profiler/kineto 记录 op shape（单个 8-16K `max_tokens=1` request 即可）。
  - 给 `torch.mm`/`linear`/Inductor 外部 kernel 调用加临时诊断，只记录 shape 与调用栈，禁止进入默认。
  - 对照 `vllm_server.log` 编译图与 `hipkernel.csv` 调用数（top rows calls `192/448/384`）做层数归因。
- 产出：`experiments/r3.0-gemm-shape-attribution-<date>/summary.json`，列 `kernel_row -> op -> shape -> calls -> share`。

### Step 1 — 候选 A：PyTorch TunableOp

现场核实当前 PyTorch 2.10 的 tunable API/环境变量后，做最小 A/B：

- 通过 Python API 或等效 env 开启 `torch.cuda.tunable.enable(True)` / `tuning_enable(True)`，设置独立结果文件到实验目录。
- 第一轮只跑 8-16K，`--accuracy none`，3 reps；若有正向再跑三档和 accuracy smoke。
- 必须隔离 warmup/compile/tuning 成本：吞吐只采稳定 reps，中间日志写明是否包含 tuning。
- 产出 tunable result file；若为空或没有覆盖目标 GEMM，判定为不可用，不继续。

### Step 2 — 候选 B：Inductor GEMM autotune

单独测试 Inductor autotune 开关，不与 TunableOp 混开：

- 先只验证配置是否真的生效（日志或 `torch._inductor.config` dump）。
- 候选开关以任务现场源码为准，当前 survey 显示这些默认是关的：
  - `max_autotune`
  - `max_autotune_gemm`
  - `search_autotune_cache`
- 任何环境变量名必须从当前 PyTorch/Inductor 源码或运行时 config 现场确认；不要凭记忆硬写。
- 同样先 8-16K throughput-only A/B，正向再扩三档和 accuracy smoke。

### Step 3 — 候选 C：库/缓存路径与环境变量

如果 A/B 指向“配置未覆盖目标 GEMM”，再调查库选择和缓存路径：

- rocBLAS/Tensile/hipBLASLt 是否被 PyTorch wheel 静态/私有打包，为什么 CLI 不暴露。
- `VLLM_CACHE_ROOT`、`TRITON_CACHE_DIR`、Inductor cache 与 tunable result file 是否复用导致冷热污染。
- DTK/HIP/Torch 相关 env 每个都必须进 R5 环境变量说明，不能偷偷默认化。

### Step 4 — 守门员

最小验收顺序：

1. shape attribution 完成，目标 GEMM 明确。
2. 8-16K same-container A/B：3 reps，TTFT-P99 或 output throughput 正向，TPOT 不明显倒退。
3. 三档 guard：4-8K / 8-16K / 16-32K 全部不倒退。
4. accuracy smoke；round close 前 full accuracy。

## 判定

- 正向：8-16K/16-32K TTFT-P99 下降或 output throughput 上升，三档无倒退，accuracy smoke Δ<1%。
- 负向：只减少冷启动/compile 时间但稳定吞吐不变，或 tuning 成本污染 benchmark，或任何一档 throughput 明显倒退，立即关掉。
- 若 TunableOp/Inductor 均不能覆盖 `Cijk_*` 目标 row，本卡输出“工具链不可用”结论，转 R5/SLA 或 R4 INT8 prefill feasibility，不在本卡写 kernel。

## 产出

- `experiments/r3.0-gemm-<date>/`：shape attribution、candidate config/env、raw logs、summary、verdict。
- `memory/50-arch-bottleneck.md` changelog：追加 R3.0 的真实可用性/收益，不改旧行。
- 环境变量说明草稿：若候选进默认，必须记录 env 名称、默认值、作用、关闭方式。

## Changelog

- 2026-07-07 create（Codex；R3.2 Step0 显示 GEMM 已成 post-R3.1 prefill 主热点；survey 发现无 standalone GEMM bench CLI，但 PyTorch TunableOp 存在、Inductor GEMM autotune 默认关闭，因此下一步转 shape attribution + env-gated A/B）。
