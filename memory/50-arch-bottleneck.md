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
- **🔴 MTP = 违禁杠杆（opus 审计提醒，写下来防队友"发现"）**：本模型 vLLM 头号 decode 吞吐旋钮是 `qwen3_next_mtp`（多 token 预测）speculative —— **正是红线投机解码**。任何 MTP / speculative / draft 路径一律越界，别开。
- **⚠️ 已知 live bug（opus 审计查到，vLLM issue #35238）**：Qwen3.5-27B DeltaNet 层 torch.compile **dtype mismatch(float vs half)**；相关 **FP8-DeltaNet 路径产生乱码而非崩溃**。两者都会逼人退回 enforce_eager（decode 减半）。→ **任何动 torch.compile / cudagraph / 低精度的改动，必须过 output-equivalence gate**（固定 prompt 集，greedy 输出逐 token 对齐 eager baseline），否则可能静默 ship 坏生成、吃精度乘数清零。

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
- DCU=gfx936 确认；`du_mma` 有 **FP8 转换 builtin**，但 ⚠️ **FP8 矩阵吞吐(MFMA)是否快未确认**（opus 审计：builtin 存在 ≠ throughput 快；double/single/half/INT8 有文档、FP8 没有）→ 需 DTK release notes + 本地实测。峰值带宽/算力由外部作业补（1206 GB/s / 395 TFLOPS）。
- **[待本地验证] INT8 MFMA throughput**（文档有、通常 2–4× bf16）→ 是**合规**低精度候选（prefill compute 侧、权重仍 bf16 驻留 HBM，非持久化），见 r2-r5 §4。
- **[待本地验证] gfx936 的 CK/Triton flash-attention 是否 build 支持 head_dim=256**（opus 审计：CK 上游支持到 256、Triton 功能完整，但需针对 gfx936 target 编译）→ 决定 R3 是"启用/移植现成后端"还是"手写"。
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
- 2026-07-07 R2.1 enforce_eager A/B（`experiments/r2-eager-enforce-ab-20260707/`）：locked `runai_streamer` 8-16K diagnostic, 3 prompts x 3 reps, accuracy none. Current graph/compile path median TPOT-P99 `69.910539ms`, output `7.882584 tok/s`; `--enforce-eager` median TPOT-P99 `118.195300ms`, output `5.737616 tok/s`; delta `+48.285ms` TPOT and `-27.21%` output throughput. Baseline log explicitly captured decode `FULL` CUDA graphs, while eager disabled both torch.compile and CUDAGraphs. Conclusion: R2.1 is not a "graph absent" rescue; next step is decode-only launch-count confirmation and narrow residual host-overhead tuning.
- 2026-07-07 Claude 整合 opus4.8 web-grounded 审计（存档 `plans/audit-opus-20260707.md`）：新增 **MTP=违禁杠杆**、**issue #35238 DeltaNet dtype/FP8 乱码 → output-equivalence gate 必做**、**FP8 MFMA throughput 未确认**（推翻本文件早先"FP8 支持确认"的过度乐观——builtin≠throughput）、**INT8 MFMA 是合规 prefill 低精度候选（待测）**、**gfx936 CK/Triton FA head_dim=256 build 待验证**、**prefill GEMM 仅 ~33% 峰值效率(autotune near-free)**。审计对 R2.1 的"enforce_eager 会误导"担心已被本仓 decode-FULL 日志实锤排除。
- 2026-07-07 R3.1 flash-attn prefill candidate（`experiments/r3.1-flash-attn-20260707/`, overlay SHA `d1984df6021dd8795625fee6dd6d477da3465668273f779d67e227e0a999a64d`）：gfx936/head_dim=256 bf16 candidate uses `flash_attn.flash_attn_interface.vllm_flash_attn_varlen_func` behind `XDB_R31_FLASH_ATTN_PREFILL=1`; default-off falls back to original `unified_attention`. Correctness smoke passed for head_dim256/GQA4kv/causal/chunk-boundary and module-level `TritonAttentionImpl.forward` (`q=512,k=2048` max diff `0.000244`; decode `q=1` fallback exact). Locked 8-16K A/B: baseline `experiments/r3-fa-baseline-8to16-20260707-1817/summary.json` median `7.233721 tok/s`, TTFT-P99 `15603.455ms`, TPOT-P99 `70.666ms`; candidate `experiments/r3-fa-candidate-8to16-20260707-1905/` median `11.455965 tok/s`, TTFT-P99 `3830.262ms`, TPOT-P99 `70.338ms`（+58.37% output, -75.45% TTFT）. Candidate补档 `experiments/r3-fa-candidate-extra-buckets-20260707-2010/summary.json`: 4-8K `13.416425 tok/s` / `1780.959ms` TTFT-P99 / `69.210ms` TPOT-P99; 16-32K `8.454834 tok/s` / `5451.579ms` TTFT-P99 / `71.988ms` TPOT-P99, both above `guard-a55f3c3-overlay-fullsmoke-20260707-0010` reference. Smoke accuracy candidate `experiments/r3-fa-candidate-accuracy-smoke-20260707-1948/summary.json`: hotpotqa `67.71` (=baseline), gov_report `34.45` vs `35.00` (-0.55), retrieval/aggregation `100.00` unchanged. Verdict: R3.1 is the current high-ROI positive candidate; keep env-gated until final submission decision.
- 2026-07-07 R3.1 local full accuracy gate（`experiments/r3-fa-candidate-fullacc-20260707-2130/`, source commit `847d1bef10b0b5bb71b7e427535b610a20a4d263`）：reused locked R3.1 service (`max_seq_len=32768`, `load_format=runai_streamer`, `XDB_R31_FLASH_ATTN_PREFILL enabled`) and ran local full accuracy. Scores: hotpotqa `77.96`, gov_report `32.71`, retrieval_multi_point `100.00 (30/30)`, aggregation_keyword_aggregation `100.00 (30/30)`. This passes local no-collapse sanity; caveat: not a same-container baseline full-accuracy A/B, so gov_report remains a watch item until official scoring or a same口径 baseline full run is available.
- 2026-07-07 R2.0 decode-only profile correction（`experiments/r2-decode-profile-r31-20260707-2151/`, R3.1 source on, decode path unchanged）：streaming 8-16K prompt, trace starts after first non-empty token, `64` decode chunks. Wall `4553.406ms` (`71.147ms/chunk`), hipprof kernel busy sum `4453.442ms` (`69.585ms/chunk`), residual wall-minus-kernel `99.963ms` = **`1.562ms/chunk` only**. `hipGraphLaunch=64` for 64 chunks, matching one graph replay per token/chunk; full-attn decode `kernel_unified_attention_3d` is `4.158%` of kernel time (`2.893ms/chunk`); GDN recurrent decode kernel is `0.910%` (`0.633ms/chunk`). Correction to earlier external 45/49/69ms hypothesis: in this local trace, the remaining TPOT is mostly GPU kernel/weight-bandwidth time, not a large host/detokenize gap. R2.3 host-overlap is therefore low ROI; R2 should close unless a future profile contradicts this.
- 2026-07-07 R3.2 post-R3.1 prefill profile（`experiments/r3.2-post-r31-prefill-profile-20260707-2210/`, R3.1 source on）：one locked 8-16K prefill request (`13964` prompt tokens, `max_tokens=1`) under hipprof. Request wall `3299.468ms`; hipkernel total busy `3163.713ms` (`95.886%` of wall). Post-R3.1 hotspot shifted to Tensile GEMM: `Cijk_*` kernels `67.101%` of kernel time. Flash-attn prefill is `8.994%`; old unified full-attn path is absent (`0%`). GDN core chunk kernels are only `9.303%` (`chunk_gated_delta` `5.050%`, `chunk_fwd_o` `2.231%`, `recompute_w_u` `1.540%`, `chunk_scaled_dot_kkt` `0.482%`), or `13.277%` including causal-conv/merge/l2norm/gating helpers. Conclusion: R3.2 GDN is lower ROI than the pre-R3.1 `~30%` assumption; next higher-ROI local task is R3.0/R2.4 GEMM library/autotune unless a 16-32K profile contradicts this single-request trace.
- 2026-07-07 R3.0 GEMM/autotune survey（`experiments/r3.0-gemm-autotune-survey-20260707-2249/`）：current container on `BW/gfx936`, PyTorch `2.10.0` HIP `6.3.26093`, vLLM `0.18.1`, competition source `847d1bef`. Standalone GEMM tuning CLIs are not exposed (`hipblaslt-bench`/`rocblas-bench`/`rocblas-gemm-tune`/Tensile client/`omniperf` all not found); only `hipprof` is available. PyTorch `torch.cuda.tunable` is present (`enable`, `tuning_enable`, `set_filename`, `tune_gemm_in_file`), while Inductor `max_autotune`, `max_autotune_gemm`, and `search_autotune_cache` are currently `False`. Conclusion: R3.0 should start with shape attribution for the hot `Cijk_*` rows, then env/config-gated TunableOp or Inductor autotune A/B; do not waste time searching for missing standalone rocBLAS/hipBLASLt bench tools in this container.
- 2026-07-08 R3.1 official AC result（`experiments/r3.1-official-ac-20260708/`, user-reported official platform row）：status `AC`, final score `74.6924`, actual throughput `13.78 / 12.89 / 11.18` for `4-8K / 8-16K / 16-32K`, SLA penalty `0.0`, accuracy penalty `0.5644`. Weighted throughput with `0.2/0.5/0.3` is `12.555`; score before accuracy penalty would be `75.2568` if no other hidden penalty. Compared with local R3.1 candidate guard (`13.416425 / 11.455965 / 8.454834`), the local proxy was conservative, especially for 16-32K. Conclusion: R3.1 is officially positive and should be treated as the current safe baseline; R3.0/R2.4 GEMM autotune remains the next mainline because post-R3.1 profile is GEMM-dominant.
- 2026-07-08 R3.0 Step0 shape attribution（`experiments/r3.0-gemm-shape-attribution-20260708-112017-wheel/`）：Python hook on `UnquantizedLinearMethod.apply` is **not** a valid language-graph attribution source under torch.compile; it only captured visual tower warmup shapes. Compiled graph signature + model config reconcile post-R3.1 Cijk calls exactly: 4 prefill chunks × language projection families `1216` + logits `4` = observed Cijk `1220`. Target shapes for TunableOp/Inductor A/B: MLP `gate_up_proj [34816,5120]`, MLP `down_proj [5120,17408]`, GDN `in_proj_qkvz [16384,5120]`, output projection `[5120,6144]`, full-attn `qkv_proj [14336,5120]`. Exact WGM row-to-shape mapping remains medium-confidence because committed hipkernel CSV lacks per-call M/N/K; require TunableOp result file or equivalent shape record in Step1.
- 2026-07-08 R3.0 TunableOp/Inductor status（`experiments/r3.0-tunableop-coverage-20260708-133548/`, `experiments/r3.0-inductor-autotune-smoke-clean-20260708-141641/`, `experiments/r3.0-inductor-autotune-screen-20260708-144124/`）：TunableOp loaded but only recorded visual-encoder GEMMs (`tn_3456_65536_1152`, `tn_1152_65536_1536`), with zero hits on target language GEMM shapes, so stop TunableOp. Inductor `TORCHINDUCTOR_MAX_AUTOTUNE=1` + `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1` applied and passed a clean 1-prompt smoke; old-container 8-16K candidate-only screen median was `7.865905 tok/s`, TTFT-P99 `13231.200ms`, TPOT-P99 `70.1517ms`, but it lacks same-container baseline because the container disappeared. New-container baseline attempts (`r3.0-baseline-screen-newcontainer-*`) were invalidated by concurrent 0.8B smoke services and then job `659779` was gone (`scnetctl status`: `job: none`, `worker: unreachable`). Conclusion: R3.0 A/B is still incomplete; next clean container should start with a fresh same-container baseline, then Inductor candidate using the prepared config-dump hook.
- 2026-07-09 R3.0 Inductor same-container A/B close（`experiments/r3.0-inductor-ab-20260709/`, container `Instances_2607090940238205_0_0` / job `661607`）：locked `runai_streamer`, isolated port `18001`, 8-16K 3 prompts x 3 reps, accuracy none. Baseline `r3.0-baseline-8to16-port18001-20260709-0952` median output `8.096026 tok/s`, TTFT-P99 `12779.450ms`, TPOT-P99 `69.743966ms`; Inductor `TORCHINDUCTOR_MAX_AUTOTUNE=1` + `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1` candidate `r3.0-inductor-8to16-port18001-20260709-1024` median output `8.091543 tok/s`, TTFT-P99 `12787.138ms`, TPOT-P99 `70.015290ms`. Delta output `-0.055%`, TTFT `+0.060%`, TPOT `+0.389%`; config switch applies but has no stable positive signal. Verdict: stop Inductor GEMM autotune, do not expand to three buckets/accuracy, do not enter defaults.
- 2026-07-09 R4.1 zero-cost GEMM roofline precheck（`experiments/r4.1-prefill-gemm-roofline-20260709/`, task card `plans/task-r4.1-prefill-gemm-roofline.md`）：using existing R3.2 8-16K prefill profile (`13964` prompt tokens, `Cijk_*` `2122.873ms`) and R3.0 shape set, included dense projection work is `679.403 TFLOP`; aggregate Cijk throughput is `320.040 TFLOPS`, about `81.0%` of external gfx936 `395 TFLOPS` peak. Verdict: yellow-zone, not a 2-3x GEMM-kernel opportunity; next diagnostic is 16-32K prefill roofline plus target-shape INT8/FP8 compute microbench if GEMM remains below ~360 TFLOPS. `lm_head/logits` was only `0.24%` in the committed profile, so skipping intermediate logits is not a main target unless a new profile contradicts it.
