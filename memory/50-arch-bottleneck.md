# 50 · 模型架构与瓶颈画像

> 本文件是"优化往哪使力"的事实地基。数字均来自 baseline 画像，非最终目标值。
> 来源：`experiments/baseline-full-reuse-20260706-0155/`（competition wheel @06-22 build，
> warm / `--reuse-server`，本地 10-prompt proxy）+ 同批 `codex-full10-*` 的 `vllm_server.log`。
> ⚠️ 本地 proxy ≠ 官方评测集；绝对值仅供内部对照，趋势与结构性结论才是重点。

## 架构：混合 Gated-DeltaNet 线性注意力（非普通 dense transformer）

`Qwen3.5-27B` = **GDN(Gated DeltaNet) 线性注意力层 + 少量标准 full-attention 层的混合体**
（Qwen3-Next 那一路的放大版），很可能还带 **MoE**。**是否 MoE / 层配比 待 R0.1 用 config.json 确认。**

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

## 待 R0 确认
- 是否 MoE、GDN/full-attn 层配比、激活参数量、head_dim、KV heads（config.json）。
- DCU 带宽/算力/gfx/是否支持 FP8（`torch.cuda.get_device_properties` + `hy-smi -a`）。
- throughput 数据集 max_tokens（输出长度）→ 定 prefill/decode 投入比。
- 精度口径 EM vs F1（官方群）→ 定量化激进程度。

## Changelog
- 2026-07-06 create（Claude 读 baseline 实验 + vllm_server.log 得出混合GDN架构与prefill瓶颈判断；推翻"KV量化是主战场"的纯文档预判）。
- 2026-07-06 R0.1/R0.2 correction（Codex live probe on SCNet job 655597）：`config.json` confirms 64 text layers = 48 `linear_attention` + 16 `full_attention`, `full_attention_interval=4`, **not MoE** (`num_experts`/`num_experts_per_tok` absent), `head_dim=256`, `num_attention_heads=24`, `num_key_value_heads=4`, `hidden_size=5120`; DCU is `BW`/`gfx936:sramecc+:xnack-`, 80 CU, 64 GiB class VRAM, wavefront 64, HIP 6.3 runtime; DTK exposes `du_mma` FP8 fragments/conversion builtins, but `hy-smi` did not print peak memory bandwidth or peak TFLOPS directly.
