# 30 · 代码仓库与队友自动化

## 两个 vLLM 仓库（都在 `/public/home/xdzs2026_c166/`）
- **`vllm_cscc/`** —— 基线干净浅克隆，remote 指向 `http://developer.sourcefind.cn/codes/OpenDAS/vllm_cscc.git` 分支 `v0.18.1`。已编译 wheel 在 `dist/` @06-21 19:17。
- **`vllm_cscc_competition/`** —— 优化工作副本。当前分支 `contest-p1-ffn-pool-20260621`，HEAD `a55f3c3 perf(qwen3.5): replace hot rearranges with views`。
  - 未提交改动：`M vllm/v1/attention/ops/triton_unified_attention.py`、`M vllm/version.py`（后者大概率 build 时自动写入版本号，无意义）。
  - 历史快照 `build_stale_20260622_021336/` 未跟踪。
  - **git "dubious ownership"**：root 身份访问需 `git config --global --add safe.directory /public/home/xdzs2026_c166/vllm_cscc_competition`。
  - 已编译 wheel @06-22 12:40，fingerprint `54ec7e66d0d4becf2656c0f71ebebe8161f212dd086fbf8b59d7d8225e4ed643`。

## competition 分支提交链（从新到旧）
```
a55f3c3 perf(qwen3.5): replace hot rearranges with views
fde463d perf(qwen3.5): remove qwen temp allocations
293566c perf(qwen3.5): reuse core attn and buffer pools
0ba4953 fix(build): clean generated version metadata
993a944 perf(activation): reuse swiglu output buffers   ← activation_p1.patch 对应这条
33323a1 perf(qwen3next): chunk long gdn prefills
fa71803 [Arch] Support bmz and nmz   (grafted origin/v0.18.1 起点)
```
→ `activation_p1.patch` 已落地为 commit `993a944`，不再是游离 patch。

## 未入选/已弃方向
- `pending_gdn_pool/pending_qwen_gdn_pool.patch` —— GDN pool 方向，GDN chunk 已合（`33323a1`），但 pool 版进了 pending 没选中。
- `fp8kv_noscale_direct_20260622_132651/` —— FP8 KV cache 实验过，结论"不值得继续押"。

## Codex 隔离调查 worktree（只读，不动队友脏文件）
- 路径 `/public/home/xdzs2026_c166/vllm_cscc_codex_baseline_gap_224533/`
- 分支 `codex/baseline-gap-20260705-224533` 从 `a55f3c3` 派生
- 用途：诊断 model loader strict-check 假阳性的隔离 worktree。

## 当前已知阻塞（待修复）
fresh 容器装基线或优化任一 wheel，`start_vllm.sh` 都在 model loader 阶段失败：
报 49 个权重 `linear_attn.out_proj.weight` / `self_attn.o_proj.weight` "not initialized"，
**全部来自 shard `model.safetensors-00010-of-00011.safetensors`**。源码 `qwen3_5.py` 的 `load_weights` 逻辑上应把 `o_proj` 加入 `loaded_params`，怀疑 mapper 命名未命中 params_dict 或 strict-check 假阳性。
6-22 OpenCompass 精度日志能跑通 → 推测当时 site-packages 有未提交的临时修正。修复链路：先在隔离 worktree 临时绕 strict-check 验证精度正常 → 把修正做进 wheel → 跑全量官方 throughput/accuracy。

## 已跑出的数字（仅供参考，非官方分）
- 本次（7-5 fresh）`run_throughput.sh all 10`：4-8K 12.253 / 8-16K 6.558 / 16-32K 3.217 tok/s。
- 裸加权（0.2/0.5/0.3）= 6.694741 ≈ 6-22 baseline 6.693327 (+0.021%)。
- 6-22 优化分支 vs baseline：activation only +4.10%。16-32K 曾 +20.039%，但 8-16K 反而 −51%。
- 套官方公式估算 ≈ **38.69 分**（4-8K 11.41 / 8-16K 18.61 / 16-32K 8.67）。第 9 名约 86.32，缺口主要在 8-16K 与 16-32K。
- accuracy smoke（`all 1`）：hotpotqa 100.00、gov_report 31.75。仅链路验证，非正式精度。

## 用户 baseline 反推
前 9 名截图反推官方 baseline：4-8K ≈ 12.96、8-16K ≈ 10.04、16-32K ≈ 5.77。拟合残差 2.5e-5，公式可信。

## 队友自动化基础设施
家目录下三件套：
- `watch_job_feishu.sh` —— 每 20s 轮询 `squeue`，状态变化时发飞书通知。
- `job_ready_hook.sh` —— 作业 RUNNING 时由 prolog 触发；读 `${HOME}/codex_callback_token.txt` 向 `127.0.0.1:18765`（默认）发带 token HTTP 回调，唤醒本机 Codex session 续跑。**当前 autorun 被关掉**，需人工在已有 Codex session 续跑（见 `pending_gdn_pool/NEXT_JOB.md`）。
- `feishu_sign.py` + 飞书 webhook —— 状态通知到飞书。
- `NEXT_JOB.md` 标准 ready 动作：`cd testdata; unset http_proxy…; ./run_throughput.sh all 10`；量化方向先跑 accuracy。

## 已观察到队友活动（与你不直接相关）
- `xdzs2026_c311` 目录 700 不可读。
  - `maoym`：用 `hipprof --hip-trace` 跑过 profile。
  - `chenwp`：用自己的 venv 起过 vllm serve。
- 都是 bench/profile 一轮就退，没有长期持有进程。

## 安全遗留
- 早期曾明文贴过一把私钥，是否已轮换未确认。
- 现用平台私钥 7-20 过期。

## Changelog
- 2026-07-06 seed（合并 Claude memory 与 Codex 提炼；修正 activation_p1.patch 状态为已合入；明确 loader 假阳性阻塞）。
- 2026-07-06 live correction（Codex SCNet job 655597）：`vllm_cscc_competition` is currently clean on `contest-p1-ffn-pool-20260621` at `d29e9db3f` (`perf(attention): add qwen35 rocm gqa attention path`), not `a55f3c3`; current competition wheel sha256 is `a0f09295a60dc1e5f4f7e9a096f540f29165168047c3caaf37233b6e4cb8cfde`. Baseline repo still has only build-noise `M vllm/version.py`.
