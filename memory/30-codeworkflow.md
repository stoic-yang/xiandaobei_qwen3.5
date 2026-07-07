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
- 2026-07-06 R1 wheel fingerprint correction（`experiments/r1-wheel-fingerprint-20260706-2320/`）：repo HEAD is still `d29e9db3f`, but the installed competition wheel `a0f09295...` and site-packages do **not** include the `d29e9db3` `triton_unified_attention.py`; that file matches `33323a1` through `a55f3c3` and differs from source `d29e9db3`. Any benchmark using only `pip install dist/*.whl` measures wheel content, not necessarily source HEAD. Rebuild wheel or explicitly overlay source before claiming a d29 measurement.
- 2026-07-07 R1 guard update（SCNet job 656013）：`experiments/guard-a55f3c3-overlay-fullsmoke-20260707-0010/` completed the full guard for `a55f3c3` overlay with medians 12.156717 / 7.231679 / 4.655501 output tok/s and weighted 7.443833; smoke accuracy log reports hotpotqa 100.00, gov_report 30.51, retrieval recalculated 100.00 (1/1), aggregation recalculated 100.00 (1/1), with an OpenCompass aggregation JSON-vs-recalc mouthpiece mismatch. `experiments/guard-d29e9db3-overlay-fullsmoke-20260706-2355/` only proves the d29 source attention overlay SHA (`acf4b51...`) and abnormal first warmup request (128.30s); it was stopped at 4-8K rep1 and is **not** a completed d29 sign row. The 6+1 R1 sign table remains incomplete.

## meta 仓库 worktree 拓扑（2026-07-07 opencode 复核）
原始 `meta` 仓库的 `/Users/keynary/Code/xiandaobei/{meta,meta-r0-r1,meta-infra-verify}` 是同一 git 仓库的三个 worktree，挂在三条 codex/* 分支上：
- `meta`               → `codex/scnet-submit-auto-20260706`（提交自动化：`chrome_submit_adapter.mjs`/`submit_job.py`/`automation/submission.json`）
- `meta-r0-r1`         → `codex/r0-r1-probe-20260706`       （R0/R1 守门、infra 池骨架、6+1 commit sign table 主干）
- `meta-infra-verify`  → `codex/infra-verify-20260707`      （r0-r1 + 2 commit：`gfx936 microbench`、`candidate_executor.py`，seam 1/3/4 已验、seam 2 sbatch 命门仍 blocked）

分支关系：`infra-verify` 祖先含 `r0-r1`；`scnet-submit-auto` 与 `r0-r1` 自 `41cbafb` 分叉**互不祖先**但内容不重叠，可干净 merge。**main 上没有任何 codex/* 内容**。

### 三处 `scripts/*.py` 工作树差异（2026-07-07 已收）
`r0-r1` 的 dirty 版已确认是权威最新：guard_bench +189 行新增 `--overlay-source-dir`/`--foreground`/后台 upload-start-poll；pool_manager 把 `NotImplementedError` 换成 `POOL_SUBMIT_CMD` 空值 gate；scnetctl smoke rows 1→10。这些已在 `codex/r0-r1-probe-20260706` 上 commit `72fee79` 固化。**`meta/` 与 `meta-infra-verify/` 各自的 `scnetctl.py` dirty 版未** commit（仍按原样冻结，留 Codex 后续决策）。

### 远程同步阻塞（待 Codex 修一次）
容器 `/public/home/xdzs2026_c166/meta` 的 `.git/FETCH_HEAD` 为 root/ftp 拥有，`git pull --ff-only` 被拒（`meta` 与 `meta-r0-r1` 仓库更新推送上去后远程同步不进来）。修法二选一：容器内 `sudo chown xdzs2026_c166:xdzs2026_c166 .git/FETCH_HEAD .git/refs .git/logs` 或改用 `git -C ... pull --ff-only --no-edit refs/remotes/origin/main`（绕开 FETCH_HEAD）。

### 7 个 R1 guard experiment 目录处置记录（2026-07-07 opencode）
`meta-r0-r1/experiments/` 下 7 个未跟踪 dir 已全部按 AGENTS.md "1 exp = 1 dir = 1 commit" 规则 commit 进 `codex/r0-r1-probe-20260706`：
- PASS（有 `summary.json`）: `guard-d29e9db3-overlay-fullsmoke10-20260707-0122` (weighted 7.4566)、`guard-fde463d-overlay-fullsmoke10-20260707-0236` (weighted 7.4592)。
- partial（无 `summary.json`）：`guard-d29e9db3-overlay-fullsmoke10-20260707-0900`（模型拷贝窗口被打断）、`...-reuse-20260707-1005`、`...-reuse-20260707-1010`、`guard-d29e9db3-hotserver-nooverlay-fullsmoke10-20260707-1011`（poll 末态 8-16K rep=3；README 仅 opencode stub，待 Codex 填最终 verdict）。
- 证据账（非 guard 运行）: `r1-container-start-20260707-0854`（656380/656384 chrome-start-vs-resubmit 对照）。
**未完成 R1 sign table 仍差 `33323a1 / 993a944 / 0ba4953 / 293566c` 四个 commit 的同容器三档 guard。**

## Changelog (continued)
- 2026-07-07 opencode: 仓库 worktree 拓扑复核 + scripts dirty 收编 + 7 个 R1 experiment 目录 commit 入仓；追加本节。改 r0-r1 工作 9 commits（`72fee79` 起 `2a96294` 止）。**未推送 GitHub**。
- 2026-07-07 Codex correction: `guard-d29e9db3-hotserver-nooverlay-fullsmoke10-20260707-1011` later completed, replacing the opencode partial stub with final medians 12.211258 / 7.223185 / 4.652457 output tok/s and weighted 7.449581; runtime fingerprint keeps d29 attention site SHA `acf4b51...`. `guard-d29-revert333-srcdir-fullsmoke10-20260707-1126` tested candidate head `51cb6f325ab53854e94f5d4b5018712f4f662d7f` via `/root/overlay-d29-revert333-51cb6f325ab5`; it is **not** baseline-safe vs 1011 (deltas -0.0148% / -0.0528% / -0.1630%, weighted -0.0610%). `guard_bench.py` now skips full `accuracy_debug/output` copies unless `--copy-accuracy-output` is set; raw accuracy log remains collected by default.
