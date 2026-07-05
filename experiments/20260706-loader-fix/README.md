# 20260706-loader-fix — Model loader "假阳性"阻塞解除

**Verdict: PASS** · 装上 competition 优化 wheel 后 49 个 o_proj missing 报错消失，vllm serve 能进入 safetensors 加载阶段。之前 Codex 7-5 22:54 装的是 **baseline wheel**（"baseline-gap 调查"残留），不是优化 wheel，所以一直报 strict check 失败。

## 真正的原因（不是"假阳性"也不是 mapper bug）
`vllm_cscc_competition/dist/vllm-0.18.1+das.dtk2604-*.whl`（@06-22 12:40，含 6 个 commit：993a944 / 33323a1 / 0ba4953 / 293566c / fde463d / a55f3c3）的 `qwen3_5.py` 与 `vllm_cscc/dist/*.whl`（@06-21 19:17 基线）的 `qwen3_5.py` 在 line 197 附近处理 `core_attn_out` 的方式不同：

| 内容 | baseline wheel | competition wheel |
|---|---|---|
| line 197 alloc | `torch.zeros(...)` | `torch.empty(...)` |
| line 222 shape | `rearrange(...,"... h d -> ... (h d)")` | `core_attn_out.flatten(1)` |

优化分支把 `torch.zeros` 改成 `torch.empty` 同时把 `rearrange` 改为 `flatten` —— 这两处与 `AutoWeightsLoader` 的 strict-check 行为耦合？（精确机理未深挖，但实测装 comp wheel 后 49 个 missing 全部不再报）。装 baseline wheel 时反复 strict-fail，装 comp wheel 时顺利通过——可复现、决定的差异。

## 第二个坑（与 loader 无关）
`start_vllm.sh` 默认 `MODEL_DIR=${MODEL_DIR:-$HOME/Qwen3.5-27B}`，root 账户 `$HOME=/root` → `/root/Qwen3.5-27B`（本地盘但**不存在**，家目录才有 52G）→ vllm 报 `Repo id must be in the form 'repo_name'... '/root/Qwen3.5-27B'`。

**Fix**: `export MODEL_DIR=/public/home/xdzs2026_c166/Qwen3.5-27B` 或先 `cp -r` 模型到 `/root/Qwen3.5-27B`（推荐后者 — 直读网络盘加载 shard 7 min/片，估总 70 min；cp 到本地盘后启用本地 IO，vllm 启动可压到 5–10 min）。

## 操作时间线（CST 2026-07-06）
- 00:45 接手诊断。第一次 ssh banner 超时（`xiandaobei-worker` HostName `.2` 是错的，本容器实际 IP `.7`），改回 `.7` 后通。
- 00:52 诊断脚本：`qwen3_5.py` site-packages hash `76eeda1...` = **baseline wheel**；competition worktree hash `dd98896...` 不一致。
- 01:03 重装 `vllm_cscc_competition/dist/*.whl` → site-packages hash 变 `dd98896...`。首次 vllm serve 启动：strict check 通过；新错 `Repo id must ...'/root/Qwen3.5-27B'`（路径配置问题）。
- 01:10 启动 vllm serve with `MODEL_DIR=/public/home/xdzs2026_c166/Qwen3.5-27B`。
- 01:12 `Resolved architecture: Qwen3_5ForConditionalGeneration`; `Starting to load model`。
- 01:19 加载第 1 片 shard 7 分钟，估总 70 min —— 网络盘读取过慢，决定中止并 `cp -r` 到 `/root` 本地盘。
- 01:21 kill vllm serve；setsid 启动后台 `cp -r` 写入 `/root/Qwen3.5-27B`，end-marker `/root/.cp_done_marker`，log `/public/home/xdzs2026_c166/cp_model.log`。
- ⏳ cp 待完成（下次会话验证）后用 `MODEL_DIR=/root/Qwen3.5-27B` 起服务，预计 5–10 min 启动 → smoke → `run_throughput.sh all 10`。

## Git 版本管理
- **不动队友 `vllm_cscc_competition` 工作副本**（有未提交修改 `triton_unified_attention.py` + version.py）
- wheel 直接用 `vllm_cscc_competition/dist/*.whl`（持久化、已编译产物），本实验**不需要重新编译**
- Codex 隔离 worktree `vllm_cscc_codex_baseline_gap_224533`（分支 `codex/baseline-gap-20260705-224533`）只用作只读，本实验无须动它
- 本实验全过程日志/快照入 **meta 仓** `experiments/20260706-loader-fix/`

## 工件
- `site-packages-before/qwen3_5.py, qwen3_next.py` — 重装前的 site-packages qwen3 文件副本（baseline wheel 状态）
- `start_vllm_competition.log` — 第一次启动（baseline wheel 路径配置错的日志）
- `start_vllm_comp_final.log` — 第二次启动（comp wheel + 修正 MODEL_DIR，进入 shard loading 阶段后中止）
- `throughput_4-8K_smoke.log` — 没跑成功（vllm 未 ready）
- `cp_model.log` — 模型 cp 到 /root 的后台 log

## 下次窗口的接续动作
1. ssh xiandaobei-worker (Host IP 可能又变 → 用 scontrol+docker inspect 重新定位，sync-meta.sh 不解决的)
2. 检查 `cat /root/.cp_done_marker`：若存在则 cp 已完成；若无则看 `cp_model.log` 状态
3. 启动 `MODEL_DIR=/root/Qwen3.5-27B nohup bash -lc ./testdata/start_vllm.sh &`
4. 轮询 `curl 127.0.0.1:8001/health` 等 200
5. smoke `curl /v1/chat/completions` 验一次推理
6. `./testdata/run_throughput.sh all 10` 收全程吞吐数据 → 落 `experiments/20260706-baseline-fresh/`
7. 跑 accuracy 验证精度链路通

## Changelog
- 2026-07-06 01:25 seed by opencode.