# 20 · 环境与连通

## 平台账号与节点
- 赛事官网 `pra.xtnl.org.cn` 与算力平台 `scnet.cn` 是两套独立系统。
- 算力实际在 scnet.cn 超算互联网：控制台 → 容器服务 → 镜像管理 → 克隆镜像 `qwen3.5-dtk26.04:0509` → 容器实例选分区 `hx1hdexclu08` 队列、SSH 开发工具、镜像选克隆的那张。

## SSH 三段跳（实测打通，记忆里"不对公网开放"的结论已作废）
链路：
```
Mac → zzeshell.scnet.cn:65032  (user=xdzs2026_c166, key=~/.ssh/zzeshell_xdzs2026_c166)
     → ssh e03r1n07          (计算节点 zz-login01/02 负载均衡)
     → root@<container-ip>:22 (key=~/.ssh/xiandaobei_scnet_ed25519)
```
- `~/.ssh/config` 已配别名 `xiandaobei-login`（zzeshell）和 `xiandaobei-worker`（容器，经 ProxyCommand 走 login + e03r1n07 `ssh -W` 转发）。
- **容器 IP 每次重建都变**（见过 173.0.59.x 的 .2/.6/.7）—— 每次新容器要用 `scontrol show job` + `docker inspect` 重新拿 IP，然后 `sed` 改 `~/.ssh/config` 里 `xiandaobei-worker` 的 `HostName`。
- 容器名格式 `<jobid>_e03r1n07`；用 `docker ps -a` 查。
- `zzeshell` 背后是 `zz-login01/02` LB，主机指纹会切换 —— 用 `UserKnownHostsFile ~/.ssh/known_hosts_xiandaobei` 隔离。
- 平台登录私钥 `~/.ssh/zzeshell_xdzs2026_c166` 文件名带 `RsaKeyExpireTime_2026-07-20_19-03-24`。轮换机制未办。

## 容器存储与时长
- 用户家目录 `/public/home/xdzs2026_c166/` 是**网络盘，持久**（代码/模型/wheel/testdata 放这里）。
- `/root` 与 `site-packages` 是**容器 layer，易失** —— 容器回收后丢失（含 `pip install` 装的包、`/root/Qwen3.5-27B` 副本、nohup 进程、tmux）。
- 容器作业 **4 小时硬性到点回收**（Slurm wall-clock）。机时近乎无限，到点重开即可；真正成本只是"重建税"。
- 重开窗口：网页控制台点"确认并启动" → 新容器（新 IP、新 4h）。

## 每次新容器的最小前置（≈ 7–12 min 下限）
1. 重定位容器 IP：`scontrol show job` + `docker inspect` → 改 `~/.ssh/config`。
2. `pip install --no-deps /public/home/xdzs2026_c166/vllm_cscc_competition/dist/vllm-*.whl`（site-packages 没了）。30–60 s。
3. **取消代理**（大坑）：容器内 `http_proxy=http://preset:...@10.13.17.166:3128` 默认开，`curl 127.0.0.1:8001/health` 会被 Squid 拦截返回 HTML 报错页。每次 shell 先 `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY`，或脚本里前置。
4. 起 `start_vllm.sh`（权重载入显存是物理下限）。**5–10 min**。
5. `curl 127.0.0.1:8001/health` 健康检查。10 s。

**可跳过**：52G 模型 `cp -r` 到 `/root` —— 当前 `start_vllm.sh` 默认读 `$HOME/Qwen3.5-27B`，直接用家目录即可。
**可跳过**：`bdist_wheel` 重新编译 —— `dist/*.whl` 持久；只有改了 C++/kernel 才要重编，且用增量 build 跑 3–5 min 而非 clean。

## DCU 确认为 Hygon
- 工具 `hy-smi`（对应 nvidia-smi），内部把加速卡标 `HCU`，与规则里 `DCU` 同一物。
- 空闲实测：Temp 50°C、AvgPwr 83W、**PwrCap 1000W**、VRAM% 0、HCU% 0。
- **显存绝对容量与带宽仍未知** —— 需 `python -c "import torch; print(torch.cuda.get_device_properties(0))"`（ROCm PyTorch 沿用 `torch.cuda`）。

## 模型下载
```
pip install modelscope
modelscope download --model Qwen/Qwen3.5-27B --local_dir ./Qwen3.5-27B
```
已有：`/public/home/xdzs2026_c166/Qwen3.5-27B/` 52G，11 个 safetensors 分片齐全，config.json 在。架构 dense/MoE 需看 config 确认。

## testdata（非镜像自带，curl 下载 testdata.tar.gz 解压得到）
- 吞吐：`4-8K_throughput.jsonl` / `8-16K_throughput.jsonl` / `16-32K_throughput.jsonl`
- 精度：`hotpotqa.jsonl`、`gov_report.jsonl`、`retrieval_multi_point.jsonl`、`aggregation_keyword_aggregation.jsonl`
- 脚本：`start_vllm.sh`、`run_throughput.sh`、`run_accuracy.sh`（需 `chmod +x`）
- **这是选手自测子集，不等于官方评测集**；最终分以评测机为准。
- `start_vllm.sh` 监听 `127.0.0.1:8001`，served-model-name `Qwen3.5-27B`，参数见 `memory` 不再赘述。
- `run_throughput.sh <all|4-8K|8-16K|16-32K> <N>` 输出 TTFT/TPOT/ITL/E2EL，官方只看输出吞吐量、TTFT P99、TPOT P99。
- `run_accuracy.sh`：hotpotqa 指标 F1、gov_report ROUGE；retrieval/aggregation 走脚本自解析，日志里这两项显示 0 是正常噪音，真值在 "Recalculated RULER Results"。

## 提交前官方 checklist
- vLLM 源码改后能重新编译通过
- 装 wheel 后模型服务能正常启动
- curl 单次推理正常
- `run_throughput.sh` 正常运行
- `run_accuracy.sh` 正常运行

## 仍待回答
- EM vs F1 权威口径。
- 平台私钥过期续签路径。
- 显存绝对容量/带宽。

## Changelog
- 2026-07-06 seed（合并 Claude memory 与 Codex 7-5 实测：SSH 三段跳打通，废弃"不对公网开放"结论；明确 4h 硬性 + 重建税；列出最小前置 5 步）。