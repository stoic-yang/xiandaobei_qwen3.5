# 容器池 / 预热 / 并行 —— 加速基础设施

> 定位：让"干活更快"的元工作。**元工作总投入 ≤ 1 天，主力精力留给 R3 算子**（真正拿分）。
> 相关代码：[`scripts/pool_manager.py`](../scripts/pool_manager.py)（维持型池骨架）。

## 目标：维持型池（maintain K running + B pending）
在竞争性集群上"持续保有热容器"，一箭双雕：
1. **抗抢卡**：始终在 Slurm 队列挂着 B 个 pending 请求，晚高峰有卡释放立刻拿到，而不是
   临时要用才提交、结果排长队。（pending 排队不占卡、不烧机时）
2. **天然预热**：池里始终维持 K 个 running 热容器，某个 4h 到期时本就有别的热容器顶上
   —— 不再需要专门的"到期前预热接班"逻辑（早期 L0 的双缓冲被这个更强的框架自然包含）。

## 已定决策（2026-07-06 与用户敲定）
- 管理器**跑在登录节点**（tmux/nohup）：squeue/scontrol/sbatch 本地可用，不受 mac 睡眠影响。
- **并发多开无硬限制，但晚高峰会排队** → 用维持型池持续排队占位。
- 机时充足（1000 GPU-hours）。

## ⚠️ 别囤积（好心办坏事的边界）
- **pending 不烧机时、不占卡** → 放心多挂 B 个当缓冲，纯赚。
- **running 烧机时** → 空占 running 是浪费：6 容器 7×24 空转 ≈ **144 机时/天，几天烧光 1000**，
  还可能踩集群公平性/被限流。
- 所以 **K（维持的 running 数）按负载走**：有一批候选要并行筛时调高 `POOL_K`，平时降回 1。
  **"多排 pending、少空转 running"** 是既抗抢卡又省机时的姿势；`POOL_MAX` 再兜底防塞爆队列。

## 概念澄清：机时 ≠ 并发卡数
- **机时(GPU-hours) = 总预算**：1000 烧不完，不是约束。
- **并发卡数 = 同一刻能同时占几张卡**：与机时无关；用户观察当前无硬限，但晚高峰靠排队。
- 参数：`POOL_K`=维持 running 数、`POOL_B`=常驻 pending 缓冲、`POOL_MAX`=总数安全帽。

## 🔑 命门：submit_job() 必须能命令行 sbatch（维持型的生死线）
维持型池 = 循环 sbatch 提交排队 + squeue 轮询分类，**强依赖 sbatch**（chrome 点击没法
持续自动排队）。先在登录节点探明网页"创建容器"背后的 sbatch：
```bash
scontrol show job <当前jobid>                                       # 看 Command= / 提交脚本路径
sacct -j <jobid> --format=JobID,JobName,Partition,SubmitLine%200    # SubmitLine 可能直接给出提交命令
squeue -u $USER -o "%.18i %.30j %.9P %.8T %R"
sacctmgr show assoc user=$USER format=Partition,MaxJobs,GrpTRES,MaxSubmitJobs   # 并发/队列上限
```
- 找到等价 sbatch → 填进 `submit_job()`，**维持型池成立**。
  - 2026-07-07 已从 job `656121` 的 `sacct SubmitLine` 探明当前可复用脚本：
    `sbatch -p hx1hdexclu08 /public/home/xdzs2026_c166/SothisAI/instance/ssh/Instances_2607070059422056_0_0/job_xdzs2026_c166_20260707_010007`。
  - 2026-07-07 08:54 复核更正：直接重投上述脚本生成 job `656380`，但 15 秒后
    `COMPLETED`，日志报 `_dockerlist_656380 does not exist` / `No such container:
    656380_e03r1n11`。随后通过 Chrome「确认并启动」同一 stopped instance 生成
    job `656384` 并成功 RUNNING/attach。结论：`sacct SubmitLine` 是有用线索，
    但旧脚本**不是已验证的可复用自动建池入口**；`pool_manager.py` 默认不再自动
    sbatch，只有 `POOL_SUBMIT_CMD='sbatch --parsable -p hx1hdexclu08 <script>'`
    经过“提交后 RUNNING + `scnetctl.py attach` 成功”验证后才允许打开。
- 只能网页/chrome → 维持型池无法自动化，退化为"脚本提醒 + 人工点"（价值大打折扣）。

## 维持循环（见 pool_manager.py，每 60s）
1. running+pending 补齐到 `min(K+B, MAX)`（一直排队，抗抢卡）。
2. 新 RUNNING 容器首次预热一次（装 wheel + 起 vllm）。
3. WARMING 容器 health 通过 → READY（可派活）。
4. 已回收（到期/消失）的移出池记录。
状态落 `~/pool_state.json` 防管理器重启失忆。

## 分层
- **基础版（先做）**：`POOL_K=1 POOL_B=2` —— 维持 1 个干活容器 + 2 个排队缓冲。
  验证命门 + 拿到抗抢卡/预热收益。骨架已给。
- **并行版（命门通过 + 有候选时）**：调高 `POOL_K` 并行筛候选。跑候选时**每容器自带
  baseline 对照，只比相对提升率 Δ，绝对吞吐不跨容器比**（跨容器有基线漂移，见 roadmap 并行纪律）。
- **L2（有余量再说）**：按候选队列长度动态调 K + 主动 scancel 空闲 running（需与调度器协调 busy）。

## Codex 接手 TODO
1. **探命门**（上面的命令），判定 sbatch 可行性 —— 维持型池的生死线，第一件事。
2. 填 `submit_job()`；按实际环境校准 pool_manager.py 里的远程转义 / docker 网络名 / health 判定（均标了 NOTE）。
3. `POOL_K=1 POOL_B=2` 起管理器跑通，观察抢卡/预热/回收一轮，再评估调高 K 并行。

## Changelog
- 2026-07-06 create（Claude 设计 + L0 单容器热备骨架；用户定：登录节点/先L0/机时够）。
- 2026-07-06 升级为维持型池（用户洞察：晚高峰排队 → 持续排队占位）。L0 双缓冲被"维持 K+B"
  自然包含；强调别囤积（多排 pending 少空转 running）；命门从加分项升级为生死线。
