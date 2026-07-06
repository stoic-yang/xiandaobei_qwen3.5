# 容器池 / 预热 / 并行 —— 加速基础设施

> 定位：让"干活更快"的元工作。**元工作总投入 ≤ 1 天，主力精力留给 R3 算子**（真正拿分）。
> 相关代码：[`scripts/pool_manager.py`](../scripts/pool_manager.py)（L0 骨架）。

## 目标
双缓冲（double buffering）消除每 4h 容器到期后 ~12min 的"重建税"空窗：在 primary
到期前预热 standby，到期时无缝接手。进而延伸为并行池，供 R2/R3 并行筛候选。

## 已定决策（2026-07-06 与用户敲定）
- 管理器**跑在登录节点**（tmux/nohup）：squeue/scontrol/sbatch 本地可用，不受 mac 睡眠影响。
- **先做 L0**（单容器热备），验证命门 + 拿到预热收益，再决定是否上 L1 并行池。
- 机时充足（1000 GPU-hours），L0 不是约束。

## ⚠️ 概念澄清：机时 ≠ 并发卡数（别混，否则 L1 会误判）
- **机时(GPU-hours) = 总预算**：决定总共能烧多久。1000 机时烧不完，不是约束。
- **并发卡数 = 同一刻能同时占几张卡**：由 Slurm 分区/QOS 配额决定，**与机时无关**。
- L0 预热只需**并发 ≥ 2**（旧+新短暂重叠），门槛极低，基本稳。
- **L1 并行池的 K 受"并发卡数上限"约束**，要查的是 Slurm QOS/分区限制，不是机时。

## 🔑 命门：create_container() 能否命令行无人值守
容器是 Slurm 作业 → 网页"创建容器"背后**几乎一定是一条 sbatch**。先在登录节点探明：
```bash
scontrol show job <当前jobid>                          # 看 Command= / 提交脚本路径
sacct -j <jobid> --format=JobID,JobName,Partition,SubmitLine%200   # SubmitLine 可能直接给出提交命令
squeue -u $USER -o "%.18i %.30j %.9P %.8T %R"
sacctmgr show assoc user=$USER format=Partition,MaxJobs,GrpTRES     # 并发上限（给 L1）
```
- 找到等价 sbatch → 填进 `create_container()`，**全自动池成立**。
- 只能网页/chrome → `create_container()` 走"发通知让人工创建"的降级路径（脆弱，session 掉了就断）。

## 状态机（见 pool_manager.py）
```
CREATING ─(分到节点)→ WARMING ─(vllm health 通过)→ READY ⇄ BUSY
                                                        │
                                     (剩余寿命<阈值)─────┘
                                            ↓
                                       DRAINING ─(到期)→ DEAD
```
tick 循环（每 60s）：刷新剩余寿命+health → primary 将到期且无备 → 预热 standby →
standby health 通过转 READY → primary 到期则 standby 转正。状态落 `~/pool_state.json`
防管理器重启失忆。

## 分层
- **L0（先做）**：1 主 + 1 备预热。解决重建税空窗 + 验证命门。骨架已给。
- **L1（命门通过后）**：K>1 并行池 + 候选队列调度。跑候选时**每容器自带 baseline 对照，
  只比相对提升率 Δ，绝对吞吐不跨容器比**（跨容器有基线漂移，见 roadmap 并行纪律）。
- **L2（有余量再说）**：自愈、更强的持久化恢复。

## Codex 接手 TODO
1. 探命门（上面的命令），判定 sbatch 可行性。
2. 填 `create_container()`；按实际环境校准 pool_manager.py 里的远程转义 / docker 网络名 / health 判定（均标了 NOTE）。
3. L0 跑通（tmux 起管理器，观察一次到期是否无缝切换）后，再评估 L1。

## Changelog
- 2026-07-06 create（Claude 设计 + L0 骨架；用户定：登录节点/先L0/机时够）。
