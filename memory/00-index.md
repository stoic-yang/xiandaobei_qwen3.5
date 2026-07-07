# Memory Index

Last updated: 2026-07-06 by opencode (initial seed from Claude memories + Codex rollouts).

## ⚠️ 执行 agent 先读
- [../execution-discipline.md](../execution-discipline.md) — 无人值守远程执行纪律（长任务后台化 / 低频轮询 ≥5min / 子任务外包 codex exec / 汇报少而准）。**跑长任务前必读。**

## Files

- [10-project.md](10-project.md) — 赛题/规则/评分/红线/绿区/截止 2026-07-15 12:00
- [20-env.md](20-env.md) — 环境/SSH 三段跳/容器易失/4h 回收/DCU=Hygon/代理陷阱
- [30-codeworkflow.md](30-codeworkflow.md) — vllm_cscc[_competition]/队友 Slurm+飞书自动化/NEXT_JOB
- [40-user.md](40-user.md) — 用户背景与偏好
- [50-arch-bottleneck.md](50-arch-bottleneck.md) — 混合GDN架构/瓶颈=prefill非KV/关键数字/当前净负

## Plans

- [../plans/roadmap.md](../plans/roadmap.md) — 分轮次优化计划书总纲（R0探针→R1止损→R2普惠→R3算子→R4量化→R5合规）
- [../plans/r2-r5-detail.md](../plans/r2-r5-detail.md) — R2–R5 实证细化（decode ~24ms host gap/flash-attn靶心head_dim=256/量化被规则封锁/官方分校准动作）
- [../plans/infra-pool.md](../plans/infra-pool.md) — 容器池/预热/并行加速基础设施（L0 骨架 scripts/pool_manager.py + sbatch 命门）
- [../plans/audit-opus-20260707.md](../plans/audit-opus-20260707.md) — opus4.8 web-grounded 审计存档（采纳/超越/待验证对齐表 + issue #35238 等来源链接）

### 任务卡（可直接派给实现 agent）
- [../plans/task-r2.1-cudagraph.md](../plans/task-r2.1-cudagraph.md) — R2.1 CUDA graph 覆盖度 ✅已诊断关闭（graph 已 FULL、省 48ms，无覆盖度肉）
- [../plans/task-r2.0-decode-profile.md](../plans/task-r2.0-decode-profile.md) — R2.0 decode-only profile（收尾 R2，定性剩余 ~20ms + full-attn decode 占比）
- [../plans/task-r3.1-flash-attention.md](../plans/task-r3.1-flash-attention.md) — 🎯R3.1 prefill flash-attention on gfx936（head_dim=256，唯一大胜负手）

## Changelog

- 2026-07-06 seed: migrated from `~/.claude/projects/-Users-keynary-Code-xiandaobei/memory/*` + Codex rollout summaries.
- 2026-07-06 Claude: add 50-arch-bottleneck.md（混合GDN架构与prefill瓶颈判断）+ plans/roadmap.md（分轮次计划书）。
- 2026-07-06 Claude: add plans/infra-pool.md + scripts/pool_manager.py（L0 容器池热备骨架，登录节点运行，create_container 命门待 Codex 填）。
- 2026-07-07 Claude: add plans/r2-r5-detail.md（R0/R1 实证驱动细化 R2–R5：decode host gap 数字锚点、flash-attn靶心、量化封锁、官方分校准两动作）。
- 2026-07-07 Claude: add plans/audit-opus-20260707.md + 整合审计入 memory/50、r2-r5-detail、task-r3.1、task-r2.0（INT8 合规上行 / FP8+FA-build 待验证 / MTP 违禁 / issue #35238 → output-equivalence gate / R3 dump backend / config autotune 前置）。
- 2026-07-07 Claude: add execution-discipline.md（无人值守远程执行纪律，用户经 opus4.8 分析 Codex 执行后提炼）。