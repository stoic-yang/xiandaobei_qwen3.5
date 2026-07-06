# Memory Index

Last updated: 2026-07-06 by opencode (initial seed from Claude memories + Codex rollouts).

## Files

- [10-project.md](10-project.md) — 赛题/规则/评分/红线/绿区/截止 2026-07-15 12:00
- [20-env.md](20-env.md) — 环境/SSH 三段跳/容器易失/4h 回收/DCU=Hygon/代理陷阱
- [30-codeworkflow.md](30-codeworkflow.md) — vllm_cscc[_competition]/队友 Slurm+飞书自动化/NEXT_JOB
- [40-user.md](40-user.md) — 用户背景与偏好
- [50-arch-bottleneck.md](50-arch-bottleneck.md) — 混合GDN架构/瓶颈=prefill非KV/关键数字/当前净负

## Plans

- [../plans/roadmap.md](../plans/roadmap.md) — 分轮次优化计划书（R0探针→R1止损→R2普惠→R3算子→R4量化→R5合规）
- [../plans/infra-pool.md](../plans/infra-pool.md) — 容器池/预热/并行加速基础设施（L0 骨架 scripts/pool_manager.py + sbatch 命门）

## Changelog

- 2026-07-06 seed: migrated from `~/.claude/projects/-Users-keynary-Code-xiandaobei/memory/*` + Codex rollout summaries.
- 2026-07-06 Claude: add 50-arch-bottleneck.md（混合GDN架构与prefill瓶颈判断）+ plans/roadmap.md（分轮次计划书）。
- 2026-07-06 Claude: add plans/infra-pool.md + scripts/pool_manager.py（L0 容器池热备骨架，登录节点运行，create_container 命门待 Codex 填）。