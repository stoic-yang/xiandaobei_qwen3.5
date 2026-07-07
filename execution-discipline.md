# 远程执行纪律（无人值守）· execution-discipline

> **适用**：所有在 SCNet 容器 / 登录节点跑长任务的执行 agent（Codex / GLM / OpenCode）。
> **来源**：用户让 opus4.8 分析 Codex 实际执行后提炼；补进项目文档以统一执行风格、少烧 token、断点可续。
> **配套**：[`AGENTS.md`](AGENTS.md)（文件放哪 / 协作规则）、`.claude/skills/xiandaobei-operator`（容器 / benchmark / 提交操作）、`plans/infra-pool.md`（三层解耦 / 候选队列）。

## 0. 无人值守原则
- 假设用户不在电脑旁。**绝不停下来等用户确认、绝不用 request_user_input**。
- 不确定时：先按**任务卡决策表**执行；没有决策表 → 选**最保守、可回退**的选项，并在汇报里写清"我替你做了什么决定、为什么"。
- 被阻塞（权限、连接彻底断、缺关键信息）时：把【当前状态 / 已完成部分 / 下一步】落盘成 handoff 笔记（`journal/` 或 `experiments/<id>/`），然后**切到下一件能推进的事**，**不要原地等待或反复重试**。

## 1. 长任务：脱离式后台运行，不做轮内盯梢
- 预计 **> 2 分钟**的命令（训练、benchmark、编译、模型下载、`guard_bench`）**必须用 tmux/nohup 完全脱离**当前 exec 会话：输出重定向到日志，退出码写状态文件（`echo $? > .exit`）。脱离后这个任务就不再属于"仍在运行的 exec 会话"，回合可以正常收尾。
- **等待结果时禁止高频轮询**。规则：
  - 每次检查间隔 **≥ 5 分钟**；
  - 优先用一次**阻塞式调用**等状态变化，例如
    `timeout 280 bash -c 'until [ -f .exit ]; do sleep 20; done; cat .exit'`，返回 RUNNING / DONE 两态；
  - 两次检查之间**不发消息、不 update_plan**。
- 等待期间若同一任务卡里还有**不依赖该结果**的工作，先做那些；确实无事再进低频等待。

## 2. 子任务外包（subagent 模式）
- 大体量分析（长日志、profile 输出、大 diff 审查）**不要拉进主线程上下文**。用 `codex exec`（headless、低 reasoning effort）作一次性子 agent：输入 = 文件路径 + 明确问题，输出 = **一页 verdict 落盘**；主线程只读 verdict。
- 子 agent 后台运行：`nohup codex exec "..." > verdict.md 2>&1 &`，主线程按第 1 条低频等待收结果。
- **监控本身不要用子 agent 烧 token**：用 bash watcher 落盘状态；子 agent 只负责"结果出来之后的分析判读"。

## 3. 汇报纪律：少而准
- 只在**三种时刻**发有信息量的消息：① 拿到阶段性结论；② 做了需要记录的自主决策；③ 回合收尾总结。
- **禁止连续发"还在跑 / 还没结束"类消息**；等待状态合并进下一条有结论的汇报。
- `update_plan` 只在**里程碑变化**时更新。

## Changelog
- 2026-07-07 create（Claude；用户经 opus4.8 分析 Codex 执行后提炼的无人值守执行纪律，补进项目文档）。
