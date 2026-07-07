#!/usr/bin/env python3
"""
候选执行器 —— 让 codex 可随时停工，系统继续把排好的活跑完

定位（三层解耦架构的"常驻层"）
------------------------------
codex（智能层，会因额度停工）只负责"排候选 + 分析结果"；本执行器（常驻层，跑在
登录节点 tmux/nohup，不依赖 codex）负责"把排好的候选一个个跑完"。两者通过 meta 仓的
candidates.jsonl 队列 + experiments/ 结果解耦：
  codex 停工前把待试候选写进队列 → 执行器在 codex 睡觉时跑完 → codex 续上读结果排下一批。
这样 codex 从"必须一直在"降级为"间歇来排活和分析"，额度打断无损。

与 pool_manager 分工（两者都跑登录节点、都不依赖 codex）
------------------------------------------------------
  pool_manager.py        —— 维持容器池（保证有 READY 热容器），管容器生命周期。
  candidate_executor.py  —— 消费 READY 容器，执行候选队列里的活。
executor 读 pool_manager 写的 ~/pool_state.json 找 READY 容器。

断点可续：候选状态落 candidates.jsonl（pending→running→done/failed）；executor 重启后把
超时未完成的退回 pending。**单实例常驻**（别开多个 executor，避免抢同一候选）。

⚠️ 骨架说明：run_candidate() 里"apply 改动 + 装 wheel/增量重编 + 跑守门员"的细节按实际由
Codex 填（复用 guard_bench.py / scnetctl.py，别重造测量）；队列/挑容器/状态机逻辑是设计意图。
"""
from __future__ import annotations
import json, os, time, datetime, subprocess, re

# ---------------- 配置 ----------------
META = os.path.expanduser("~/meta")
CANDIDATES = os.path.join(META, "candidates.jsonl")    # codex 写入待跑候选
POOL_STATE = os.path.expanduser("~/pool_state.json")   # pool_manager 写的容器状态
TICK_S = 30
CLAIM_TIMEOUT_S = 90 * 60      # running 超过此仍没 done → 视为 executor 曾崩溃，退回 pending
# 各类候选预估耗时（挑容器时要求剩余寿命 > 此，DRAINING 边界，避免半截被 4h 回收）
EST_S = {"microbench": 10 * 60, "smoke": 20 * 60, "full": 55 * 60}

def now(): return datetime.datetime.now()
def log(m): print(f"[{now():%m-%d %H:%M:%S}] {m}", flush=True)

def sh(cmd, timeout=60):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except subprocess.TimeoutExpired:
        return ""

# ---------------- 候选队列（jsonl，单实例常驻，原子读改写）----------------
# 候选格式（每行一个 JSON，codex 写入）：
#   {"id": "...", "desc": "...", "status": "pending",
#    "apply": {"branch": "..."} | {"patch": "..."} | {"env": {...}},   # 怎么套改动
#    "bench": {"type": "microbench|smoke|full", "buckets": ["8-16K"], "accuracy": "smoke|none"},
#    "container": null, "result": null, "claimed_at": null}
def load_candidates() -> list[dict]:
    if not os.path.exists(CANDIDATES): return []
    return [json.loads(l) for l in open(CANDIDATES) if l.strip()]

def save_candidates(cands: list[dict]):
    tmp = CANDIDATES + ".tmp"
    with open(tmp, "w") as f:
        for c in cands:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    os.replace(tmp, CANDIDATES)   # 原子替换，防写一半

# ---------------- 容器选择（读 pool_manager 状态）----------------
def ready_containers() -> list[dict]:
    if not os.path.exists(POOL_STATE): return []
    conts = json.load(open(POOL_STATE)).get("containers", {})
    return [dict(job_id=j, **d) for j, d in conts.items() if d.get("state") == "READY"]

def job_remaining_s(job_id: str) -> int:
    out = sh(f"scontrol show job {job_id} 2>/dev/null")
    m = re.search(r"EndTime=(\S+)", out)
    if not m or m.group(1) in ("Unknown", "N/A"): return -1
    try:
        return int((datetime.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S") - now()).total_seconds())
    except ValueError:
        return -1

def pick_container(cand: dict, busy: set) -> dict | None:
    need = EST_S.get(cand.get("bench", {}).get("type", "smoke"), 20 * 60)
    for c in ready_containers():
        if c["job_id"] in busy:
            continue                                   # 已被别的候选占用
        if job_remaining_s(c["job_id"]) < need:
            continue                                   # DRAINING 边界：剩余寿命不够别派
        return c
    return None

# ---------------- 执行一个候选（核心 seam，Codex 填）----------------
def run_candidate(container: dict, cand: dict) -> str:
    """
    ⚠️ 由 Codex 填：在 container 上执行 cand，返回 experiments/<id>/ 结果目录。步骤（复用现有脚本）：
      1) apply 改动：cand["apply"] 是 branch(git checkout) / patch(git apply) / env(导环境变量)。
      2) 若改了 C++/kernel → 增量编译 wheel（别 clean rebuild）并装；纯 env/microbench 可跳过。
      3) 跑守门员：bench.type → microbench(打 27B 真实 shape) / smoke(每类10条) / full；
         用 guard_bench.py / scnetctl.py，别自己重写测量。
      4) 结果落 experiments/<id>/summary.json，含**相对该容器内 baseline 的 Δ**（不跨容器比绝对值）。
    """
    raise NotImplementedError("Codex: fill apply + build + guard_bench, return experiments/<id>/ — see plans/infra-pool.md")

# ---------------- 主循环 ----------------
def tick():
    cands = load_candidates()
    if not cands:
        return
    dirty = False
    # 1) 恢复超时的 running（executor 曾崩溃/容器被回收）
    for c in cands:
        if c["status"] == "running" and c.get("claimed_at"):
            age = (now() - datetime.datetime.fromisoformat(c["claimed_at"])).total_seconds()
            if age > CLAIM_TIMEOUT_S:
                log(f"候选 {c['id']} running 超时 → 退回 pending")
                c.update(status="pending", container=None, claimed_at=None)
                dirty = True
    # 2) 取一个 pending
    busy = {c.get("container") for c in cands if c["status"] == "running" and c.get("container")}
    pend = next((c for c in cands if c["status"] == "pending"), None)
    if pend is None:
        if dirty: save_candidates(cands)
        return
    # 3) 找容器（DRAINING 边界 + 排除 busy）
    container = pick_container(pend, busy)
    if container is None:
        if dirty: save_candidates(cands)
        return                                         # 没合适容器，等 pool_manager 补/等寿命够
    # 4) 认领 + 执行
    pend.update(status="running", container=container["job_id"], claimed_at=now().isoformat())
    save_candidates(cands)
    log(f"候选 {pend['id']} → 容器 {container['job_id']} 开跑")
    try:
        pend["result"] = run_candidate(container, pend)
        pend["status"] = "done"
        log(f"候选 {pend['id']} done → {pend['result']}")
    except NotImplementedError as e:
        pend.update(status="pending", container=None, claimed_at=None)   # 骨架未实现，别标死
        log(f"run_candidate 未实现，退回 pending：{e}")
    except Exception as e:
        pend.update(status="failed", error=str(e)[:200])
        log(f"候选 {pend['id']} FAILED：{e}")
    save_candidates(cands)

def main():
    log("candidate_executor 启动（登录节点常驻；codex 可随时停工，本执行器继续跑排好的候选）")
    while True:
        try:
            tick()
        except Exception as e:
            log(f"tick 异常：{e}")
        time.sleep(TICK_S)

if __name__ == "__main__":
    main()
