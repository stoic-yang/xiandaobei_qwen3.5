#!/usr/bin/env python3
"""
容器池管理器 —— 维持型池（maintain K running + B pending）

目标
----
在竞争性集群上"持续保有热容器"，一箭双雕：
  1. 抗抢卡：始终在 Slurm 队列里挂着 B 个 pending 请求，晚高峰有卡释放立刻拿到，
     而不是临时要用才提交、结果排长队。（pending 排队不占卡、不烧机时）
  2. 天然预热：池里始终维持 K 个 running 热容器，某个 4h 到期时本就有别的热容器顶上
     —— 不再需要专门的"到期前预热接班"逻辑。

运行位置：登录节点（tmux/nohup）。squeue/scontrol/sbatch 本地可用，不受 mac 睡眠影响。
    ssh xiandaobei-login
    tmux new -s pool 'POOL_K=1 POOL_B=2 python3 ~/meta/scripts/pool_manager.py'

⚠️ 别囤积（重要）
  - pending 不烧机时 → 放心多挂 B 个当缓冲，这是纯赚。
  - running 烧机时 → 空占 running 是浪费（6 容器 7×24 空转 ≈ 144 机时/天，几天烧光 1000）。
  - 所以 K（维持的 running 数）按负载走：有候选并行筛时调高 POOL_K，平时=1。
    "多排 pending、少空转 running"是既抗抢卡又省机时的姿势。MAX 帽再兜底防塞爆队列。

设计解耦：不依赖"容器怎么创建"。submit_job() 由 Codex 填（命门：能否命令行 sbatch）。
维持型池**强依赖 sbatch** —— chrome 点击没法持续自动排队。见 plans/infra-pool.md。

配合：与 guard_bench.py / scnetctl.py 分工——池只管"永远有热容器"，那两个在热容器上
跑测量。主动释放空闲 running（超 K 时 scancel）需与调度器协调 busy 状态，留 L1，本骨架
不做，以免误杀正在跑候选的容器。

⚠️ 骨架说明：远程命令转义、docker 网络名、health 判定按实际由 Codex 校准；维持逻辑
（补 pending 到 K+B、首次 running 预热、health 转 READY、回收清理）是设计意图，保持不变。
"""
from __future__ import annotations
import json, subprocess, time, shlex, datetime, os, re
from dataclasses import dataclass, asdict

# ---------------- 配置 ----------------
USER = os.environ.get("USER", "xdzs2026_c166")
TARGET_RUNNING = int(os.environ.get("POOL_K", "1"))    # K：维持的 running 数，按负载调（有候选=高，平时=1）
PENDING_BUFFER = int(os.environ.get("POOL_B", "2"))    # B：常驻 pending 缓冲，抗晚高峰抢卡（不烧机时）
MAX_TOTAL      = int(os.environ.get("POOL_MAX", "8"))  # 安全帽：running+pending 上限，防囤积/塞爆队列
HEALTH_PORT = 8001
TICK_S = 60
STATE_FILE = os.path.expanduser("~/pool_state.json")
MODEL_DIR = "/public/home/xdzs2026_c166/Qwen3.5-27B"
WHEEL_GLOB = "/public/home/xdzs2026_c166/vllm_cscc_competition/dist/*.whl"
TESTDATA_DIR = "/public/home/xdzs2026_c166/testdata"
DEFAULT_SUBMIT_CMD = ""
SUBMIT_CMD = os.environ.get("POOL_SUBMIT_CMD", DEFAULT_SUBMIT_CMD)

# 容器状态：WARMING(已 RUNNING，正装 wheel/起 vllm) → READY(health 通过，可派活)
@dataclass
class Container:
    job_id: str
    state: str = "WARMING"
    node: str = ""
    ip: str = ""

def now(): return datetime.datetime.now()
def log(m): print(f"[{now():%m-%d %H:%M:%S}] {m}", flush=True)

def sh(cmd, timeout=30):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except subprocess.TimeoutExpired:
        return ""

# ---------------- 原语（登录节点本地跑 squeue/sbatch；到节点用 ssh <node>）----------------
def classify_jobs():
    """本用户作业按 Slurm 状态分类：(running_ids, pending_ids)。"""
    running = sh(f'squeue -u {USER} -h -t RUNNING -o "%i"').split()
    pending = sh(f'squeue -u {USER} -h -t PENDING -o "%i"').split()
    return running, pending

def node_of(job_id):
    return sh(f'squeue -u {USER} -h -j {job_id} -o "%N"')

def container_ip(job_id, node):
    if not node: return ""
    name = f"{job_id}_{node}"
    # NOTE(codex): docker 网络名/模板按实际调整；曾见 IP 173.0.59.x
    return sh(f"ssh {node} \"docker inspect -f '{{{{.NetworkSettings.IPAddress}}}}' {name}\"").strip()

def check_health(c: Container):
    if not (c.node and c.ip): return False
    url = f"http://{c.ip}:{HEALTH_PORT}/health"
    code = sh(f"ssh {c.node} 'curl -s -o /dev/null -w \"%{{http_code}}\" -m 5 --noproxy \"*\" {url}'", 20)
    return code.strip() == "200"

def submit_job() -> str:
    """
    提交一个容器作业进 Slurm 队列，返回 job_id（失败返回 ""）。

    2026-07-07 复核：旧 SubmitLine 直接 sbatch 会生成短命作业但不一定创建
    容器（656380 15 秒退出，缺 _dockerlist），不能作为默认自动建池入口。

    只有当 POOL_SUBMIT_CMD 指向经过实测的可复用命令（提交后 RUNNING 且
    scnetctl attach 成功）时才自动提交；否则退回 Chrome 启动。
    """
    if not SUBMIT_CMD:
        log("POOL_SUBMIT_CMD 未配置为已验证命令；需要 Chrome/平台启动容器")
        return ""
    out = sh(SUBMIT_CMD, 30)
    m = re.search(r"\b(\d+)\b", out)
    if m:
        return m.group(1)
    log(f"submit_job 未拿到 job_id：cmd={SUBMIT_CMD!r} output={out!r}")
    return ""

def warm_container(c: Container):
    """新 RUNNING 容器里装 wheel + 后台起 vllm（每容器首次一次）。"""
    name = f"{c.job_id}_{c.node}"
    remote = (
        "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; "
        f"pip install --no-deps {WHEEL_GLOB} >/tmp/pool_pip.log 2>&1; "
        f"export MODEL_DIR={MODEL_DIR}; "
        f"cd {TESTDATA_DIR} && nohup ./start_vllm.sh >/tmp/pool_vllm.log 2>&1 &"
    )
    sh(f"ssh {c.node} \"docker exec {name} bash -lc {shlex.quote(remote)}\"", 120)
    log(f"{c.job_id} 开始预热（装 wheel + 起 vllm）")

# ---------------- 状态持久化（防管理器重启失忆）----------------
def load_state():
    if os.path.exists(STATE_FILE): return json.load(open(STATE_FILE))
    return {"containers": {}}   # job_id -> Container dict

def save_state(s): json.dump(s, open(STATE_FILE, "w"), indent=2)

# ---------------- 维持循环 ----------------
def tick(s):
    running, pending = classify_jobs()
    conts = s["containers"]

    # 1) 持续补齐到 K+B（一直排队，抗抢卡）；MAX 帽防囤积/塞爆队列
    target_total = min(TARGET_RUNNING + PENDING_BUFFER, MAX_TOTAL)
    deficit = target_total - (len(running) + len(pending))
    for _ in range(max(0, deficit)):
        try:
            jid = submit_job()          # ⚠️ 命门
            if jid: log(f"提交作业 {jid} 进队列（排队抗抢卡）")
        except NotImplementedError as e:
            log(f"submit_job 未实现，需人工创建：{e}"); break

    # 2) 新 RUNNING 容器：首次预热一次
    for jid in running:
        if jid not in conts:
            c = Container(jid, "WARMING", node_of(jid))
            warm_container(c)
            conts[jid] = asdict(c)

    # 3) WARMING → health 通过 → READY
    for jid, d in list(conts.items()):
        c = Container(**d)
        if c.state == "WARMING":
            c.ip = c.ip or container_ip(jid, c.node)
            if check_health(c):
                c.state = "READY"; log(f"{jid} READY（可派活）")
            conts[jid] = asdict(c)

    # 4) 已回收（到期/消失）的容器：移出池记录
    alive = set(running) | set(pending)
    for jid in [j for j in conts if j not in alive]:
        log(f"{jid} 已回收，移出池"); del conts[jid]

    ready = sum(1 for d in conts.values() if d["state"] == "READY")
    log(f"池：running={len(running)} pending={len(pending)} ready={ready} (K={TARGET_RUNNING} B={PENDING_BUFFER})")
    save_state(s)
    return s

def main():
    log(f"pool_manager 维持型启动（维持 K={TARGET_RUNNING} running + B={PENDING_BUFFER} pending，MAX={MAX_TOTAL}）")
    s = load_state()
    while True:
        try: s = tick(s)
        except Exception as e: log(f"tick 异常：{e}")
        time.sleep(TICK_S)

if __name__ == "__main__":
    main()
