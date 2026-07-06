#!/usr/bin/env python3
"""
容器池管理器 —— L0：单容器热备预热（warm standby）

目标
----
消除每 4 小时 Slurm wall-clock 到期后 ~12min 的"重建税"空窗。在当前容器
（primary）到期前，提前创建并预热一个接班容器（standby），等它 vllm health
通过、READY 待命；primary 到期时无缝切换 —— 新容器的启动开销被藏在旧容器的
工作期背后（双缓冲）。

运行位置
--------
登录节点（tmux / nohup），例如：
    ssh xiandaobei-login
    tmux new -s pool 'python3 /public/home/xdzs2026_c166/meta/scripts/pool_manager.py'
squeue / scontrol / sbatch 都在登录节点本地可用，且不受本地 mac 睡眠影响。

设计解耦（关键）
----------------
本骨架不依赖"容器怎么创建"。创建被隔离在 create_container() 后面，由 Codex 填：
  - 若网页"创建容器"背后是一条 sbatch（大概率，因为容器就是 Slurm 作业）→
    填 sbatch 命令，全自动。探查清单见 plans/infra-pool.md。
  - 若只能网页/chrome → 在 create_container() 里触发通知让人工创建（降级路径）。

范围
----
L0 = 1 主 + 最多 1 备。并行池（K>1）+ 候选队列调度是 L1，命门通过后再扩。
与 guard_bench.py / scnetctl.py 配合：池负责"永远有热容器"，那两个负责在热容器上
跑测量。本脚本只管容器生命周期，不跑 benchmark。

⚠️ 骨架说明：远程命令的多级引用/转义、docker 网络名、health 返回码判定，均按
实际环境由 Codex 校准；状态机与预热/切换逻辑是设计意图，保持不变。
"""
from __future__ import annotations
import json, subprocess, time, shlex, datetime, os, re
from dataclasses import dataclass, asdict

# ---------------- 配置 ----------------
USER = os.environ.get("USER", "xdzs2026_c166")
DRAIN_THRESHOLD_S = 20 * 60          # primary 剩余寿命 < 20min → 预热 standby
                                     #   ≈ 启动耗时(~12min) + 安全余量(~8min)
HEALTH_PORT = 8001
TICK_S = 60
STATE_FILE = os.path.expanduser("~/pool_state.json")
MODEL_DIR = "/public/home/xdzs2026_c166/Qwen3.5-27B"
WHEEL_GLOB = "/public/home/xdzs2026_c166/vllm_cscc_competition/dist/*.whl"
TESTDATA_DIR = "/public/home/xdzs2026_c166/testdata"   # start_vllm.sh 所在

# 容器状态：CREATING → WARMING → READY → DRAINING → DEAD
@dataclass
class Container:
    job_id: str
    state: str = "CREATING"
    node: str = ""
    ip: str = ""

def now() -> datetime.datetime:
    return datetime.datetime.now()

def log(msg: str) -> None:
    print(f"[{now():%m-%d %H:%M:%S}] {msg}", flush=True)

def sh(cmd: str, timeout: int = 30) -> str:
    """在登录节点本地执行；到计算节点/容器由调用方在 cmd 里显式 ssh <node> / docker exec。"""
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    except subprocess.TimeoutExpired:
        return ""

# ---------------- 原语 ----------------
def list_running_jobs() -> list[str]:
    out = sh(f'squeue -u {USER} -h -t RUNNING -o "%i"')
    return [l.strip() for l in out.splitlines() if l.strip()]

def node_of(job_id: str) -> str:
    return sh(f'squeue -u {USER} -h -j {job_id} -o "%N"')

def job_remaining_s(job_id: str) -> int:
    """scontrol EndTime - now，单位秒；拿不到返回 -1。"""
    out = sh(f"scontrol show job {job_id} 2>/dev/null")
    m = re.search(r"EndTime=(\S+)", out)
    if not m or m.group(1) in ("Unknown", "N/A"):
        return -1
    try:
        end = datetime.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")
        return int((end - now()).total_seconds())
    except ValueError:
        return -1

def container_ip(job_id: str, node: str) -> str:
    """计算节点上 docker inspect 拿容器 IP。容器名格式 <jobid>_<node>（见 memory/20-env.md）。"""
    if not node:
        return ""
    name = f"{job_id}_{node}"
    # NOTE(codex): 网络名/模板按实际调整；曾见 IP 173.0.59.x
    return sh(f"ssh {node} \"docker inspect -f '{{{{.NetworkSettings.IPAddress}}}}' {name}\"").strip()

def check_health(c: Container) -> bool:
    if not (c.node and c.ip):
        return False
    url = f"http://{c.ip}:{HEALTH_PORT}/health"
    code = sh(f"ssh {c.node} 'curl -s -o /dev/null -w \"%{{http_code}}\" -m 5 --noproxy \"*\" {url}'", 20)
    return code.strip() == "200"

def create_container() -> str:
    """
    ⚠️ 命门 —— 由 Codex 填，返回新容器的 Slurm job_id。

    先探明网页"创建容器"背后的 sbatch 等价（见 plans/infra-pool.md 探查清单）：
        job = sh("sbatch --parsable <partition/image/wall/gres 参数> <submit.sh>")
        return job.strip()
    若确实只能网页/chrome，则在此发通知并返回 ""（降级为人工创建）。
    """
    raise NotImplementedError("Codex: fill sbatch (or chrome bridge) — see plans/infra-pool.md")

def warm_container(c: Container) -> None:
    """新容器里装 wheel + 后台起 vllm。复用 start_vllm.sh（见 memory/20-env.md 最小前置）。"""
    name = f"{c.job_id}_{c.node}"
    remote = (
        "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; "
        f"pip install --no-deps {WHEEL_GLOB} >/tmp/pool_pip.log 2>&1; "
        f"export MODEL_DIR={MODEL_DIR}; "
        f"cd {TESTDATA_DIR} && nohup ./start_vllm.sh >/tmp/pool_vllm.log 2>&1 &"
    )
    sh(f"ssh {c.node} \"docker exec {name} bash -lc {shlex.quote(remote)}\"", 120)
    c.state = "WARMING"
    log(f"standby {c.job_id} 开始预热（装 wheel + 起 vllm）")

# ---------------- 状态持久化（防管理器重启失忆）----------------
def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    return {"primary": None, "standby": None}

def save_state(s: dict) -> None:
    json.dump(s, open(STATE_FILE, "w"), indent=2)

def _c(d): return Container(**d) if d else None

# ---------------- 主循环（L0）----------------
def tick(s: dict) -> dict:
    primary = _c(s.get("primary"))
    standby = _c(s.get("standby"))

    # 0) 冷启动/恢复：没有 primary 就把当前 RUNNING 作业认作 primary
    if primary is None:
        jobs = list_running_jobs()
        if jobs:
            primary = Container(job_id=jobs[0], state="READY", node=node_of(jobs[0]))
            log(f"认领 primary = {primary.job_id}")

    # 1) primary 剩余寿命低 且 无备 → 预热 standby
    if primary and standby is None:
        rem = job_remaining_s(primary.job_id)
        if 0 < rem < DRAIN_THRESHOLD_S:
            primary.state = "DRAINING"
            log(f"primary {primary.job_id} 剩 {rem//60}min → 预热接班容器")
            try:
                jid = create_container()          # ⚠️ 命门
                if jid:
                    standby = Container(job_id=jid, state="CREATING")
            except NotImplementedError as e:
                log(f"create_container 未实现，需人工创建：{e}")

    # 2) standby 分到节点 → 预热；health 通过 → READY
    if standby:
        if standby.state == "CREATING" and job_remaining_s(standby.job_id) > 0:
            standby.node = node_of(standby.job_id)
            if standby.node:
                warm_container(standby)
        if standby.state == "WARMING":
            standby.ip = container_ip(standby.job_id, standby.node)
            if check_health(standby):
                standby.state = "READY"
                log(f"standby {standby.job_id} READY 待命")

    # 3) primary 到期 → 备转正，无缝接手
    if primary and job_remaining_s(primary.job_id) <= 0:
        log(f"primary {primary.job_id} 到期")
        if standby and standby.state == "READY":
            log(f"standby {standby.job_id} 转正为 primary（无缝接手）")
            primary, standby = standby, None
        else:
            primary = None   # 没备好，下个 tick 重认（退化为普通重建）

    s["primary"] = asdict(primary) if primary else None
    s["standby"] = asdict(standby) if standby else None
    save_state(s)
    return s

def main():
    log("pool_manager L0 启动（单容器热备预热）")
    s = load_state()
    while True:
        try:
            s = tick(s)
        except Exception as e:
            log(f"tick 异常：{e}")
        time.sleep(TICK_S)

if __name__ == "__main__":
    main()
