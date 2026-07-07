# xiandaobei race meta — AGENTS.md

This repo is the **single source of truth** for the 先导杯 2026 赛题一
(Qwen3.5-27B vLLM inference optimization on Hygon DCU) project, shared by
Codex / Claude Code / OpenCode and any human teammate.

## Where things live

- `memory/` — long-term, slow-changing facts that no agent should rewrite
  casually. Edit only after cross-session confirmation; append a dated
  changelog line, do not delete old lines.
- `journal/YYYY-MM-DD.md` — append-only timeline of what happened each day.
  Any agent may append a timestamped entry; never edit past entries.
- `experiments/<id>/` — per-experiment account: intent / method / diff /
  raw logs / metrics / verdict. The agent that ran it owns the dir; others
  read-only.
- `snapshots/<id>/` — pointers to reproducible artifacts (git SHA,
  wheel fingerprint, model dir). **No wheels, no model weights** ever
  committed; `.gitignore` drops them.
- `source/` (in repo root, above this `meta/` dir on the laptop) — official
  PDFs, not managed here.

## Canonical remotes

- GitHub: `git@github.com:stoic-yang/xiandaobei_qwen3.5.git`
- Local clone: `/Users/keynary/Code/xiandaobei/meta/`
- Container clone: `/public/home/xdzs2026_c166/meta/`

## Rules for all agents

1. Before claiming project state, `git -C <repo> pull` first.
2. Long-term facts → `memory/`; what happened today → `journal/`; one
   experiment → `experiments/<id>/`. Never mix.
3. `journal/` is append-only; `memory/*.md` changes get a dated changelog
   line at the bottom.
4. One experiment = one dir = one commit `exp: <id> <pass|fail> <metrics>`.
5. Models/wheels stay in the container; only SHAs and paths get committed.
6. If something contradicts memory, trust git history + raw logs in
   `experiments/`, then update `memory/` with a dated correction line.
7. Each agent may still keep its native private memory (e.g.
   `~/.claude/projects/.../memory/`), but it must only point to
   `memory/00-index.md` here, not duplicate facts.

## Project skill

Before remote experiments, benchmarks, container-pool work, or submission
preflight, load `.claude/skills/xiandaobei-operator/SKILL.md` and then
`.claude/skills/xiandaobei-operator/references/fast-iteration.md`. The
non-negotiables are: generated SSH config with ControlMaster; locked
`guard_bench.py --locked-start-script --load-format runai_streamer` when a
rebuilt container lacks `/root/Qwen3.5-27B`; explicit log verification that
`max_seq_len=32768`; long jobs in background/logged form; smoke accuracy for
daily regression and full accuracy only at round close/pre-submit; same-container
A/B with relative delta only; and pool automation only after the `sbatch` command
path is proven.

## Efficient execution (all agents)

Rollout analysis (2026-07-06) found the biggest time sink is the SSH layer:
with no connection reuse, every remote command re-handshakes the 3-hop tunnel,
which is slow and randomly drops (`exit 255: Connection ... closed by remote`),
which then triggers retry loops, per-session re-discovery of connect flags, and
long re-planning. One session alone: 933 commands, 1142 reasoning turns, ~4.4h
of command wall-time, 65 independent `ssh xiandaobei-worker` calls. Follow these
so no agent re-walks that path.

1. **Reuse ONE SSH connection (biggest win).** Multiplex the 3-hop tunnel.
   Put in `~/.ssh/config` (or the `scnetctl.py`-generated config) for the login
   and worker hosts:
   ```
   ControlMaster auto
   ControlPath ~/.ssh/cm-xiandaobei-%C
   ControlPersist 10m
   ```
   or pass `-o ControlMaster=auto -o ControlPath=~/.ssh/cm-xiandaobei-%C -o ControlPersist=10m`
   on every ssh. First call opens the tunnel (~5s); the rest reuse it (<0.5s).
   Never open a fresh `ssh xiandaobei-worker …` per command.
2. **Batch remote steps into one session.** Multiple remote steps go in a single
   heredoc script over one ssh (or `scnetctl.py attach` + one script), not N
   separate ssh calls.
3. **One fixed connect recipe — do not re-discover it.** The working recipe is
   in `memory/20-env.md` + `scnetctl.py` (alias `xiandaobei-worker-auto`). Do
   not burn a session permuting BatchMode / ConnectTimeout / ServerAlive / `-i`.
   Same for env prerequisites (unset proxy, competition wheel not baseline,
   `MODEL_DIR`): read them from memory, don't rediscover.
4. **Validate in batches, not per edit.** Do not `py_compile` / `node --check` /
   `git status` after every micro-change (seen 10–24× on one file). Edit a
   batch, validate once.
5. **This laptop is macOS.** No GNU `timeout` (use `gtimeout`, or a background
   PID + `sleep`+`kill` fallback). New worktrees have no upstream
   (`git push -u` / `git pull origin <branch>`). Read paths from memory instead
   of guessing (dead-path `find`s waste a turn).
6. **Background long jobs; poll.** vLLM start / compile / bench take minutes.
   Launch with `nohup … &` (or the existing `job_ready_hook` / Feishu callback)
   and poll, instead of blocking the session in the foreground.
7. **Stop when the acceptance criterion is met.** Do not re-run repo-check /
   dry-runs to self-reassure (seen 12×). Hit the stated exit criterion, record
   the anchor, move on.

Root cause: fixing rule 1 removes most of rules 2–3 and the retry / re-plan
churn. Fix the connection first.
