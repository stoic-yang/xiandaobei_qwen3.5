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