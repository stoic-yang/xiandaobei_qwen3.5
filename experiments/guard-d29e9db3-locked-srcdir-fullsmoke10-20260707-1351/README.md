# guard-d29e9db3-locked-srcdir-fullsmoke10-20260707-1351

## Verdict

**INVALID / guard script bug.** This run switched overlay input to the prepared
local source directory `/root/overlay-d29e9db3-locked`, and the overlay itself
was applied successfully. It then failed before vLLM startup because the remote
guard script's Python heredoc was not dedented.

Root cause: `scripts/guard_bench.py` inserted multiple `--env` exports where
the second generated line was not indented, preventing `textwrap.dedent` from
removing the outer indentation of the generated remote script. The failure was:

```text
guard_remote.sh: line 385: warning: here-document at line 129 delimited by end-of-file (wanted `PY')
IndentationError: unexpected indent
```

Fixed by commit `90459ea` (`fix(guard): preserve heredoc dedent with env exports`).

## Anchors

- Raw driver log: `driver.log`
- Overlay manifest: `overlay_manifest.txt`
- Remote script copy: `guard_remote.sh`
- Remote run dir:
  `/public/home/xdzs2026_c166/codex_runs/guard-d29e9db3-locked-srcdir-fullsmoke10-20260707-1351`
