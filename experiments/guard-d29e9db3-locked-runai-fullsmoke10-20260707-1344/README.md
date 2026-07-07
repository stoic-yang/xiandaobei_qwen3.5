# guard-d29e9db3-locked-runai-fullsmoke10-20260707-1344

## Verdict

**INVALID / aborted before vLLM startup.** This run was the first attempt to
measure d29 with the new locked `--max-model-len 32768` + `runai_streamer`
guard path. It used `--overlay-rev d29e9db3`, which made the remote guard read
source files with `git show` from the shared home-directory repository.

The run was stopped manually after `git show d29e9db3:...qwen3_5.py` stalled on
the shared filesystem. No server was started and no benchmark result exists.

## Useful Evidence

- Confirms that guard hot paths should not use `git show` from
  `/public/home/xdzs2026_c166/vllm_cscc_competition` for overlays.
- Follow-up runs should prepare `/root/overlay-<rev>` once with `git archive`
  and then use `--overlay-source-dir`.

## Anchors

- Raw driver log: `driver.log`
- Poll log: `poll.log`
- Remote run dir:
  `/public/home/xdzs2026_c166/codex_runs/guard-d29e9db3-locked-runai-fullsmoke10-20260707-1344`
