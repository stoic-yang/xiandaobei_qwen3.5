#!/usr/bin/env bash
# Sync the `meta` git repo between laptop and SCNet container without
# requiring the container to have outbound internet.
#
# Uses `git bundle` over scp — pure git data exchange, no proxy changes,
# zero impact on teammates.
#
# Usage (run on the laptop):
#   ./scripts/sync-meta.sh push     # laptop → container (default)
#   ./scripts/sync-meta.sh pull     # container → laptop, then push to GitHub
#   ./scripts/sync-meta.sh both    # push then pull
#
# Env:
#   META_LOCAL   default: /Users/keynary/Code/xiandaobei/meta
#   META_REMOTE  default: /public/home/xdzs2026_c166/meta
#   SSH_ALIAS    default: xiandaobei-worker

set -euo pipefail

ACTION="${1:-push}"
META_LOCAL="${META_LOCAL:-/Users/keynary/Code/xiandaobei/meta}"
META_REMOTE="${META_REMOTE:-/public/home/xdzs2026_c166/meta}"
SSH_ALIAS="${SSH_ALIAS:-xiandaobei-worker}"
BUNDLE_NAME="meta-bundle-$(date +%Y%m%d-%H%M%S).bundle"

log() { printf '\033[1;34m[sync-meta]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[sync-meta] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[ -d "$META_LOCAL/.git" ] || fail "META_LOCAL=$META_LOCAL not a git repo"
[ -n "$ACTION" ] || fail "no action"

push_to_container() {
  log "laptop → container ($BUNDLE_NAME)"
  cd "$META_LOCAL"
  git fetch origin --quiet 2>/dev/null || true
  git bundle create "/tmp/$BUNDLE_NAME" --all 2>&1 | tail -1
  scp -o ConnectTimeout=20 "/tmp/$BUNDLE_NAME" "$SSH_ALIAS:$META_REMOTE.bundle" >/dev/null
  ssh -o ConnectTimeout=20 "$SSH_ALIAS" "cd $META_REMOTE \
    && git config --global --add safe.directory $META_REMOTE 2>/dev/null || true \
    && git fetch $META_REMOTE.bundle 2>&1 | tail -3 \
    && git merge --ff-only FETCH_HEAD 2>&1 | tail -3 \
    && rm -f $META_REMOTE.bundle \
    && git log --oneline | head -3"
  rm -f "/tmp/$BUNDLE_NAME"
}

pull_from_container() {
  log "container → laptop ($BUNDLE_NAME)"
  ssh -o ConnectTimeout=20 "$SSH_ALIAS" "cd $META_REMOTE \
    && git bundle create $META_REMOTE.bundle --all 2>&1 | tail -1"
  scp -o ConnectTimeout=20 "$SSH_ALIAS:$META_REMOTE.bundle" "/tmp/$BUNDLE_NAME" >/dev/null
  cd "$META_LOCAL"
  git fetch "/tmp/$BUNDLE_NAME" 2>&1 | tail -2
  git merge --ff-only FETCH_HEAD 2>&1 | tail -2
  rm -f "/tmp/$BUNDLE_NAME"
  ssh -o ConnectTimeout=20 "$SSH_ALIAS" "rm -f $META_REMOTE.bundle"
  log "pushing to GitHub origin"
  git push origin main 2>&1 | tail -5
}

case "$ACTION" in
  push) push_to_container ;;
  pull) pull_from_container ;;
  both) push_to_container; pull_from_container ;;
  *) fail "unknown action: $ACTION (use push|pull|both)" ;;
esac

log "done."