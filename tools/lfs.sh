#!/usr/bin/env bash
set -e

# Installs this checkout's LFS client, then runs it. Setup, updates, and branch
# switches run the copy in the checkout they switch to.
cd "$(dirname "$0")/.."
# the shim lives in the git dir so it works in every worktree and on every
# branch, and survives the venv being rebuilt
shim="$(git rev-parse --git-common-dir)/git-lfs"
cp scripts/git-lfs "$shim"
chmod +x "$shim"
# resolved when git runs it, so copies of the checkout (like the updater's) keep working
filter='"$(git rev-parse --git-common-dir)/git-lfs"'
git config --local filter.lfs.clean "$filter clean -- %f"
git config --local filter.lfs.smudge "$filter smudge -- %f"
git config --local filter.lfs.process "$filter filter-process"
git config --local filter.lfs.required false
hooks="$(git rev-parse --git-path hooks)"
mkdir -p "$hooks"
printf '#!/bin/sh\nexec %s pre-push "$@"\n' "$filter" > "$hooks/pre-push"
chmod +x "$hooks/pre-push"
# stock hooks from "git lfs install" exit 2 without git-lfs on PATH, and we don't use lfs locks
for hook in post-checkout post-commit post-merge; do
  if grep -qs "git lfs $hook" "$hooks/$hook"; then
    rm "$hooks/$hook"
  fi
done
exec "$shim" "$@"
