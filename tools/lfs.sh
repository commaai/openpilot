#!/usr/bin/env bash
set -e

cd "$(git rev-parse --show-toplevel)"
shim="$(git rev-parse --path-format=absolute --git-common-dir)/git-lfs"
if [[ "$1" == "install" || "$1" == "pull" ]]; then
  cp scripts/git-lfs "$shim"
  chmod +x "$shim"
  filter='"$(git rev-parse --git-common-dir)/git-lfs"'
  git config --local filter.lfs.clean "$filter clean -- %f"
  git config --local filter.lfs.smudge "$filter smudge -- %f"
  git config --local filter.lfs.process "$filter filter-process"
  git config --local filter.lfs.required false
  hooks="$(git rev-parse --git-path hooks)"
  printf '#!/bin/sh\nexec "$(git rev-parse --git-common-dir)/git-lfs" pre-push "$@"\n' > "$hooks/pre-push"
  chmod +x "$hooks/pre-push"
  # Stock hooks require git-lfs on PATH, and we don't use LFS locks.
  for hook in post-checkout post-commit post-merge; do
    if grep -qs "git lfs $hook" "$hooks/$hook"; then
      rm "$hooks/$hook"
    fi
  done
fi
if [[ "$1" != "install" ]]; then
  "$shim" "$@"
fi
