#!/usr/bin/env bash
# Enable the git-xet transfer agent when available: it uploads through Xet chunks and downloads
# large LFS objects from the Hub's CDN with parallel ranged requests instead of one connection
# per object. Absent binaries are a no-op since ordinary git-lfs behavior is used.
# "install" first fetches the pinned prebuilt binary (checksummed against SHA256SUMS by install.sh).
# Fail-open: a missing or broken git-xet must never fail the LFS pull that follows it.
set -eu

REPO="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"

if [ "$#" -ge 1 ] && [ "$1" = "install" ] && ! command -v git-xet >/dev/null 2>&1; then
  # Fetch into ~/.local/bin; pinned to an immutable release tag, checksum-verified by install.sh.
  if ! curl -fsSL "${GIT_XET_INSTALL_URL:-https://raw.githubusercontent.com/haraschax/xet-core/git-xet-v0.2.2-dev.2/git_xet/install.sh}" | sh; then
    echo "git-xet download failed, using plain git-lfs"
    exit 0
  fi
  PATH="${HOME}/.local/bin:${PATH}"
  export PATH
fi

if ! command -v git-xet >/dev/null 2>&1; then
  echo "git-xet not found, using plain git-lfs"
  exit 0
fi

LFS_URL="$(git config --file "${REPO}/.lfsconfig" lfs.url 2>/dev/null || true)"

if [ -n "$LFS_URL" ] && git-xet install --local --path "$REPO" --lfs-url "$LFS_URL" 2>/dev/null; then
  echo "git-xet standalone transfers enabled"
else
  git-xet install --local --path "$REPO" 2>/dev/null || echo "git-xet install failed, using plain git-lfs"
fi

# git-lfs spawns the agent with PATH inherited from its own process; record an absolute path
# so the agent resolves regardless of the caller's PATH.
if git -C "$REPO" config --local --get lfs.customtransfer.xet.args >/dev/null 2>&1; then
  git -C "$REPO" config --local lfs.customtransfer.xet.path "$(command -v git-xet)"
fi

exit 0