#!/usr/bin/env bash
# Enable the git-xet transfer agent when available: it uploads through Xet chunks and downloads
# large LFS objects from the Hub's CDN with parallel ranged requests instead of one connection
# per object. Absent binaries are a no-op since ordinary git-lfs behavior is used.
# "install" first fetches the prebuilt binary (pinned by GIT_XET_RELEASE in install.sh).
# Uploads-only git-xet (upstream huggingface/xet-core before native downloads) still works via
# the plain registration path.
set -e

REPO="$(git rev-parse --show-toplevel)"

if [ "$1" = "install" ] && ! command -v git-xet >/dev/null 2>&1; then
  # CI helper: fetch the pinned prebuilt archive into ~/.local/bin.
  curl -fsSL "${GIT_XET_INSTALL_URL:-https://raw.githubusercontent.com/haraschax/xet-core/git-xet-downloads/git_xet/install.sh}" | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if ! command -v git-xet >/dev/null 2>&1; then
  echo "[-] git-xet not found, using plain git-lfs"
  exit 0
fi

LFS_URL="$(git config --file "${REPO}/.lfsconfig" lfs.url 2>/dev/null || true)"

if [ -n "$LFS_URL" ] && git-xet install --local --lfs-url "$LFS_URL" 2>/dev/null; then
  echo "[-] git-xet standalone transfers enabled"
else
  git-xet install --local
  echo "[-] git-xet installed (uploads only)"
fi
