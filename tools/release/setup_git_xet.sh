#!/usr/bin/env bash
# Configure git-xet for this repository so LFS downloads use parallel CDN requests instead of one
# connection per object. With "install", first fetch the pinned prebuilt linux binary from
# https://github.com/haraschax/xet-core/releases/tag/git-xet-v0.2.2-dev.2 and verify its sha-256.
# Every failure path logs and exits 0: a missing or broken git-xet must never fail the LFS pull
# that follows it.
set -eu

REPO="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
LFS_URL="$(git config --file "${REPO}/.lfsconfig" lfs.url 2>/dev/null || true)"

if [ "${1:-}" = "install" ] && ! command -v git-xet >/dev/null 2>&1; then
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)
      arch=x86_64; sha256=55fbd4f5a5db5185349b8834df787ff6b58ab7000044accf50f04749abef4d24 ;;
    Linux-aarch64|Linux-arm64)
      arch=aarch64; sha256=b04ea5b6a6a6567c2b4dbc518f2b1b8263a41c7398a96c6faf197a9d2d0c018c ;;
    *)
      echo "no git-xet build for $(uname -s)/$(uname -m), using plain git-lfs"; exit 0 ;;
  esac
  tmp="$(mktemp -d)" || exit 0
  if curl -fsSL "https://github.com/haraschax/xet-core/releases/download/git-xet-v0.2.2-dev.2/git-xet-linux-${arch}.tar.gz" -o "$tmp/git-xet.tgz" \
    && printf '%s  %s\n' "$sha256" "$tmp/git-xet.tgz" | sha256sum -c - >/dev/null 2>&1 \
    && tar -xzf "$tmp/git-xet.tgz" -C "$tmp" git-xet \
    && mkdir -p "${HOME}/.local/bin" \
    && install -m 755 "$tmp/git-xet" "${HOME}/.local/bin/git-xet"; then
    PATH="${HOME}/.local/bin:${PATH}"
    export PATH
  fi
  rm -rf "$tmp"
fi

if ! command -v git-xet >/dev/null 2>&1; then
  echo "git-xet not found, using plain git-lfs"
  exit 0
fi

if [ -n "$LFS_URL" ] && git-xet install --local --path "$REPO" --lfs-url "$LFS_URL" 2>/dev/null; then
  echo "git-xet standalone transfers enabled"
else
  git-xet install --local --path "$REPO" 2>/dev/null || echo "git-xet install failed, using plain git-lfs"
fi

# git-lfs spawns the agent with its own PATH; record an absolute path so it resolves anywhere.
# Only meaningful once an install wrote the full key set (never leave "path" without "args").
if git -C "$REPO" config --local --get lfs.customtransfer.xet.args >/dev/null 2>&1; then
  git -C "$REPO" config --local lfs.customtransfer.xet.path "$(command -v git-xet)"
fi

exit 0