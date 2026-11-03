#!/usr/bin/env bash
set -e

# Runs this checkout's LFS client. Setup, updates, and branch switches run the
# copy in the checkout they switch to.
cd "$(dirname "$0")/.."
exec ./lfs.py "$@"
