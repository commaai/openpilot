#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

if ! command -v "mull-runner-17" > /dev/null 2>&1; then
  sudo apt-get update && sudo apt-get install -y curl clang-17
  curl -1sLf 'https://dl.cloudsmith.io/public/mull-project/mull-stable/setup.deb.sh' | sudo -E bash
  sudo apt-get update && sudo apt-get install -y mull-17
fi
