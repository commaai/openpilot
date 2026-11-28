#!/usr/bin/env bash
set -euo pipefail

data=$(mktemp -d /data/test-install.XXXXXX)
trap 'rm -rf "$data"' EXIT

curl --fail --show-error --silent --location -o "$data/installer" \
  -A "AGNOSSetup-$(cat /VERSION)" -H "X-openpilot-device-type: mici" "$1"
chmod +x "$data/installer"

# Give the shipped installer an empty /data and the same user and environment as AGNOS setup.
sudo systemd-run --wait --pipe --collect --uid=comma -p "BindPaths=$data:/data" -p RuntimeMaxSec=18min \
  bash -c 'source /etc/profile && exec /data/installer'

test -x "$data/continue.sh"
cd "$data/openpilot"
test -x ./launch_openpilot.sh
expected_commit=$(git rev-parse --verify "$2^{commit}")
test "$(git rev-parse HEAD)" = "$expected_commit"
git submodule status --recursive | awk '/^[-+U]/ { print; bad=1 } END { exit bad }'
git lfs ls-files | awk '$2 == "-" { print "Unresolved LFS pointer:", $0; bad=1 } END { exit bad }'
