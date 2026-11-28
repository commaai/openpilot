#!/usr/bin/env bash
set -euo pipefail

data=$(mktemp -d /data/test-install.XXXXXX)
trap 'rm -rf "$data"' EXIT

curl --fail --show-error --silent --location -o "$data/installer" \
  -A "AGNOSSetup-$(cat /VERSION)" -H "X-openpilot-device-type: mici" "$1"
chmod +x "$data/installer"

# Run like comma.service, with a clean install location and the device's Git config.
sudo systemd-run --wait --pipe --collect --uid=comma --slice=app.slice -p MemoryMax=infinity -p OOMScoreAdjust=-500 \
  -p "BindPaths=$data:/data" -p RuntimeMaxSec=18min \
  /usr/comma/run_in_app_group.sh bash -c 'source /etc/profile && exec /data/installer'

test -x "$data/continue.sh"
cd "$data/openpilot"
test -x ./launch_openpilot.sh
expected_commit=$(git rev-parse --verify "$2^{commit}")
test "$(git rev-parse HEAD)" = "$expected_commit"
git submodule status --recursive | awk '/^[-+U]/ { print; bad=1 } END { exit bad }'
git lfs ls-files | awk '$2 == "-" { print "Unresolved LFS pointer:", $0; bad=1 } END { exit bad }'
