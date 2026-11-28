#!/usr/bin/env bash
set -euo pipefail

data=$(mktemp -d /data/test-install.XXXXXX)
trap 'rm -rf "$data"' EXIT
cd "$data"

curl -fsSL -A "AGNOSSetup-$(cat /VERSION)" -H "X-openpilot-device-type: mici" "$1" -o installer
chmod +x installer

sudo unshare --mount sh -ec '
  mount --bind . /data
  exec runuser --login comma -c /data/installer
'

test -x continue.sh
cd openpilot
test -x ./launch_openpilot.sh
expected_commit=$(git rev-parse --verify "$2^{commit}")
test "$(git rev-parse HEAD)" = "$expected_commit"
git submodule status --recursive | awk '/^[-+U]/ { print; exit 1 }'
git lfs ls-files | awk '$2 == "-" { print "Unresolved LFS pointer:", $0; exit 1 }'
