#!/usr/bin/env bash
set -euo pipefail

data=$(mktemp -d /data/test-install.XXXXXX)
trap 'rm -rf "$data"' EXIT

curl --fail --show-error --silent --location -o "$data/installer" \
  -A "AGNOSSetup-$(cat /VERSION)" -H "X-openpilot-device-type: mici" "$1"
chmod +x "$data/installer"

# Run like comma.service, with a clean install location and the device's Git config.
sudo systemd-run --scope --slice=app.slice --expand-environment=no unshare --mount sh -ec '
  mount --bind "$1" /data
  cd /
  exec setpriv --reuid=comma --regid=comma --init-groups --reset-env \
    /usr/comma/run_in_app_group.sh bash -lc "exec /data/installer"
' sh "$data"

test -x "$data/continue.sh"
cd "$data/openpilot"
test -x ./launch_openpilot.sh
expected_commit=$(git rev-parse --verify "$2^{commit}")
test "$(git rev-parse HEAD)" = "$expected_commit"
git submodule status --recursive | awk '/^[-+U]/ { print; bad=1 } END { exit bad }'
git lfs ls-files | awk '$2 == "-" { print "Unresolved LFS pointer:", $0; bad=1 } END { exit bad }'
