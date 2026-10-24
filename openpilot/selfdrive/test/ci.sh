#!/usr/bin/env bash
set -eo pipefail
source ~/.bash_profile
source /etc/profile
set -euo pipefail

UP_LIB='git+https://github.com/commaai/hardware@e02dc9f#subdirectory=ultimate_provisioning'
dut_ip=10.254.254.2
up() { uvx -q --from "$UP_LIB" ultimate_provisioning "$@"; }
phone_ssh() {
  ssh -A -T -oBatchMode=yes -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null \
      -oConnectTimeout=10 -oServerAliveInterval=5 -oServerAliveCountMax=3 "comma@$dut_ip" "$@"
}
cleanup() {
  local rc=$?
  trap - EXIT
  set +e
  if ! phone_ssh 'timeout 60 flock /tmp/alfred.lock true'; then rc=1; up off; fi
  if [[ -n ${can_pid:-} ]]; then
    kill -TERM -- "-$can_pid" 2>/dev/null
    for _ in $(seq 20); do kill -0 "$can_pid" 2>/dev/null || break; sleep 0.1; done
    kill -KILL -- "-$can_pid" 2>/dev/null
    wait "$can_pid" 2>/dev/null
  fi
  up release || rc=$?
  exit "$rc"
}
trap 'exit 130' INT
trap 'exit 143' HUP TERM
trap 'cleanup >/tmp/alfred-openpilot-cleanup.log 2>&1' EXIT

supervisor="trap '' PIPE; trap ':' TERM; trap 'trap - HUP; kill -TERM 0; wait \"\$child\"; exit 143' HUP; (trap '' HUP; exec bash -se) <&0 & child=\$!; wait \"\$child\""
phone() {
  local seconds=$1
  {
    cat <<'PROFILE'
set -eo pipefail
trap 'exit 143' TERM
source ~/.bash_profile
source /etc/profile
PROFILE
    local name
    for name in TEST_DIR GIT_BRANCH GIT_COMMIT AZURE_TOKEN CI_ARTIFACTS_TOKEN GITHUB_COMMENTS_TOKEN; do
      printf 'export %s=%q\n' "$name" "${!name}"
    done
    cat <<'PROFILE'
export CI=1 TERM=xterm-256color PYTHONWARNINGS=error PYTHONFAULTHANDLER=1
export COMMA_CACHE=/data/tmp/comma_download_cache PYTEST_ADDOPTS='-n0 -s'
export GIT_SSH_COMMAND='ssh -i /data/gitkey'
if [ -f /data/openpilot/launch_env.sh ]; then source /data/openpilot/launch_env.sh; fi
export LD_LIBRARY_PATH="$(python -c 'import ffmpeg; print(ffmpeg.LIB_DIR)'):/usr/local/lib:${LD_LIBRARY_PATH:-}"
ln -snf "$TEST_DIR" /data/pythonpath
PROFILE
    cat
  } | phone_ssh "exec setpriv --pdeathsig HUP timeout --signal=HUP --kill-after=10 $(printf %q "$seconds") flock --no-fork /tmp/alfred.lock bash -c $(printf %q "$supervisor")"
}

setup() {
  local attempt
  for attempt in 1 2 3; do
    if up ready && {
      cat <<'DUT'
if [ -f /AGNOS ]; then
  rm -rf /tmp/tmp* ~/.commacache /dev/shm/* /dev/tmp/tmp*
  if ! systemctl is-active --quiet systemd-resolved; then
    sudo systemctl start systemd-resolved
    sleep 3
  fi
fi
DUT
      cat openpilot/selfdrive/test/setup_device_ci.sh
    } | phone 2100; then return; fi
    echo "Device setup attempt $attempt failed" >&2
  done
  return 1
}

start_can() {
  local uv
  uv=$(type -ap uv | grep -vF /usr/comma/shims | head -n1)
  PYTHONUNBUFFERED=1 setsid "$uv" run -q --no-project --active --python python3 --with "$UP_LIB" \
    env PYTHONPATH="$PWD:$PWD/opendbc_repo" python3 openpilot/tools/replay/can_replay.py --up >/tmp/can_replay.log 2>&1 &
  can_pid=$!
  for _ in $(seq 120); do
    if ! kill -0 "$can_pid" 2>/dev/null; then cat /tmp/can_replay.log; return 1; fi
    if grep -q carFingerprint /tmp/can_replay.log; then
      sleep 2
      if kill -0 "$can_pid" 2>/dev/null; then return; fi
      cat /tmp/can_replay.log
      return 1
    fi
    sleep 1
  done
  cat /tmp/can_replay.log
  return 1
}

case $1 in
  setup) setup ;;
  run) printf 'cd %q\n%s\n' "$TEST_DIR" "$3" | phone "$2" ;;
  can) start_can; printf 'cd %q\n%s\n' "$TEST_DIR" "$3" | phone "$2" ;;
  *) echo "Unknown CI action: $1" >&2; exit 1 ;;
esac
