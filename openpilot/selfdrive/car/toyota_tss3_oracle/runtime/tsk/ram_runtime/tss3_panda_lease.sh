#!/usr/bin/env bash
# Cooperative Panda ownership handoff for TSS3 in-car tools.
# Caller must define die(). The matching openpilot pandad.py wrapper handles
# SIGWINCH, releases only its native child, and acknowledges the exact lease.

DIRECT_PANDA_LEASE_PATH="${DIRECT_PANDA_LEASE_PATH:-/tmp/openpilot-pandad-direct-lease}"
DIRECT_PANDA_LEASE_READY_PATH="${DIRECT_PANDA_LEASE_READY_PATH:-/tmp/openpilot-pandad-direct-ready}"
unset DIRECT_PANDA_PAUSED_SUPERVISOR_PID DIRECT_PANDA_LEASE_SUPERVISOR_PID DIRECT_PANDA_LEASE_ID || true
PANDAD_WRAPPER_PID=""
DIRECT_PANDA_LEASE_ID=""

_live_native_pandad_children() {
  local child state out=""
  for child in $(pgrep -x pandad 2>/dev/null || true); do
    state="$(ps -o stat= -p "$child" 2>/dev/null | awk '{print $1}')"
    [[ "$state" == Z* ]] || out+=" $child"
  done
  printf '%s' "$out"
}

_read_exact_file() {
  local path="$1"
  [[ -f "$path" ]] || return 1
  cat "$path" 2>/dev/null
}

_release_direct_panda_lease() {
  local observed live_children="" reacquired=0
  [[ -n "$DIRECT_PANDA_LEASE_ID" ]] || return 0

  observed="$(_read_exact_file "$DIRECT_PANDA_LEASE_PATH" || true)"
  if [[ -n "$observed" && "$observed" != "$DIRECT_PANDA_LEASE_ID" ]]; then
    printf 'f33 Panda lease: refusing to remove a different live lease: %s\n' "$observed" >&2
    return 1
  fi
  rm -f "$DIRECT_PANDA_LEASE_PATH"

  if [[ -n "$PANDAD_WRAPPER_PID" ]] && ! kill -0 "$PANDAD_WRAPPER_PID" 2>/dev/null; then
    printf 'f33 Panda lease: pandad wrapper %s exited before Panda ownership could be restored\n' "$PANDAD_WRAPPER_PID" >&2
    return 1
  fi
  if [[ -n "$PANDAD_WRAPPER_PID" ]]; then
    for _ in $(seq 1 120); do
      if [[ ! -f "$DIRECT_PANDA_LEASE_READY_PATH" ]]; then
        for child in $(pgrep -P "$PANDAD_WRAPPER_PID" -x pandad 2>/dev/null || true); do
          state="$(ps -o stat= -p "$child" 2>/dev/null | awk '{print $1}')"
          if [[ "$state" != Z* ]]; then
            reacquired=1
            break
          fi
        done
        [[ $reacquired -eq 1 ]] && break
      fi
      sleep 0.05
    done
    if [[ $reacquired -ne 1 ]]; then
      live_children="$(_live_native_pandad_children)"
      printf 'f33 Panda lease: pandad wrapper %s did not reacquire Panda; native children:%s\n' \
        "$PANDAD_WRAPPER_PID" "${live_children:- none}" >&2
      return 1
    fi
  fi

  unset DIRECT_PANDA_LEASE_SUPERVISOR_PID DIRECT_PANDA_LEASE_ID
  PANDAD_WRAPPER_PID=""
  DIRECT_PANDA_LEASE_ID=""
  return 0
}

_direct_panda_lease_exit() {
  local rc=$?
  trap - EXIT INT TERM
  if ! _release_direct_panda_lease; then
    [[ $rc -ne 0 ]] || rc=2
  fi
  exit "$rc"
}

quiesce_panda_owner() {
  local expected_wrapper_pid="${1:-}" caught_handoff="${2:-}" wrappers existing existing_pid token tmp ready live_children=""
  if [[ -n "$expected_wrapper_pid" ]]; then
    [[ "$expected_wrapper_pid" =~ ^[0-9]+$ ]] || die "invalid expected pandad supervisor PID: $expected_wrapper_pid"
    kill -0 "$expected_wrapper_pid" 2>/dev/null || die "expected pandad supervisor exited: $expected_wrapper_pid"
    PANDAD_WRAPPER_PID="$expected_wrapper_pid"
  else
    wrappers="$(pgrep -f '^openpilot\.selfdrive\.pandad\.pandad$' 2>/dev/null || true)"
    [[ $(awk 'NF{n++} END{print n+0}' <<<"$wrappers") -le 1 ]] || die "multiple pandad supervisors found: ${wrappers//$'\n'/,}"
    PANDAD_WRAPPER_PID="$(awk 'NF{print; exit}' <<<"$wrappers")"
  fi
  if [[ -z "$PANDAD_WRAPPER_PID" ]] && pgrep -f 'openpilot\.system\.manager\.manager|/openpilot/system/manager/manager\.py|system/manager/manager\.py' >/dev/null 2>&1; then
    die "openpilot manager is running without its always-run pandad supervisor"
  fi

  if [[ -n "$PANDAD_WRAPPER_PID" ]]; then
    existing="$(_read_exact_file "$DIRECT_PANDA_LEASE_PATH" || true)"
    if [[ -n "$existing" ]]; then
      existing_pid="${existing%% *}"
      if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
        die "another direct-Panda lease is active: $existing"
      fi
      rm -f "$DIRECT_PANDA_LEASE_PATH" "$DIRECT_PANDA_LEASE_READY_PATH"
    fi

    token="$(cat /proc/sys/kernel/random/uuid 2>/dev/null || printf '%s-%s-%s' "$$" "$RANDOM" "$(date +%s)")"
    DIRECT_PANDA_LEASE_ID="$$ $token"
    tmp="${DIRECT_PANDA_LEASE_PATH}.f33.$$"
    printf '%s\n' "$DIRECT_PANDA_LEASE_ID" >"$tmp"
    mv -f "$tmp" "$DIRECT_PANDA_LEASE_PATH"
    export DIRECT_PANDA_LEASE_SUPERVISOR_PID="$PANDAD_WRAPPER_PID"
    export DIRECT_PANDA_LEASE_ID

    trap _direct_panda_lease_exit EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    kill -WINCH "$PANDAD_WRAPPER_PID" || die "could not request Panda lease from pandad supervisor $PANDAD_WRAPPER_PID"
    for _ in $(seq 1 200); do
      ready="$(_read_exact_file "$DIRECT_PANDA_LEASE_READY_PATH" || true)"
      [[ "$ready" == "$DIRECT_PANDA_LEASE_ID" ]] && break
      kill -0 "$PANDAD_WRAPPER_PID" 2>/dev/null || die "pandad supervisor exited while granting Panda lease"
      sleep 0.005
    done
    ready="$(_read_exact_file "$DIRECT_PANDA_LEASE_READY_PATH" || true)"
    [[ "$ready" == "$DIRECT_PANDA_LEASE_ID" ]] || die "pandad supervisor did not acknowledge cooperative Panda lease"
  fi

  if [[ "$caught_handoff" != caught-handoff ]]; then
    live_children="$(_live_native_pandad_children)"
    [[ -z "$live_children" ]] || die "Panda still owned by native pandad:${live_children}"
    [[ -z "$(pgrep -x boardd 2>/dev/null || true)" ]] || die "boardd still owns Panda"
  fi
}
