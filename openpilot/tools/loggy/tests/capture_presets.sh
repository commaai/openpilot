#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGGY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="${LOGGY_CAPTURE_DIR:-/tmp}"
WIDTH=${LOGGY_CAPTURE_WIDTH:-1280}
HEIGHT=${LOGGY_CAPTURE_HEIGHT:-720}
XVFB_BIN="${LOGGY_XVFB_BIN:-/home/batman/openpilot/.venv/bin/Xvfb}"
DISPLAY_ARG=()

cleanup_xvfb() {
  if [[ -n "${LOGGY_CAPTURE_XVFB_PID:-}" ]]; then
    kill "$LOGGY_CAPTURE_XVFB_PID" >/dev/null 2>&1 || true
  fi
}

choose_xvfb_display() {
  if [[ -n "${LOGGY_CAPTURE_DISPLAY:-}" ]]; then
    echo "$LOGGY_CAPTURE_DISPLAY"
    return
  fi

  local n
  for n in $(seq 87 187); do
    if [[ ! -e "/tmp/.X${n}-lock" && ! -S "/tmp/.X11-unix/X${n}" ]]; then
      echo ":$n"
      return
    fi
  done

  echo ":$((200 + RANDOM % 200))"
}

if [[ -x "$XVFB_BIN" ]]; then
  DISPLAY="$(choose_xvfb_display)"
  export DISPLAY
  XVFB_LOG="/tmp/loggy_capture_xvfb_${DISPLAY#:}.log"
  "$XVFB_BIN" "$DISPLAY" -screen 0 "${WIDTH}x${HEIGHT}x24" -nolisten tcp >"$XVFB_LOG" 2>&1 &
  LOGGY_CAPTURE_XVFB_PID=$!
  trap cleanup_xvfb EXIT
  sleep 1
  if ! kill -0 "$LOGGY_CAPTURE_XVFB_PID" >/dev/null 2>&1; then
    echo "Error: Xvfb failed to start on $DISPLAY; see $XVFB_LOG" >&2
    exit 1
  fi
elif command -v xvfb-run >/dev/null 2>&1; then
  DISPLAY_ARG=(xvfb-run -a -s "-screen 0 ${WIDTH}x${HEIGHT}x24")
else
  echo "Error: Xvfb is required for this capture helper" >&2
  exit 127
fi

run_under_display() {
  local binary="$1"
  local out="$2"
  "${DISPLAY_ARG[@]}" "$binary" --demo --width "$WIDTH" --height "$HEIGHT" --output "$out"
}

run_capture() {
  local binary="$1"
  local preset="$2"
  local out="$OUT_DIR/loggy-${preset}-capture.png"

  run_under_display "$LOGGY_DIR/$binary" "$out"

  echo "Captured $preset preset to $out"
}

run_capture loggy_cabana cabana
run_capture loggy_jotpluggler jotpluggler
