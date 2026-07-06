#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGGY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$LOGGY_ROOT"

rg_count() {
  local pattern="$1"
  shift
  local matches
  matches="$(rg -n "$pattern" "$@" 2>/dev/null || true)"
  if [[ -z "$matches" ]]; then
    echo 0
  else
    printf '%s\n' "$matches" | wc -l | tr -d ' '
  fi
}

awk_count() {
  local program="$1"
  shift
  awk "$program" "$@" | tail -1 | tr -d ' '
}

check() {
  local name="$1"
  local count="$2"
  local max="$3"
  if (( count > max )); then
    echo "STYLE: $name grew: $count > $max" >&2
    exit 1
  fi
  printf 'STYLE: %-22s %s <= %s\n' "$name" "$count" "$max"
}

check "error out-params" \
  "$(rg_count 'std::string \*error|std::string\* error|std::string \*error_text|std::string\* error_text' backend panes shell -g'*.h' -g'*.cc')" \
  0
check "null-guard writes" \
  "$(rg_count 'if \([^\n]+ != nullptr\).*\*|if \(error\) \*error' backend panes shell -g'*.h' -g'*.cc')" \
  0
check "std::function" \
  "$(rg_count 'std::function' backend panes shell -g'*.h' -g'*.cc')" \
  0
check "getter pairs" \
  "$(rg_count '\(\) (const )?\{ return [a-z_]+_; \}' backend/session.h)" \
  0
check "pane header fns" \
  "$(awk_count '/^[A-Za-z].*\(/ {n++} END {print n+0}' panes/*.h)" \
  14
# 73->84 (2026-07-05): scan/history job+page types now cross the pane<->backend boundary;
# the REVIEW 2.3 boundary rule outranks the raw count.
# 84->85 (2026-07-06): owner-directed `struct Theme` (shell/theme.h) — every color token as one
# plain aggregate, the single source of truth apply_theme() reads from; it replaces several
# one-off free functions (net token count still goes down) but is itself one more named struct.
check "named header structs" \
  "$(rg_count '^struct [A-Za-z]+' backend panes shell -g'*.h' -g'!generated_*')" \
  85
# LOC history (prior in git): 20605->20950 parity-fix wave; ->21020 ship fixes; ->21124
# red-team data-honesty wave (honest route duration, deprecated series, export isolation);
# ->21368 visual-identity pass (struct Theme, binary per-signal spans + M/L markers, camera/plot
# chrome) + root cleanup; ->21462 (2026-07-06) playhead-semantics wave — Binary/Signal grids
# moved off can_events() copies onto can_event_summary + byte_change_times (tracker-bound,
# O(log n)), History page keyed off a quantized summary count with CSV export split to full
# route (cabana semantics), Browser/Plot legend sampling decoupled from chart zoom, playback
# autostart on route load; ->21481 camera decode keeps publishing when playback outpaces it
# (segment-scoped abort + best-effort stale publish in video.cc). Genuine defect-cluster
# capability, not padding.
check "product LOC" \
  "$(find backend panes shell \( -name '*.cc' -o -name '*.h' \) ! -name 'generated_*' -print0 | xargs -0 cat | wc -l | tr -d ' ')" \
  21481
# 850->852 (2026-07-06): maybe_autostart_playback (cabana/jotpluggler parity: play on load).
check "runtime.cc size" \
  "$(wc -l < shell/runtime.cc | tr -d ' ')" \
  852
check "pane-local statics" \
  "$(rg_count '^\s+static ' panes -g'*.cc')" \
  0
check "backend header camel" \
  "$(rg_count '\b[a-z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*\s*\(' backend -g'*.h' -g'!generated_*')" \
  0
# Theme sweep (2026-07-06, visual-identity pass): shell/theme.{h,cc}, shell/runtime.cc,
# shell/workspace.cc, panes/{binary,camera,plot,browser}.cc are fully on Theme tokens (0
# literals). Outstanding, different owners' files this pass didn't touch: panes/map.cc (16, its
# own hand-drawn map palette), panes/signal.cc (2), panes/messages.cc (2) — baseline reflects
# that honestly rather than claiming 0. Target 0 once those sweeps land.
check "color_rgb literals outside theme.cc" \
  "$(rg_count 'color_rgb\(' panes shell -g'*.cc' -g'!theme.cc')" \
  20
