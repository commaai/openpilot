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
check "named header structs" \
  "$(rg_count '^struct [A-Za-z]+' backend panes shell -g'*.h' -g'!generated_*')" \
  84
# 21045->21124 (2026-07-06): red-team wave — honest route duration, deprecated series
# extraction, export/selection isolation. Prior history in git.
check "product LOC" \
  "$(find backend panes shell \( -name '*.cc' -o -name '*.h' \) ! -name 'generated_*' -print0 | xargs -0 cat | wc -l | tr -d ' ')" \
  21124
# generation-counter discriminator test, constraint comments. Prior history:
# 20605->20950 (2026-07-06): parity-fix wave (all five batches) — live messages heatmap,
# camera seek guard, find-bits stats, DBC editing protection incl. BE drag-resize, preset
# rebalance + polish. Audit-driven capability, not padding.
check "product LOC" \
  "$(find backend panes shell \( -name '*.cc' -o -name '*.h' \) ! -name 'generated_*' -print0 | xargs -0 cat | wc -l | tr -d ' ')" \
<<<<<<< HEAD
  21045
=======
  21111
>>>>>>> 3455389f7 (loggy: honest route duration, deprecated series, export correctness)
check "runtime.cc size" \
  "$(wc -l < shell/runtime.cc | tr -d ' ')" \
  850
check "pane-local statics" \
  "$(rg_count '^\s+static ' panes -g'*.cc')" \
  0
check "backend header camel" \
  "$(rg_count '\b[a-z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*\s*\(' backend -g'*.h' -g'!generated_*')" \
  0
