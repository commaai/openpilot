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
# 14->15 (2026-07-06): draw_plot_context_menu — the pane context-menu hook (PaneType
# draw_context_menu) is the seam that keeps plot menu items out of runtime.cc.
check "pane header fns" \
  "$(awk_count '/^[A-Za-z].*\(/ {n++} END {print n+0}' panes/*.h)" \
  17
# 73->84 (2026-07-05): scan/history job+page types now cross the pane<->backend boundary;
# the REVIEW 2.3 boundary rule outranks the raw count.
# 84->85 (2026-07-06): owner-directed `struct Theme` (shell/theme.h) — every color token as one
# plain aggregate, the single source of truth apply_theme() reads from; it replaces several
# one-off free functions (net token count still goes down) but is itself one more named struct.
check "named header structs" \
  "$(rg_count '^struct [A-Za-z]+' backend panes shell -g'*.h' -g'!generated_*')" \
  86
# LOC history (prior in git): 20605->20950 parity-fix wave; ->21020 ship fixes; ->21124
# red-team data-honesty wave (honest route duration, deprecated series, export isolation);
# ->21368 visual-identity pass (struct Theme, binary per-signal spans + M/L markers, camera/plot
# chrome) + root cleanup; ->21462 (2026-07-06) playhead-semantics wave — Binary/Signal grids
# moved off can_events() copies onto can_event_summary + byte_change_times (tracker-bound,
# O(log n)), History page keyed off a quantized summary count with CSV export split to full
# route (cabana semantics), Browser/Plot legend sampling decoupled from chart zoom, playback
# autostart on route load; ->21481 camera decode keeps publishing when playback outpaces it
# (segment-scoped abort + best-effort stale publish in video.cc); ->21619 interaction
# defect batch — binary edge-vs-create drag semantics, plot empty-series persistence + drop
# hint, workspace splitter drag (previously absent entirely), resize-status undo staleness,
# map failure banner, signal pane sizing that could drop BeginTable and the editor below it;
# ->21669 theme-token completion — map carto palette gets Theme tokens with real light values
# (map was staying dark in the light theme), color_rgb() deleted from the public API; ->21672
# History pager shares the export row (default preset showed headers and zero rows); ->21695
# red-team round 2 — decode abort consumed once at dequeue (lost cross-segment aborts), UI-thread
# stall in set_camera_index, stale-fill floor after cache-hit seeks, active_key preserved on
# best-effort fills, timeline None barriers survive incremental re-merge; ->21818 round-2
# survivors — table/plot-selection theme tokens, zoom seek-on-release + one-undo-per-gesture,
# browser uncapped via skeleton cache, camera invalidate_displayed after pane recreate, Ctrl+Z,
# resize keeps min/max, autostart yields to user pause, sparse-series sample-hold, History key
# quantized to 4 Hz while playing, map cold-cache is a miss not a failure, start_ string leaks;
# ->22098 (2026-07-06) owner UX round — load-phase frame pacing (budgeted store drain, sub-batch
# staging, camera index cadence + incremental decoder update), camera cover-crop when Fit off,
# jotpluggler parity (empty-pane type picker, full split/change-type/close context menu, tab
# close/rename/duplicate/context menu, browser double-click-to-plot), Ctrl+Z = DBC undo like Qt
# cabana, editor buffers keyed on DBC generation, wheel-zoom history coalescing, History tail
# settle, abort raised under the lock; ->22108 plot input map matches jotpluggler (left-drag
# pan, right-drag box zoom, right-click menu) with the pane popup opened from the pane scope;
# ->22114 LOGGY_AUTOSAVE_DIR override so tests/parallel sessions isolate workspace drafts;
# ->22225 (2026-07-06) never-drop-frames round 2, measured via the new LOGGY_FRAME_TRACE:
# CAN storage becomes lazy-merged chunk lists (mergeEvents re-copied an id's whole history per
# drained sub-batch — 20 ms+ per frame late in load), camera index entries drop their per-entry
# path string (~19k allocs x4 views per rebuild), logs use sort+inplace_merge of the incoming
# slice, pane close X, split creates an empty plot, jotpluggler 10-color curve palette;
# ->22266 opus behavior-diff round 1 — input map corrected to the references' actual layout
# (left-drag box-zoom, right-drag pan; the earlier "jot pans with left" claim was a subagent
# misread of jot's source), async auto-DBC parse (DBCManager::adopt), white plot canvas, amber
# selection box, jot legend chrome + SE mouse readout (custom hover tooltip dropped), 2.25 line
# weight, jot grid/tracker colors, cabana 36px binary cells, 64-chunk staging sub-batches;
# ->22308 plot controls into the right-click menu via the PaneType draw_context_menu hook
# (toolbar + chip rows gone, jot look), drain carryover buffer, camera rebuild staggered off
# drain frames, settings modal fixed field widths (AlwaysAutoResize shrink loop).
# Genuine defect-cluster capability, not padding.
# ->22576 (2026-07-06) manual-review batch — camera crop default + overlay removal, jotpluggler
# browser (density, special sources, layout selector, edge drop zones), decoded-CAN-series with
# per-signal plot/delete buttons, binary history strip, corner close X, Ctrl+Z undo fallback,
# peak-preserving decimation + RangeFit, cursor-follow scroll, parallel shutdown. pane header
# fns 15->17 (browser/plot special-item + added-series helpers), structs 85->86 (SpecialItemPane),
# runtime.cc 1074->1217 (pane drop zones + layout apply).
# ->22845 (2026-07-06) enum state-block plots (deferred manual-review #6): cereal enum names
# plumbed through ingest (route drain_enum_metadata) into a session enum registry, DBC value
# descriptions attached in materialize_decoded_signal, and jotpluggler's StateBlock renderer
# (build_state_blocks/state_block_color/state_block_label/draw_state_blocks) ported into plot.cc.
# A whole capability (named-state lanes), not padding.
check "product LOC" \
  "$(find backend panes shell \( -name '*.cc' -o -name '*.h' \) ! -name 'generated_*' -print0 | xargs -0 cat | wc -l | tr -d ' ')" \
  22845
# 850->852 (2026-07-06): maybe_autostart_playback (cabana/jotpluggler parity: play on load).
# 852->912 (2026-07-06): splitter drag (apply_splitter_delta/draw_split_handle) — the workspace
# tree had no interactive divider between siblings at all; the whole feature, not padding.
# ->930 (2026-07-06): undo/redo_workspace helpers shared by menu + new Ctrl+Z/Ctrl+Shift+Z
# hotkeys, and the autostart-yields-to-user-pause guard.
# ->1041 (2026-07-06): tab close/rename/duplicate/context menu + full pane context menu (the
# jotpluggler-parity round). If runtime.cc keeps growing, split the workspace chrome (tab bar,
# pane menus, splitters) into shell/workspace_ui.cc — same seam as the earlier A8 split.
# ->1074 (2026-07-06): pane close X + LOGGY_FRAME_TRACE stage attribution. The workspace_ui.cc
# split threshold from the previous note stands.
check "runtime.cc size" \
  "$(wc -l < shell/runtime.cc | tr -d ' ')" \
  1217
check "pane-local statics" \
  "$(rg_count '^\s+static ' panes -g'*.cc')" \
  0
check "backend header camel" \
  "$(rg_count '\b[a-z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*\s*\(' backend -g'*.h' -g'!generated_*')" \
  0
# Theme sweep complete (2026-07-06): every pane/shell color reads a Theme token; the map's carto
# palette (roads/water/marker) got its own token group with real light values, so the map no
# longer stays dark in the light theme. color_rgb() is deleted from theme.h — this check now
# trips on any attempt to reintroduce a literal-color helper in a pane.
check "color_rgb literals outside theme.cc" \
  "$(rg_count 'color_rgb\(' panes shell -g'*.cc' -g'!theme.cc')" \
  0
