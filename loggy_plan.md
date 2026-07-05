# Loggy Master Plan — from nothing to 100% done

You are the **root agent** for building `loggy`. You have zero prior context; this document is
your complete brief. Read it fully before writing any code. The design background (a Q&A with
Adeeb, the owner) is in `loggy.md` at the repo root — read that second.

## 0. Mission

Build **one tool, `openpilot/tools/loggy/`**, that replaces both Cabana (CAN reverse-engineering
GUI) and Jotpluggler (cereal-log plotting GUI) with a single Dear ImGui app on a shared backend.
"Cabana" and "Jotpluggler" survive as **preset workspaces** and thin launcher scripts — not modes,
not walls: any pane type can live in any workspace. No feature from either tool may be dropped.
The old tools stay untouched and working as references throughout.

Your personal job is **integration and quality**: you spawn subagents to build the pieces (task
specs below), you review and merge their work, you keep the code beautiful and small, and you
verify — with your own eyes, via the GUI harness in §5 — that loggy is fast, correct, and never
drops a frame. You do not relitigate the decisions in §2. Escalate to Adeeb only for genuine
product forks; otherwise decide and move.

## 1. Where everything lives

- Repo root contains `openpilot/` (nested June-2026 restructure): tools are at `openpilot/tools/*`.
- This branch: `jot_cabana` (master-based, has Qt cabana + C++ jotpluggler).
- Branch `imgui_cabana_fable`: complete Qt→ImGui port of cabana. Contains **`cabana_core`**
  (`openpilot/tools/cabana/{dbc,streams,utils}/ + commands.cc + settings.cc`, ~6.9k lines,
  Qt-free) — the primary copy source. If `/tmp/worktrees/tmp.8N4A5FfOCP/imgui_cabana_fable/`
  exists use it, else `git worktree add /tmp/loggy_ref imgui_cabana_fable`. Read its
  `openpilot/tools/cabana/MIGRATION.md`.
- Reference binaries (prebuilt): Qt cabana `/home/batman/openpilot/openpilot/tools/cabana/_cabana`,
  jotpluggler `<imgui_cabana_fable worktree>/openpilot/tools/jotpluggler/jotpluggler`.
- Source trees to mine: `openpilot/tools/jotpluggler/` (this branch, ~13.8k lines) and the two
  cabana trees. Approx map of jotpluggler: `app.h` = master types (`RouteData`, `AppSession`,
  `SketchLayout`), `sketch_layout.cc` = route load + extraction engine, `runtime.cc` = GLFW/ImGui
  runtime + async loaders + camera decoder, `plot.cc`, `map.cc`, `logs.cc`, `browser.cc`,
  `custom_series.cc`, `generate_event_extractors.py` (codegen: cereal schema → series extractors).
- Demo route `a2a0ccea32023010`-family (`5beb9b58bd12b691|0000010a--a51155e496` resolves via
  `--demo`); network works, downloads cache.
- Build: SCons from repo root, e.g. `scons -j$(nproc) openpilot/tools/jotpluggler` — copy that
  pattern for `openpilot/tools/loggy`.

## 2. Decisions already made — do not reopen

1. One binary `loggy`; presets `cabana` and `jotpluggler` (layout JSON); thin launcher scripts
   with the old names. Old CLI feel preserved (`--demo`, route arg, `--stream`, etc.).
2. Clean new code in `tools/loggy/`, but **copy freely** from cabana_core, the imgui cabana
   panels, and jotpluggler — then clean what you copy. "Beautiful" outranks "from scratch".
3. **One series store, one namespace.** DBC-decoded CAN signals are paths beside cereal fields
   (e.g. `/can/0/0x77/SIG` beside `/carState/vEgo`). Every pane reads the same store. DBC edits
   invalidate and re-derive affected series.
4. Backend built on `tools/replay`'s **`LogReader` + `FrameReader` + `PyDownloader`** with a
   loggy-owned **segment scheduler** — NOT the `Replay` class.
5. Store API is on-demand (panes request range views each frame); v1 implementation eagerly
   decodes per segment. Memory windowing/compression deferred behind this seam — design the API
   so it can arrive later without touching panes.
6. Cabana's DBC subsystem (`dbc.cc/dbcfile.cc/dbcmanager.cc` + Catch2 tests from cabana_core) is
   canonical. Jotpluggler's `dbc.h` regex parser dies. DBC *editing UI* ships in the cabana
   preset only, but the panes exist for all.
7. Time model: **view x-range and playback tracker are decoupled** (jotpluggler style). Cabana's
   zoom-drives-seek behavior is expressed on top where needed.
8. One plotting implementation (ImPlot) serves both "charts" and "plots".
9. Computed series (Python custom series, derivative, scale/offset) are a **backend** series
   type — plottable and exportable everywhere. Keep the `math_eval.py` subprocess model.
10. Camera/video: `FrameReader` direct decode for routes; VisionIPC only for live streams.
11. All live sources (device msgq/zmq, panda USB, socketcan) available regardless of preset.
12. Clean new formats for layouts/settings (json11 / layout JSON); no migration from Qt-era state.
13. Loading prioritizes **time-to-first-use**: metadata, timeline, and the first/visible segment
    interactive ASAP; rest fills in background; seeks reprioritize immediately.
14. LOC discipline: final tree must land well under the ~25k sum of the two old tools. Target
    ≤ ~15k. Prefer deleting to abstracting; abstract only when the second caller exists.

## 3. Non-negotiables (your quality gates)

- **60 fps, always.** The render thread never touches disk, network, capnp parsing, HEVC decode,
  or a contended lock. Producers stage under a mutex; the UI drains once per frame
  (`store.beginFrame()`); observer events fire on the UI thread only. This is the cabana_core
  `AbstractStream::update()` contract — copy its header comment and keep the discipline.
- **Responsiveness is the product.** First interactive view of the demo route in ~2s; a seek to
  an unloaded region shows *something* immediately and reprioritizes ingestion.
- **Beauty.** Consistent theme, one idiom for panels (pane registry), no globals-of-convenience
  beyond an explicit `Session&` handed to panes, no dead code, no copied-but-unused features.
  You review every subagent diff for this before merging; send work back rather than patching it.
- **Proof over claims.** A task is done when YOU have built it, driven it in the GUI harness,
  and seen the screenshot. Keep a `PROGRESS.md` in `tools/loggy/` with phase status and evidence
  paths; update it after every merge.

## 4. Shape of the tree

```
openpilot/tools/loggy/
  SConscript  main.cc  loggy (launcher)  cabana (launcher)  jotpluggler (launcher)
  shell/      runtime.cc theme.cc workspace.cc pane.h (registry) transport.cc dialogs.cc settings.cc
  backend/    session.h/cc  ingest.cc (scheduler)  store.h/cc (events+series)
              extract.cc (+ generate_event_extractors.py)  dbc/ (lifted cabana_core)
              streams_live.cc  video.cc (FrameReader cache)  computed.cc  export.cc  undo.h
  panes/      plot.cc messages.cc binary.cc signal.cc historylog.cc logs.cc map.cc camera.cc
              video.cc browser.cc find_signal.cc find_bits.cc dbc_editor.cc
  layouts/    cabana.json jotpluggler.json + ported presets
  tests/      Catch2: dbc round-trip, store, scheduler, extraction golden values
```

Key contracts (keep this thin — these five lines, not a framework):
- `Session` owns store, scheduler, dbc, playback clock, selection context, undo stacks.
- Panes are free functions registered with `(name, draw_fn, serialize hooks)`; they receive
  `Session&` + their pane state; they may hold store views only until frame end.
- `store.series(path, t0, t1, max_points)` → decimated borrowed view; `store.canEvents(id, range)`
  likewise. Missing data returns what exists + coverage info; never blocks.
- Selection context: panes in a group share `selected_msg_id`/hover; groups are per-workspace.

## 5. GUI testing harness (verified working — use it constantly)

Fully isolated from the user's desktop; you can click around like a human and see the result.

```bash
# 1. Private display. MUST use the venv wrapper — it sets LIBGL_DRIVERS_PATH so GLX/llvmpipe
#    exists. The raw Xvfb binary gives "couldn't find RGB GLX visual" and GLFW apps abort.
/home/batman/openpilot/.venv/bin/Xvfb :87 -screen 0 1920x1080x24 -nolisten tcp &
DISPLAY=:87 glxinfo | grep renderer     # expect llvmpipe, OpenGL 4.5

# 2. Launch SIZED TO THE SCREEN — otherwise root-window captures have black bars around the app.
#    jotpluggler/loggy: pass --width 1920 --height 1080 (give loggy these flags from day one).
#    Qt apps (no flags): xdotool windowsize <id> 1920 1080; windowmove <id> 0 0.
#    Fallback: import -window <id> crops to the app window (coords stay 1:1 while it sits at 0,0).
DISPLAY=:87 ./loggy --demo --width 1920 --height 1080 &
DISPLAY=:87 xdotool mousemove 160 413 click 1
DISPLAY=:87 xdotool type "carState/vEgo"
# drag & drop: ImGui needs stepped motion —
#   mousedown 1; 5-6 mousemove steps toward target with ~0.15s sleeps; mouseup 1
DISPLAY=:87 import -window root shot.png       # then Read the png — actually LOOK at it
# video evidence: /usr/bin/ffmpeg (venv ffmpeg lacks x11grab)
/usr/bin/ffmpeg -f x11grab -video_size 1920x1080 -framerate 20 -i :87 out.mp4
```

Traps (each cost real time; don't repeat them):
- `pkill -f`/`pgrep -f` with the display number or binary path **matches your own bash wrapper
  and kills your shell**. Kill by scanning `/proc/<pid>/cmdline` of `pgrep -x <name>`.
- No window manager on Xvfb: Qt windows spawn tiny. Fix with
  `xdotool windowsize <id> 1920 1080; windowmove <id> 0 0` (find ids via `xwininfo -root -tree`).
- Screenshot pixel coords map 1:1 to xdotool coords.
- llvmpipe is software GL — do NOT judge fps by swap rate here. Judge with the in-app frame-time
  HUD (§ST-1A): CPU ms per frame (update + build + render) must stay < 8ms so real GPUs hold 60.
- Stale locks: `rm -f /tmp/.X87-lock /tmp/.X11-unix/X87` after killing a dead server.

Verification pattern for every feature: run old tool and loggy on two displays, perform the same
actions on both, screenshot both, compare behavior (not pixels).

## 6. Execution phases and subagent tasks

Run subagents with these briefs (augment each with exact file pointers you gather first — give
them the copy-source paths, not "go find it"). Independent tasks in the same phase run in
parallel. Every implementation brief ends with the same three demands: (a) builds via scons,
(b) subagent ran it headless (or unit tests) itself, (c) reports LOC added and what it copied
vs wrote. You then verify in the GUI harness before marking done. Keep briefs scoped so no
subagent lands more than ~1.5k lines in one task; split otherwise.

### Phase 0 — Orientation (you, no subagents)
Read `loggy.md`, this file, MIGRATION.md. Launch both reference tools in the harness, drive
them for 15 minutes each (open demo route, expand browser, drag signals, edit a DBC signal in
cabana, seek, play). This calibrates the feel bar. Create `tools/loggy/PROGRESS.md`.

### Phase 1 — Foundations (2 subagents, parallel)
**ST-1A shell runtime.** Copy the GLFW+ImGui+ImPlot runtime, font/icon loading (Inter,
JetBrainsMono, bootstrap icons), and Darcula theme from jotpluggler `runtime.cc` + imgui cabana
`imgui/app.cc`; dedupe into `shell/runtime.cc` + `theme.cc`. Add a debug frame-time HUD
(current/p99 CPU ms, toggleable). SConscript target `loggy` linking imgui/glfw/implot + common.
Acceptance: empty themed window, HUD, headless screenshot, clean build.
**ST-1B workspace engine.** Port jotpluggler's `SketchLayout` (workspace tabs, recursive split
tree, pane drag-to-split/move, layout JSON read/write, autosave drafts, snapshot undo) from
`app.h`/`layout.cc`/`layout_io.cc`/`sketch_layout.cc` into `shell/workspace.cc` + `pane.h`
registry. Strip route/stream coupling — panes are opaque registered types. Acceptance: two dummy
pane types; create/split/drag/save/reload driven by xdotool on a test display.

### Phase 2 — Backend engine (3 subagents, parallel)
**ST-2A store + segment scheduler.** `backend/store` + `backend/ingest`. Route resolution + file
listing via `PyDownloader`; N-worker segment pipeline (download → LogReader → publish);
priority queue keyed by distance to tracker/visible ranges, re-sorted on seek; CAN event arena +
sorted merge (copy cabana `MonotonicBuffer`/`mergeEvents` model); series chunk store; staged
batches drained by `beginFrame()`. CLI harness binary that loads the demo route and prints
time-to-first-segment and total-load timings. Acceptance: harness output + Catch2 tests for
merge/priority/drain; first segment usable while later ones still load.
**ST-2B DBC subsystem.** Lift `dbc/` + `commands.cc` UndoStack + tests from cabana_core
verbatim into `backend/dbc`, de-cabana the includes, keep Catch2 tests green. Wire
fingerprint→DBC auto-load (`car_fingerprint_to_dbc.json` codegen). **Investigate & fix:** master
Qt cabana fails to parse `opendbc/dbc/ford_lincoln_base_pt.dbc` (jotpluggler parses it fine) —
loggy's parser must load it; add a regression test.
**ST-2C cereal extraction.** Port `generate_event_extractors.py` codegen + `SeriesAccumulator`/
`StreamAccumulator` from jotpluggler `sketch_layout.cc` into `backend/extract`, feeding ST-2A's
series store per segment (eager v1), incl. enums/value-descriptions and DEPRECATED handling.
Acceptance: golden test — `/carState/vEgo` sample count and 5 spot values match jotpluggler on
the demo route.

### Phase 3 — Vertical slice (2 subagents, then YOU integrate)
**ST-3A transport + timeline.** Playback clock (play/pause/rate/loop/step), timeline bar with
engaged/alert coloring and seek, tracker time, decoupled shared view-range. Copy feel from both
tools' transport bars.
**ST-3B plot pane + messages/binary panes.** Plot pane on `store.series` (multi-series, tracker
readout, stairstep for ints, decimation); messages table pane (live bytes, freq, count, filters,
`CanData::compute` heatmap — port it from cabana_core); binary view pane (bit grid + heatmap);
selection context linking them.
**You:** wire panes + transport into the workspace, create draft presets, then run the slice
gate: demo route opens, plots fill progressively while video-less playback runs, seek to an
unloaded region reprioritizes (watch scheduler logs), drag a series from nothing → plotted,
select message → binary view follows. HUD < 8ms throughout. Screenshot set into PROGRESS.md.
**This is the moment to fix architectural smells — do not proceed with debt.**

### Phase 4 — Cabana parity (subagents; parallelize in pairs)
**ST-4A signal editor.** Signal view pane (all properties incl. multiplex, value descriptions,
inline sparkline) + drag-on-binary-view signal create/resize + all edits through UndoStack.
Reference: imgui cabana `signal_view.cc`/`binary_view.cc`.
**ST-4B DBC file management + settings.** New/open/save/save-as/clipboard, opendbc list, recent
files, per-bus assignment (`SourceSet`), settings dialog + json11 persistence.
**ST-4C history log + exports.** History log pane (hex/decoded, comparators, paging) + CSV
exports: whole stream, per message, per signal.
**ST-4D chart parity.** On the plot pane: signal selector dialog, zoom with zoom-undo, value
tooltip across panes, series type line/step/scatter — expressed in workspace idiom (split panes
replace chart tabs).
**ST-4E analysis tools.** Find Signal (iterative bit-range filtering w/ history) and Find
Similar Bits panes, ported from imgui cabana.
**ST-4F video pane.** Route video via `FrameReader` (copy jotpluggler's decode/prefetch/LRU
stack in `runtime.cc`/`camera.cc`), camera tabs, qlog thumbnails on the timeline, alert overlay.

### Phase 5 — Jotpluggler parity (subagents; parallel)
**ST-5A browser pane** (tree from schema + live values, search, sparklines, drag source,
DEPRECATED toggle). **ST-5B logs pane** (logMessage/errorLogMessage/OS logs/alerts, level mask,
search, time modes, follow). **ST-5C map pane** (port `map.cc`: Overpass vector basemap +
cache, GPS trace colored by engagement, follow, Google Maps link). **ST-5D camera panes**
(road/driver/wide/qcam synced to tracker, sidebar-preview equivalent). **ST-5E computed
series** (`backend/computed` + `math_eval.py` subprocess + editor pane with templates/preview;
derivative & scale/offset transforms). **ST-5F presets** (port all 17 jotpluggler layouts +
author `cabana.json`, `jotpluggler.json`; launcher scripts select preset by argv[0]/flag).

### Phase 6 — Live sources (2 subagents)
**ST-6A live cereal.** msgq/zmq subscription (port jotpluggler `StreamPoller` + live extraction)
into the same store with rolling buffer + follow mode; `--stream --address`.
**ST-6B CAN hardware streams + open flow.** Port `pandastream`/`socketcanstream`/`devicestream`
from cabana_core; stream selector startup dialog + comma-API routes browser (port from imgui
cabana `stream_selector.cc`/`routes_dialog.cc`); live video via VisionIPC.

### Phase 7 — Hardening → 100%
**ST-7A parity audit (Explore-style subagent per tool):** walk §7 checklists feature by feature
against the reference binaries in the harness; file a defect list with screenshots. You fix or
dispatch fixes until the list is empty.
**ST-7B perf audit:** HUD p99 < 8ms CPU during: load-storm (open route, immediately seek 3×),
playback with 6 plots + video + binary view, live stream. Time-to-first-use ≤ ~2s demo route.
Record before/after mp4s.
**ST-7C beauty & LOC pass:** dead code, duplicated helpers, needless abstractions, comment
hygiene; report final LOC (target ≤ ~15k).
**ST-7D ship:** README, CI wiring (Catch2 tests + headless `--output` PNG captures of both
presets like jotpluggler's), PROGRESS.md finalized with the evidence trail.

**Definition of 100% done:** every checklist item below demonstrated in loggy via the harness;
both presets launch via old-name launchers; tests + headless captures wired for CI; perf gates
met; LOC target met; zero known regressions vs either reference tool.

## 7. Feature parity checklists (nothing may be missing at the end)

**From Cabana:** message table (name/bus/addr/node/freq/count/live bytes, per-column filters,
inactive toggle) · binary view (bit grid, live + bit-flip heatmaps, MSB/LSB markers, drag to
create/resize signals, overlap detection) · signal editor (name/size/endian/sign/offset/factor/
min/max/unit/comment/receiver/type/multiplex value, value descriptions, precision, color,
sparkline w/ adjustable window) · history log (hex + decoded, comparator filters, paging, CSV) ·
charts (multi-signal, line/step/scatter, zoom + zoom-undo, tooltip, drag between plots, split) ·
video (synced, camera tabs, slider thumbnails, alert/engaged overlay, route info) · playback
(seek/pause/speed/skip/loop, Space) · DBC (new/open/save/save-as/clipboard both ways, opendbc
menu, recents, per-bus SourceSet, fingerprint auto-load, full undo/redo + command list) · Find
Signal · Find Similar Bits · CSV exports (stream/message/signal) · suppress highlighted/defined
bits · settings (fps, cache, theme, drag direction) · streams: replay, device zmq (bridge),
panda USB (bus config, CAN-FD), socketcan · routes browser (comma API, period filter) · stream
selector · help overlay (F1) · SIGTERM-clean shutdown.

**From Jotpluggler:** plot panes (multi-curve, shared synced x, tracker + legend values,
stairstep, derivative, scale/offset, y-limits editor, hover highlight) · workspace (tabs,
recursive splits, drag curves/panes, context menus, undo, autosave drafts) · browser (schema
tree, search, sparklines, drag, deprecated toggle, Ctrl+F) · logs viewer (py + OS + alerts,
level mask, source filter, text search, expandable ctx, route/boot/wall time, follow) · map
(Overpass basemap + disk cache, engaged-colored GPS trace, follow/zoom/pan, Google Maps link,
cache mgmt UI) · cameras (4 feeds synced, prefetch, fit modes) · custom Python series (editor,
templates, live preview, numpy env) · CAN decode via fingerprint/manual DBC override · live
stream (msgq local + zmq remote, rolling buffer, pause, follow) · route chip (copy, useradmin/
connect links, segment slice editing, rlog/qlog selector, route info) · headless PNG export
(`--output`) · playback rate/loop · 17 preset layouts.

## 8. Standing footguns

- jotpluggler's `setenv("ZMQ")` process-global switch is thread-hostile — redesign that seam
  when porting live streams (explicit endpoint config, not env mutation).
- cabana's DBC writer silently drops `BA_`/`BO_TX_BU_`/signal-less `BO_` — legacy behavior,
  keep byte-identical for now, document it.
- `ReplayStream`'s destructor-ordering comments in cabana_core encode real deadlock fixes —
  if you copy threading patterns, copy the comments.
- imgui port's known Qt-parity gaps (per-signal CSV export, multi-tab charts, chart DnD, splitter
  restore) are NOT excuses — loggy's bar is the Qt cabana feature list (§7), in workspace idiom.
- PNG export shells out to ImageMagick `convert`; custom series shell out to `python3
  math_eval.py` via temp files. Both acceptable; keep them contained in one module each.
