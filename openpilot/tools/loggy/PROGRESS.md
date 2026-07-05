# Loggy Progress

## Phase 0 - Orientation

- Status: complete for initial implementation pass.
- Read `loggy_plan.md`, `loggy.md`, and the ImGui Cabana migration note at
  `/tmp/worktrees/tmp.8N4A5FfOCP/imgui_cabana_fable/openpilot/tools/cabana/imgui/MIGRATION.md`.
- Reference source paths:
  - Current Jotpluggler: `openpilot/tools/jotpluggler/`
  - ImGui Cabana reference: `/tmp/worktrees/tmp.8N4A5FfOCP/imgui_cabana_fable/openpilot/tools/cabana/`
  - Cabana poll/update contract: `streams/abstractstream.h`

## Phase 1A - Shell Runtime

- Status: initial baseline complete.
- Added initial `openpilot/tools/loggy` SCons target and launchers.
- Launchers present: `loggy`, `cabana`, `jotpluggler`, plus compatibility aliases `loggy_cabana` and `loggy_jotpluggler`.
- Added Dear ImGui + ImPlot + GLFW runtime, Darcula-style theme, Inter/JetBrains Mono/bootstrap-icons fonts, CLI sizing, and `--output` framebuffer capture.
- Added debug frame-time HUD with current and p99 CPU milliseconds.

## Phase 1B - Workspace Engine

- Status: visible shell integration complete for the scaffold.
- Added pane registry, pane instances, workspace tabs, recursive split nodes, basic add/split/move/close APIs, JSON save/load, draft autosave helpers, and snapshot undo/redo history.
- Added default Cabana and Jotpluggler workspace constructors with dummy pane types.
- Added `openpilot/tools/loggy/tests/workspace_smoke` for registry/default/split/move/close/JSON/history coverage.
- Runtime now constructs a `Session`, renders preset workspaces from the split tree, and supports context-menu pane split/close plus tab creation.
- Fixed workspace tab selection so ImGui receives one-frame programmatic selection requests instead of pinning the current tab every frame; mouse clicks now switch between preset tabs.

## Backend Session

- Status: initial skeleton integrated.
- Added `Session` with workspace ownership, route/stream config, playback clock, shared view range, timeline model, store, scheduler, and per-selection-group context. Undo stacks remain to be added.
- Added temporary `Session::seedDemoData()` that stages synthetic series/CAN/timeline data through the same `Store::stage()` path real ingestion will use. This is a bridge for pane/UI development until LogReader/PyDownloader ingestion lands.

## Phase 2A - Store + Segment Scheduler

- Status: first route-ingestion slice integrated.
- Added `Store::stage()` producer-side staging and `Store::beginFrame()` UI-thread drain boundary.
- Added nonblocking `series(path, t0, t1, max_points)` and `canEvents(id, range)` committed-data queries with coverage reporting.
- Added `SegmentScheduler` segment records, tracker/visible-range priority ordering, reprioritization, `takeNext()`, and store batch publishing.
- Added route resolution for remote/local route file lists, rlog/qlog selection, segment slicing, and nominal segment ranges.
- Added background `RouteIngestor` using replay `LogReader`, `FileReader`, `PyDownloader`, and replay decompression utilities, publishing extracted `StoreBatch`es through the scheduler.
- Route ingestion now extracts selfdriveState timeline spans and stages them for the UI thread alongside Store batches.
- Route ingestion now extracts logMessage/errorLogMessage, operatingSystemLog, and selfdriveState alert entries, stages them for the UI thread, and stores them on `Session`.
- Runtime calls `session.beginFrame()` once per frame, updating scheduler tracker/visible priorities and draining the store on the UI thread.

## Phase 3A - Transport + Timeline

- Status: route-backed timeline slice integrated.
- Added playback clock, play/pause/rate/step/loop behavior, independent shared view range, timeline span model, seek mapping, and smoke tests.
- Runtime displays a bottom transport bar with playback controls, tracker time, clickable timeline strip, route/status text, and route-backed engagement/alert spans from real selfdriveState events.

## Phase 2B - DBC Subsystem

- Status: parser/serializer/manager first slice integrated.
- Lifted Cabana's Qt-free DBC parser, serializer, manager, and event helper into `openpilot/tools/loggy/backend/dbc` under the `loggy` namespace.
- Preserved documented legacy writer omissions for `BA_`, `BO_TX_BU_`, and signal-less `BO_` output.
- Added `openpilot/tools/loggy/tests/dbc_parser` with inline parser coverage, legacy writer behavior coverage, and `ford_lincoln_base_pt.dbc` regression coverage.

## Phase 2C - Event Extraction

- Status: first generated-dispatch route slice integrated.
- Added `SeriesAccumulator`, segment extraction result types, metadata capture hooks, raw CAN frame accumulation, and StoreBatch finishing in `openpilot/tools/loggy/backend/extract.{h,cc}`.
- Added `generate_event_extractors.py`, adapted from the Jotpluggler schema walker, with SCons wiring to generate `backend/generated_event_extractors.h` from current cereal/opendbc Cap'n Proto schemas.
- Generated extractor output is treated as a build artifact and ignored by git.
- Added `openpilot/tools/loggy/tests/extract_smoke` for accumulator ordering, metadata, and raw CAN chunk coverage.
- Added `openpilot/tools/loggy/tests/route_ingest_smoke` to resolve a real route, load a selected segment through replay `LogReader`, extract cereal/CAN series, publish via `SegmentScheduler`, and drain into `Store`.

## Phase 3B - First Real Panes

- Status: first data-backed pane slice integrated.
- Plot panes query shared `Store::series()` data and render vEgo/aEgo with ImPlot.
- Browser panes list loaded `Store::seriesPaths()` in a searchable table and provide drag payloads
  that Plot panes accept to add a dropped series to their saved pane state.
- Map panes render a GPS trace from `/gpsLocationExternal/{latitude,longitude,hasFix,bearingDeg}`
  store series, fit the route to the pane, and show the playback tracker as a heading marker.
- Messages panes render an all-CAN-ID table from `Store::canMessageIds()` plus lightweight
  `Store::canEventSummary()` rows, with ID text filtering, bus filtering, frequency/count, and
  live byte cells.
- Binary panes read the selected/default CAN message and render byte/bit state from the shared store.
- Signal panes read the selected CAN message, show DBC-decoded signal rows when a DBC definition
  is loaded, and otherwise show live bit-candidate rows with current bit value and flip counts.
- Logs panes render staged route log entries with filter, level threshold, follow mode, and compact route/log table columns.
- Workspace registry now mounts the real browser/plot/map/messages/binary/signal/history/logs pane draw functions; placeholder pane rendering remains only for pane types that do not yet have implementations.

## Phase 4B - DBC File Management + Settings

- Status: path-based DBC management, New/Open/Save/Save As, clipboard DBC import/export, opendbc browser, recents, persisted source assignments/root, and settings-helper slices integrated; native file dialogs, settings dialog UI, and full per-bus reassignment UX remain.
- Added a DBC pane with path/source controls for New, Open, Save, Save As, Close, Close All, Copy, and Paste against the shared `DBCManager`.
- Added helpers for persisted pane state, backend source-set parsing (`all`, `*`, and explicit bus lists), loaded-file summarization, and DBC lookup by source set.
- Added a json11-backed settings helper for recent DBC files, DBC source assignments, and opendbc root path with normalization, missing-file defaults, malformed-file errors, and path round-trip helpers.
- Session now loads settings from `--settings`, XDG config, or `~/.config/loggy/settings.json`, auto-opens valid persisted DBC source assignments at startup, and exposes save-on-demand settings updates to panes.
- DBC pane successful Open/Save As actions remember the path, persist the current source assignment, and show a Recent DBC combo that restores matching source keys.
- DBC pane New creates an untitled in-memory DBC for the selected sources; Copy exports the selected source set's DBC text to the clipboard; Paste parses clipboard DBC text into the selected source set.
- DBC pane now includes an opendbc browser with configurable root, text filter, bounded directory scan, Use/Open row actions, default-root reset, and root persistence.
- Fixed DBC manager close behavior so closing a source removes that mapping and restores `SOURCE_ALL` fallback; fixed `DBCFile::saveAs` so failed writes do not mutate the stored filename.
- Cabana preset now opens a second `DBC` tab containing the DBC management pane while the main Cabana tab keeps Messages/Binary/History/Signal together.
- Pane helper coverage verifies source parsing failures, state JSON round-trip, loaded-file table summaries, `SOURCE_ALL` lookup fallback behavior, empty DBC creation, clipboard export, clipboard import, empty clipboard failure handling, opendbc root scanning, filtering, row limits, and missing-root errors.
- Settings/session coverage verifies recent-file de-duplication/limits, assignment/root round-trip, missing-file defaults, malformed-file handling, malformed-field filtering, isolated settings paths, and startup DBC assignment loading.

## Phase 4C - History Log + Exports

- Status: history-log, comparator filtering, paging, CSV export helpers, and path-based file export slices integrated; native file dialogs and broader export UX remain.
- History panes read the selected CAN message from the shared Cabana selection group, render newest-first time/bus-time/length/hex/decoded columns for CAN frames in the current view range, and persist the selected CAN ID plus text filter, comparator, and paging state in pane state.
- History panes expose DBC-signal comparator filters (`>`, `=`, `!=`, `<`, `>=`, `<=`) when a DBC message definition is available, and page rows with a configurable row count plus previous/next controls.
- History helper coverage verifies state parsing/selected-ID preservation, comparator/page state, raw hex rows, newest-first paging, bus-time propagation, text filtering, comparator filtering, max-row limiting through the shared store query path, and DBC decoded values when a message definition is available.
- Added backend CSV helpers for whole-CAN-stream, per-message, and per-signal export data, with escaping, deterministic ordering, raw hex bytes, and DBC decoded values where applicable.
- History panes expose a `Copy CSV` action for the selected CAN message in the current view range; GUI verification confirmed the clipboard payload contains CSV rows for selected message `0:47`.
- History panes expose a path-based export row with `Save Msg`, `Save Stream`, and `Save Signal`; file-write tests verify directory creation, write success, and empty-path failure handling.
- Cabana preset now opens Messages/Binary/History/Signal together so selected-message investigation has the expected table, bits, history, and signal views in one workspace.

## Evidence

- Build: `scons -j$(nproc) openpilot/tools/loggy/_loggy`
- Harness: `/home/batman/openpilot/.venv/bin/Xvfb :87 -screen 0 1920x1080x24 -nolisten tcp`
- GLX: llvmpipe, OpenGL 4.5, verified with `DISPLAY=:87 glxinfo`.
- Screenshot: `/tmp/loggy_phase1a.png` captured with
  `DISPLAY=:87 openpilot/tools/loggy/loggy --demo --width 1920 --height 1080 --output /tmp/loggy_phase1a.png`.
- Visual check: clean 1920x1080 Darcula shell with menu bar, status bar, route/preset text, and frame-time HUD (`CPU 0.42 ms`, `p99 0.42 ms` in the captured frame).
- Launcher smoke: `/tmp/loggy_cabana_launcher.png` captured with
  `DISPLAY=:87 openpilot/tools/loggy/loggy_cabana --width 640 --height 360 --output /tmp/loggy_cabana_launcher.png`.
- Old-name launcher smoke: `/tmp/loggy_oldname_cabana.png` captured with
  `DISPLAY=:87 openpilot/tools/loggy/cabana --width 640 --height 360 --output /tmp/loggy_oldname_cabana.png`.
- Workspace smoke: `scons -j$(nproc) openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/workspace_smoke && openpilot/tools/loggy/tests/workspace_smoke`.
- Jotpluggler launcher smoke after workspace linkage: `/tmp/loggy_jotpluggler_launcher.png` captured with
  `DISPLAY=:87 openpilot/tools/loggy/loggy_jotpluggler --demo --width 800 --height 450 --output /tmp/loggy_jotpluggler_launcher.png`.
- Workspace visual evidence:
  - `/tmp/loggy_cabana_workspace.png` from
    `DISPLAY=:87 openpilot/tools/loggy/loggy_cabana --demo --width 1920 --height 1080 --output /tmp/loggy_cabana_workspace.png`
  - `/tmp/loggy_jotpluggler_workspace.png` from
    `DISPLAY=:87 openpilot/tools/loggy/loggy_jotpluggler --demo --width 1920 --height 1080 --output /tmp/loggy_jotpluggler_workspace.png`
  - Visual check: Cabana preset shows Messages/Binary/Signal panes; Jotpluggler preset shows Browser/Plot/Logs panes; no scroll/clipping; HUD p99 below 1 ms in both captures.
- DBC parser smoke: `scons -j$(nproc) openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/workspace_smoke openpilot/tools/loggy/tests/dbc_parser && openpilot/tools/loggy/tests/workspace_smoke && openpilot/tools/loggy/tests/dbc_parser`.
  Result: all DBC tests passed (`45 assertions in 5 test cases`).
- Full current smoke: `scons --cache-disable -j$(nproc) openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/workspace_smoke openpilot/tools/loggy/tests/dbc_parser openpilot/tools/loggy/tests/transport_smoke openpilot/tools/loggy/tests/settings_smoke openpilot/tools/loggy/tests/store_scheduler openpilot/tools/loggy/tests/export_smoke openpilot/tools/loggy/tests/panes_smoke openpilot/tools/loggy/tests/extract_smoke openpilot/tools/loggy/tests/route_ingest_smoke`, followed by the non-network smoke binaries and one qlog route-ingest demo smoke.
  Results: workspace passed silently; DBC passed (`45 assertions in 5 test cases`); transport printed `transport_smoke passed`; settings passed (`29 assertions in 3 test cases`); store/scheduler passed (`34 assertions in 4 test cases`); export passed (`19 assertions in 1 test case`); panes passed (`158 assertions in 10 test cases`); extraction passed (`30 assertions in 3 test cases`); route ingest loaded one demo qlog segment in `0.917s`, producing `10976` Store series, `189` CAN ids, `1` timeline span, and `13` log entries.
- Transport UI evidence: `/tmp/loggy_transport_workspace.png` captured with
  `DISPLAY=:87 openpilot/tools/loggy/jotpluggler --demo --width 1920 --height 1080 --output /tmp/loggy_transport_workspace.png`.
  Visual check: Browser/Plot/Logs workspace plus bottom transport/timeline/status bar; no overlap; HUD p99 below 1 ms.
- Seeded store evidence: `/tmp/loggy_seeded_store.png` captured with
  `DISPLAY=:87 openpilot/tools/loggy/jotpluggler --demo --width 1920 --height 1080 --output /tmp/loggy_seeded_store.png`.
  Visual check: timeline engagement/alert spans render and status shows `2 series 1 CAN ids`, proving staged data drained into the shared store on the UI frame.
- Real plot pane evidence: `/tmp/loggy_real_plot_pane.png` captured with
  `DISPLAY=:87 openpilot/tools/loggy/jotpluggler --demo --width 1920 --height 1080 --output /tmp/loggy_real_plot_pane.png`.
  Visual check: Plot pane renders seeded vEgo/aEgo series through ImPlot with no blank canvas or overlap; HUD p99 around 1 ms.
- Real CAN pane evidence: `/tmp/loggy_real_can_panes.png` captured with
  `DISPLAY=:87 openpilot/tools/loggy/cabana --demo --width 1920 --height 1080 --output /tmp/loggy_real_can_panes.png`.
  Visual check: Messages pane shows CAN id `0x123`, event count/bytes, and the Binary pane follows the selected/default message bits; HUD p99 below 1 ms.
- Real route-ingestion evidence: `/tmp/loggy_route_ingest_qlog.png` captured from a live GUI session with
  `DISPLAY=:87 openpilot/tools/loggy/jotpluggler 5beb9b58bd12b691/0000010a--a51155e496/q --width 1920 --height 1080 --show`.
  Visual check after background ingestion: Plot pane renders real `/carState/vEgo` and `/carState/aEgo` data over the full qlog route; status bar shows `complete`, `16/16 segments`, `10976 series`, and `290 CAN ids`; HUD p99 is about `2.17 ms`.
- Real route timeline evidence: `/tmp/loggy_route_timeline_qlog.png` captured from the same qlog route after adding timeline extraction.
  Visual check: bottom timeline strip shows route-backed green engagement spans from selfdriveState events; Plot pane still renders real route data; status bar shows `complete`, `16/16 segments`, `10976 series`, and `290 CAN ids`; HUD p99 is about `2.25 ms`.
- Real route logs evidence: `/tmp/loggy_route_logs_qlog.png` captured from the same qlog route after adding log extraction and the Logs pane.
  Visual check: Logs pane shows `44/44 logs` with alert/log rows, levels, origins, sources, and messages; Plot pane and route timeline remain populated; status bar shows `complete`, `16/16 segments`, `10976 series`, and `290 CAN ids`; HUD p99 is about `2.33 ms`.
- Real route messages-table evidence: `/tmp/loggy_message_table_qlog.png` captured from a live GUI session with
  `DISPLAY=:87 openpilot/tools/loggy/cabana 5beb9b58bd12b691/0000010a--a51155e496/q --width 1920 --height 1080 --show`.
  Visual check: Messages pane shows a filtered-capable table with `290/290 CAN ids`, per-row bus/ID/frequency/count/latest bytes, selected row `0:47`, and the Binary pane follows the same selected ID; status bar shows `complete`, `16/16 segments`, `10976 series`, and `290 CAN ids`; HUD p99 is about `1.63 ms`.
- Real route browser evidence: `/tmp/loggy_browser_qlog.png` captured from a live GUI session with
  `DISPLAY=:87 openpilot/tools/loggy/jotpluggler 5beb9b58bd12b691/0000010a--a51155e496/q --width 1920 --height 1080 --show`.
  Visual check: Browser pane shows a searchable table with `1000/10976 series` beside populated Plot and Logs panes; status bar shows `complete`, `16/16 segments`, `10976 series`, and `290 CAN ids`; HUD p99 is about `6.64 ms`.
- Browser-to-Plot drag evidence: `/tmp/loggy_browser_drag_plot_qlog.png` captured after dragging the first Browser row into the Plot pane in the same qlog route session.
  Visual check: Plot pane updates from `2 series` to `3 series` and renders the dropped accelerometer series in the legend/plot without leaving the HUD budget; HUD p99 is about `6.68 ms`.
- Real route map evidence: `/tmp/loggy_map_qlog.png` captured from a live GUI session with
  `DISPLAY=:87 openpilot/tools/loggy/jotpluggler 5beb9b58bd12b691/0000010a--a51155e496/q --width 1920 --height 1080 --show`.
  Visual check: Map pane renders `345 GPS points`, route bounds `32.74948, -117.23334` to `32.83509, -117.19469`, and a yellow playback tracker marker alongside Browser/Plot/Logs; status bar shows `complete`, `16/16 segments`, `10976 series`, and `290 CAN ids`; HUD p99 is about `6.56 ms`.
- Real route signal evidence: `/tmp/loggy_signal_qlog.png` captured from a live GUI session with
  `DISPLAY=:87 openpilot/tools/loggy/cabana 5beb9b58bd12b691/0000010a--a51155e496/q --width 1920 --height 1080 --show`.
  Visual check: Signal pane follows selected message `0:47` from the Messages table, shows `64 bit candidates` with current value/flip columns, and remains linked with Binary for the same selected ID; status bar shows `complete`, `16/16 segments`, `10976 series`, and `290 CAN ids`; HUD p99 is about `1.86 ms`.
- Real route history evidence: `/tmp/loggy_history_qlog.png` captured from a live GUI session with
  `DISPLAY=:87 openpilot/tools/loggy/cabana 5beb9b58bd12b691/0000010a--a51155e496/q --width 1920 --height 1080 --show`.
  Visual check: History pane follows selected message `0:47` from the Messages table, shows `13 events` with time/bus-time/length/hex/decoded columns, and remains linked with Binary and Signal for the same selected ID; status bar shows `complete`, `16/16 segments`, `10976 series`, and `290 CAN ids`; HUD p99 is about `2.20 ms`.
- Real route history export evidence: `/tmp/loggy_history_export_qlog.png` captured from a live GUI session with
  `DISPLAY=:87 openpilot/tools/loggy/cabana 5beb9b58bd12b691/0000010a--a51155e496/q --width 1920 --height 1080 --show`.
  Visual check: History pane still follows selected message `0:47`, shows the `Copy CSV` action without toolbar/table overlap, and `DISPLAY=:87 xclip -selection clipboard -o` after clicking it returned CSV headed `mono_time,bus_time,bus,address,length,hex,decoded`; HUD p99 is about `2.04 ms`.
- Real route history paging evidence: `/tmp/loggy_history_paging_qlog.png` captured from a live GUI session with
  `DISPLAY=:87 openpilot/tools/loggy/cabana 5beb9b58bd12b691/0000010a--a51155e496/q --width 1920 --height 1080 --show`.
  Visual check: History pane shows newest-first rows for selected message `0:47`, with row-count input, page indicator, previous/next controls, and `Copy CSV` sharing the toolbar without overlap; HUD p99 is about `1.97 ms`.
- Real route file-export evidence:
  - `/tmp/loggy_history_file_export_qlog.png` captured before saving, showing the History export path input plus `Save Msg`, `Save Stream`, and `Save Signal` controls without overlap.
  - `/tmp/loggy_history_file_saved_qlog.png` captured after clicking `Save Msg`, showing `Saved /tmp/loggy_history.csv`; `sed -n '1,4p' /tmp/loggy_history.csv` confirmed a CSV headed `mono_time,bus_time,bus,address,length,hex,decoded` with selected message `0x47` rows; HUD p99 is about `2.23 ms`.
- Real route DBC management evidence:
  - `/tmp/loggy_dbc_tab_blank.png` captured from a live qlog Cabana session after switching to the `DBC` tab, showing the path/source controls and empty loaded-file table after `Close All`; HUD p99 is about `1.04 ms`.
  - `/tmp/loggy_dbc_open_qlog.png` captured after opening `opendbc_repo/opendbc/dbc/ford_lincoln_base_pt.dbc`, showing one loaded row with `331` messages and `2150` signals for `all` sources; HUD p99 is about `1.26 ms`.
- DBC opendbc browser evidence:
  - `/tmp/loggy_opendbc_filter.png` captured from `DISPLAY=:87 openpilot/tools/loggy/cabana --settings /tmp/loggy_opendbc_settings.json --width 1920 --height 1080 --show` after switching to the `DBC` tab, typing filter `ford`, and clicking `Scan`; visual check shows the configured opendbc root, filter field, `Found 7 opendbc files`, and a table including `FORD_CADS`, `ford_fusion_2018_pt`, and `ford_lincoln_base_pt`; HUD p99 is about `1.41 ms`.
  - `/tmp/loggy_opendbc_open.png` captured after clicking `Open` on `ford_lincoln_base_pt` in the opendbc browser; visual check shows the loaded DBC row with `331` messages and `2150` signals for `all` sources and the recent/path fields updated; HUD p99 is about `1.54 ms`.
  - `/tmp/loggy_opendbc_saved_root.png` captured after clicking `Save Root`; `sed -n '1,120p' /tmp/loggy_opendbc_settings.json` confirmed `opendbc_root`, `recent_files`, and `assignments: {"all": "...ford_lincoln_base_pt.dbc"}` were persisted; HUD p99 is about `1.46 ms`.
- DBC New/clipboard evidence:
  - `/tmp/loggy_dbc_new_qlog.png` captured after clicking `New` in a Cabana DBC tab with an isolated settings file, showing an in-memory `untitled` DBC row for `all` sources and status `Created untitled DBC for all`; HUD p99 is about `1.06 ms`.
  - `/tmp/loggy_dbc_clipboard_paste.png` captured after clicking `Copy`, verifying `DISPLAY=:87 xclip -selection clipboard -o` returned DBC text headed `VERSION ""`, then clicking `Close All` and `Paste`; visual check shows a `clipboard` DBC row for `all` sources and status `Pasted DBC for all`; HUD p99 is about `1.58 ms`.
- DBC settings persistence evidence:
  - `/tmp/loggy_dbc_settings_open.png` captured from `DISPLAY=:87 openpilot/tools/loggy/cabana --settings /tmp/loggy_dbc_settings.json --width 1920 --height 1080 --show` after opening `opendbc_repo/opendbc/dbc/ford_lincoln_base_pt.dbc`; visual check shows the Recent combo, `Saved settings`, and one loaded row with `331` messages and `2150` signals; `sed -n '1,120p' /tmp/loggy_dbc_settings.json` confirmed `recent_files` plus `assignments: {"all": "...ford_lincoln_base_pt.dbc"}`; HUD p99 is about `1.22 ms`.
  - `/tmp/loggy_dbc_settings_autoload.png` captured after restarting Cabana with the same `--settings` file and switching to the `DBC` tab without pressing Open; visual check shows the DBC row auto-loaded from the persisted `all` assignment with `331` messages and `2150` signals; HUD p99 is about `1.42 ms`.
