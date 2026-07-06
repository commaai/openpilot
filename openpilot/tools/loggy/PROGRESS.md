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
- Added a static pane definition table, pane instances, workspace tabs, recursive split nodes, basic add/split/move/close APIs, JSON save/load, draft autosave helpers, and snapshot undo/redo history.
- Added default Cabana and Jotpluggler workspace constructors with initial placeholder pane types.
- Added `openpilot/tools/loggy/tests/workspace_smoke` for pane-table/default/split/move/close/JSON/history coverage.
- Runtime now constructs a `Session`, renders preset workspaces from the split tree, and supports context-menu pane split/close plus tab creation.
- Fixed workspace tab selection so ImGui receives one-frame programmatic selection requests instead of pinning the current tab every frame; mouse clicks now switch between preset tabs.
- Runtime now loads saved workspace drafts for file-backed layouts, tracks workspace undo/redo history for tab/split/close mutations, autosaves draft changes, and exposes Workspace menu actions for Undo, Redo, Save Layout, and Clear Draft.

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

- Status: parser/serializer/manager plus fingerprint-to-DBC generation slices integrated.
- Lifted Cabana's Qt-free DBC parser, serializer, and manager into `openpilot/tools/loggy/backend/dbc` under the `loggy` namespace.
- Preserved documented legacy writer omissions for `BA_`, `BO_TX_BU_`, and signal-less `BO_` output.
- SCons now generates `car_fingerprint_to_dbc.h` from opendbc platform metadata and materializes generated opendbc DBC files under ignored build artifacts for runtime auto-load.
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
  `Store::canEventSummary()` rows, with ID/name/node text filtering, bus filtering, active/all
  filtering, DBC-backed message names/transmitters when loaded, frequency/count, and live byte
  cells.
- Binary panes read the selected/default CAN message and render byte/bit state from the shared store.
- Signal panes read the selected CAN message, show DBC-decoded signal rows when a DBC definition
  is loaded, and otherwise show live bit-candidate rows with current bit value and flip counts.
- Logs panes render staged route log entries with filter, level threshold, follow mode, and compact route/log table columns.
- The static workspace pane table now mounts the real browser/plot/map/messages/binary/signal/history/logs pane draw functions; placeholder pane rendering remains only for pane types that do not yet have implementations.

## Phase 4A - Signal Editor

- Status: command-backed message metadata edit, existing-signal edit/remove/value-table, precision/color editing, Binary drag-create/resize with overlap detection, Binary defined-bit highlight/suppress controls, and Signal sparkline slices integrated; richer dirty/save UX remains.
- Added a session-owned DBC `UndoStack` plus `EditSignalCommand`, adapted from Cabana's command model without a global singleton.
- Added validation and helper APIs for applying edits to existing DBC signals, including duplicate-name, missing-target, start-bit, and size checks.
- Signal panes now select DBC-backed signal rows, render an inline editor for core fields (name, start bit, size, endian, signed, factor, offset, min/max, unit, receiver, comment, type, mux value), and apply changes through the DBC undo stack.
- Signal panes expose command-backed Apply, Reset, Remove, Undo, and Redo controls for the selected DBC signal.
- Signal panes now remove selected DBC signals through `RemoveSignalCommand`, with Undo/Redo; multiplexor removal also removes multiplexed children.
- Signal panes now expose a compact `Value Table` editor using DBC `VAL_`-style `value "description"` pairs, with inline parsing, validation, filtering, and command-backed Apply/Undo/Redo.
- Signal panes now expose editable precision and RGB color controls with Auto buttons; explicit values persist on the DBC `Signal` model, survive `Signal::update()`, and travel through the existing DBC undo/redo command path while ordinary factor/offset/name/bit edits keep auto-derived precision/color.
- Signal panes now expose a command-backed Message editor for DBC message name, byte size, transmitter, and comment, with Apply/Reset/Undo/Redo controls, existing-signal preservation, derived-mask refresh, and validation that prevents shrinking a message below its signals.
- Binary panes now support left-dragging across bit cells to create a DBC signal through `AddSignalCommand`; creation can also create the DBC message for the selected CAN id when a DBC file is loaded for the source.
- Binary drag-created signals now reject selections that overlap existing DBC-defined bits before pushing `AddSignalCommand`, surfacing the overlapping bit in the pane status instead of creating conflicting signals.
- Binary panes now start an edge-resize interaction when a drag begins on a DBC-defined little-endian signal bit, apply the resized copy through `EditSignalCommand`, and reject ranges that overlap other DBC signals.
- Binary panes now persist `Defined` and `Suppress` controls that highlight DBC-defined bits from the DBC message mask or mute them while inspecting unknown bits.
- Signal panes now compute and render inline DBC signal sparklines from CAN events, with a persisted 1-120 second Spark window control.
- Command coverage verifies add/edit/remove/value-description/precision/color/message-metadata/undo/redo behavior, validation failures, message-size signal-fit rejection, and multiplexor removal cascades; pane helper coverage verifies Binary bit-range signal drafting, DBC overlap detection, defined-bit state/mask detection, DBC signal-at-bit lookup, Binary resize draft generation, resize overlap rejection, value-table parsing/formatting, Signal-pane apply/remove/precision/color/message-edit/undo/redo integration, sparkline decoding, and Spark window state persistence.

## Phase 4B - DBC File Management + Settings

- Status: path-based DBC management, native DBC/opendbc path choosers, New/Open/Save/Save As, clipboard DBC import/export, opendbc browser, recents, persisted source assignments/root, loaded-file source reassignment, DBC command history, fingerprint auto-load/manual override, settings-helper, first settings-dialog slices, target-FPS/HUD app preferences, configurable map cache root, map drag-direction preference, and theme selection integrated.
- Added a DBC pane with path/source controls for New, Open, Save, Save As, Close, Close All, Copy, and Paste against the shared `DBCManager`.
- Added helpers for persisted pane state, backend source-set parsing (`all`, `*`, and explicit bus lists), loaded-file summarization, and DBC lookup by source set.
- Added a json11-backed settings helper for recent DBC files, DBC source assignments, and opendbc root path with normalization, missing-file defaults, malformed-file errors, and path round-trip helpers.
- Session now loads settings from `--settings`, XDG config, or `~/.config/loggy/settings.json`, auto-opens valid persisted DBC source assignments at startup, and exposes save-on-demand settings updates to panes.
- Route and live ingestion now capture `carParams.carFingerprint`; Session maps known fingerprints to generated opendbc DBC names, opens the resolved DBC for all sources, and surfaces car/auto/active DBC status to panes and route UI.
- DBC pane now exposes Route DBC status plus a manual DBC override field; overrides accept a DBC name or path, persist to settings, can return to Auto, and roll back if the requested DBC cannot be resolved.
- DBC pane successful Open/Save As actions remember the path, persist the current source assignment, and show a Recent DBC combo that restores matching source keys.
- DBC pane New creates an untitled in-memory DBC for the selected sources; Copy exports the selected source set's DBC text to the clipboard; Paste parses clipboard DBC text into the selected source set.
- DBC pane now includes an opendbc browser with configurable root, text filter, bounded directory scan, Use/Open row actions, default-root reset, and root persistence.
- File menu now exposes a `Settings...` modal with the active settings path, editable opendbc root, editable DBC override, Save/Reload/Clear Override actions, and DBC/session status feedback using the same `Session::saveSettings()` and manual override paths as the DBC pane.
- The DBC pane now exposes native `Browse`/`Choose` buttons for DBC open, DBC Save As, and opendbc root selection while preserving the existing editable path fields as deterministic fallback/edit state.
- Settings JSON now stores an `app` preference block with clamped target FPS, Frame-Time HUD visibility, optional map cache root, natural/inverted map drag direction, and theme selection; the runtime applies the FPS cap after swap, persists the HUD preference through the settings modal, keeps `--no-hud` authoritative for that launch, applies the configured cache root to the session-owned map cache manager, feeds the map drag preference into the Map pane, and applies the saved theme at startup and after Settings saves.
- Loaded DBC rows now expose an Assign editor backed by `DBCManager::assignSources`, allowing an already loaded file to move between source sets while preserving the shared file object and persisting the new assignment.
- DBC pane now exposes the shared DBC undo stack as a compact command history with Undo, Redo, command count/index, clean marker, and row-level jump-to-command controls.
- Settings persistence now syncs assignments from the loaded DBC manager state, clearing stale keys for loaded paths and source conflicts so moving one file does not resurrect competing autoload mappings for another loaded DBC on restart.
- Settings save-failure paths now roll back in-memory opendbc root/manual DBC override changes instead of reporting a failed save while keeping unsaved live state; settings loading now rejects directory paths as malformed files instead of throwing from `ifstream`.
- Fixed DBC manager close behavior so closing a source removes that mapping and restores `SOURCE_ALL` fallback; fixed `DBCFile::saveAs` so failed writes do not mutate the stored filename.
- Cabana preset now opens a second `DBC` tab containing the DBC management pane while the main Cabana tab keeps Messages/Binary/History/Signal together.
- Pane helper coverage verifies source parsing failures, state JSON round-trip, loaded-file table summaries, `SOURCE_ALL` lookup fallback behavior, loaded-file source reassignment, conflict-aware loaded assignment sync, empty DBC creation, clipboard export, clipboard import, empty clipboard failure handling, opendbc root scanning, filtering, row limits, and missing-root errors; command coverage verifies undo-stack command-list entries, clean markers, and redo-next state.
- Settings/session coverage verifies recent-file de-duplication/limits, assignment/root round-trip, app target-FPS/HUD/cache-root/drag-direction/theme preference round-trip, clamping, and default fallback, stale assignment clearing for reassigned paths, missing-file defaults, malformed-file handling, directory-path rejection, malformed-field filtering, isolated settings paths, startup DBC assignment loading, map cache-root application at session startup, and manual override rollback after save failure.

## Phase 4C - History Log + Exports

- Status: history-log, comparator filtering, paging, CSV export helpers, path-based file export, and native CSV save path selection integrated; broader export UX remains.
- History panes read the selected CAN message from the shared Cabana selection group, render newest-first time/bus-time/length/hex/decoded columns for CAN frames in the current view range, and persist the selected CAN ID plus text filter, comparator, and paging state in pane state.
- History panes expose DBC-signal comparator filters (`>`, `=`, `!=`, `<`, `>=`, `<=`) when a DBC message definition is available, and page rows with a configurable row count plus previous/next controls.
- History helper coverage verifies state parsing/selected-ID preservation, comparator/page state, raw hex rows, newest-first paging, bus-time propagation, text filtering, comparator filtering, max-row limiting through the shared store query path, and DBC decoded values when a message definition is available.
- Added backend CSV helpers for whole-CAN-stream, per-message, and per-signal export data, with escaping, deterministic ordering, raw hex bytes, and DBC decoded values where applicable.
- History panes expose a `Copy CSV` action for the selected CAN message in the current view range; GUI verification confirmed the clipboard payload contains CSV rows for selected message `0:47`.
- History panes expose a path-based export row with `Save Msg`, `Save Stream`, and `Save Signal`; file-write tests verify directory creation, write success, and empty-path failure handling.
- History panes now add a native `Browse` path chooser for the CSV export target while keeping direct path editing available for scripted or deterministic workflows.
- Cabana preset now opens Messages/Binary/History/Signal together so selected-message investigation has the expected table, bits, history, and signal views in one workspace.

## Phase 4D - Chart Parity

- Status: plot display controls, signal selector, zoom undo, and plot-to-plot drag slices integrated; cross-pane value tooltip polish and broader chart parity remain.
- Plot panes now persist a pane-level series style override (`Auto`, `Line`, `Step`, `Scatter`) while preserving old per-series `stairs` compatibility.
- Plot panes expose a compact Y-axis limit popup with optional min/max values, persist `y_limits` in pane state, and apply explicit limits without changing default auto-fit behavior.
- Plot panes now expose a `+ Series` selector popup backed by loaded Store series paths, preserving existing imported/computed plot state when adding a path.
- Plot panes now persist bounded `x_zoom_history` entries and expose Undo Zoom for x-range changes.
- Plot panes now render draggable series labels that reuse the Browser series-path payload, allowing one Plot pane to drop a series into another.
- Plot panes now show hover tooltips with cursor time and sampled values for each visible series; legend values remain tied to the playback tracker.
- Pane helper coverage verifies style/Y-limit JSON round-trip, old `stairs` compatibility, dropped-series state preservation, zoom-history state preservation, effective style selection, and Y-bound guard behavior.

## Phase 4E - Analysis Tools

- Status: Find Signal and Find Bits pane slices, persisted bounded scan-history controls, and named candidate creation integrated; reference-level polish remains.
- Added `Find Signal` and `Find Bits` pane types and a Cabana preset `Analysis` tab containing both panes.
- Find Signal scans CAN messages over the current view range with bus/address, bit-size, endian/sign, factor/offset, and comparator controls; results can select the source message or create a DBC signal through the shared DBC undo stack.
- Find Signal now exposes a persisted DBC signal Name field for candidate creation, trims explicit names, and falls back to deterministic `SIG_<addr>_<start>_<size>` names instead of generic `NEW_SIGNAL_N` labels.
- Find Bits scans bits on a selected bus against a source message bit, ranks rows by mismatch percentage, preserves the Cabana strict `total > min_msgs` behavior, and can activate a matching message through the shared selection group.
- Find Signal and Find Bits panes now persist a capped, de-duplicated scan history with compact History/Apply controls so repeated narrowing can return to prior scan parameters.
- Helper coverage verifies Find Signal candidate generation and comparator filtering, persisted scan history serialization/application, stale history-index clamping, default/custom DBC signal creation names, duplicate-name rejection, DBC signal creation with undo through the Analysis path, Find Bits mismatch ranking, strict min-count filtering, persisted scan history serialization/application, and selection activation.

## Phase 4F / 5D - Camera Panes

- Status: route-video index, async FrameReader decode boundary, first decoded-frame LRU/prefetch slice, camera-pane shell slices, alert/engaged overlay, and live VisionIPC frame receive path integrated; successful real-frame proof on downloaded route camera files, thumbnails, route-info overlay polish, and sidebar preview remain.
- Route resolution now preserves road, driver, wide-road, and qcamera file paths on `RouteSegment` records instead of dropping them after file discovery.
- Added `backend/video` camera view specs, segment-file lists, encode-index parsing from `/roadEncodeIdx`, `/driverEncodeIdx`, `/wideRoadEncodeIdx`, and `/qRoadEncodeIdx`, `segmentIdEncode` fallback support, unavailable-segment filtering, and tracker-time frame lookup.
- Session now refreshes camera feed indexes when route segments arrive or encode-index series are touched, exposing per-view indexes to panes without panes reaching into scheduler/store internals.
- Camera panes now replace the previous dummy renderer with persisted camera view and Fit state, view switching, indexed file/frame status, async decode status, UI-thread texture upload for decoded RGBA frames, decode cache/queue status, and a route-video canvas that tracks playback time.
- Camera decoding now keeps a small successful-frame LRU per view and queues a two-frame lookahead behind the requested tracker frame, while only the requested frame publishes UI results or visible load/decode errors.
- Stream-mode camera panes now use a separate lazy VisionIPC source for road, driver, and wide-road live frames, converting NV12 buffers to the existing RGBA upload path while keeping OpenGL texture updates on the UI thread; qRoad reports an explicit unsupported live-stream status because this repo does not define a qRoad VisionIPC stream.
- Camera panes now derive the current timeline span from `TimelineModel::kind_at_time()` and render compact engaged/alert badges inside the video canvas without adding a separate state source.
- Camera panes now draw a compact route/live info overlay inside the video canvas with view, segment/frame/time, decode/cache status, or VisionIPC waiting/error state, trimmed to the canvas and positioned away from the alert badge.
- Seeded stream/demo data now includes synthetic encode-index series and synthetic camera segment paths, allowing camera layout UI development and CI-style captures before real camera-file downloads land.
- Loggy now links the existing replay library plus VisionIPC/FFmpeg/libyuv video dependencies needed by `FrameReader`.
- Helper coverage verifies camera index construction, unavailable-segment filtering, `segmentIdEncode` fallback, nearest-frame lookup, pane state round-trip, Session-backed camera snapshots, camera overlay classification, route-info overlay text, live VisionIPC support/status mapping, synthetic VisionIPC frame receive/conversion, and asynchronous decoder load-failure reporting; focused verification also builds `_loggy` with the cached/prefetching decoder and reruns `store_scheduler` plus `panes_smoke`.

## Phase 5A - Browser Pane

- Status: live-value/sparkline, path-tree, DEPRECATED toggle, Ctrl+F focus, schema-path display, and metadata-backed annotation slices integrated; richer drag source UX remains.
- Browser panes now show a tracker-sampled live value and compact inline sparkline for visible series rows, using the shared Store and a persisted 1-120 second Spark window.
- Browser rendering enriches only ImGui-clipped visible rows so large route stores do not force all listed series to query every frame.
- Browser panes now persist Tree/table display mode and a Deprecated visibility toggle.
- Browser panes can render loaded slash-path series as a collapsible hierarchy with group leaf counts while preserving draggable series leaves.
- Deprecated paths are hidden by default and can be included on demand.
- Browser panes now focus their search field with Ctrl+F when no text input is active, and path cells render slash-paths as compact schema-style breadcrumbs.
- Browser rows now surface Store-backed metadata annotations for enum names and deprecated schema paths; `Store::seriesPaths()` includes metadata-only paths so schema annotations can appear before data points arrive.
- Helper coverage verifies Browser state persistence, tracker interpolation, sparkline windowing, min/max capture, path segmenting, schema-path formatting, Ctrl+F focus gating, deprecated-path filtering, metadata annotations, hierarchy leaf counts, and tree/table state round-trip.

## Phase 5B - Logs Pane

- Status: source/origin/time-mode, level-mask, expandable-detail, and selected-row navigation slices integrated; broader log-specific polish remains.
- Logs panes now persist source filter, origin filter, route/boot/wall time display mode, level mask, and selected log index alongside existing text, legacy level, follow, and max-row state.
- Logs filtering now supports source and origin constraints while retaining message/source/function/context text matching.
- Logs filtering can include or exclude Trace, Info, Warn, Error, and Critical buckets independently; legacy `min_level` state migrates to the equivalent mask.
- Logs table time cells can display route, boot, or wall time.
- Message rows with function/context details can expand inline to show those details.
- Logs panes now expose Prev/Next/Clear selected-row navigation over the current filtered result set, disable follow while navigating, and highlight/scroll to the selected log row.
- Helper coverage verifies state round-trip, source/origin filters, time-mode labels/formatting, legacy text/level filtering, level-mask migration/filtering, detail formatting, selected-row position, wraparound navigation, and empty-result handling.

## Phase 5C - Map Pane

- Status: engagement-colored trace, follow/zoom-to-cursor/pan/link controls, setting-backed drag direction, Overpass vector fetch, disk cache, configurable cache root, cached-feature rendering, stale-cache suppression, and cache-management popup/stat refresh integrated; final parity/perf audit remains.
- Map trace points now carry a timeline span classification from the shared `TimelineModel`.
- Map panes render the GPS trace segment-by-segment using the same engaged/alert/disengaged color vocabulary as the timeline.
- Map pane state now persists follow mode, basemap visibility, zoom, and optional center coordinates; panes expose Follow, Basemap, zoom in/out, Fit, Maps, Fetch, and Clear controls.
- Map canvas supports cursor-anchored mouse-wheel zoom and left-drag pan, disabling follow when the user manually navigates; Settings can choose direct/natural map dragging or inverted drag direction.
- Map helpers now generate a Google Maps directions URL for the current trace.
- Added a session-owned `MapBasemapManager` worker so cache reads, cache clears, and explicit Overpass fetches stay off the UI draw path.
- Map helpers now build corridor Overpass queries, parse `out tags geom` road/water JSON, serialize JSON cache files under the user cache directory, and reject stale cache keys.
- Cached roads and water render under the GPS trace using the existing Loggy projection only when the loaded basemap key matches the current trace, avoiding stale map features while the correct cache/fetch is pending.
- Map panes now expose a `Cache` popup with cache root/path copy, formatted cache size, current cache key, nonblocking stats refresh, and clear controls; inline cache status stays on its own row so controls do not clip in compact panes.
- `MapBasemapManager` now accepts a settings-backed cache root, uses that root for cache reads/writes/stats/clear, clears stale in-memory basemaps when roots change, and ignores stale worker completions from a previous root generation.
- Seeded stream/demo data now includes a small synthetic GPS trace, making Map panes and cache controls immediately testable in stream captures before route ingestion or network-backed demo data arrives.
- Helper coverage verifies route trace preparation, `None`/`Engaged`/`AlertWarning` point classification, map state round-trip/clamping, cursor-anchored zoom math, natural/inverted map-pan math, Google Maps URL formatting, Overpass request generation, basemap JSON parsing, cache JSON/file round-trip, cache summary/status formatting, current-trace basemap matching, manager cache load, custom-root switching, async cache clear, and nonblocking cache-stat refresh.

## Phase 5E - Computed Series

- Status: plot-level derivative/scale, session-backed transform materialization, custom-Python evaluator, dependency invalidation helpers, editor/template/preview slices, and Computed pane CSV export controls integrated; progressive async recompute remains.
- Plot series state now parses, preserves, and applies Jotpluggler-style `transform: derivative` with automatic or fixed `derivative_dt`.
- Plot series state now parses, preserves, and applies Jotpluggler-style `transform: scale` with `scale` and `offset`.
- Plot rendering now honors imported `#RRGGBB` curve colors and gives same-source transformed curves distinct ImPlot item IDs.
- Store batches can now replace `/computed/` series without wiping route-ingested series, full-resolution series snapshots are available for recompute/export, and drains report touched series paths.
- Added backend computed-series spec/status helpers plus materialization for scale/offset and derivative transforms into stable `/computed/...` output paths.
- Computed specs now expose small dependency invalidation helpers that report whether linked, additional, or code-referenced source paths were touched and should trigger recompute.
- Session startup now normalizes plot transform and custom-Python series to backend `/computed/...` output paths while preserving source/operation/custom metadata in pane state.
- Session frame drains synchronously materialize backend transform and custom-Python specs when source series are first available or touched again, making computed outputs plottable as normal Store series.
- Added Loggy-owned `backend/math_eval.py`, preserving Jotpluggler's manifest semantics for `linked_source`, `additional_sources`, `time`, `value`, `t(path)`, `v(path)`, and resampled `v1...` variables.
- Custom Python materialization now snapshots referenced Store series, evaluates Python through the temp-file subprocess contract, supports scalar array and `(time, value)` tuple returns, captures code-discovered `t()`/`v()` dependencies, and reports per-series status.
- Added a Computed pane with persisted name, linked source, additional sources, globals, function body, selected template, last output path, and run status.
- Computed pane templates now populate derivative, difference, smoothing, and integral Python snippets, while preserving an editable custom body.
- Computed pane Run materializes a Custom Python output into the shared Store, exposes the resulting `/computed/...` path as a drag source, supports clipboard copy, and renders a bounded preview table from the committed output.
- Computed output series are exportable through a generic `series_csv` helper once materialized into the shared Store, and the Computed pane now exposes persisted Export CSV path, native Browse path selection, Copy CSV, Save CSV, and export status controls.
- Helper coverage verifies transform parsing, display-option state round-trip preservation, scale output, automatic/fixed-`dt` derivative output, tracker values, store replacement/full snapshots, computed output path/dependency/status/export helpers, dependency invalidation helpers, transform materialization, custom Python linked-source/tuple/additional-source/code-reference materialization, recompute replacement, session plot-state normalization/materialization, computed editor state/template/source/spec/preview helpers, and series CSV export.

## Phase 5F - Presets, Layouts, and Route Controls

- Status: file-backed preset, bundled Jotpluggler layout, route-chip/info slices, and first comma API route-browser picker integrated; richer route metadata, autosave/runtime menu polish, full custom-Python semantics, and decoded camera playback remain.
- `layouts/cabana.json` and `layouts/jotpluggler.json` are now real non-empty Loggy workspace layouts instead of empty preset fallback markers.
- Session startup now prefers `openpilot/tools/loggy/layouts/<preset>.json` for `--preset`/launcher startup and falls back to C++ defaults only if the file is missing.
- All 17 current Jotpluggler layout JSON files are bundled under `openpilot/tools/loggy/layouts/` and load by `--layout <name>`.
- The file-backed and C++ fallback Jotpluggler presets now include a Computed tab so the custom-series editor is discoverable without a custom layout.
- Workspace loading now recognizes Jotpluggler layout leaves with `curves`, `kind`, and `camera_view`, converting them into Loggy plot, map, and camera panes.
- Imported plot panes preserve Jotpluggler curve color, derivative/scale metadata, custom Python specs, Y limits, and original range metadata in pane state for later computed-series support.
- Runtime split rendering now submits a trailing dummy item after split-node cursor extension, fixing ImGui assertion failures seen while capturing imported layouts.
- Runtime footer now exposes a `Route` chip for route sessions, showing current route spec, selector description, segment counts, series count, and CAN count.
- Runtime now exposes a `Route` popup from the footer and File menu with route open/reopen, Copy, Copy Onebox, slice edit/apply, rlog/qlog/auto selector changes, and Useradmin/Connect link buttons.
- Runtime now exposes a `Browse...` action inside the `Route` popup, fetching comma devices/routes through `PyDownloader::getDevices()` and `PyDownloader::getDeviceRoutes()` on worker threads and draining results on the UI frame.
- Route browser backend helpers cover Cabana-style period labels, comma API URL construction, preserved/non-preserved route JSON parsing, route list labels, and stricter route text validation.
- Session now exposes `restartRoute()`, which stops live polling, clears route-owned Store/timeline/log/camera state, resets playback/view range, and starts a fresh route ingest for selector or slice changes.
- Route helper coverage verifies selector labels, route full-spec formatting, slice parsing, Useradmin/Connect URL formatting, and invalid `restartRoute()` handling without starting network ingestion.
- Workspace smoke coverage loads the bundled preset JSONs plus all 17 bundled Jotpluggler layout files, verifies representative plot, map/camera, scale, and custom-Python metadata import, and verifies `jotpluggler` preset session startup comes from the JSON file.

## Phase 6A / 6B - Live Sources

- Status: local MSGQ, remote ZMQ, Device Bridge, SocketCAN, Panda USB with bus-speed/CAN-FD configuration, live VisionIPC camera panes, live extraction boundary, Store rolling-buffer eviction, stream pause/follow controls, route browser/comma API picker, and live source reconnect UI integrated.
- Added `backend/live` with local/remote source config, stream address normalization, service subscription filtering, live poller snapshots, live batch merging, and worker-thread MSGQ/ZMQ poll loops using generated cereal services and queue sizes.
- Live cereal extraction now feeds serialized `cereal::Event` messages through Loggy's generated `appendEventReader` path into `StoreBatch`, while also extracting selfdrive timeline spans and log/alert rows for the existing Timeline and Logs panes.
- Live batches preserve the first observed `carParams.carFingerprint`, allowing live streams to share the same DBC auto-load path as route ingestion.
- Live source config now carries an explicit source kind and the Live Source modal exposes Local MSGQ, Remote ZMQ, Device Bridge, Panda USB, and SocketCAN choices without breaking legacy address-based `--stream --address` behavior.
- Added `--socketcan <dev>` CLI startup and a Linux SocketCAN poller that reads classic/CAN-FD frames from a CAN raw socket and stages them as live CAN `StoreBatch` events through the same UI-thread drain path.
- Added `--device <host>` CLI startup and a Linux Device Bridge source that starts `openpilot/cereal/messaging/bridge` with a CAN whitelist, subscribes to the local bridge output through the existing MSGQ live path, and kills/waits the bridge child during poller teardown.
- Added `--panda`, `--panda-serial <serial>`, and repeatable `--panda-bus <bus>:<can_kbps>[:fd|off[:data_kbps]]` CLI startup plus a `backend/panda_live` adapter around Cabana's Qt-free Panda USB wrapper. The adapter converts Panda CAN frames into Loggy live CAN batches, defaults Panda safety to `NO_OUTPUT`, applies per-bus CAN speed and CAN-FD data-speed configuration, and preserves an empty serial as "first Panda".
- The Live Source popup now exposes three Panda bus rows with CAN speed, CAN-FD enable, and data-speed controls. Defaults match Cabana (`500` kbps CAN, `2000` kbps data speed, FD disabled), supported speed choices match Cabana, and unsupported values normalize back to safe defaults before a stream starts.
- Cabana's Panda USB control-loop error handling now marks non-timeout libusb failures disconnected instead of retrying forever, so Loggy can surface no-hardware/permission failures and exit cleanly during captures.
- `_loggy` now has an explicit SCons dependency on `openpilot/cereal/messaging/bridge` so the device bridge binary is available for CLI/runtime Device Bridge startup.
- `--stream` now starts the local live poller in addition to the existing seeded stream fallback, and `--stream-buffer <seconds>` configures the live follow window exposed in the status bar.
- Session drains pending live batches before the UI-thread Store drain, updates live route/follow range from incoming data, merges live timeline/log side data, and surfaces live connection/message/batch/error status in the runtime footer.
- Store now exposes a UI-thread `trimBefore()` retention boundary, and stream frames prune committed series, CAN events, live logs, and live timeline spans behind the configured live buffer start after draining staged data.
- Runtime footer now exposes `Pause Live` / `Resume Live` and `Follow live` controls backed by Session methods; follow re-enables an immediate jump to the newest live window, while paused live pollers drop pending data without blocking teardown.
- Runtime now exposes a `Live Source` popup from the footer and File menu, with address and buffer controls wired to Session reconnect/stop methods; reconnect normalizes local addresses and restarts the poller without restarting the app.
- Remote non-local `--address` now uses `BridgeZmqSubSocket`/`BridgeZmqPoller` directly instead of the local MSGQ `SubSocket::create` path, avoiding the non-local MSGQ assert and the old process-global `ZMQ` switch.
- Stream-mode camera panes now request live road/driver/wide-road frames directly from VisionIPC `camerad` instead of relying on cereal encode-index subscriptions, preserving the existing route `FrameReader` path for non-stream sessions and showing connection/unsupported statuses when no camera stream is available.
- Store query handling now keeps zero-span single-point chunks queryable when the point timestamp is inside the requested range, which live first samples rely on.
- Live/store/workspace smoke coverage verifies stream address normalization, service filtering, serialized cereal event extraction into Store series, in-process remote ZMQ receive/parse/publish, timeline spans, log rows, Session reconnect/pause/follow state, and physical Store eviction of stale series chunks, series paths, CAN events, CAN IDs, and clipped coverage ranges.

## Phase 7D - Hardening

- Status: first shutdown hardening, README, headless preset-capture helper, and smoke-baseline runner slices integrated; broader parity audit, perf audit, LOC pass, and final CI adoption remain.
- Runtime now installs scoped SIGINT/SIGTERM handlers for `run()`, records shutdown requests without doing non-signal-safe work in the handler, restores prior handlers on exit, and lets the normal GLFW/ImGui/Session teardown path run.
- SIGTERM smoke coverage launches `_loggy` under an isolated Xvfb display, sends SIGTERM, and verifies the process exits cleanly with status `0`.
- Runtime now exposes a compact Help/About popup from `Help -> About Loggy` and F1, covering core shortcuts, route/live controls, and workspace basics without interfering with Space playback or F12 HUD toggling.
- Extraction signatures now explicitly use the global replay `::Event` type, and the dead DBC manager observer helper has been removed.
- A low-effort UI review pass found `File -> Live Source...` was unreachable in route/demo mode; Runtime now keeps the menu item available and opens a route-mode explanatory modal instead of silently blocking the action.
- A follow-up low-fast UI review found Help/Route modal Escape dismissal was unreliable under xdotool. Runtime now captures GLFW Escape press events, forwards them to ImGui, and closes Help, Route, Live Source, and Remote Routes modals without letting text inputs swallow the key.
- Added `openpilot/tools/loggy/README.md` with build/run/test/capture commands and `tests/capture_presets.sh`, which captures Cabana and Jotpluggler presets under a virtual display using the known-good venv Xvfb wrapper when available.
- Added the `loggy_smoke_build` SCons alias and `tests/run_smoke.sh`, giving CI and manual baseline checks one command for the deterministic non-GUI smoke suite plus optional route ingest and virtual-display preset captures.
- Added a compact native-dialog helper that builds and launches `zenity`/`kdialog`/`osascript` file/folder selections, plus `tests/native_dialog_smoke` pure command/result coverage; pane tests do not launch GUI dialogs.
- REVIEW cleanup A1/A2 removed the dead in-process DBC observer system (`Event<>`, seven DBC manager event members, emission sites, and the orphan header) and replaced the mutable `PaneRegistry`/registration step with a compile-time static pane table using plain function pointers.
- Follow-up low-effort QA used window-targeted Xvfb/xdotool clicks and found no reproducible Loggy UI defect in F1/F12/Space, Binary Defined/Suppress, DBC auto/manual controls, route popup, or Live Source route-mode flows; the earlier missed-click finding remains a harness-targeting limitation rather than an app bug.
- Continuous low-fast QA rechecked F1/Help dismissal, context-menu dismiss, and Live Source modal Escape behavior using disposable Xvfb sessions. It did not find actionable reproducible UI bugs; route/footer/Binary targeting gaps remain harness limitations rather than confirmed app defects.
- Follow-up low-fast QA found `tests/capture_presets.sh` could collide in parallel because it defaulted to a fixed `:87` display; the helper now chooses an unused display unless `LOGGY_CAPTURE_DISPLAY` is explicit and reports a per-display Xvfb log on startup failure.
- Latest low-fast QA ran a bounded read-only visual pass with an explicit private Xvfb display for every GUI launch, found no reproducible Signal/DBC or camera/map visual regression, and reported all app/Xvfb processes torn down at command exit.
- Low-fast QA sidecar for the Map cache slice ran read-only Cabana preset visual coverage under private Xvfb, found no reproducible UI bug, and reported a black menu-click artifact as harness noise rather than an app regression; artifacts: `/tmp/loggy_qa_root_baseline.png` and `/tmp/loggy_qa_menu93.png`.
- Low-fast QA sidecar for the Settings app-preferences slice ran read-only deterministic smoke coverage plus private-Xvfb Cabana preset clicks/captures, found no reproducible GUI bug, blank pane, text overlap, or bad menu state, and reported `xdotool windowactivate` as a no-window-manager harness limitation rather than an app issue; artifacts include `/tmp/loggy-qa-captures/loggy-cabana-capture.png`, `/tmp/loggy-qa-captures/loggy-jotpluggler-capture.png`, and `/tmp/loggy-qa.hSmrCy/*.png`.
- Low-fast QA sidecar for the configurable map-cache-root slice ran read-only `run_smoke.sh --skip-build --with-capture` plus private-Xvfb Jotpluggler Settings/Map checks, found the Settings modal and presets rendering cleanly, and reported only a synthetic-click harness miss when trying to open the Map cache popup; artifacts include `/tmp/loggy_menu_open_205.png`, `/tmp/loggy_settings_modal_207.png`, `/tmp/loggy_map_cache_popup_208.png`, `/tmp/loggy_map_cache_popup_209.png`, `/tmp/loggy-cabana-capture.png`, and `/tmp/loggy-jotpluggler-capture.png`.
- Low-fast QA sidecar for the map drag-direction slice ran read-only deterministic smokes, virtual-display preset captures, and private-Xvfb Settings/map-drag checks, found no reproducible UI defect, and reported `xdotool windowactivate` failure as a no-window-manager harness issue worked around with direct window-targeted events; artifacts include `/tmp/loggy-qa-captures/interactive2/settings.png`, `/tmp/loggy-qa-captures/mapdrag/before.png`, `/tmp/loggy-qa-captures/mapdrag/after.png`, `/tmp/loggy-qa-captures/loggy-cabana-capture.png`, and `/tmp/loggy-qa-captures/loggy-jotpluggler-capture.png`.
- Low-fast QA sidecar for the theme slice ran read-only Settings/theme interaction coverage under private `Xvfb :97`, switched Darcula to Light and back, found no crash or obvious visual/layout regression, and reported explicit app/Xvfb cleanup; artifacts include `/tmp/loggy_initial_97.png`, `/tmp/loggy_settings_open_97.png`, `/tmp/loggy_theme_light_97.png`, and `/tmp/loggy_theme_darcula_97.png`.
- Low-fast QA sidecar for the native-dialog slice ran read-only route/live/map/camera/computed exploratory coverage under private `Xvfb :123` and `:124`, found no reproducible UI bug, and reported explicit app/Xvfb/helper cleanup; artifacts include `/tmp/loggy_qa_1783284155/*.png` and `/tmp/loggy_cam_qa_1783284189/*.png`.
- Low-fast QA sidecar for the message-editor slice ran read-only Cabana/Jotpluggler/camera-map exploratory coverage under private Xvfb displays `:90`, `:92`, `:94`, `:95`, `:96`, and `:97`, found no reproducible visual glitch or crash, and reported all app/Xvfb/helper processes cleaned up; artifacts include `/tmp/loggy_qa/cabana_initial2.png`, `/tmp/loggy_qa/dbc_tab.png`, `/tmp/loggy_qa/jotpluggler_initial.png`, and `/tmp/loggy_qa/cameras_map.png`.
- Smoke-baseline runner evidence: `LOGGY_SMOKE_JOBS=1 openpilot/tools/loggy/tests/run_smoke.sh`.
  Result: `loggy_smoke_build` built `_loggy` plus all smoke binaries; workspace passed silently; transport passed; DBC parser passed (`51 assertions in 5 test cases`); DBC commands passed (`100 assertions in 5 test cases`); settings passed (`34 assertions in 3 test cases`); store/scheduler passed (`122 assertions in 10 test cases`, with expected video-open diagnostic); live passed (`61 assertions in 3 test cases`); export passed (`21 assertions in 1 test case`); computed passed (`63 assertions in 2 test cases`); panes passed (`567 assertions in 15 test cases`); extract passed (`30 assertions in 3 test cases`). The default runner skipped real-route ingest and GUI capture unless explicitly requested.
- Settings theme evidence: focused `scons --cache-disable -j1 openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/settings_smoke && openpilot/tools/loggy/tests/settings_smoke` passed (`59 assertions in 3 test cases`); virtual-display capture smoke `LOGGY_CAPTURE_DIR=/tmp/loggy-theme-slice-captures openpilot/tools/loggy/tests/run_smoke.sh --skip-build --with-capture` passed and produced `/tmp/loggy-theme-slice-captures/loggy-cabana-capture.png` plus `/tmp/loggy-theme-slice-captures/loggy-jotpluggler-capture.png`; private-Xvfb manual captures show the Settings Theme combo at `/tmp/loggy_settings_theme_popup.png` and persisted Light theme runtime at `/tmp/loggy_light_theme_capture.png`.
- Native dialog/export chooser evidence: focused `scons --cache-disable -j1 openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/native_dialog_smoke openpilot/tools/loggy/tests/panes_smoke openpilot/tools/loggy/tests/export_smoke` passed, followed by `native_dialog_smoke`, `export_smoke` (`21 assertions in 1 test case`), and `panes_smoke` (`647 assertions in 15 test cases`). Private-Xvfb captures show DBC Browse/Choose controls at `/tmp/loggy_native_dialog_dbc_only.png`, History export Browse controls at `/tmp/loggy_native_dialog_history_button.png`, a real Zenity CSV save dialog on private display at `/tmp/loggy_native_dialog_save_csv.png`, and successful cancel/resume at `/tmp/loggy_native_dialog_save_csv_after_cancel.png`.
- DBC message-editor evidence: focused `scons --cache-disable -j1 openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/dbc_commands openpilot/tools/loggy/tests/panes_smoke` passed, followed by `dbc_commands` (`136 assertions in 7 test cases`) and `panes_smoke` (`668 assertions in 16 test cases`). Private-Xvfb capture `/tmp/loggy_message_editor_signal_pane_waited.png` shows the Signal pane's Message editor with name, bytes, transmitter, comment, Apply/Reset/Undo/Redo controls, and DBC-backed signal rows after demo data load.
- Find Signal candidate-name evidence: focused `scons --cache-disable -j1 openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/panes_smoke && openpilot/tools/loggy/tests/panes_smoke` passed (`676 assertions in 16 test cases`). Private-Xvfb capture `/tmp/loggy_analysis_find_signal_name.png` shows the Analysis tab's Find Signal pane with the new candidate `Name` field beside bus/address controls.
- (Historical per-slice smoke-runner logs compacted 2026-07-05; see git history. Latest run:)
- Current smoke-baseline runner after REVIEW A1/A2 cleanup: `LOGGY_SMOKE_JOBS=1 openpilot/tools/loggy/tests/run_smoke.sh`.
  Result: `loggy_smoke_build` rebuilt `_loggy`, `workspace_smoke`, `dbc_parser`, `dbc_commands`, `export_smoke`, and `panes_smoke`; workspace passed silently; transport passed; DBC parser passed (`51 assertions in 5 test cases`); DBC commands passed (`136 assertions in 7 test cases`); settings passed (`59 assertions in 3 test cases`); native dialog smoke passed silently; store/scheduler passed (`122 assertions in 10 test cases`, with expected video-open diagnostic); live passed (`61 assertions in 3 test cases`); export passed (`21 assertions in 1 test case`); computed passed (`71 assertions in 2 test cases`); panes passed (`676 assertions in 16 test cases`); extract passed (`30 assertions in 3 test cases`). Route ingest and GUI capture remain opt-in flags for the runner.

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
  Result: all DBC tests passed (`51 assertions in 5 test cases`).
- Full current smoke: `scons --cache-disable -j$(nproc) openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/workspace_smoke openpilot/tools/loggy/tests/dbc_parser openpilot/tools/loggy/tests/dbc_commands openpilot/tools/loggy/tests/transport_smoke openpilot/tools/loggy/tests/settings_smoke openpilot/tools/loggy/tests/store_scheduler openpilot/tools/loggy/tests/live_smoke openpilot/tools/loggy/tests/export_smoke openpilot/tools/loggy/tests/computed_smoke openpilot/tools/loggy/tests/panes_smoke openpilot/tools/loggy/tests/extract_smoke openpilot/tools/loggy/tests/route_ingest_smoke`, followed by the non-network smoke binaries and one qlog route-ingest demo smoke.
  Results: workspace passed silently; DBC passed (`51 assertions in 5 test cases`); DBC commands passed (`73 assertions in 5 test cases`); transport printed `transport_smoke passed`; settings passed (`31 assertions in 3 test cases`); store/scheduler passed (`77 assertions in 7 test cases`); live passed (`30 assertions in 2 test cases`); export passed (`21 assertions in 1 test case`); computed passed (`63 assertions in 2 test cases`); panes passed (`438 assertions in 14 test cases`); extraction passed (`30 assertions in 3 test cases`); route ingest loaded one demo qlog segment in `1.62708s`, producing `12333` Store series, `446` CAN ids, `1` timeline span, and `956` log entries.
- Focused live-buffer retention smoke: `scons --cache-disable -j$(nproc) openpilot/tools/loggy/tests/store_scheduler openpilot/tools/loggy/tests/live_smoke openpilot/tools/loggy/tests/workspace_smoke openpilot/tools/loggy/_loggy && openpilot/tools/loggy/tests/store_scheduler && openpilot/tools/loggy/tests/live_smoke && openpilot/tools/loggy/tests/workspace_smoke`, followed by `panes_smoke`, `computed_smoke`, and `export_smoke`.
  Results: store/scheduler passed (`98 assertions in 8 test cases`, including Store retention trimming); live passed (`30 assertions in 2 test cases`); workspace passed silently; panes passed (`438 assertions in 14 test cases`); computed passed (`63 assertions in 2 test cases`); export passed (`21 assertions in 1 test case`).
- Focused live controls smoke: `scons --cache-disable -j$(nproc) openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/workspace_smoke openpilot/tools/loggy/tests/live_smoke && openpilot/tools/loggy/tests/workspace_smoke && openpilot/tools/loggy/tests/live_smoke`.
  Results: workspace passed silently with Session reconnect/pause/follow assertions; live passed (`30 assertions in 2 test cases`).
- Focused remote-ZMQ live smoke: `scons --cache-disable -j$(nproc) openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/live_smoke openpilot/tools/loggy/tests/workspace_smoke openpilot/tools/loggy/tests/panes_smoke && openpilot/tools/loggy/tests/live_smoke && openpilot/tools/loggy/tests/workspace_smoke && openpilot/tools/loggy/tests/panes_smoke`.
  Results: live passed (`35 assertions in 3 test cases`, including in-process BridgeZmq pub/sub ingestion into Store); workspace passed silently; panes passed (`438 assertions in 14 test cases`).
- SIGTERM shutdown smoke: `scons --cache-disable -j$(nproc) openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/workspace_smoke && openpilot/tools/loggy/tests/workspace_smoke`, then launched `DISPLAY=:88 openpilot/tools/loggy/_loggy --preset jotpluggler --width 640 --height 360 --show --no-hud`, sent SIGTERM, and waited for exit.
  Result: `sigterm_exit=0`.
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
- Settings modal evidence:
  - Focused smoke: `scons --cache-disable -j1 openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/workspace_smoke openpilot/tools/loggy/tests/settings_smoke && openpilot/tools/loggy/tests/workspace_smoke && openpilot/tools/loggy/tests/settings_smoke`. Result: workspace passed silently with manual override save-failure rollback coverage; settings passed (`38 assertions in 3 test cases`) with directory-path rejection coverage.
  - `/tmp/loggy_settings_modal_3.png` captured from a disposable private Xvfb `:103` session after opening `File -> Settings...`; visual check shows the settings modal centered over Cabana with config path, opendbc root field, DBC override field, Save/Reload/Clear Override/Cancel actions, and no label clipping.
  - Virtual-display capture smoke: `openpilot/tools/loggy/tests/run_smoke.sh --skip-build --with-capture`. Result: deterministic smokes passed, then private-Xvfb Cabana/Jotpluggler captures refreshed `/tmp/loggy-cabana-capture.png` and `/tmp/loggy-jotpluggler-capture.png`.
- Settings app-preferences evidence:
  - Focused smoke: `scons --cache-disable -j1 openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/settings_smoke openpilot/tools/loggy/tests/workspace_smoke && openpilot/tools/loggy/tests/settings_smoke && openpilot/tools/loggy/tests/workspace_smoke`. Result: `_loggy` linked, settings passed (`50 assertions in 3 test cases`) with target-FPS/HUD round-trip, legacy root-field compatibility, and clamping coverage; workspace passed silently.
  - `/tmp/loggy_settings_preferences_popup.png` captured from a disposable private Xvfb `:188` session after opening `File -> Settings...`; visual check shows Target FPS, the `15-240` range label, Frame-Time HUD checkbox, and existing DBC settings controls without clipping or overlap at 1280x720.
  - Virtual-display capture smoke: `LOGGY_CAPTURE_DIR=/tmp/loggy-settings-slice-captures openpilot/tools/loggy/tests/run_smoke.sh --skip-build --with-capture`. Result: deterministic smokes passed, then private-Xvfb Cabana/Jotpluggler captures were written to `/tmp/loggy-settings-slice-captures/loggy-cabana-capture.png` and `/tmp/loggy-settings-slice-captures/loggy-jotpluggler-capture.png`.
- Settings map-cache-root evidence:
  - Focused smoke: `scons --cache-disable -j1 openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/settings_smoke openpilot/tools/loggy/tests/workspace_smoke openpilot/tools/loggy/tests/panes_smoke && openpilot/tools/loggy/tests/settings_smoke && openpilot/tools/loggy/tests/workspace_smoke && openpilot/tools/loggy/tests/panes_smoke`. Result: `_loggy` linked, settings passed (`53 assertions in 3 test cases`) with map-cache-root persistence/malformed-field coverage, workspace passed silently with session startup applying the configured root, and panes passed (`638 assertions in 15 test cases`) with custom-root manager load/clear/stat coverage.
  - `/tmp/loggy_settings_cache_root_popup.png` captured from a disposable private Xvfb `:188` session after opening `File -> Settings...`; visual check shows Map cache root, Default action, effective root text, existing FPS/HUD controls, and no clipping or overlap at 1280x720.
  - Virtual-display capture smoke: `LOGGY_CAPTURE_DIR=/tmp/loggy-cache-root-slice-captures openpilot/tools/loggy/tests/run_smoke.sh --skip-build --with-capture`. Result: deterministic smokes passed, then private-Xvfb Cabana/Jotpluggler captures were written to `/tmp/loggy-cache-root-slice-captures/loggy-cabana-capture.png` and `/tmp/loggy-cache-root-slice-captures/loggy-jotpluggler-capture.png`.
- Settings map-drag-direction evidence:
  - Focused smoke: `scons --cache-disable -j1 openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/settings_smoke openpilot/tools/loggy/tests/panes_smoke && openpilot/tools/loggy/tests/settings_smoke && openpilot/tools/loggy/tests/panes_smoke`. Result: `_loggy` linked, settings passed (`55 assertions in 3 test cases`) with drag-direction persistence coverage, and panes passed (`647 assertions in 15 test cases`) with direct/natural and inverted map-pan helper coverage.
  - `/tmp/loggy_settings_drag_direction_popup.png` captured from a disposable private Xvfb `:188` session after opening `File -> Settings...`; visual check shows the Natural map drag checkbox with the existing FPS/HUD/cache-root settings and no clipping or overlap at 1280x720.
  - Virtual-display capture smoke: `LOGGY_CAPTURE_DIR=/tmp/loggy-drag-direction-slice-captures openpilot/tools/loggy/tests/run_smoke.sh --skip-build --with-capture`. Result: deterministic smokes passed, then private-Xvfb Cabana/Jotpluggler captures were written to `/tmp/loggy-drag-direction-slice-captures/loggy-cabana-capture.png` and `/tmp/loggy-drag-direction-slice-captures/loggy-jotpluggler-capture.png`.
- DBC source reassignment evidence:
  - `/tmp/loggy_dbc_assign_source.png` captured from `DISPLAY=:87 openpilot/tools/loggy/cabana --settings /tmp/loggy_assign_settings.json --width 1920 --height 1080 --show` after opening `opendbc_repo/opendbc/dbc/ford_lincoln_base_pt.dbc`, changing the loaded row Assign field from `all` to `1`, and clicking `Set`; visual check shows the loaded row `Sources` and `Assign` fields both at `1`, and `sed -n '1,200p' /tmp/loggy_assign_settings.json` confirmed `assignments: {"1": "opendbc_repo/opendbc/dbc/ford_lincoln_base_pt.dbc"}` with no stale `all` mapping.
- DBC signal edit evidence:
  - `/tmp/loggy_signal_edit_apply.png` captured from `DISPLAY=:87 openpilot/tools/loggy/cabana --settings /tmp/loggy_signal_edit_settings.json --width 1920 --height 1080 --show` after opening `/tmp/loggy_signal_edit.dbc`, switching to the Cabana tab, editing selected DBC signal `speed` to `vehicle_speed`, and clicking `Apply`; visual check shows the editor and table row renamed to `vehicle_speed` with Undo enabled.
  - `/tmp/loggy_signal_edit_undo.png` captured after clicking `Undo` in the same session; visual check shows the editor and table row restored to `speed` with Redo enabled.
  - Focused smoke: `scons --cache-disable -j1 openpilot/tools/loggy/tests/panes_smoke openpilot/tools/loggy/tests/dbc_commands && openpilot/tools/loggy/tests/panes_smoke && openpilot/tools/loggy/tests/dbc_commands`. Result: `panes_smoke` passed (`536 assertions in 14 test cases`) and `dbc_commands` passed (`84 assertions in 5 test cases`), including Signal precision/color auto-vs-override and undo/redo coverage.
  - Focused command-history smoke: `scons --cache-disable -j1 openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/dbc_commands openpilot/tools/loggy/tests/panes_smoke && openpilot/tools/loggy/tests/dbc_commands && openpilot/tools/loggy/tests/panes_smoke && git diff --check`. Result: `dbc_commands` passed (`100 assertions in 5 test cases`) and `panes_smoke` passed (`552 assertions in 15 test cases`), including DBC command-list and camera route-info overlay coverage.
- DBC signal remove evidence:
  - `/tmp/loggy_signal_remove_after.png` captured from `DISPLAY=:87 openpilot/tools/loggy/cabana --settings /tmp/loggy_signal_remove_settings.json --width 1920 --height 1080 --show` after opening `/tmp/loggy_signal_remove.dbc`, switching to the Cabana tab, selecting DBC signal `speed`, and clicking `Remove`; visual check shows `speed` removed, `flag` still present, and Undo enabled.
  - `/tmp/loggy_signal_remove_undo.png` captured after clicking `Undo` in the same session; visual check shows `speed` restored and Redo enabled.
- DBC signal value-description evidence:
  - `/tmp/loggy_signal_valdesc_apply.png` captured from `DISPLAY=:87 openpilot/tools/loggy/cabana --settings /tmp/loggy_signal_valdesc_settings.json --width 1920 --height 1080 --show` after autoloading `/tmp/loggy_signal_valdesc.dbc`, editing selected signal `speed` from `0 "stopped" 3 "cruise"` to `0 "stopped" 3 "cruise" 7 "fault"`, and clicking `Apply`; visual check shows the new Value Table text and Undo enabled.
  - `/tmp/loggy_signal_valdesc_undo.png` captured after clicking `Undo` in the same session; visual check shows the Value Table restored to `0 "stopped" 3 "cruise"` with Redo enabled.
- Binary drag-create signal evidence:
  - `/tmp/loggy_binary_create_after.png` captured from `DISPLAY=:87 openpilot/tools/loggy/cabana --demo --settings /tmp/loggy_binary_create_settings.json --width 1920 --height 1080 --show` after autoloading `/tmp/loggy_binary_create.dbc` with an empty message definition, then dragging across Binary row-0 bits 7 through 4 for selected CAN id `0:47`; visual check shows `Created DBC signal`, History rows decoded as `NEW_SIGNAL_1=2`, and Signal switched from bit candidates to `1 DBC signals` with `NEW_SIGNAL_1` selected and Undo enabled.
  - `/tmp/loggy_binary_create_undo.png` captured after clicking `Undo` in the same session; visual check shows History decoded values cleared and Signal returned to `64 bit candidates`.
  - Focused smoke: `scons --cache-disable -j1 openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/workspace_smoke openpilot/tools/loggy/tests/panes_smoke openpilot/tools/loggy/tests/dbc_commands && openpilot/tools/loggy/tests/workspace_smoke && openpilot/tools/loggy/tests/panes_smoke && openpilot/tools/loggy/tests/dbc_commands && git diff --check`. Result: workspace passed silently, `panes_smoke` passed (`543 assertions in 14 test cases`), and `dbc_commands` passed (`84 assertions in 5 test cases`), including Binary DBC-overlap rejection coverage.
- Binary resize evidence:
  - Focused smoke: `scons --cache-disable -j1 openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/panes_smoke && openpilot/tools/loggy/tests/panes_smoke`. Result: panes passed (`621 assertions in 15 test cases`), including DBC signal-at-bit lookup, little-endian Binary resize draft generation, and overlap rejection against other DBC signals.
  - Virtual-display capture smoke: `openpilot/tools/loggy/tests/run_smoke.sh --skip-build --with-capture`. Result: all deterministic smokes passed, preset captures generated `/tmp/loggy-cabana-capture.png` and `/tmp/loggy-jotpluggler-capture.png`, and visual inspection showed normal Cabana/Jotpluggler layouts without clipped controls or host-desktop windows.
- Signal sparkline evidence:
  - `/tmp/loggy_signal_sparkline.png` captured from `DISPLAY=:87 openpilot/tools/loggy/_loggy --preset cabana --demo --settings /tmp/loggy_signal_sparkline_settings.json --width 1920 --height 1080 --show --no-hud` with `/tmp/loggy_signal_sparkline.dbc`; visual check shows the Spark window control at `30s`, the Signal table Spark column, decoded DBC row `byte0_value` with value `32`, and a visible inline sparkline.
- Plot display-control evidence:
  - `/tmp/loggy_plot_style_scatter.png` captured from `DISPLAY=:87 openpilot/tools/loggy/_loggy --preset jotpluggler --demo --width 1920 --height 1080 --show --no-hud` after selecting `Scatter`; visual check shows the Style combo at `Scatter`, scatter markers in the Plot pane, and a hover tooltip with sampled `vEgo`/`aEgo` values.
  - `/tmp/loggy_plot_y_limits.png` captured after setting Y max to `10`; visual check shows the persisted `[auto, 10]` indicator and the plot Y axis clamped at `10`.
  - `/tmp/loggy_plot_hover_values.png` captured with the cursor over the plot; visual check shows tooltip rows for cursor time plus sampled `vEgo` and `aEgo` values while the legend remains present.
- Analysis tools evidence:
  - `/tmp/loggy_analysis_results.png` captured from `DISPLAY=:87 openpilot/tools/loggy/_loggy --preset cabana --demo --width 1920 --height 1080 --show --no-hud` after switching to the new `Analysis` tab and running both scans; visual check shows `Find Signal` with `Found 512 candidates` and rows for selected CAN id `0:47`, plus `Find Bits` with `Found 512 bit matches` ranked for source `0:47`.
  - Focused smoke after scan-history controls: `scons --cache-disable -j1 openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/panes_smoke && openpilot/tools/loggy/tests/panes_smoke`. Result: panes passed (`592 assertions in 15 test cases`), including Find Signal/Find Bits history serialization/application and stale history-index clamping.
- Browser live-value/sparkline evidence:
  - `/tmp/loggy_browser_values_sparklines.png` captured from `DISPLAY=:87 openpilot/tools/loggy/_loggy --preset jotpluggler --demo --width 1920 --height 1080 --show --no-hud`; visual check shows the Browser Spark window control at `30s`, `Value` and `Spark` columns, tracker-sampled values, and inline sparklines for visible route series.
- Browser tree/deprecated evidence:
  - `/tmp/loggy_browser_tree.png` captured from `DISPLAY=:87 openpilot/tools/loggy/_loggy --preset jotpluggler --demo --width 1920 --height 1080 --show --no-hud`; visual check shows the Browser path hierarchy, group leaf counts, live values, sparklines, Tree and Deprecated controls, separate series count, and no toolbar clipping.
- Browser metadata annotation evidence:
  - Focused smoke: `scons --cache-disable -j1 openpilot/tools/loggy/tests/panes_smoke openpilot/tools/loggy/_loggy && openpilot/tools/loggy/tests/panes_smoke && git diff --check`. Result: panes passed (`567 assertions in 15 test cases`), including Store metadata-only series paths, enum annotations, deprecated annotations, and Computed CSV export controls.
- Logs controls evidence:
  - `/tmp/loggy_logs_controls.png` captured from `DISPLAY=:87 openpilot/tools/loggy/_loggy --preset jotpluggler --demo --width 1920 --height 1080 --show --no-hud`; visual check shows the Logs pane with Source filter, Level/Origin/Time controls, Follow, row count, populated rows, and no toolbar overlap.
- Logs level-mask/detail evidence:
  - `/tmp/loggy_logs_mask.png` captured from `DISPLAY=:87 openpilot/tools/loggy/_loggy --preset jotpluggler --demo --width 1920 --height 1080 --show --no-hud`; visual check shows the `All` level-mask control, Source/Origin/Time/Follow controls, expandable detail arrows in message rows, populated logs, and no toolbar overlap.
  - Focused smoke after selected-row navigation: `scons --cache-disable -j1 openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/panes_smoke && openpilot/tools/loggy/tests/panes_smoke`. Result: panes passed (`592 assertions in 15 test cases`), including Logs selected-row persistence, position, wraparound navigation, and empty-result handling.
- Map engagement-color evidence:
  - `/tmp/loggy_map_engagement.png` captured from `DISPLAY=:87 openpilot/tools/loggy/_loggy --preset jotpluggler --demo --width 1920 --height 1080 --show --no-hud`; visual check shows the Map pane route trace rendered through timeline-derived colors with the tracker marker intact and no overlap.
- Map controls evidence:
  - `/tmp/loggy_map_controls.png` captured from `DISPLAY=:87 openpilot/tools/loggy/_loggy --preset jotpluggler --demo --width 1920 --height 1080 --show --no-hud`; visual check shows the Map pane with real GPS trace, Follow, zoom in/out, Fit, and Maps controls fitting without toolbar overlap.
- Map basemap/cache evidence:
  - Focused smoke: `scons --cache-disable -j$(nproc) openpilot/tools/loggy/tests/panes_smoke openpilot/tools/loggy/_loggy && openpilot/tools/loggy/tests/panes_smoke`.
    Result: panes passed (`469 assertions in 14 test cases`), including Overpass request/parser/cache helpers and `MapBasemapManager` cache load/clear.
  - Focused smoke after cursor-anchored zoom/cache-match polish: `scons --cache-disable -j1 openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/panes_smoke openpilot/tools/loggy/tests/computed_smoke && openpilot/tools/loggy/tests/panes_smoke && openpilot/tools/loggy/tests/computed_smoke`. Result: panes passed (`604 assertions in 15 test cases`), including cursor-anchored map zoom and current-trace basemap matching; computed passed (`71 assertions in 2 test cases`), including dependency invalidation helpers.
  - `/tmp/loggy_map_basemap_controls_final.png` captured from `DISPLAY=:95 openpilot/tools/loggy/_loggy --preset jotpluggler --demo --width 1280 --height 720 --show --no-hud`; visual check shows route-loaded Map pane with compact Basemap, Fetch, Clear, Fit, and Maps controls fitting without clipping.
  - `/tmp/loggy_map_basemap_fetch_attempt.png` captured after clicking Fetch in the same session; visual check shows Overpass roads/water rendered behind the colored GPS trace, status `Fetched basemap`, `525 feat`, and one cache file.
  - Focused smoke after cache-management popup/seeded-GPS slice: `scons --cache-disable -j1 openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/workspace_smoke openpilot/tools/loggy/tests/panes_smoke && openpilot/tools/loggy/tests/workspace_smoke && openpilot/tools/loggy/tests/panes_smoke`. Result: workspace passed silently with seeded GPS coverage; panes passed (`630 assertions in 15 test cases`) with cache summary/status formatting and manager stat-refresh coverage.
  - `/tmp/loggy_map_cache_popup_3.png` captured from a disposable private Xvfb `:107` session after launching `--preset jotpluggler --stream` and opening the Map pane `Cache` popup; visual check shows immediate seeded GPS trace, the cache popup root/size/key, Copy Path/Refresh/Clear actions, and the inline status on its own row without toolbar clipping.
  - Focused smoke after configurable cache-root slice: `scons --cache-disable -j1 openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/settings_smoke openpilot/tools/loggy/tests/workspace_smoke openpilot/tools/loggy/tests/panes_smoke && openpilot/tools/loggy/tests/settings_smoke && openpilot/tools/loggy/tests/workspace_smoke && openpilot/tools/loggy/tests/panes_smoke`. Result: settings passed (`53 assertions in 3 test cases`), workspace passed silently, and panes passed (`638 assertions in 15 test cases`) with `MapBasemapManager::setCacheRoot()` and custom-root cache load/clear/stat checks.
  - Virtual-display capture smoke after map cache changes: `openpilot/tools/loggy/tests/run_smoke.sh --skip-build --with-capture`. Result: deterministic smokes passed, then private-Xvfb Cabana/Jotpluggler captures refreshed `/tmp/loggy-cabana-capture.png` and `/tmp/loggy-jotpluggler-capture.png`.
- Plot transform evidence:
  - `/tmp/loggy_plot_transforms.png` captured from `DISPLAY=:87 openpilot/tools/loggy/_loggy --layout /tmp/loggy_transform_layout.json --stream --width 1280 --height 720 --output /tmp/loggy_plot_transforms.png`; visual check shows same-source scale and derivative curves rendered simultaneously with distinct legend rows, tracker values, imported colors, and no overlap.
- Plot interaction evidence:
  - `/tmp/loggy_plot_interactions.png` captured from `DISPLAY=:87 openpilot/tools/loggy/_loggy --preset jotpluggler --stream --width 1280 --height 720 --output /tmp/loggy_plot_interactions.png`; visual check shows the `+ Series` selector entry point, disabled Undo Zoom control, draggable series labels, and no plot-header overlap.
- Live stream status evidence:
  - `/tmp/loggy_live_stream_status.png` captured from `DISPLAY=:87 openpilot/tools/loggy/_loggy --preset jotpluggler --stream --stream-buffer 12 --width 1280 --height 720 --output /tmp/loggy_live_stream_status.png`; visual check shows local live status in the footer (`live connected`, message/batch counters, `12s buffer`) while preserving existing stream fallback data.
- Live controls evidence:
  - `/tmp/loggy_live_controls.png` captured from `DISPLAY=:89 openpilot/tools/loggy/_loggy --preset jotpluggler --stream --stream-buffer 12 --width 1280 --height 720 --output /tmp/loggy_live_controls.png --no-hud`; visual check shows `Pause Live`, checked `Follow live`, and the live status counters fitting in the footer without overlap.
- Live source popup evidence:
  - `/tmp/loggy_live_source_popup.png` captured after launching `DISPLAY=:90 openpilot/tools/loggy/_loggy --preset jotpluggler --stream --stream-buffer 12 --width 1280 --height 720 --show --no-hud` and clicking `Source`; visual check shows the modal address field, buffer control, Reconnect/Stop/Cancel buttons, and readable footer controls.
- Remote ZMQ status evidence:
  - `/tmp/loggy_remote_zmq_status.png` captured from `DISPLAY=:91 openpilot/tools/loggy/_loggy --preset jotpluggler --stream --address 192.0.2.1 --stream-buffer 12 --width 1280 --height 720 --output /tmp/loggy_remote_zmq_status.png --no-hud`; visual check shows remote address `192.0.2.1`, connected live status, and no remote-not-wired error.
- Computed backend evidence:
  - `openpilot/tools/loggy/tests/store_scheduler`, `openpilot/tools/loggy/tests/export_smoke`, and `openpilot/tools/loggy/tests/computed_smoke` verify replaceable `/computed/` Store series, full-resolution snapshots, touched-path drain reporting, generic series CSV export, stable computed output paths, dependency/status helpers, scale/offset materialization, automatic/fixed-`dt` derivative materialization, and recompute replacement without duplicate chunks.
  - Focused computed smoke after invalidation helpers: `scons --cache-disable -j1 openpilot/tools/loggy/tests/computed_smoke && openpilot/tools/loggy/tests/computed_smoke`. Result: computed passed (`71 assertions in 2 test cases`), covering linked, additional, code-referenced, and negative dependency touch cases.
- Session computed transform evidence:
  - `/tmp/loggy_session_computed.png` captured from `DISPLAY=:87 openpilot/tools/loggy/_loggy --layout /tmp/loggy_session_computed_layout.json --stream --width 1280 --height 720 --output /tmp/loggy_session_computed.png`; visual check shows source `vEgo`, backend-materialized `scaled`, and backend-materialized derivative `dv` series rendered together from a layout that only specified Jotpluggler-style transform metadata.
- Session custom Python evidence:
  - `/tmp/loggy_session_custom_python.png` captured from `DISPLAY=:87 openpilot/tools/loggy/_loggy --layout /tmp/loggy_session_custom_layout.json --stream --width 1280 --height 720 --output /tmp/loggy_session_custom_python.png`; visual check shows source `vEgo`, backend-materialized `scaled`, backend-materialized derivative `dv`, and backend-materialized custom Python `custom` (`return value * 3`) rendered together from plot metadata.
- Computed editor evidence:
  - `/tmp/loggy_computed_editor.png` captured from `DISPLAY=:87 openpilot/tools/loggy/_loggy --layout /tmp/loggy_computed_editor_layout.json --stream --width 1280 --height 720 --show` after clicking Run; visual check shows the Computed pane controls, `ok: 301 points`, a generated `/computed/...` output path with Copy affordance, and a populated preview table beside a live Plot pane.
  - Focused smoke from the Browser metadata run above verifies persisted Computed export path/status, default computed export path generation, and CSV output from a materialized custom Python series.
- Camera index pane evidence:
  - `/tmp/loggy_camera_index_panes.png` captured from `DISPLAY=:87 openpilot/tools/loggy/_loggy --layout cameras-and-map --stream --width 1280 --height 720 --output /tmp/loggy_camera_index_panes.png`; visual check shows the imported Road/Wide Road/Driver camera panes using real Loggy camera indexes with `1 files`, `301 frames`, current segment/decode/frame metadata, Fit controls, and no dummy pane rendering.
- Camera async decode-status evidence:
  - `/tmp/loggy_camera_decode_status.png` captured from `DISPLAY=:87 openpilot/tools/loggy/_loggy --layout cameras-and-map --stream --width 1280 --height 720 --output /tmp/loggy_camera_decode_status.png`; visual check shows each imported camera pane requesting a frame through the async decoder and surfacing the expected synthetic-file `failed to load camera segment` status without blocking or falling back to a dummy pane.
- Camera alert/engaged overlay evidence:
  - Focused smoke: `scons --cache-disable -j1 openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/panes_smoke && openpilot/tools/loggy/tests/panes_smoke`. Result: `507 assertions in 14 test cases`, including camera snapshot overlay classifications for disengaged, engaged, and alert-warning tracker times.
  - `/tmp/loggy_camera_overlay_alert.png` captured from a disposable `:122` Xvfb session after seeking the cameras-and-map stream layout to `18.20s`; visual check shows Road/Wide Road/Driver camera panes rendering bottom-right `alert warning` badges without overlapping camera status text.
- Camera route-info overlay evidence:
  - Focused smoke from the command-history run above verifies route no-video, indexed-frame, and unsupported-live overlay line generation through `build_camera_route_info_lines()`.
- Live VisionIPC camera evidence:
  - Focused smoke: `scons --cache-disable -j1 openpilot/tools/loggy/tests/store_scheduler && openpilot/tools/loggy/tests/store_scheduler`. Result: `122 assertions in 10 test cases`, including a synthetic VisionIPC server publishing an NV12 road-camera frame and `LiveCameraFrameSource` receiving/converting it to a decoded RGBA frame.
  - `/tmp/loggy_live_vipc_waiting.png` captured from `DISPLAY=:120 openpilot/tools/loggy/_loggy --layout openpilot/tools/loggy/layouts/cameras-and-map.json --stream --width 1280 --height 720 --output /tmp/loggy_live_vipc_waiting.png`; visual check shows Road/Wide Road/Driver panes in live VisionIPC mode with clean no-camera waiting status and no stale route-video fallback.
- Jotpluggler layout import evidence:
  - `/tmp/loggy_layout_longitudinal.png` captured from `DISPLAY=:87 openpilot/tools/loggy/_loggy --layout openpilot/tools/jotpluggler/layouts/longitudinal.json --demo --width 1280 --height 720 --output /tmp/loggy_layout_longitudinal.png`; visual check shows the imported four-plot longitudinal layout with carried Y-limit labels and no ImGui assertion.
  - `/tmp/loggy_layout_cameras_map.png` captured from `DISPLAY=:87 openpilot/tools/loggy/_loggy --layout openpilot/tools/jotpluggler/layouts/cameras-and-map.json --demo --width 1280 --height 720 --output /tmp/loggy_layout_cameras_map.png`; visual check shows the imported map plus Road/Wide Road/Driver Camera panes in the expected split layout.
- File-backed preset/bundled layout evidence:
  - `/tmp/loggy_file_preset_jotpluggler.png` captured from `DISPLAY=:87 openpilot/tools/loggy/_loggy --preset jotpluggler --stream --width 1280 --height 720 --output /tmp/loggy_file_preset_jotpluggler.png`; visual check shows the file-backed `jotpluggler.json` Browser/Plot/Logs/Map preset with the explicit two-series plot state.
  - `/tmp/loggy_bundled_layout_longitudinal.png` captured from `DISPLAY=:87 openpilot/tools/loggy/_loggy --layout longitudinal --stream --width 1280 --height 720 --output /tmp/loggy_bundled_layout_longitudinal.png`; visual check shows `--layout longitudinal` resolving from Loggy's bundled layouts directory and rendering the four-plot layout.
  - Focused workspace smoke from the Binary/autosave run above covered runtime-facing workspace draft load semantics through `Session::workspace_layout_path()`, `Session::loaded_workspace_draft()`, `load_workspace_or_draft()`, and draft clearing against a unique temporary layout.
- Route chip/info evidence:
  - Focused smoke: `scons --cache-disable -j$(nproc) openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/workspace_smoke && openpilot/tools/loggy/tests/workspace_smoke`.
    Result: workspace passed silently, including route full-spec/link/slice helper assertions and invalid route restart validation.
  - `/tmp/loggy_route_footer_before_popup.png` captured from `DISPLAY=:96 openpilot/tools/loggy/_loggy --preset jotpluggler --demo --width 1280 --height 720 --show --no-hud`; visual check shows the footer `Route` chip with route spec, selector description, segment counts, series count, and CAN count.
  - `/tmp/loggy_route_popup.png` captured after clicking the footer `Route` chip in the same session; visual check shows the Route popup with Open, Copy, Copy Onebox, Slice/Apply Slice, selector combo, Useradmin, and Connect controls over a loaded route.
- Route browser evidence:
  - Focused smoke: `scons --cache-disable -j1 openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/workspace_smoke openpilot/tools/loggy/tests/live_smoke && openpilot/tools/loggy/tests/workspace_smoke && openpilot/tools/loggy/tests/live_smoke`.
    Result: workspace passed silently with route-browser period/url/parse/validation assertions; live passed (`50 assertions in 3 test cases`).
  - `/tmp/loggy_route_browser_popup.png` captured from `DISPLAY=:107 openpilot/tools/loggy/_loggy --preset jotpluggler --demo --width 1280 --height 720 --show --no-hud` after opening `Route -> Browse...`; visual check shows the Remote Routes modal with Device combo, Refresh, Period combo, fetched route rows, hover fullname tooltip, and Open/Cancel controls fitting without overlap.
- Modal dismissal evidence:
  - Low-fast QA reported stuck Help/Route modals via `/tmp/loggy_root87_help_open.png` and `/tmp/loggy_root87_route_popup1.png`.
  - `/tmp/loggy_modal6_help_after_escape.png` shows the Help modal dismissed after F1 then Escape under Xvfb.
  - `/tmp/loggy_modal8_route_open.png` and `/tmp/loggy_modal8_route_after_escape_0_4s.png` show the Route popup before and after Escape, with the modal gone in the next captured frame.
- Fast-worker integration evidence:
  - Help/F1 worker build: `scons -j4 openpilot/tools/loggy/_loggy`; screenshot `/tmp/loggy_help_overlay.png` captured by the worker from `_loggy --width 1280 --height 720 --output /tmp/loggy_help_overlay.png --show --no-hud`.
  - Focused pane integration smoke: `scons --cache-disable -j$(nproc) openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/panes_smoke && openpilot/tools/loggy/tests/panes_smoke`.
    Result: panes passed (`494 assertions in 14 test cases`), including DBC-backed message name/node filtering, active/all message rows, Browser Ctrl+F focus gating, and schema-path formatting.
  - DBC fingerprint/override focused smoke: `scons --cache-disable -j$(nproc) openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/settings_smoke openpilot/tools/loggy/tests/live_smoke openpilot/tools/loggy/tests/workspace_smoke openpilot/tools/loggy/tests/panes_smoke && openpilot/tools/loggy/tests/settings_smoke && openpilot/tools/loggy/tests/live_smoke && openpilot/tools/loggy/tests/workspace_smoke && openpilot/tools/loggy/tests/panes_smoke`.
    Results: settings passed (`34 assertions in 3 test cases`); live passed (`37 assertions in 3 test cases`, including live `carParams.carFingerprint` capture); workspace passed silently with manual DBC override startup/generated-name coverage; panes passed (`495 assertions in 14 test cases`).
  - DBC auto/manual visual evidence:
    - `/tmp/loggy_dbc_auto_controls.png` captured from `DISPLAY=:99 openpilot/tools/loggy/_loggy --layout /tmp/loggy_dbc_only_layout.json --demo --width 1280 --height 720 --show --no-hud`; visual check shows Route DBC car fingerprint `FORD_BRONCO_SPORT_MK1`, auto/active `ford_lincoln_base_pt`, the manual `DBC Override` field, Auto/Apply buttons, and a loaded generated DBC row.
    - `/tmp/loggy_livesource_route_mode_fixed.png` captured from route-mode Cabana demo after selecting `File -> Live Source...`; visual check shows the Live Source modal opens and explains live source is unavailable while a route is open.
  - Latest broad current smoke: `scons --cache-disable -j$(nproc) openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/workspace_smoke openpilot/tools/loggy/tests/live_smoke openpilot/tools/loggy/tests/panes_smoke openpilot/tools/loggy/tests/store_scheduler openpilot/tools/loggy/tests/computed_smoke openpilot/tools/loggy/tests/export_smoke openpilot/tools/loggy/tests/extract_smoke openpilot/tools/loggy/tests/route_ingest_smoke openpilot/tools/loggy/tests/settings_smoke openpilot/tools/loggy/tests/dbc_parser openpilot/tools/loggy/tests/dbc_commands`, followed by each smoke binary.
    Results: workspace passed silently; live passed (`37 assertions in 3 test cases`); panes passed (`504 assertions in 14 test cases`); store/scheduler passed (`98 assertions in 8 test cases`, with expected video-open diagnostic); computed passed (`63 assertions in 2 test cases`); export passed (`21 assertions in 1 test case`); extract passed (`30 assertions in 3 test cases`); route ingest exited `0` after one qlog segment with `10976` Store series and `189` CAN ids; settings passed (`34 assertions in 3 test cases`); DBC parser passed (`51 assertions in 5 test cases`); DBC commands passed (`73 assertions in 5 test cases`).
- Binary defined-bit suppression evidence:
  - Focused smoke: `scons --cache-disable -j$(nproc) openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/panes_smoke && openpilot/tools/loggy/tests/panes_smoke`.
    Result: panes passed (`504 assertions in 14 test cases`), including Binary pane defined-bit state round-trip and DBC mask detection.
- README/headless capture evidence:
  - Build/capture: `scons --cache-disable -j$(nproc) openpilot/tools/loggy/_loggy openpilot/tools/loggy/loggy_cabana openpilot/tools/loggy/loggy_jotpluggler && LOGGY_CAPTURE_DISPLAY=:102 bash openpilot/tools/loggy/tests/capture_presets.sh`.
    Result: generated nonblank `/tmp/loggy-cabana-capture.png` and `/tmp/loggy-jotpluggler-capture.png` preset captures at 1280x720.
- SocketCAN live-source evidence:
  - Focused smoke: `scons --cache-disable -j$(nproc) openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/live_smoke openpilot/tools/loggy/tests/workspace_smoke && openpilot/tools/loggy/tests/live_smoke && openpilot/tools/loggy/tests/workspace_smoke`.
    Result: live passed (`48 assertions in 3 test cases`, including source-kind labels, SocketCAN availability helper, direct CAN batch staging); workspace passed silently with explicit SocketCAN restart/config validation and explicit Remote ZMQ localhost preservation.
  - Headless startup capture: `/tmp/loggy_socketcan_status.png` from `DISPLAY=:104 openpilot/tools/loggy/_loggy --preset cabana --socketcan vcan-test --width 1280 --height 720 --output /tmp/loggy_socketcan_status.png --no-hud`; visual check shows `SocketCAN vcan-test`, live controls, seeded CAN panes, and clean `Failed to find SocketCAN device vcan-test` status without a crash.
  - Broad current smoke after SocketCAN: `scons --cache-disable -j$(nproc) openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/workspace_smoke openpilot/tools/loggy/tests/live_smoke openpilot/tools/loggy/tests/panes_smoke openpilot/tools/loggy/tests/store_scheduler openpilot/tools/loggy/tests/computed_smoke openpilot/tools/loggy/tests/export_smoke openpilot/tools/loggy/tests/extract_smoke openpilot/tools/loggy/tests/route_ingest_smoke openpilot/tools/loggy/tests/settings_smoke openpilot/tools/loggy/tests/dbc_parser openpilot/tools/loggy/tests/dbc_commands`, followed by each smoke binary.
    Results: workspace passed silently; live passed (`48 assertions in 3 test cases`); panes passed (`504 assertions in 14 test cases`); store/scheduler passed (`98 assertions in 8 test cases`, with expected video-open diagnostic); computed passed (`63 assertions in 2 test cases`); export passed (`21 assertions in 1 test case`); extract passed (`30 assertions in 3 test cases`); route ingest exited `0` after one qlog segment with `10976` Store series and `189` CAN ids; settings passed (`34 assertions in 3 test cases`); DBC parser passed (`51 assertions in 5 test cases`); DBC commands passed (`73 assertions in 5 test cases`).
- Device Bridge live-source evidence:
  - Focused smoke: `scons --cache-disable -j$(nproc) openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/live_smoke openpilot/tools/loggy/tests/workspace_smoke && openpilot/tools/loggy/tests/live_smoke && openpilot/tools/loggy/tests/workspace_smoke && ls -l openpilot/cereal/messaging/bridge`.
    Result: `_loggy` built with the bridge dependency, `openpilot/cereal/messaging/bridge` was present, live passed (`50 assertions in 3 test cases`, including Device Bridge labels/targets), and workspace passed silently with explicit Device Bridge startup-config preservation.
  - Headless startup capture: `/tmp/loggy_device_bridge_status.png` from `DISPLAY=:105 openpilot/tools/loggy/_loggy --preset cabana --device 192.0.2.1 --width 1280 --height 720 --output /tmp/loggy_device_bridge_status.png --no-hud`; visual check shows `Device Bridge 192.0.2.1`, live controls, seeded CAN panes, and no lingering `openpilot/cereal/messaging/bridge` process after capture exit.
  - `git diff --check` passed after the Device Bridge slice.
  - Broad current smoke after Device Bridge: `scons --cache-disable -j$(nproc) openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/workspace_smoke openpilot/tools/loggy/tests/live_smoke openpilot/tools/loggy/tests/panes_smoke openpilot/tools/loggy/tests/store_scheduler openpilot/tools/loggy/tests/computed_smoke openpilot/tools/loggy/tests/export_smoke openpilot/tools/loggy/tests/extract_smoke openpilot/tools/loggy/tests/route_ingest_smoke openpilot/tools/loggy/tests/settings_smoke openpilot/tools/loggy/tests/dbc_parser openpilot/tools/loggy/tests/dbc_commands`, followed by each smoke binary.
    Results: workspace passed silently; live passed (`50 assertions in 3 test cases`); panes passed (`504 assertions in 14 test cases`); store/scheduler passed (`98 assertions in 8 test cases`, with expected video-open diagnostic); computed passed (`63 assertions in 2 test cases`); export passed (`21 assertions in 1 test case`); extract passed (`30 assertions in 3 test cases`); route ingest exited `0` after one qlog segment with `10976` Store series and `189` CAN ids; settings passed (`34 assertions in 3 test cases`); DBC parser passed (`51 assertions in 5 test cases`); DBC commands passed (`73 assertions in 5 test cases`).
- Panda USB live-source evidence:
  - Focused smoke: `scons --cache-disable -j1 openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/workspace_smoke openpilot/tools/loggy/tests/live_smoke && openpilot/tools/loggy/tests/workspace_smoke && openpilot/tools/loggy/tests/live_smoke`.
    Result: `_loggy`, `workspace_smoke`, and `live_smoke` built with `backend/panda_live` and Cabana `panda.cc`; workspace passed silently with explicit Panda restart/config preservation; live passed (`53 assertions in 3 test cases`, including Panda labels/targets and availability helpers).
  - Headless startup capture: `/tmp/loggy_panda_status.png` from `timeout --kill-after=2s 20s xvfb-run -a openpilot/tools/loggy/_loggy --preset cabana --panda --width 1280 --height 720 --output /tmp/loggy_panda_status.png`; visual check shows `Panda USB first Panda`, live controls, seeded CAN panes, and clean `Failed to connect to Panda USB: No Panda USB detected` status without a crash.
  - Panda bus-config focused smoke: `scons --cache-disable -j1 openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/workspace_smoke openpilot/tools/loggy/tests/live_smoke && openpilot/tools/loggy/tests/workspace_smoke && openpilot/tools/loggy/tests/live_smoke`.
    Result: workspace passed silently with custom Panda bus config preservation and unsupported-speed normalization; live passed (`61 assertions in 3 test cases`, including Panda speed choice/default normalization coverage).
  - Panda bus-config CLI check: `openpilot/tools/loggy/_loggy --panda-bus 0:333` exits `2` with `Invalid Panda bus config: invalid Panda CAN speed`.
  - Configured Panda startup capture: `/tmp/loggy_panda_bus_status.png` from `timeout --kill-after=2s 20s xvfb-run -a openpilot/tools/loggy/_loggy --preset cabana --panda-bus 0:250:fd:1000 --panda-bus 1:125:off:5000 --width 1280 --height 720 --output /tmp/loggy_panda_bus_status.png`.
  - Configured Panda popup capture: `/tmp/loggy_panda_bus_popup.png` from a disposable `:118` Xvfb session after opening the footer Source popup; visual check shows Bus 0 `250` + FD + `1000`, Bus 1 `125` + FD off + disabled `5000`, and Bus 2 default `500`/`2000`, with no clipping or overlap.
  - Broad current smoke after Panda USB: `scons --cache-disable -j1 openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/workspace_smoke openpilot/tools/loggy/tests/live_smoke openpilot/tools/loggy/tests/panes_smoke openpilot/tools/loggy/tests/store_scheduler openpilot/tools/loggy/tests/computed_smoke openpilot/tools/loggy/tests/export_smoke openpilot/tools/loggy/tests/extract_smoke openpilot/tools/loggy/tests/route_ingest_smoke openpilot/tools/loggy/tests/settings_smoke openpilot/tools/loggy/tests/dbc_parser openpilot/tools/loggy/tests/dbc_commands`, followed by each smoke binary.
    Results after bus-config controls: workspace passed silently; live passed (`61 assertions in 3 test cases`); panes passed (`504 assertions in 14 test cases`); store/scheduler passed (`98 assertions in 8 test cases`, with expected video-open diagnostic); computed passed (`63 assertions in 2 test cases`); export passed (`21 assertions in 1 test case`); extract passed (`30 assertions in 3 test cases`); route ingest exited `0` after one qlog segment with `10976` Store series and `189` CAN ids; settings passed (`34 assertions in 3 test cases`); DBC parser passed (`51 assertions in 5 test cases`); DBC commands passed (`73 assertions in 5 test cases`).
- Broad current smoke after route browser/modal fixes: `scons --cache-disable -j1 openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/workspace_smoke openpilot/tools/loggy/tests/live_smoke openpilot/tools/loggy/tests/panes_smoke openpilot/tools/loggy/tests/store_scheduler openpilot/tools/loggy/tests/computed_smoke openpilot/tools/loggy/tests/export_smoke openpilot/tools/loggy/tests/extract_smoke openpilot/tools/loggy/tests/route_ingest_smoke openpilot/tools/loggy/tests/settings_smoke openpilot/tools/loggy/tests/dbc_parser openpilot/tools/loggy/tests/dbc_commands`, followed by each smoke binary.
  Results: workspace passed silently; live passed (`50 assertions in 3 test cases`); panes passed (`504 assertions in 14 test cases`); store/scheduler passed (`98 assertions in 8 test cases`, with expected video-open diagnostic); computed passed (`63 assertions in 2 test cases`); export passed (`21 assertions in 1 test case`); extract passed (`30 assertions in 3 test cases`); route ingest exited `0` after one qlog segment with `10976` Store series and `189` CAN ids; settings passed (`34 assertions in 3 test cases`); DBC parser passed (`51 assertions in 5 test cases`); DBC commands passed (`73 assertions in 5 test cases`).
- Style-review alignment evidence:
  - `loggy_plan.md` now makes `openpilot/tools/loggy/REVIEW.md` authoritative, removes stale registry/observer guidance, describes the static pane table and subsystem-owned runtime split, and requires workers to keep helpers file-local and avoid nullable out-params/test-shaped APIs.
  - Added `openpilot/tools/loggy/tests/style_ratchet.sh` and wired it into `tests/run_smoke.sh`.
    Current ratchet baselines: error out-params `0`, null-guard writes `0`, `std::function` `0`, getter pairs `0`, pane header functions `14`, named header structs `79`, product LOC `21300`, `runtime.cc` `850`, pane-local statics `0`, backend header camelCase methods `0`.
  - Verification: `LOGGY_CAPTURE_DIR=/tmp/loggy-plan-final-capture LOGGY_SMOKE_JOBS=4 openpilot/tools/loggy/tests/run_smoke.sh --skip-build --with-capture` passed. It ran the style ratchet, all deterministic smokes, and private-Xvfb preset captures.
  - Capture evidence: `/tmp/loggy-plan-final-capture/loggy-cabana-capture.png` and `/tmp/loggy-plan-final-capture/loggy-jotpluggler-capture.png`; both were visually checked as nonblank/coherent.
  - REVIEW v2 layering fix: moved `MessageSummary`, message-id helpers, and CSV writers from `panes/messages.*`/`backend/export.h` into `backend/csv.{h,cc}`; deleted the `backend/export.h` shim. `panes/messages.h` is back to the single draw-function header shape.
  - REVIEW v2 struct/pimpl pass: removed the route segment-log header struct, locked pimpl comments to the FFmpeg/VisionIPC and libusb-backed classes, and lowered named header structs from `82` to `79`.
  - REVIEW v2 HUD fix: frame HUD p99 is now computed from a rolling five-second sample window instead of a lifetime/stale startup maximum.
  - REVIEW v2 one-click selection fix: Messages row activation now updates the shared selection on the first left-click. Xvfb evidence from display `:133`: `/tmp/loggy_review_v2_afterfix_click0_before.png`, `/tmp/loggy_review_v2_afterfix_click1_once.png`, `/tmp/loggy_review_v2_afterfix_click2_twice.png`; Binary-pane crop diff was `3479` pixels before-to-once and `0` pixels once-to-twice.
  - REVIEW v2 broad verification: `scons --cache-disable -j4 openpilot/tools/loggy/_loggy openpilot/tools/loggy/tests/workspace_smoke openpilot/tools/loggy/tests/panes_smoke openpilot/tools/loggy/tests/store_scheduler openpilot/tools/loggy/tests/computed_smoke` passed, followed by `workspace_smoke`, `panes_smoke`, `store_scheduler`, `computed_smoke`, and `style_ratchet.sh`.
  - REVIEW v2 capture verification: `LOGGY_CAPTURE_DIR=/tmp/loggy-review-v2-final2 LOGGY_SMOKE_JOBS=4 openpilot/tools/loggy/tests/run_smoke.sh --skip-build --with-capture` passed. Captures: `/tmp/loggy-review-v2-final2/loggy-cabana-capture.png` and `/tmp/loggy-review-v2-final2/loggy-jotpluggler-capture.png`, both visually checked as coherent.

## Completion push — phase 1 (2026-07-05)

- Baseline commit `1976165e5` ("loggy: baseline before completion push") through phase-1 commit
  `30fad8895` ("loggy: phase 1 — preset panes, rolling HUD p99, QA fixes"): presets gained
  camera/plot panes with a plot-dominant Jotpluggler layout; the frame HUD p99 is now a rolling
  5-second ring buffer instead of a lifetime max; the wheel-scroll false-highlight and the
  1280x720 clipping bug are fixed; single-click Messages selection and the backend
  rename/layering work (REVIEW v1/v2) were verified as already landed rather than re-done.
- Verification sweep (this pass): grepped the whole `openpilot/tools/loggy/` tree (product +
  tests) for rename-sweep casualties — `.set_/init_/get_` calls on capnp reader/builder objects,
  ImGui/ImPlot/GLFW identifiers, and `_(`-suffixed member-call typos. Found and fixed one
  remaining casualty: `DeviceBridgeProcess::start_()` in `backend/live.cc` (plus its call site in
  `LiveCerealPoller::start()` and the `"Failed to start_ messaging bridge"` error string) — a
  sibling of the already-fixed `tests/live_smoke.cc` `setEnabled`/`poller.start_(` bugs, renamed
  back to `start()` / `"Failed to start messaging bridge"`. No other casualties found; all capnp
  accessor calls (`event.getCarState()`, `car_state.setVEgo()`, etc.) and all ImGui/ImPlot/GLFW
  calls remain correctly camelCase.
- `LOGGY_SMOKE_JOBS=$(nproc) openpilot/tools/loggy/tests/run_smoke.sh` is green after the fix:
  style ratchet passed at all current baselines, and every deterministic smoke binary passed
  (transport, dbc_parser `51/5`, dbc_commands `136/7`, settings `59/3`, store_scheduler `122/10`,
  live_smoke `61/3`, computed_smoke `71/2`, panes_smoke `4/1`, extract_smoke `30/3`; workspace_smoke
  passed silently).
- **Correction note:** `tests/panes_smoke.cc` is now an intentional 81-line boundary stub (the
  old 668-assertion helper-level suite was retired by REVIEW v1-A4; it only checks
  `Store::series_paths_matching` today). REVIEW.md's own coverage note (section 6, "Coverage
  note") already flags this and asks whether the Find Signal / history-comparator logic that
  moved out of the gutted pane-level tests kept coverage in the backend suites. Checked
  `tests/dbc_commands_test.cc` (DBC edit/add/remove signal and message undo-redo cases) and
  `tests/store_scheduler_test.cc` (Store/SegmentScheduler/video/live-camera cases): **neither
  references `find_signal`, `history_log`, or any comparator logic** — grepping the whole
  `tests/` tree turns up zero hits for `make_find_signal_job`, `prepare_find_signal_candidates`,
  or `prepare_history_log_page`. This logic (in `panes/find_signal.cc` and `panes/historylog.cc`)
  has in fact *not* moved into `backend/`; it still lives at the pane layer and has no dedicated
  test coverage anywhere — `workspace_smoke.cc` only asserts that a pane of type `"find_signal"`
  is registered/present in layouts, which is workspace-wiring coverage, not logic coverage. **This
  is a real gap**, not just a stale doc: Find Signal candidate scoring/ranking and the
  history-log page/CSV-export helpers are currently unverified by any automated test.
