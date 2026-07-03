# cabana: Qt → Dear ImGui migration

Replace cabana's Qt dependency with Dear ImGui, keeping 1:1 functionality and
near-visual matching. cabana is the **last Qt user in this repo** — when this
migration finishes, Qt (and its qmake detection, `rcc` assets, and Qt5Charts)
is deleted from openpilot entirely.

The stack is already proven in-repo by `tools/jotpluggler`:
Dear ImGui 1.92 (docking branch) + ImPlot + GLFW/OpenGL3 from the prebuilt
`imgui` wheel (commaai/dependencies `release-imgui`), Inter/JetBrainsMono +
bootstrap-icons fonts, `replay_lib` for routes/video, libusb for panda.

## Strategy

Strangler migration with a **shared core**:

1. De-Qt cabana's data layer in place (`streams/`, `dbc/`, `commands.cc`,
   `settings.cc`). The existing Qt UI keeps working on top of it through a
   thin bridge, so the tool never breaks mid-migration.
2. Grow the new ImGui UI in this directory (`tools/cabana/imgui/`, binary
   `_cabana_imgui`) panel by panel, sharing that core.
3. At parity: flip the entrypoint, delete the Qt UI and all Qt build support.

jotpluggler scaffolding is copied, not shared, during the migration (runtime,
theme, capture, camera view, timeline, byte widgets). Extracting a common
imgui lib for both tools is a post-flip cleanup, not a blocker.

### Core design change: signals → poll

`AbstractStream` currently pushes to the UI through Qt signals with queued
connections. The de-Qt'd core inverts this: producer threads enqueue under the
existing mutex, and the UI thread calls `stream->update()` once per frame,
which drains the queue, refreshes `last_msgs`, and reports what changed (new
messages, events merged, seek completed, sources updated). Async replay/live
callbacks queue internally and deliver on `update()`.

- ImGui UI: calls `update()` at frame start — natural immediate-mode fit.
- Qt UI (during migration): a ~100-line `QObject` bridge calls `update()`
  from a `QTimer` at `settings.fps` and re-emits the same signals widgets
  already connect to.

Mechanical swaps in the core: `QString`→`std::string`,
`QColor`→RGBA struct (bit-castable to `ImU32`), `QRegularExpression`→`std::regex`,
`QThread`→`std::thread`, `QProcess`→`posix_spawn`, `QUndoStack`→small custom
undo stack, `QSettings`→JSON (clean break; Qt `saveState()` blobs are replaced
by imgui-side layout persistence). Keep cabana's own DBC parser/serializer
(`dbc/dbcfile.cc` does round-trip with header preservation and has the only
real test coverage) — jotpluggler's `dbc.h` is read-only and stays where it is.

## Phases

Each phase lands as one or more PRs; Qt cabana stays fully functional until
the flip in Phase 8.

### Phase 0 — shell (this directory) ✅ in progress
- [x] `_cabana_imgui` build target (no Qt anywhere in its dep tree)
- [x] GLFW + ImGui + ImPlot runtime (docking enabled)
- [x] Fonts: Inter Regular/SemiBold + JetBrainsMono + bootstrap icons merged
- [x] Theme tokens: light + dark ("Darcula", from `utils/util.cc setTheme()`)
- [x] CLI parity with `cabana.cc` (flags parsed, streams stubbed until later phases)
- [x] Welcome screen, menu bar + status bar skeleton
- [x] Headless `--output foo.png` capture for screenshots/golden images

### Phase 1 — de-Qt the core (enables everything else)
Sub-PRs in order; existing Qt UI adapts via the bridge and stays green:
- [ ] `dbc/`: QString/QColor/QRegularExpression out; port Catch2 tests
- [ ] `commands.cc`: custom UndoStack (with command merging + clean tracking)
- [ ] `settings.cc`: JSON persistence, drop Qt blob fields
- [ ] `streams/`: poll-based `update()` API, `std::thread`, posix_spawn
- [ ] Qt bridge + widget adaptation; add headless stream-core tests (merge/seek/mux)

### Phase 2 — messages table + binary view (usable read-only tool)
- [ ] Replay stream wiring (open route from CLI, transport bar: pause/seek/speed)
- [ ] Messages table: sort, per-column filters, live byte cells with fade
      coloring, suppress-bits controls, active/inactive dimming
- [ ] Binary view: bit grid, hex column, bit-flip heatmap, hover/selection
- [ ] History log (read-only)

### Phase 3 — signal editing (cabana's core job at parity)
- [ ] Signal tree: inline edit, color chips, sparklines, value descriptions
- [ ] Drag-on-bits to create/resize signals (`drag_direction` setting honored)
- [ ] Message edit dialog, undo/redo UI, multi-bus DBC management
- [ ] DBC save/load/save-as/clipboard, recent files, fingerprint auto-load

### Phase 4 — charts (ImPlot)
- [ ] Chart panes with signal selector; drag-drop signals between charts
- [ ] Box-zoom + zoom undo stack driving `setTimeRange`; scrub cursor synced
      to playback; value tooltips; line/step/scatter series types
- [ ] Envelope decimation (jotpluggler `plot.cc`) replaces SegmentTree caching

### Phase 5 — video + timeline
- [ ] Camera pane: VisionIPC client thread → GL texture (upload path from
      jotpluggler `CameraFeedView`), camera tabs
- [ ] Timeline slider: alert/engagement coloring, qlog thumbnails on hover
- [ ] Full transport parity (speed dropdown, skip-to-end, time display modes)

### Phase 6 — live streams
- [ ] Panda (USB), SocketCAN, msgq/ZMQ device streams on the ported core
- [ ] Stream selector startup dialog + per-stream config forms
- [ ] Live logging (`log_livestream`)

### Phase 7 — tools & dialogs
- [ ] Find signal, find similar bits
- [ ] Route browser (comma API), route info, CSV export
- [ ] Settings dialog, keyboard shortcuts parity, help overlay

### Phase 8 — flip
- [ ] Parity checklist audit + side-by-side screenshots (both themes)
- [ ] Session state save/restore (layout, selection, open charts)
- [ ] Rename binary/entrypoint, deprecation window for `_cabana`
- [ ] Delete Qt UI sources, Qt SConscript branches, CI Qt deps

## Verification

- Screenshot every PR via `--output` headless capture; golden-image
  comparisons once panels stabilize.
- Phase 1 makes the DBC tests Qt-free (always built) and adds stream-core
  tests (event merge, seek, mux decode) that previously needed a QObject.
- Keep Qt cabana running side-by-side for A/B until the flip.
- Demo-route smoke test: open `--demo`, capture, assert message decode.

## Open decisions

- **Floating charts window**: try `ImGuiConfigFlags_ViewportsEnable`
  (docking branch supports multi-viewport); if janky on Linux/tiling WMs,
  charts stay dockable-in-window — the one accepted visual deviation.
- **HiDPI**: jotpluggler ignores monitor content scale; decide whether to add
  `glfwGetWindowContentScale` handling during the visual-polish pass.
- **Settings compat**: clean break from QSettings (fresh recent-files list).
