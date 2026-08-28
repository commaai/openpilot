we're migrating cabana away from Qt and to eventually entirely use imgui

we are doing it incrementally, in small pieces that are easy to execute and verify.
we will repeat this until we're all done.

# Approach

The Qt-free core (streams/, dbc/, core/, utils/{util,strings,export}, commands, settings, routes, panda)
is done and shared. Rather than de-Qt the remaining widgets in place, a second frontend is built
alongside the Qt one in `ui/`, modeled on `tools/jotpluggler` (GLFW + imgui + implot, docking).
Each Qt widget gets ported into `ui/` as an imgui panel; when the imgui build reaches parity the Qt
files and the Qt SConscript env are deleted wholesale.

```
_cabana      Qt frontend (mainwin.cc + widgets), unchanged while porting
_cabana_ui   imgui frontend: tools/cabana/ui/*.cc, built from the base env, no Qt
```

`ui/` layout (base infra, done):
- `main.cc`: arg parsing, same options as `cabana.cc`
- `app.h`: `Options`, `App`/`UiState`, runtime + panel entry points
- `app.cc`: `GlfwRuntime`, `ImGuiRuntime`, `createStream()` (same stream selection as cabana.cc), `startStream()`/`closeStream()` (own the stream, set `can`), render loop that drains `utils::drainMainThreadQueue()` every frame, SIGINT/SIGTERM exit
- `style.cc`: light/dark theme from `settings.theme`, Inter/JetBrainsMono fonts with bootstrap icons merged, `pushMonoFont()`/`pushBoldFont()`
- `layout.cc`: main menu bar, dockspace with default split (Messages left, Detail center, Charts bottom, Video right), status bar, panels, Open Stream / Settings / Error popups, shortcuts

Conventions for ported panels:
- one `drawXxxPanel(App *)` per Qt widget, in its own `ui/<name>.cc`; per-panel state lives in `UiState` (transient) or `Settings` (persisted), not in statics (immutable caches like the opendbc listing and the font handles are fine)
- talk to the core through `can`, `dbc()`, `UndoStack::instance()`, `settings`; core observables fire on the main thread (the render loop drains the queue), so handlers can touch UI state directly
- no `ImGui::GetIO().IniFilename`; layout persistence goes through `Settings` when it is ported
- verification: run `_cabana_ui --demo` under Xvfb, drive it with xdotool, capture with `ffmpeg -f x11grab`; screenshots/GIFs go in the PR. No test-only CLI options.

# Implementation plan

Each step is a PR. Line counts are the Qt code being replaced.

1. Messages panel (`messageswidget.cc`, 467): filter/search row, sortable columns, suppress-bits toggles,
   byte change coloring from `CanData::colors`, multi-line hex option, right-click menu,
   persist `settings.active_msg_id` / `selected_msg_ids`. Replaces the table stub in `layout.cc`.
2. Binary view (`binaryview.cc`, 510): bit grid with per-bit flip counts and signal coloring,
   drag-select to create a signal (`settings.drag_direction`), hover/tooltips.
3. Signal view (`signalview.cc`, 719) + sparkline (`chart/sparkline.cc`, 101): collapsible signal editor
   rows (name, size, endianness, factor/offset, min/max, unit, comment, value descriptions), inline
   sparkline via implot, add/remove/reorder through `UndoStack` commands.
4. History log (`historylog.cc`, 251): per-message event table with signal value columns, filter by value, time range follow.
5. Detail panel (`detailwidget.cc`, 323): tab bar binding 2-4 together, message edit form (name/size/node/comment),
   remove message, warnings for undefined/overlapping signals.
6. Video panel (`videowidget.cc`, 434; `cameraview.cc`, 121): VisionIPC frame -> GL texture, camera tabs,
   timeline slider with engaged/alert bands from the qlog, playback speed, time range (loop) selection,
   thumbnails on hover, `settings.absolute_time`.
7. Charts (`chart/chart.cc`, 769; `chartswidget.cc`, 660; `signalselector.cc`, 107; `tiplabel.cc`, 58): implot
   chart tiles with shared x-axis/zoom, series types (line/step/scatter), column count and range settings,
   signal selector popup, hover tip, undock to floating window, persist `settings.active_charts`.
8. Stream selector (`streamselector.cc`, 330; `routesdialog.cc`, 110; `routes.cc` already core): replay tab
   (route/local file, camera flags), routes browser from the comma API, panda (serial + bus config),
   device (msgq/zmq), socketcan; replaces the replay-only popup in `layout.cc`.
9. Settings dialog (`settingsdialog.cc`, 94): all fields of `CabanaSettingsState`, theme switch at runtime via
   `applyTheme()` (fonts are loaded once in `loadFonts()`, do not reload them), log path.
10. Tools (`tools/findsignal.cc`, 286; `findsimilarbits.cc`, 161; `routeinfo.cc`, 40): dockable tool windows.
11. File actions in `mainwin.cc` (752): open/save/save-as DBC (file picker: imgui text path + directory
    listing, no new native dependency), export CSV, recent files, opendbc list, clipboard, fingerprint ->
    DBC auto load (`dbc/car_fingerprint_to_dbc.json`), remind-save-changes on close, help overlay, session
    state (dock layout + geometry persisted in `Settings` instead of the Qt byte arrays), full screen.
12. Cutover: `cabana` wrapper runs `_cabana_ui`, delete `mainwin.*`, all `*widget*`, `chart/`, `tools/*.cc`
    Qt files, `utils/qtutil.*`, `utils/elidedlabel.*`, `assets/assets.qrc`, the Qt env in `SConscript`,
    `Settings` Qt byte-array fields; update README, CI, `tests/`.

# Cabana Qt API inventory

these are all still in cabana. we remove them from this list once they're gone.
each bullet is an atomic unit of work.

our workflow is:
- pick the easiest of the bulleted items from below
- implement it and make sure it builds
- spin up reviewer agents to review the code in a clean context and a separate one to click around in xvfb as a gui test
- then implement the fixes from the above reviewer agents

some rules
- do not add more Qt usage ever
- nothing in `ui/` may include Qt, and nothing in `ui/` may depend on a file that does

- `QObject`, `QMetaObject`, `QMetaType`
- `QApplication`, `QCoreApplication`, `QGuiApplication`
- `QString`, `QStringList`, `QStringBuilder`, `QChar`, `QLatin1Char`
- `QVariant`
- `QTimer`
- `QWidget`, `QMainWindow`, `QWindow`
- `QDialog`, `QDialogButtonBox`, `QMessageBox`, `QProgressDialog`
- `QFileDialog`
- `QMenu`, `QMenuBar`, `QAction`, `QActionGroup`, `QWidgetAction`
- `QToolBar`, `QToolButton`, `QPushButton`
- `QCheckBox`, `QRadioButton`, `QButtonGroup`, `QAbstractButton`
- `QComboBox`, `QLineEdit`, `QTextEdit`, `QSpinBox`, `QSlider`
- `QLabel`, `QGroupBox`, `QFrame`
- `QTabBar`, `QTabWidget`, `QSplitter`, `QScrollArea`, `QScrollBar`
- `QDockWidget`, `QStatusBar`, `QProgressBar`
- `QFormLayout`, `QGridLayout`, `QHBoxLayout`, `QVBoxLayout`
- `QSizePolicy`
- `QAbstractItemModel`, `QAbstractTableModel`, `QModelIndex`
- `QAbstractItemView`, `QTableView`, `QTreeView`
- `QTableWidget`, `QTableWidgetItem`, `QListWidget`, `QListWidgetItem`
- `QItemSelection`, `QItemSelectionModel`, `QItemSelectionRange`
- `QHeaderView`, `QStyledItemDelegate`, `QStyleOptionViewItem`
- `QValidator`, `QIntValidator`
- `QColor`, `QRgb`, `QPalette`
- `QBrush`, `QPen`
- `QPainter`, `QPainterPath`, `QStylePainter`
- `QImage`, `QPixmap`, `QPixmapCache`, `QStaticText`
- `QFont`, `QFontDatabase`, `QFontMetrics`, `QTextDocument`
- `QStyle`, `QStyleOption`, `QStyleOptionFrame`, `QStyleOptionSlider`
- `QPoint`, `QPointF`, `QRect`, `QRectF`, `QRegion`
- `QSize`, `QSizeF`
- `QEvent`, `QPaintEvent`, `QResizeEvent`, `QShowEvent`, `QCloseEvent`
- `QMouseEvent`, `QWheelEvent`, `QNativeGestureEvent`, `QContextMenuEvent`
- `QKeySequence`, `QShortcut`, `QToolTip`
