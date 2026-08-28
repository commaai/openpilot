we're migrating cabana away from Qt and to eventually entirely use imgui

we are doing it incrementally, in small pieces that are easy to execute and verify.
we will repeat this until we're all done.

# Approach

The Qt-free core (streams/, dbc/, core/, utils/{util,strings,export}, commands, settings, routes, panda)
is done and shared. The frontend is rebuilt as a second binary next to the Qt one, modeled on
`tools/jotpluggler` (GLFW + imgui + implot, docking):

```
_cabana      Qt frontend (mainwin.cc + widgets), unchanged while porting
_cabana_ui   imgui frontend: tools/cabana/ui/, built from the base env, no Qt
```

Every Qt file is ported **line for line** into `ui/`: same file name, same classes, same method names,
same order of statements, same strings, same edge cases. Only the rendering/event layer changes from
Qt to imgui. When parity is reached the Qt files and the Qt SConscript env are deleted wholesale.
Behavior must be 1:1 with the Qt cabana of today; if the Qt code has a quirk, keep it and add a
one-line comment.

## ui/ layout

```
ui/main.cc              cabana.cc          arg parsing, stream construction, run()
ui/app.{h,cc}           (new)              GlfwRuntime, ImGuiRuntime, render loop, signal handling
ui/style.cc             utils::setTheme    fonts + theme (applyTheme() is safe at runtime)
ui/imgui_util.h         utils/qtutil.h     imgui helpers shared by all widgets
ui/mainwin.{h,cc}       mainwin.{h,cc}     MainWindow: menus, dock layout, file/stream actions, status bar
ui/widgets/             messageswidget, detailwidget, binaryview, signalview, historylog, videowidget, cameraview
ui/chart/               chart, chartswidget, signalselector, sparkline, tiplabel
ui/dialogs/             streamselector, routesdialog, settingsdialog + messagebox (QMessageBox), filedialog (QFileDialog)
ui/tools/               findsignal, findsimilarbits, routeinfo
```

## Porting conventions (read before porting a file)

- Keep the Qt file's structure: same includes order where applicable, same class names, same methods in the
  same order, same comments, same user-visible strings (`tr("...")` becomes a plain literal).
  A reviewer must be able to diff `foo.cc` against `ui/widgets/foo.cc` function by function.
- Rendering: each widget gets `void draw()` that renders inline into the current imgui window (immediate
  mode replaces `paintEvent` + event handlers). Top-level dock widgets are opened by `MainWindow`, which
  owns the `ImGui::Begin/End` and the window title; the widget draws its content. Child widgets are drawn
  by their parent's `draw()` in the same order the Qt layout added them.
- Qt signals become `Observable<...>` members with the same name (`tools/cabana/core/observable.h`);
  `QObject::connect` becomes `connections_.push_back(x.connect(...))`. Slots keep their names.
- Qt models (`QAbstractTableModel` etc.) become plain classes holding the same `items_`/rows and the same
  refresh methods; `dataChanged`/`layoutChanged` calls are dropped (imgui redraws every frame).
- `QTimer::singleShot(0, ...)` -> `MainWindow::nextFrame` / `utils::runOnMainThread` from a thread;
  periodic `QTimer` -> check `ImGui::GetTime()` in `draw()`.
- Modal dialogs (`exec()`) are non-blocking: `MessageBox::{information,warning,question}` and
  `FileDialog::{getOpenFileName,getSaveFileName,getExistingDirectory}` take a continuation; a method that
  needs the answer takes a `std::function<void()> then` parameter (see `MainWindow::remindSaveChanges`).
  Non-modal Qt dialogs (`show()`) become objects with `bool draw()` returning false once closed.
- `QString` -> `std::string` (the core already exposes std::string); `QColor` -> `CabanaColor`/`ImU32`;
  `QRect/QPoint` -> `ImRect/ImVec2`; `QFont` variants -> `pushMonoFont()`/`pushBoldFont()`.
- Icons: `utils::icon("name")` -> the bootstrap glyph merged into the fonts (see jotpluggler `icons.cc`);
  tool buttons are `ImGui::Button`/`SmallButton` with the glyph.
- `setWhatsThis(...)` -> `std::string whatsThis() const` returning the same text (used by the F1 overlay).
- Persisted Qt byte-array state (`saveHeaderState`, splitter/geometry) is out of scope for the port;
  keep the method and return `{}` with a TODO. Session ids (`serializeMessageIds`, `serializeChartIds`)
  keep working with std::string.
- Do not add Qt. Do not add new CLI options. ASCII only in code. No new third-party dependencies.
- Verification: `scons openpilot/tools/cabana/_cabana_ui`, then run under Xvfb, drive with xdotool,
  capture with `ffmpeg -f x11grab`; screenshots/GIFs go in the PR. Compare against `_cabana` side by side.
  Under Xvfb with mesa llvmpipe, pausing a replay can stall the main thread inside `glXSwapBuffers` (a gallium
  fence wait, ~30% of runs); it is not a cabana bug, unpause before quitting or run with `LP_NUM_THREADS=0`.

# Status

All Qt files are ported into `ui/` and `_cabana_ui` builds and runs the demo route with the full window
(messages, binary/signal/log views, video with timeline and camera tabs, charts, all dialogs and tools).
Verified against `_cabana` side by side under Xvfb; remaining work, each one a PR:

1. Known deviations to close (found by the side by side test, all in `ui/`):
   - dialog button order is `[OK] [Cancel]` left aligned (Qt: `[Cancel] [OK]` right aligned)
   - sparklines draw a marker per sample (Qt: bare polyline); expanded signal rows use input boxes (Qt: flat labels)
   - Find Similar Bits has no row number column; the "..." menu uses check marks, the speed menu a check mark (Qt: radio)
   - binary view hover paints the signal row saturated (Qt tints it); after Close stream every cell is hatch filled
   - F1 overlay: bold runs are flat; the Message View text was reworded
   - `OpenPandaWidget` opens the panda in its constructor like the Qt widget does (both connect over USB when the
     dialog opens); keep an eye on it when a panda is attached
2. Persisted Qt byte-array state: `saveHeaderState`/`restoreHeaderState`, window geometry, dock layout and the
   video/charts splitter (`Settings::geometry`, `window_state`, `video_splitter_state`, `message_header_state`)
   are stubs; store the imgui equivalents in `Settings` and drop the Qt fields.
3. Cutover: `cabana` wrapper runs `_cabana_ui`, delete the Qt files, the Qt env in `SConscript`, `assets/assets.qrc`;
   update README, CI, `tests/`.

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
- `ui/` files are line-for-line ports; do not restructure or "improve" while porting

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
