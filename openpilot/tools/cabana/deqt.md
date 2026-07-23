we're migrating cabana away from Qt and to eventually entirely use imgui

we are doing it incrementally, in small pieces that are easy to execute and verify.
we will repeat this until we're all done.

# Cabana Qt API inventory

these are all still in cabana. we remove them from this list once they're gone.
each bullet is an atomic unit of work.

our workflow is:
- pick the easiest of the bulleted items from below
- implement it and make] sure it builds
- spin up reviewer agents to review the code in a clean context and a separate one to click around in xvfb as a gui test
- then implement the fixes from the above reviewer agents

some rules
- do not add more Qt usage ever

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
- `QImage`, `QPixmap`, `QPixmapCache`, `QStaticText` (camera frames are no longer QImage; the only QImage left is the ImGuiHost readback blit, plus QPixmap for thumbnails, charts, and icon assets)
- `QFont`, `QFontDatabase`, `QFontMetrics`, `QTextDocument`
- `QStyle`, `QStyleOption`, `QStyleOptionFrame`, `QStyleOptionSlider`
- `QPoint`, `QPointF`, `QRect`, `QRectF`, `QRegion`
- `QSize`, `QSizeF`
- `QEvent`, `QPaintEvent`, `QResizeEvent`, `QShowEvent`, `QHideEvent`, `QCloseEvent`
- `QMouseEvent`, `QWheelEvent`, `QNativeGestureEvent`, `QContextMenuEvent`
- `QKeySequence`, `QShortcut`, `QToolTip`

# imgui beachhead

`ImGuiHost` (imguihost.{h,cc}) is a QWidget that renders an imgui frame offscreen
(EGL surfaceless-platform pbuffer on Linux, hidden GLFW window on Darwin; 3.3 core),
reads it back with glReadPixels, and QPainter-blits it — so Qt overlays can still
paint on top. All GL goes through imgui_impl_opengl3_loader.h; input is fed straight
into ImGuiIO from Qt events. `CameraWidget` is the first tenant: the vipc thread
fills raw RGBA vectors, the GUI thread uploads to a GL texture and draws it via the
background draw list.

migration path: move widgets into ImGuiHost tenants; later swap the pbuffer+readback
for an EGL window surface on winId() to drop the blit; endgame is a standalone GLFW
app (like tools/jotpluggler) and the Qt shell is deleted.
