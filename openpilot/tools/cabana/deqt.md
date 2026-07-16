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

- `QObject`, `QMetaObject`, `QMetaType`
- `QApplication`, `QCoreApplication`, `QGuiApplication`
- `QCommandLineParser`
- `QString`, `QStringList`, `QStringBuilder`, `QChar`, `QLatin1Char`
- `QByteArray`
- `QVariant`
- `QVector`, `QMap`, `QSet`, `QPointer`
- `QJsonDocument`, `QJsonObject`, `QJsonArray`, `QJsonValue`
- `QSettings`
- `QDateTime`, `QLocale`
- `QFile`, `QFileInfo`, `QDir`, `QIODevice`, `QStandardPaths`
- `QProcess`
- `QThread`
- `QTimer`, `QBasicTimer`, `QTimerEvent`
- `QSocketNotifier`
- `QDebug`, `QMessageLogContext`
- `QWidget`, `QMainWindow`, `QWindow`
- `QDialog`, `QDialogButtonBox`, `QMessageBox`, `QProgressDialog`
- `QFileDialog`
- `QMenu`, `QMenuBar`, `QAction`, `QActionGroup`, `QWidgetAction`
- `QToolBar`, `QToolButton`, `QPushButton`
- `QCheckBox`, `QRadioButton`, `QButtonGroup`, `QAbstractButton`
- `QComboBox`, `QLineEdit`, `QTextEdit`, `QSpinBox`, `QSlider`
- `QLabel`, `QGroupBox`, `QFrame`
- `QTabBar`, `QTabWidget`, `QSplitter`, `QScrollArea`, `QScrollBar`
- `QDockWidget`, `QStatusBar`, `QProgressBar`, `QRubberBand`
- `QFormLayout`, `QGridLayout`, `QHBoxLayout`, `QVBoxLayout`
- `QSizePolicy`
- `QAbstractItemModel`, `QAbstractTableModel`, `QModelIndex`
- `QAbstractItemView`, `QTableView`, `QTreeView`
- `QTableWidget`, `QTableWidgetItem`, `QListWidget`, `QListWidgetItem`
- `QItemSelection`, `QItemSelectionModel`, `QItemSelectionRange`
- `QHeaderView`, `QStyledItemDelegate`, `QStyleOptionViewItem`
- `QCompleter`
- `QValidator`, `QIntValidator`, `QDoubleValidator`
- `QRegExp`, `QRegExpValidator`
- `QRegularExpression`, `QRegularExpressionValidator`
- `QColor`, `QRgb`, `QPalette`
- `QBrush`, `QPen`
- `QPainter`, `QPainterPath`, `QStylePainter`
- `QPixmap`, `QPixmapCache`, `QIcon`, `QStaticText`
- `QFont`, `QFontDatabase`, `QFontMetrics`, `QTextDocument`
- `QStyle`, `QStyleOption`, `QStyleOptionFrame`, `QStyleOptionSlider`
- `QPoint`, `QPointF`, `QRect`, `QRectF`, `QRegion`
- `QSize`, `QSizeF`
- `QCursor`, `QClipboard`, `QScreen`, `QDesktopWidget`
- `QEvent`, `QPaintEvent`, `QResizeEvent`, `QShowEvent`, `QCloseEvent`
- `QMouseEvent`, `QWheelEvent`, `QNativeGestureEvent`, `QContextMenuEvent`
- `QDrag`, `QMimeData`
- `QDragEnterEvent`, `QDragLeaveEvent`, `QDragMoveEvent`, `QDropEvent`
- `QKeySequence`, `QShortcut`, `QToolTip`
- `QChart`, `QChartView`, `QAbstractAxis`, `QValueAxis`
- `QXYSeries`, `QLineSeries`, `QScatterSeries`
- `QLegend`, `QLegendMarker`
- `QGraphicsScene`, `QGraphicsView`, `QGraphicsItemGroup`, `QGraphicsLayout`
- `QGraphicsPixmapItem`, `QGraphicsProxyWidget`
- `QGraphicsDropShadowEffect`, `QGraphicsOpacityEffect`
- `QPropertyAnimation`, `QEasingCurve`
- `QOpenGLWidget`, `QOpenGLFunctions`
- `QOpenGLShader`, `QOpenGLShaderProgram`
- `QMatrix4x4`, `QSurfaceFormat`
- `QUndoCommand`, `QUndoStack`, `QUndoView`
