#include "tools/cabana/mainwin.h"

#include <algorithm>
#include <iostream>
#include <string>

#include <QClipboard>
#include <QDesktopWidget>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QJsonObject>
#include <QMenuBar>
#include <QMessageBox>
#include <QResizeEvent>
#include <QShortcut>
#include <QTextDocument>
#include <QUndoView>
#include <QVBoxLayout>
#include <QWidgetAction>

#include "tools/cabana/commands.h"
#include "tools/cabana/streamselector.h"
#include "tools/cabana/tools/findsignal.h"
#include "tools/cabana/utils/export.h"
#include "tools/replay/replay.h"

MainWindow::MainWindow() : QMainWindow() {
  loadFingerprints();
  createDockWindows();
  setCentralWidget(center_widget = new CenterWidget(this));
  createActions();
  createStatusBar();
  createShortcuts();

  // save default window state to allow resetting it
  default_state = saveState();

  // restore states
  restoreGeometry(settings.geometry);
  if (isMaximized()) {
    setGeometry(QApplication::desktop()->availableGeometry(this));
  }
  restoreState(settings.window_state);

  // install handlers
  static auto static_main_win = this;
  qRegisterMetaType<uint64_t>("uint64_t");
  qRegisterMetaType<SourceSet>("SourceSet");
  installDownloadProgressHandler([](uint64_t cur, uint64_t total, bool success) {
    emit static_main_win->updateProgressBar(cur, total, success);
  });
  qInstallMessageHandler([](QtMsgType type, const QMessageLogContext &context, const QString &msg) {
    if (type == QtDebugMsg) std::cout << msg.toStdString() << std::endl;
    emit static_main_win->showMessage(msg, 2000);
  });
  installMessageHandler([](ReplyMsgType type, const std::string msg) { qInfo() << msg.c_str(); });

  setStyleSheet(QString(R"(QMainWindow::separator {
    width: %1px; /* when vertical */
    height: %1px; /* when horizontal */
  })").arg(style()->pixelMetric(QStyle::PM_SplitterWidth)));

  QObject::connect(this, &MainWindow::showMessage, statusBar(), &QStatusBar::showMessage);
  QObject::connect(this, &MainWindow::updateProgressBar, this, &MainWindow::updateDownloadProgress);
  QObject::connect(dbc(), &DBCManager::DBCFileChanged, this, &MainWindow::DBCFileChanged);
  QObject::connect(UndoStack::instance(), &QUndoStack::cleanChanged, this, &MainWindow::undoStackCleanChanged);
  QObject::connect(UndoStack::instance(), &QUndoStack::indexChanged, this, &MainWindow::undoStackIndexChanged);
  QObject::connect(&settings, &Settings::changed, this, &MainWindow::updateStatus);
  QObject::connect(StreamNotifier::instance(), &StreamNotifier::changingStream, this, &MainWindow::changingStream);
  QObject::connect(StreamNotifier::instance(), &StreamNotifier::streamStarted, this, &MainWindow::streamStarted);
}

void MainWindow::loadFingerprints() {
  QFile json_file(QApplication::applicationDirPath() + "/dbc/car_fingerprint_to_dbc.json");
  if (json_file.open(QIODevice::ReadOnly)) {
    fingerprint_to_dbc = QJsonDocument::fromJson(json_file.readAll());
  }
  // get opendbc names
  for (auto fn : QDir(OPENDBC_FILE_PATH).entryList({"*.dbc"}, QDir::Files, QDir::Name)) {
    opendbc_names << QFileInfo(fn).baseName();
  }
}

void MainWindow::createActions() {
  // File menu
  QMenu *file_menu = menuBar()->addMenu(tr("&File"));
  file_menu->addAction(tr("Open Stream..."), this, &MainWindow::openStream);
  close_stream_act = file_menu->addAction(tr("Close stream"), this, &MainWindow::closeStream);
  export_to_csv_act = file_menu->addAction(tr("Export to CSV..."), this, &MainWindow::exportToCSV);
  close_stream_act->setEnabled(false);
  export_to_csv_act->setEnabled(false);
  file_menu->addSeparator();

  file_menu->addAction(tr("New DBC File"), [this]() { newFile(); }, QKeySequence::New);
  file_menu->addAction(tr("Open DBC File..."), [this]() { openFile(); }, QKeySequence::Open);

  manage_dbcs_menu = file_menu->addMenu(tr("Manage &DBC Files"));

  open_recent_menu = file_menu->addMenu(tr("Open &Recent"));
  for (int i = 0; i < MAX_RECENT_FILES; ++i) {
    recent_files_acts[i] = new QAction(this);
    recent_files_acts[i]->setVisible(false);
    QObject::connect(recent_files_acts[i], &QAction::triggered, this, &MainWindow::openRecentFile);
    open_recent_menu->addAction(recent_files_acts[i]);
  }
  updateRecentFileActions();

  file_menu->addSeparator();
  QMenu *load_opendbc_menu = file_menu->addMenu(tr("Load DBC from commaai/opendbc"));
  // load_opendbc_menu->setStyleSheet("QMenu { menu-scrollable: true; }");
  for (const auto &dbc_name : opendbc_names) {
    load_opendbc_menu->addAction(dbc_name, [this, name = dbc_name]() { loadDBCFromOpendbc(name); });
  }

  file_menu->addAction(tr("Load DBC From Clipboard"), [=]() { loadFromClipboard(); });

  file_menu->addSeparator();
  save_dbc = file_menu->addAction(tr("Save DBC..."), this, &MainWindow::save, QKeySequence::Save);
  save_dbc_as = file_menu->addAction(tr("Save DBC As..."), this, &MainWindow::saveAs, QKeySequence::SaveAs);
  copy_dbc_to_clipboard = file_menu->addAction(tr("Copy DBC To Clipboard"), this, &MainWindow::saveToClipboard);

  file_menu->addSeparator();
  file_menu->addAction(tr("Settings..."), this, &MainWindow::setOption, QKeySequence::Preferences);

  file_menu->addSeparator();
  file_menu->addAction(tr("E&xit"), qApp, &QApplication::closeAllWindows, QKeySequence::Quit);

  // Edit Menu
  QMenu *edit_menu = menuBar()->addMenu(tr("&Edit"));
  auto undo_act = UndoStack::instance()->createUndoAction(this, tr("&Undo"));
  undo_act->setShortcuts(QKeySequence::Undo);
  edit_menu->addAction(undo_act);
  auto redo_act = UndoStack::instance()->createRedoAction(this, tr("&Rndo"));
  redo_act->setShortcuts(QKeySequence::Redo);
  edit_menu->addAction(redo_act);
  edit_menu->addSeparator();

  QMenu *commands_menu = edit_menu->addMenu(tr("Command &List"));
  QWidgetAction *commands_act = new QWidgetAction(this);
  commands_act->setDefaultWidget(new QUndoView(UndoStack::instance()));
  commands_menu->addAction(commands_act);

  // View Menu
  QMenu *view_menu = menuBar()->addMenu(tr("&View"));
  auto act = view_menu->addAction(tr("Full Screen"), this, &MainWindow::toggleFullScreen, QKeySequence::FullScreen);
  addAction(act);
  view_menu->addSeparator();
  view_menu->addAction(messages_dock->toggleViewAction());
  view_menu->addAction(video_dock->toggleViewAction());
  view_menu->addSeparator();
  view_menu->addAction(tr("Reset Window Layout"), [this]() { restoreState(default_state); });

  // Tools Menu
  tools_menu = menuBar()->addMenu(tr("&Tools"));
  tools_menu->addAction(tr("Find &Similar Bits"), this, &MainWindow::findSimilarBits);
  tools_menu->addAction(tr("&Find Signal"), this, &MainWindow::findSignal);

  // Help Menu
  QMenu *help_menu = menuBar()->addMenu(tr("&Help"));
  help_menu->addAction(tr("Help"), this, &MainWindow::onlineHelp, QKeySequence::HelpContents);
  help_menu->addAction(tr("About &Qt"), qApp, &QApplication::aboutQt);
}

void MainWindow::createDockWindows() {
  messages_dock = new QDockWidget(tr("MESSAGES"), this);
  messages_dock->setObjectName("MessagesPanel");
  messages_dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea | Qt::TopDockWidgetArea | Qt::BottomDockWidgetArea);
  messages_dock->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable | QDockWidget::DockWidgetClosable);
  addDockWidget(Qt::LeftDockWidgetArea, messages_dock);

  video_dock = new QDockWidget("", this);
  video_dock->setObjectName(tr("VideoPanel"));
  video_dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
  video_dock->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable | QDockWidget::DockWidgetClosable);
  addDockWidget(Qt::RightDockWidgetArea, video_dock);
}

void MainWindow::createDockWidgets() {
  messages_widget = new MessagesWidget(this);
  messages_dock->setWidget(messages_widget);
  QObject::connect(messages_widget, &MessagesWidget::titleChanged, messages_dock, &QDockWidget::setWindowTitle);

  // right panel
  charts_widget = new ChartsWidget(this);
  QWidget *charts_container = new QWidget(this);
  charts_layout = new QVBoxLayout(charts_container);
  charts_layout->setContentsMargins(0, 0, 0, 0);
  charts_layout->addWidget(charts_widget);

  // splitter between video and charts
  video_splitter = new QSplitter(Qt::Vertical, this);
  video_widget = new VideoWidget(this);
  video_splitter->addWidget(video_widget);
  QObject::connect(charts_widget, &ChartsWidget::rangeChanged, video_widget, &VideoWidget::updateTimeRange);

  video_splitter->addWidget(charts_container);
  video_splitter->setStretchFactor(1, 1);
  video_splitter->restoreState(settings.video_splitter_state);
  video_splitter->handle(1)->setEnabled(!can->liveStreaming());
  video_dock->setWidget(video_splitter);
  QObject::connect(charts_widget, &ChartsWidget::dock, this, &MainWindow::dockCharts);
}

void MainWindow::createStatusBar() {
  progress_bar = new QProgressBar();
  progress_bar->setRange(0, 100);
  progress_bar->setTextVisible(true);
  progress_bar->setFixedSize({300, 16});
  progress_bar->setVisible(false);
  statusBar()->addWidget(new QLabel(tr("For Help, Press F1")));
  statusBar()->addPermanentWidget(progress_bar);

  statusBar()->addPermanentWidget(status_label = new QLabel(this));
  updateStatus();
}

void MainWindow::createShortcuts() {
  auto shortcut = new QShortcut(QKeySequence(Qt::Key_Space), this, nullptr, nullptr, Qt::ApplicationShortcut);
  QObject::connect(shortcut, &QShortcut::activated, []() { can->pause(!can->isPaused()); });
  // TODO: add more shortcuts here.
}

void MainWindow::undoStackIndexChanged(int index) {
  int count = UndoStack::instance()->count();
  if (count >= 0) {
    QString command_text;
    if (index == count) {
      command_text = (count == prev_undostack_count ? "Redo " : "") + UndoStack::instance()->text(index - 1);
    } else if (index < prev_undostack_index) {
      command_text = tr("Undo %1").arg(UndoStack::instance()->text(index));
    } else if (index > prev_undostack_index) {
      command_text = tr("Redo %1").arg(UndoStack::instance()->text(index - 1));
    }
    statusBar()->showMessage(command_text, 2000);
  }
  prev_undostack_index = index;
  prev_undostack_count = count;
  autoSave();
  updateLoadSaveMenus();
}

void MainWindow::undoStackCleanChanged(bool clean) {
  if (clean) {
    prev_undostack_index = 0;
    prev_undostack_count = 0;
  }
  setWindowModified(!clean);
}

void MainWindow::DBCFileChanged() {
  UndoStack::instance()->clear();
  updateLoadSaveMenus();
}

void MainWindow::openStream() {
  AbstractStream *stream = nullptr;
  StreamSelector dlg(&stream, this);
  if (dlg.exec()) {
    if (!dlg.dbcFile().isEmpty()) {
      loadFile(dlg.dbcFile());
    }
    stream->start();
    statusBar()->showMessage(tr("Route %1 loaded").arg(can->routeName()), 2000);
  }
}

void MainWindow::closeStream() {
  AbstractStream *stream = new DummyStream(this);
  stream->start();
  if (dbc()->nonEmptyDBCCount() > 0) {
    emit dbc()->DBCFileChanged();
  }
  statusBar()->showMessage(tr("stream closed"));
}

void MainWindow::exportToCSV() {
  QString dir = QString("%1/%2.csv").arg(settings.last_dir).arg(can->routeName());
  QString fn = QFileDialog::getSaveFileName(this, "Export stream to CSV file", dir, tr("csv (*.csv)"));
  if (!fn.isEmpty()) {
    utils::exportToCSV(fn);
  }
}

void MainWindow::newFile(SourceSet s) {
  closeFile(s);
  dbc()->open(s, "", "");
}

void MainWindow::openFile(SourceSet s) {
  remindSaveChanges();
  QString fn = QFileDialog::getOpenFileName(this, tr("Open File"), settings.last_dir, "DBC (*.dbc)");
  if (!fn.isEmpty()) {
    loadFile(fn, s);
  }
}

void MainWindow::loadFile(const QString &fn, SourceSet s) {
  if (!fn.isEmpty()) {
    closeFile(s);

    QString dbc_fn = fn;
    // Prompt user to load auto saved file if it exists.
    if (QFile::exists(fn + AUTO_SAVE_EXTENSION)) {
      auto ret = QMessageBox::question(this, tr("Auto saved DBC found"), tr("Auto saved DBC file from previous session found. Do you want to load it instead?"));
      if (ret == QMessageBox::Yes) {
        dbc_fn += AUTO_SAVE_EXTENSION;
        UndoStack::instance()->resetClean(); // Force user to save on close so the auto saved file is not lost
      }
    }

    QString error;
    if (dbc()->open(s, dbc_fn, &error)) {
      updateRecentFiles(fn);
      statusBar()->showMessage(tr("DBC File %1 loaded").arg(fn), 2000);
    } else {
      QMessageBox msg_box(QMessageBox::Warning, tr("Failed to load DBC file"), tr("Failed to parse DBC file %1").arg(fn));
      msg_box.setDetailedText(error);
      msg_box.exec();
    }
  }
}

void MainWindow::openRecentFile() {
  if (auto action = qobject_cast<QAction *>(sender())) {
    loadFile(action->data().toString());
  }
}

void MainWindow::loadDBCFromOpendbc(const QString &name) {
  QString opendbc_file_path = QString("%1/%2.dbc").arg(OPENDBC_FILE_PATH, name);
  loadFile(opendbc_file_path);
}

void MainWindow::loadFromClipboard(SourceSet s, bool close_all) {
  closeFile(s);

  QString dbc_str = QGuiApplication::clipboard()->text();
  QString error;
  bool ret = dbc()->open(s, "", dbc_str, &error);
  if (ret && dbc()->nonEmptyDBCCount() > 0) {
    QMessageBox::information(this, tr("Load From Clipboard"), tr("DBC Successfully Loaded!"));
  } else {
    QMessageBox msg_box(QMessageBox::Warning, tr("Failed to load DBC from clipboard"), tr("Make sure that you paste the text with correct format."));
    msg_box.setDetailedText(error);
    msg_box.exec();
  }
}

void MainWindow::changingStream() {
  center_widget->clear();
  delete messages_widget;
  delete video_splitter;
}

void MainWindow::streamStarted() {
  bool has_stream = dynamic_cast<DummyStream *>(can) == nullptr;
  close_stream_act->setEnabled(has_stream);
  export_to_csv_act->setEnabled(has_stream);
  tools_menu->setEnabled(has_stream);
  createDockWidgets();

  video_dock->setWindowTitle(can->routeName());
  if (can->liveStreaming() || video_splitter->sizes()[0] == 0) {
    // display video at minimum size.
    video_splitter->setSizes({1, 1});
  }
  // Don't overwrite already loaded DBC
  if (!dbc()->nonEmptyDBCCount()) {
    newFile();
  }

  QObject::connect(messages_widget, &MessagesWidget::msgSelectionChanged, center_widget, &CenterWidget::setMessage);
  QObject::connect(can, &AbstractStream::eventsMerged, this, &MainWindow::eventsMerged);
  QObject::connect(can, &AbstractStream::sourcesUpdated, this, &MainWindow::updateLoadSaveMenus);
}

void MainWindow::eventsMerged() {
  if (!can->liveStreaming() && std::exchange(car_fingerprint, can->carFingerprint()) != car_fingerprint) {
    video_dock->setWindowTitle(tr("ROUTE: %1  FINGERPRINT: %2")
                                    .arg(can->routeName())
                                    .arg(car_fingerprint.isEmpty() ? tr("Unknown Car") : car_fingerprint));
    // Don't overwrite already loaded DBC
    if (!dbc()->nonEmptyDBCCount() && !car_fingerprint.isEmpty()) {
      auto dbc_name = fingerprint_to_dbc[car_fingerprint];
      if (dbc_name != QJsonValue::Undefined) {
        // Prevent dialog that load autosaved file from blocking replay->start().
        QTimer::singleShot(0, this, [dbc_name, this]() { loadDBCFromOpendbc(dbc_name.toString()); });
      }
    }
  }
}

void MainWindow::save() {
  // Save all open DBC files
  for (auto dbc_file : dbc()->allDBCFiles()) {
    if (dbc_file->isEmpty()) continue;
    saveFile(dbc_file);
  }
}

void MainWindow::saveAs() {
  // Save as all open DBC files. Should not be called with more than 1 file open
  for (auto dbc_file : dbc()->allDBCFiles()) {
    if (dbc_file->isEmpty()) continue;
    saveFileAs(dbc_file);
  }
}

void MainWindow::autoSave() {
  if (!UndoStack::instance()->isClean()) {
    for (auto dbc_file : dbc()->allDBCFiles()) {
      if (!dbc_file->filename.isEmpty()) {
        dbc_file->autoSave();
      }
    }
  }
}

void MainWindow::cleanupAutoSaveFile() {
  for (auto dbc_file : dbc()->allDBCFiles()) {
    dbc_file->cleanupAutoSaveFile();
  }
}

void MainWindow::closeFile(SourceSet s) {
  remindSaveChanges();
  if (s == SOURCE_ALL) {
    dbc()->closeAll();
  } else {
    dbc()->close(s);
  }
}

void MainWindow::closeFile(DBCFile *dbc_file) {
  assert(dbc_file != nullptr);
  remindSaveChanges();
  dbc()->close(dbc_file);
  // Ensure we always have at least one file open
  if (dbc()->dbcCount() == 0) {
    newFile();
  }
}

void MainWindow::saveFile(DBCFile *dbc_file) {
  assert(dbc_file != nullptr);
  if (!dbc_file->filename.isEmpty()) {
    dbc_file->save();
    updateLoadSaveMenus();
    UndoStack::instance()->setClean();
    statusBar()->showMessage(tr("File saved"), 2000);
  } else if (!dbc_file->isEmpty()) {
    saveFileAs(dbc_file);
  }
}

void MainWindow::saveFileAs(DBCFile *dbc_file) {
  QString title = tr("Save File (bus: %1)").arg(toString(dbc()->sources(dbc_file)));
  QString fn = QFileDialog::getSaveFileName(this, title, QDir::cleanPath(settings.last_dir + "/untitled.dbc"), tr("DBC (*.dbc)"));
  if (!fn.isEmpty()) {
    dbc_file->saveAs(fn);
    UndoStack::instance()->setClean();
    statusBar()->showMessage(tr("File saved as %1").arg(fn), 2000);
    updateRecentFiles(fn);
    updateLoadSaveMenus();
  }
}

void MainWindow::saveToClipboard() {
  // Copy all open DBC files to clipboard. Should not be called with more than 1 file open
  for (auto dbc_file : dbc()->allDBCFiles()) {
    if (dbc_file->isEmpty()) continue;
    saveFileToClipboard(dbc_file);
  }
}

void MainWindow::saveFileToClipboard(DBCFile *dbc_file) {
  assert(dbc_file != nullptr);
  QGuiApplication::clipboard()->setText(dbc_file->generateDBC());
  QMessageBox::information(this, tr("Copy To Clipboard"), tr("DBC Successfully copied!"));
}

void MainWindow::updateLoadSaveMenus() {
  int cnt = dbc()->nonEmptyDBCCount();
  save_dbc->setText(cnt > 1 ? tr("Save %1 DBCs...").arg(cnt) : tr("Save DBC..."));
  save_dbc->setEnabled(cnt > 0);
  save_dbc_as->setEnabled(cnt == 1);

  // TODO: Support clipboard for multiple files
  copy_dbc_to_clipboard->setEnabled(cnt == 1);

  manage_dbcs_menu->clear();
  manage_dbcs_menu->setEnabled(dynamic_cast<DummyStream *>(can) == nullptr);

  for (int source : can->sources) {
    if (source >= 64) continue; // Sent and blocked buses are handled implicitly

    SourceSet ss = {source, uint8_t(source + 128), uint8_t(source + 192)};

    QMenu *bus_menu = new QMenu(this);
    bus_menu->addAction(tr("New DBC File..."), [=]() { newFile(ss); });
    bus_menu->addAction(tr("Open DBC File..."), [=]() { openFile(ss); });
    bus_menu->addAction(tr("Load DBC From Clipboard..."), [=]() { loadFromClipboard(ss, false); });

    // Show sub-menu for each dbc for this source.
    QString file_name = "No DBCs loaded";
    if (auto dbc_file = dbc()->findDBCFile(source)) {
      bus_menu->addSeparator();
      bus_menu->addAction(dbc_file->name() + " (" + toString(dbc()->sources(dbc_file)) + ")")->setEnabled(false);
      bus_menu->addAction(tr("Save..."), [=]() { saveFile(dbc_file); });
      bus_menu->addAction(tr("Save As..."), [=]() { saveFileAs(dbc_file); });
      bus_menu->addAction(tr("Copy to Clipboard..."), [=]() { saveFileToClipboard(dbc_file); });
      bus_menu->addAction(tr("Remove from this bus..."), [=]() { closeFile(ss); });
      bus_menu->addAction(tr("Remove from all buses..."), [=]() { closeFile(dbc_file); });

      file_name = dbc_file->name();
    }

    manage_dbcs_menu->addMenu(bus_menu);
    bus_menu->setTitle(tr("Bus %1 (%2)").arg(source).arg(file_name));
  }

  QStringList title;
  for (auto f : dbc()->allDBCFiles()) {
    title.push_back(tr("(%1) %2").arg(toString(dbc()->sources(f)), f->name()));
  }
  setWindowFilePath(title.join(" | "));
}

void MainWindow::updateRecentFiles(const QString &fn) {
  settings.recent_files.removeAll(fn);
  settings.recent_files.prepend(fn);
  while (settings.recent_files.size() > MAX_RECENT_FILES) {
    settings.recent_files.removeLast();
  }
  settings.last_dir = QFileInfo(fn).absolutePath();
  updateRecentFileActions();
}

void MainWindow::updateRecentFileActions() {
  int num_recent_files = std::min<int>(settings.recent_files.size(), MAX_RECENT_FILES);
  for (int i = 0; i < num_recent_files; ++i) {
    QString text = tr("&%1 %2").arg(i + 1).arg(QFileInfo(settings.recent_files[i]).fileName());
    recent_files_acts[i]->setText(text);
    recent_files_acts[i]->setData(settings.recent_files[i]);
    recent_files_acts[i]->setVisible(true);
  }
  for (int i = num_recent_files; i < MAX_RECENT_FILES; ++i) {
    recent_files_acts[i]->setVisible(false);
  }
  open_recent_menu->setEnabled(num_recent_files > 0);
}

void MainWindow::remindSaveChanges() {
  while (!UndoStack::instance()->isClean()) {
    QString text = tr("You have unsaved changes. Press ok to save them, cancel to discard.");
    int ret = QMessageBox::question(this, tr("Unsaved Changes"), text, QMessageBox::Ok | QMessageBox::Cancel);
    if (ret != QMessageBox::Ok) break;
    save();
  }
  UndoStack::instance()->clear();
}

void MainWindow::updateDownloadProgress(uint64_t cur, uint64_t total, bool success) {
  if (success && cur < total) {
    progress_bar->setValue((cur / (double)total) * 100);
    progress_bar->setFormat(tr("Downloading %p% (%1)").arg(formattedDataSize(total).c_str()));
    progress_bar->show();
  } else {
    progress_bar->hide();
  }
}

void MainWindow::updateStatus() {
  status_label->setText(tr("Cached Minutes:%1 FPS:%2").arg(settings.max_cached_minutes).arg(settings.fps));
}

void MainWindow::dockCharts(bool dock) {
  if (dock && floating_window) {
    floating_window->removeEventFilter(charts_widget);
    charts_layout->insertWidget(0, charts_widget, 1);
    floating_window->deleteLater();
    floating_window = nullptr;
  } else if (!dock && !floating_window) {
    floating_window = new QWidget(this);
    floating_window->setWindowFlags(Qt::Window);
    floating_window->setWindowTitle("Charts");
    floating_window->setLayout(new QVBoxLayout());
    floating_window->layout()->addWidget(charts_widget);
    floating_window->installEventFilter(charts_widget);
    floating_window->showMaximized();
  }
}

void MainWindow::closeEvent(QCloseEvent *event) {
  cleanupAutoSaveFile();
  remindSaveChanges();

  installDownloadProgressHandler(nullptr);
  qInstallMessageHandler(nullptr);

  if (floating_window)
    floating_window->deleteLater();

  // save states
  settings.geometry = saveGeometry();
  settings.window_state = saveState();
  if (!can->liveStreaming()) {
    settings.video_splitter_state = video_splitter->saveState();
  }
  settings.message_header_state = messages_widget->saveHeaderState();

  QWidget::closeEvent(event);
}

void MainWindow::setOption() {
  SettingsDlg dlg(this);
  dlg.exec();
}

void MainWindow::findSimilarBits() {
  FindSimilarBitsDlg *dlg = new FindSimilarBitsDlg(this);
  QObject::connect(dlg, &FindSimilarBitsDlg::openMessage, messages_widget, &MessagesWidget::selectMessage);
  dlg->show();
}

void MainWindow::findSignal() {
  FindSignalDlg *dlg = new FindSignalDlg(this);
  QObject::connect(dlg, &FindSignalDlg::openMessage, messages_widget, &MessagesWidget::selectMessage);
  dlg->show();
}

void MainWindow::onlineHelp() {
  if (auto help = findChild<HelpOverlay*>()) {
    help->close();
  } else {
    help = new HelpOverlay(this);
    help->setGeometry(rect());
    help->show();
    help->raise();
  }
}

void MainWindow::toggleFullScreen() {
  if (isFullScreen()) {
    menuBar()->show();
    statusBar()->show();
    showNormal();
    showMaximized();
  } else {
    menuBar()->hide();
    statusBar()->hide();
    showFullScreen();
  }
}

// HelpOverlay
HelpOverlay::HelpOverlay(MainWindow *parent) : QWidget(parent) {
  setAttribute(Qt::WA_NoSystemBackground, true);
  setAttribute(Qt::WA_TranslucentBackground, true);
  setAttribute(Qt::WA_DeleteOnClose);
  parent->installEventFilter(this);
}

void HelpOverlay::paintEvent(QPaintEvent *event) {
  QPainter painter(this);
  painter.fillRect(rect(), QColor(0, 0, 0, 50));
  auto parent = parentWidget();
  drawHelpForWidget(painter, parent->findChild<MessagesWidget *>());
  drawHelpForWidget(painter, parent->findChild<BinaryView *>());
  drawHelpForWidget(painter, parent->findChild<SignalView *>());
  drawHelpForWidget(painter, parent->findChild<ChartsWidget *>());
  drawHelpForWidget(painter, parent->findChild<VideoWidget *>());
}

void HelpOverlay::drawHelpForWidget(QPainter &painter, QWidget *w) {
  if (w && w->isVisible() && !w->whatsThis().isEmpty()) {
    QPoint pt = mapFromGlobal(w->mapToGlobal(w->rect().center()));
    if (rect().contains(pt)) {
      QTextDocument document;
      document.setHtml(w->whatsThis());
      QSize doc_size = document.size().toSize();
      QPoint topleft = {pt.x() - doc_size.width() / 2, pt.y() - doc_size.height() / 2};
      painter.translate(topleft);
      painter.fillRect(QRect{{0, 0}, doc_size}, palette().toolTipBase());
      document.drawContents(&painter);
      painter.translate(-topleft);
    }
  }
}

bool HelpOverlay::eventFilter(QObject *obj, QEvent *event) {
  if (obj == parentWidget() && event->type() == QEvent::Resize) {
    QResizeEvent *resize_event = (QResizeEvent *)(event);
    setGeometry(QRect{QPoint(0, 0), resize_event->size()});
  }
  return false;
}

void HelpOverlay::mouseReleaseEvent(QMouseEvent *event) {
  close();
}
