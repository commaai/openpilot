#include "tools/cabana/mainwin.h"

#include <iostream>
#include <QClipboard>
#include <QDesktopWidget>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QShortcut>
#include <QUndoView>
#include <QVBoxLayout>
#include <QWidgetAction>

#include "tools/cabana/commands.h"

static MainWindow *main_win = nullptr;
void qLogMessageHandler(QtMsgType type, const QMessageLogContext &context, const QString &msg) {
  if (type == QtDebugMsg) std::cout << msg.toStdString() << std::endl;
  if (main_win) emit main_win->showMessage(msg, 0);
}

MainWindow::MainWindow() : QMainWindow() {
  createDockWindows();
  detail_widget = new DetailWidget(charts_widget, this);
  setCentralWidget(detail_widget);
  createActions();
  createStatusBar();
  createShortcuts();

  // restore states
  restoreGeometry(settings.geometry);
  if (isMaximized()) {
    setGeometry(QApplication::desktop()->availableGeometry(this));
  }
  restoreState(settings.window_state);
  messages_widget->restoreHeaderState(settings.message_header_state);

  qRegisterMetaType<uint64_t>("uint64_t");
  qRegisterMetaType<ReplyMsgType>("ReplyMsgType");
  installMessageHandler([this](ReplyMsgType type, const std::string msg) {
    // use queued connection to recv the log messages from replay.
    emit showMessage(QString::fromStdString(msg), 3000);
  });
  installDownloadProgressHandler([this](uint64_t cur, uint64_t total, bool success) {
    emit updateProgressBar(cur, total, success);
  });

  main_win = this;
  qInstallMessageHandler(qLogMessageHandler);
  QFile json_file("./car_fingerprint_to_dbc.json");
  if (json_file.open(QIODevice::ReadOnly)) {
    fingerprint_to_dbc = QJsonDocument::fromJson(json_file.readAll());
  }

  QObject::connect(this, &MainWindow::showMessage, statusBar(), &QStatusBar::showMessage);
  QObject::connect(this, &MainWindow::updateProgressBar, this, &MainWindow::updateDownloadProgress);
  QObject::connect(messages_widget, &MessagesWidget::msgSelectionChanged, detail_widget, &DetailWidget::setMessage);
  QObject::connect(charts_widget, &ChartsWidget::dock, this, &MainWindow::dockCharts);
  QObject::connect(can, &AbstractStream::streamStarted, this, &MainWindow::loadDBCFromFingerprint);
  QObject::connect(dbc(), &DBCManager::DBCFileChanged, this, &MainWindow::DBCFileChanged);
  QObject::connect(UndoStack::instance(), &QUndoStack::cleanChanged, this, &MainWindow::undoStackCleanChanged);
}

void MainWindow::createActions() {
  QMenu *file_menu = menuBar()->addMenu(tr("&File"));
  file_menu->addAction(tr("New DBC File"), this, &MainWindow::newFile)->setShortcuts(QKeySequence::New);
  file_menu->addAction(tr("Open DBC File..."), this, &MainWindow::openFile)->setShortcuts(QKeySequence::Open);

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
  auto dbc_names = dbc()->allDBCNames();
  std::sort(dbc_names.begin(), dbc_names.end());
  for (const auto &name : dbc_names) {
    load_opendbc_menu->addAction(QString::fromStdString(name), this, &MainWindow::openOpendbcFile);
  }

  file_menu->addAction(tr("Load DBC From Clipboard"), this, &MainWindow::loadDBCFromClipboard);

  file_menu->addSeparator();
  file_menu->addAction(tr("Save DBC..."), this, &MainWindow::save)->setShortcuts(QKeySequence::Save);
  file_menu->addAction(tr("Save DBC As..."), this, &MainWindow::saveAs)->setShortcuts(QKeySequence::SaveAs);
  file_menu->addAction(tr("Copy DBC To Clipboard"), this, &MainWindow::saveDBCToClipboard);
  file_menu->addSeparator();
  file_menu->addAction(tr("Settings..."), this, &MainWindow::setOption)->setShortcuts(QKeySequence::Preferences);

  file_menu->addSeparator();
  file_menu->addAction(tr("E&xit"), qApp, &QApplication::closeAllWindows)->setShortcuts(QKeySequence::Quit);

  QMenu *edit_menu = menuBar()->addMenu(tr("&Edit"));
  auto undo_act = UndoStack::instance()->createUndoAction(this, tr("&Undo"));
  undo_act->setShortcuts(QKeySequence::Undo);
  edit_menu->addAction(undo_act);
  auto redo_act = UndoStack::instance()->createRedoAction(this, tr("&Rndo"));
  redo_act->setShortcuts(QKeySequence::Redo);
  edit_menu->addAction(redo_act);
  edit_menu->addSeparator();

  QMenu *commands_menu = edit_menu->addMenu(tr("Command &List"));
  auto undo_view = new QUndoView(UndoStack::instance());
  undo_view->setWindowTitle(tr("Command List"));
  QWidgetAction *commands_act = new QWidgetAction(this);
  commands_act->setDefaultWidget(undo_view);
  commands_menu->addAction(commands_act);

  QMenu *tools_menu = menuBar()->addMenu(tr("&Tools"));
  tools_menu->addAction(tr("Find &Similar Bits"), this, &MainWindow::findSimilarBits);

  QMenu *help_menu = menuBar()->addMenu(tr("&Help"));
  help_menu->addAction(tr("About &Qt"), qApp, &QApplication::aboutQt);
}

void MainWindow::createDockWindows() {
  // left panel
  messages_widget = new MessagesWidget(this);
  QDockWidget *dock = new QDockWidget(tr("MESSAGES"), this);
  dock->setObjectName("MessagesPanel");
  dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea | Qt::TopDockWidgetArea | Qt::BottomDockWidgetArea);
  dock->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);
  dock->setWidget(messages_widget);
  addDockWidget(Qt::LeftDockWidgetArea, dock);

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
  QObject::connect(charts_widget, &ChartsWidget::rangeChanged, video_widget, &VideoWidget::rangeChanged);

  video_splitter->addWidget(charts_container);
  video_splitter->setStretchFactor(1, 1);
  video_splitter->restoreState(settings.video_splitter_state);
  if (can->liveStreaming() || video_splitter->sizes()[0] == 0) {
    // display video at minimum size.
    video_splitter->setSizes({1, 1});
  }

  video_dock = new QDockWidget(can->routeName(), this);
  video_dock->setObjectName(tr("VideoPanel"));
  video_dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
  video_dock->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);
  video_dock->setWidget(video_splitter);
  addDockWidget(Qt::RightDockWidgetArea, video_dock);
}

void MainWindow::createStatusBar() {
  progress_bar = new QProgressBar();
  progress_bar->setRange(0, 100);
  progress_bar->setTextVisible(true);
  progress_bar->setFixedSize({230, 16});
  progress_bar->setVisible(false);
  statusBar()->addPermanentWidget(progress_bar);
}

void MainWindow::createShortcuts() {
  auto shortcut = new QShortcut(QKeySequence(Qt::Key_Space), this, nullptr, nullptr, Qt::ApplicationShortcut);
  QObject::connect(shortcut, &QShortcut::activated, []() { can->pause(!can->isPaused()); });
  // TODO: add more shortcuts here.
}

void MainWindow::undoStackCleanChanged(bool clean) {
  setWindowModified(!clean);
}

void MainWindow::DBCFileChanged() {
  UndoStack::instance()->clear();
  setWindowFilePath(QString("%1").arg(dbc()->name()));
}

void MainWindow::newFile() {
  remindSaveChanges();
  dbc()->open("untitled.dbc", "");
}

void MainWindow::openFile() {
  remindSaveChanges();
  QString fn = QFileDialog::getOpenFileName(this, tr("Open File"), settings.last_dir, "DBC (*.dbc)");
  if (!fn.isEmpty()) {
    loadFile(fn);
  }
}

void MainWindow::loadFile(const QString &fn) {
  if (!fn.isEmpty()) {
    QFile file(fn);
    if (file.open(QIODevice::ReadOnly)) {
      auto dbc_name = QFileInfo(fn).baseName();
      dbc()->open(dbc_name, file.readAll());
      setCurrentFile(fn);
      statusBar()->showMessage(tr("DBC File %1 loaded").arg(fn), 2000);
    }
  }
}

void MainWindow::openOpendbcFile() {
  if (auto action = qobject_cast<QAction *>(sender())) {
    remindSaveChanges();
    loadDBCFromOpendbc(action->text());
  }
}

void MainWindow::openRecentFile() {
  if (auto action = qobject_cast<QAction *>(sender())) {
    remindSaveChanges();
    loadFile(action->data().toString());
  }
}

void MainWindow::loadDBCFromOpendbc(const QString &name) {
  if (name != dbc()->name()) {
    remindSaveChanges();
    dbc()->open(name);
  }
}

void MainWindow::loadDBCFromClipboard() {
  remindSaveChanges();
  QString dbc_str = QGuiApplication::clipboard()->text();
  dbc()->open("from_clipboard.dbc", dbc_str);
  if (dbc()->messages().size() > 0) {
    QMessageBox::information(this, tr("Load From Clipboard"), tr("DBC Successfully Loaded!"));
  } else {
    QMessageBox::warning(this, tr("Load From Clipboard"), tr("Failed to parse dbc from clipboard!\nMake sure that you paste the text with correct format."));
  }
}

void MainWindow::loadDBCFromFingerprint() {
  // Don't overwrite already loaded DBC
  if (!dbc()->name().isEmpty()) {
    return;
  }

  remindSaveChanges();
  auto fingerprint = can->carFingerprint();
  video_dock->setWindowTitle(tr("ROUTE: %1  FINGERPINT: %2").arg(can->routeName()).arg(fingerprint.isEmpty() ? tr("Unknown Car") : fingerprint));
  if (!fingerprint.isEmpty()) {
    auto dbc_name = fingerprint_to_dbc[fingerprint];
    if (dbc_name != QJsonValue::Undefined) {
      loadDBCFromOpendbc(dbc_name.toString());
      return;
    }
  }
  newFile();
}

void MainWindow::save() {
  if (current_file.isEmpty()) {
    saveAs();
  } else {
    saveFile(current_file);
  }
}

void MainWindow::saveFile(const QString &fn) {
  QFile file(fn);
  if (file.open(QIODevice::WriteOnly)) {
    file.write(dbc()->generateDBC().toUtf8());
    UndoStack::instance()->setClean();
    setCurrentFile(fn);
    statusBar()->showMessage(tr("File saved"), 2000);
  }
}

void MainWindow::saveAs() {
  QString fn = QFileDialog::getSaveFileName(this, tr("Save File"), QDir::cleanPath(settings.last_dir + "/untitled.dbc"), tr("DBC (*.dbc)"));
  if (!fn.isEmpty()) {
    saveFile(fn);
  }
}

void MainWindow::saveDBCToClipboard() {
  QGuiApplication::clipboard()->setText(dbc()->generateDBC());
  QMessageBox::information(this, tr("Copy To Clipboard"), tr("DBC Successfully copied!"));
}

void MainWindow::setCurrentFile(const QString &fn) {
  current_file = fn;
  setWindowFilePath(QString("%1").arg(fn));
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
  bool discard_changes = false;
  while (!UndoStack::instance()->isClean() && !discard_changes) {
    int ret = (QMessageBox::question(this, tr("Unsaved Changes"),
                                     tr("You have unsaved changes. Press ok to save them, cancel to discard."),
                                     QMessageBox::Ok | QMessageBox::Cancel));
    if (ret == QMessageBox::Ok) {
      save();
    } else {
      discard_changes = true;
    }
  }
  UndoStack::instance()->clear();
  current_file = "";
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
  remindSaveChanges();

  main_win = nullptr;
  if (floating_window)
    floating_window->deleteLater();

  // save states
  settings.geometry = saveGeometry();
  settings.window_state = saveState();
  if (!can->liveStreaming()) {
    settings.video_splitter_state = video_splitter->saveState();
  }
  settings.message_header_state = messages_widget->saveHeaderState();
  settings.save();
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
