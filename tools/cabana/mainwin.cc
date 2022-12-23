#include "tools/cabana/mainwin.h"

#include <iostream>
#include <QApplication>
#include <QClipboard>
#include <QCompleter>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QShortcut>
#include <QScreen>
#include <QToolBar>
#include <QUndoView>
#include <QVBoxLayout>
#include <QWidgetAction>

static MainWindow *main_win = nullptr;
void qLogMessageHandler(QtMsgType type, const QMessageLogContext &context, const QString &msg) {
  if (type == QtDebugMsg) std::cout << msg.toStdString() << std::endl;
  if (main_win) emit main_win->showMessage(msg, 0);
}

MainWindow::MainWindow() : QMainWindow() {
  setWindowTitle("Cabana");
  QWidget *central_widget = new QWidget(this);
  QHBoxLayout *main_layout = new QHBoxLayout(central_widget);
  main_layout->setContentsMargins(11, 11, 11, 0);
  main_layout->setSpacing(0);

  splitter = new QSplitter(Qt::Horizontal, this);
  splitter->setHandleWidth(11);

  QWidget *messages_container = new QWidget(this);
  QVBoxLayout *messages_layout = new QVBoxLayout(messages_container);
  messages_layout->setContentsMargins(0, 0, 0, 0);

  // left panel
  dbc_combo = createDBCSelector();
  messages_layout->addWidget(dbc_combo);
  messages_widget = new MessagesWidget(this);
  messages_layout->addWidget(messages_widget);
  splitter->addWidget(messages_container);

  charts_widget = new ChartsWidget(this);
  detail_widget = new DetailWidget(charts_widget, this);
  splitter->addWidget(detail_widget);
  if (!settings.splitter_state.isEmpty()) {
    splitter->restoreState(settings.splitter_state);
  }
  main_layout->addWidget(splitter);

  // right widgets
  QWidget *right_container = new QWidget(this);
  right_container->setFixedWidth(640);
  r_layout = new QVBoxLayout(right_container);
  r_layout->setContentsMargins(11, 0, 0, 0);
  QHBoxLayout *right_hlayout = new QHBoxLayout();
  fingerprint_label = new QLabel(this);
  right_hlayout->addWidget(fingerprint_label, 0, Qt::AlignLeft);

  // TODO: click to select another route.
  right_hlayout->addWidget(new QLabel(can->routeName()), 0, Qt::AlignRight);
  r_layout->addLayout(right_hlayout);

  video_widget = new VideoWidget(this);
  r_layout->addWidget(video_widget, 0, Qt::AlignTop);
  r_layout->addWidget(charts_widget, 1);
  r_layout->addStretch(0);
  main_layout->addWidget(right_container);

  setCentralWidget(central_widget);
  createActions();
  createStatusBar();
  createShortcuts();

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

  QObject::connect(dbc_combo, SIGNAL(activated(const QString &)), SLOT(loadDBCFromName(const QString &)));
  QObject::connect(this, &MainWindow::showMessage, statusBar(), &QStatusBar::showMessage);
  QObject::connect(this, &MainWindow::updateProgressBar, this, &MainWindow::updateDownloadProgress);
  QObject::connect(messages_widget, &MessagesWidget::msgSelectionChanged, detail_widget, &DetailWidget::setMessage);
  QObject::connect(charts_widget, &ChartsWidget::dock, this, &MainWindow::dockCharts);
  QObject::connect(charts_widget, &ChartsWidget::rangeChanged, video_widget, &VideoWidget::rangeChanged);
  QObject::connect(can, &CANMessages::streamStarted, this, &MainWindow::loadDBCFromFingerprint);
  QObject::connect(dbc(), &DBCManager::DBCFileChanged, this, &MainWindow::DBCFileChanged);
  QObject::connect(detail_widget->undo_stack, &QUndoStack::indexChanged, [this](int index) {
    setWindowTitle(tr("%1%2 - Cabana").arg(index > 0 ? "* " : "").arg(dbc()->name()));
  });
}

void MainWindow::createActions() {
  QMenu *file_menu = menuBar()->addMenu(tr("&File"));
  file_menu->addAction(tr("Open DBC File..."), this, &MainWindow::loadDBCFromFile);
  file_menu->addAction(tr("Load DBC From Clipboard"), this, &MainWindow::loadDBCFromClipboard);
  file_menu->addSeparator();
  file_menu->addAction(tr("Save DBC As..."), this, &MainWindow::saveDBCToFile);
  file_menu->addAction(tr("Copy DBC To Clipboard"), this, &MainWindow::saveDBCToClipboard);
  file_menu->addSeparator();
  file_menu->addAction(tr("Settings..."), this, &MainWindow::setOption);

  QMenu *edit_menu = menuBar()->addMenu(tr("&Edit"));
  auto undo_act = detail_widget->undo_stack->createUndoAction(this, tr("&Undo"));
  undo_act->setShortcuts(QKeySequence::Undo);
  edit_menu->addAction(undo_act);
  auto redo_act = detail_widget->undo_stack->createRedoAction(this, tr("&Rndo"));
  redo_act->setShortcuts(QKeySequence::Redo);
  edit_menu->addAction(redo_act);
  edit_menu->addSeparator();

  QMenu *commands_menu = edit_menu->addMenu(tr("Command &List"));
  auto undo_view = new QUndoView(detail_widget->undo_stack);
  undo_view->setWindowTitle(tr("Command List"));
  QWidgetAction *commands_act = new QWidgetAction(this);
  commands_act->setDefaultWidget(undo_view);
  commands_menu->addAction(commands_act);

  QMenu *tools_menu = menuBar()->addMenu(tr("&Tools"));
  tools_menu->addAction(tr("Find &Similar Bits"), this, &MainWindow::findSimilarBits);

  QMenu *help_menu = menuBar()->addMenu(tr("&Help"));
  help_menu->addAction(tr("About &Qt"), qApp, &QApplication::aboutQt);
}

QComboBox *MainWindow::createDBCSelector() {
  QComboBox *c = new QComboBox(this);
  c->setEditable(true);
  c->lineEdit()->setPlaceholderText(tr("Select from an existing DBC file"));
  c->setInsertPolicy(QComboBox::NoInsert);
  c->completer()->setCompletionMode(QCompleter::PopupCompletion);
  c->completer()->setFilterMode(Qt::MatchContains);

  auto dbc_names = dbc()->allDBCNames();
  std::sort(dbc_names.begin(), dbc_names.end());
  for (const auto &name : dbc_names) {
    c->addItem(QString::fromStdString(name));
  }
  c->setCurrentIndex(-1);
  return c;
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

void MainWindow::DBCFileChanged() {
  detail_widget->undo_stack->clear();
  int index = dbc_combo->findText(QFileInfo(dbc()->name()).baseName());
  dbc_combo->setCurrentIndex(index);
  setWindowTitle(tr("%1 - Cabana").arg(dbc()->name()));
}

void MainWindow::loadDBCFromName(const QString &name) {
  if (name != dbc()->name()) {
    dbc()->open(name);
  }
}

void MainWindow::loadDBCFromFile() {
  QString file_name = QFileDialog::getOpenFileName(this, tr("Open File"), settings.last_dir, "DBC (*.dbc)");
  if (!file_name.isEmpty()) {
    settings.last_dir = QFileInfo(file_name).absolutePath();
    QFile file(file_name);
    if (file.open(QIODevice::ReadOnly)) {
      auto dbc_name = QFileInfo(file_name).baseName();
      dbc()->open(dbc_name, file.readAll());
    }
  }
}

void MainWindow::loadDBCFromClipboard() {
  QString dbc_str = QGuiApplication::clipboard()->text();
  dbc()->open("From Clipboard", dbc_str);
  QMessageBox::information(this, tr("Load From Clipboard"), tr("DBC Successfully Loaded!"));
}

void MainWindow::loadDBCFromFingerprint() {
  auto fingerprint = can->carFingerprint();
  fingerprint_label->setText(fingerprint.isEmpty() ? tr("Unknown Car") : fingerprint);
  if (!fingerprint.isEmpty()) {
    auto dbc_name = fingerprint_to_dbc[fingerprint];
    if (dbc_name != QJsonValue::Undefined) {
      loadDBCFromName(dbc_name.toString());
      return;
    }
  }
  dbc()->open("New_DBC", "");
}

void MainWindow::saveDBCToFile() {
  QString file_name = QFileDialog::getSaveFileName(this, tr("Save File"),
                                                   QDir::cleanPath(settings.last_dir + "/untitled.dbc"), tr("DBC (*.dbc)"));
  if (!file_name.isEmpty()) {
    settings.last_dir = QFileInfo(file_name).absolutePath();
    QFile file(file_name);
    if (file.open(QIODevice::WriteOnly)) {
      file.write(dbc()->generateDBC().toUtf8());
      detail_widget->undo_stack->clear();
    }
  }
}

void MainWindow::saveDBCToClipboard() {
  QGuiApplication::clipboard()->setText(dbc()->generateDBC());
  QMessageBox::information(this, tr("Copy To Clipboard"), tr("DBC Successfully copied!"));
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
    r_layout->insertWidget(2, charts_widget, 1);
    floating_window->deleteLater();
    floating_window = nullptr;
  } else if (!dock && !floating_window) {
    floating_window = new QWidget(this);
    floating_window->setWindowFlags(Qt::Window);
    floating_window->setWindowTitle("Charts - Cabana");
    floating_window->setLayout(new QVBoxLayout());
    floating_window->layout()->addWidget(charts_widget);
    floating_window->installEventFilter(charts_widget);
    floating_window->setMinimumSize(QGuiApplication::primaryScreen()->size() / 2);
    floating_window->showMaximized();
  }
}

void MainWindow::closeEvent(QCloseEvent *event) {
  if (detail_widget->undo_stack->index() > 0) {
    auto ret = QMessageBox::question(this, tr("Unsaved Changes"),
                                     tr("Are you sure you want to exit without saving?\nAny unsaved changes will be lost."),
                                     QMessageBox::Yes | QMessageBox::No);
    if (ret == QMessageBox::No) {
      event->ignore();
      return;
    }
  }

  main_win = nullptr;
  if (floating_window)
    floating_window->deleteLater();

  settings.splitter_state = splitter->saveState();
  settings.save();
  QWidget::closeEvent(event);
}

void MainWindow::setOption() {
  SettingsDlg dlg(this);
  dlg.exec();
}

void MainWindow::findSimilarBits() {
  FindSimilarBitsDlg dlg(this);
  dlg.exec();
}
