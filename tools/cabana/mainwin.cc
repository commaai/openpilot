#include "tools/cabana/mainwin.h"

#include <QApplication>
#include <QCompleter>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QMenu>
#include <QMessageBox>
#include <QPushButton>
#include <QScreen>
#include <QSplitter>
#include <QVBoxLayout>

#include "tools/replay/util.h"

static MainWindow *main_win = nullptr;
void qLogMessageHandler(QtMsgType type, const QMessageLogContext &context, const QString &msg) {
  if (main_win) main_win->showStatusMessage(msg);
}

MainWindow::MainWindow() : can_message(this), QWidget() {
  main_win = this;
  // qInstallMessageHandler(qLogMessageHandler);

  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(11, 11, 11, 5);
  main_layout->setSpacing(0);

  QHBoxLayout *h_layout = new QHBoxLayout();
  h_layout->setContentsMargins(0, 0, 0, 0);
  main_layout->addLayout(h_layout);

  QWidget *left_panel = new QWidget(this);
  QVBoxLayout *left_panel_layout = new QVBoxLayout(left_panel);

  // DBC file selector
  QHBoxLayout *dbc_file_layout = new QHBoxLayout();
  dbc_combo = new QComboBox(this);
  auto dbc_names = dbc()->allDBCNames();
  for (const auto &name : dbc_names) {
    dbc_combo->addItem(QString::fromStdString(name));
  }
  dbc_combo->model()->sort(0);
  dbc_combo->setEditable(true);
  dbc_combo->setCurrentText(QString());
  dbc_combo->setInsertPolicy(QComboBox::NoInsert);
  dbc_combo->completer()->setCompletionMode(QCompleter::PopupCompletion);
  QFont font;
  font.setBold(true);
  dbc_combo->lineEdit()->setFont(font);
  dbc_file_layout->addWidget(dbc_combo);

  QPushButton *load_from_paste = new QPushButton(tr("Load from paste"), this);
  dbc_file_layout->addWidget(load_from_paste);

  dbc_file_layout->addStretch();
  QPushButton *save_btn = new QPushButton(tr("Save DBC"), this);
  dbc_file_layout->addWidget(save_btn);
  left_panel_layout->addLayout(dbc_file_layout);

  messages_widget = new MessagesWidget(this);
  left_panel_layout->addWidget(messages_widget);

  QSplitter *splitter = new QSplitter(Qt::Horizontal, this);
  splitter->addWidget(left_panel);

  detail_widget = new DetailWidget(this);
  splitter->addWidget(detail_widget);

  splitter->setSizes({100, 500});
  h_layout->addWidget(splitter);

  // right widgets
  QWidget *right_container = new QWidget(this);
  right_container->setFixedWidth(640);
  r_layout = new QVBoxLayout(right_container);
  r_layout->setContentsMargins(11, 0, 0, 0);
  QHBoxLayout *right_hlayout = new QHBoxLayout();
  fingerprint_label = new QLabel(this);
  right_hlayout->addWidget(fingerprint_label);

  right_hlayout->addWidget(initRouteControl());
  QPushButton *settings_btn = new QPushButton("Settings");
  right_hlayout->addWidget(settings_btn, 0, Qt::AlignRight);

  r_layout->addLayout(right_hlayout);

  video_widget = new VideoWidget(this);
  r_layout->addWidget(video_widget, 0, Qt::AlignTop);

  charts_widget = new ChartsWidget(this);
  r_layout->addWidget(charts_widget);

  h_layout->addWidget(right_container);

  // status bar
  status_bar = new QStatusBar(this);
  status_bar->setContentsMargins(0, 0, 0, 0);
  status_bar->setSizeGripEnabled(true);
  progress_bar = new QProgressBar();
  progress_bar->setRange(0, 100);
  progress_bar->setTextVisible(true);
  progress_bar->setFixedSize({230, 16});
  progress_bar->setVisible(false);
  status_bar->addPermanentWidget(progress_bar);
  main_layout->addWidget(status_bar);

  qRegisterMetaType<uint64_t>("uint64_t");
  qRegisterMetaType<ReplyMsgType>("ReplyMsgType");
  installMessageHandler([this](ReplyMsgType type, const std::string msg) {
    // use queued connection to recv the log messages from replay.
    emit logMessageFromReplay(QString::fromStdString(msg), 3000);
  });
  installDownloadProgressHandler([this](uint64_t cur, uint64_t total, bool success) {
    emit updateProgressBar(cur, total, success);
  });

  QObject::connect(this, &MainWindow::logMessageFromReplay, status_bar, &QStatusBar::showMessage);
  QObject::connect(this, &MainWindow::updateProgressBar, this, &MainWindow::updateDownloadProgress);
  QObject::connect(messages_widget, &MessagesWidget::msgSelectionChanged, detail_widget, &DetailWidget::setMessage);
  QObject::connect(detail_widget, &DetailWidget::showChart, charts_widget, &ChartsWidget::addChart);
  QObject::connect(charts_widget, &ChartsWidget::dock, this, &MainWindow::dockCharts);
  QObject::connect(settings_btn, &QPushButton::clicked, this, &MainWindow::setOption);
  QObject::connect(load_from_paste, &QPushButton::clicked, this, &MainWindow::loadDBCFromPaste);
  QObject::connect(save_btn, &QPushButton::clicked, [=]() {
    // TODO: save DBC to file
  });
  QObject::connect(can, &CANMessages::eventsMerged, this, &MainWindow::loadDBCFromFingerprint);
  QObject::connect(dbc_combo, SIGNAL(activated(const QString &)), SLOT(loadDBCFromName(const QString &)));

  QFile json_file("./car_fingerprint_to_dbc.json");
  if (json_file.open(QIODevice::ReadOnly)) {
    fingerprint_to_dbc = QJsonDocument::fromJson(json_file.readAll());
  }
}

void MainWindow::loadRoute(const QString &route, const QString &data_dir, bool use_qcam) {
  LoadRouteDialog dlg(this, route, data_dir, use_qcam);
  dlg.exec();
}

void MainWindow::loadDBCFromName(const QString &name) {
  dbc()->open(name);
  dbc_combo->setCurrentText(name);
}

void MainWindow::loadDBCFromPaste() {
  LoadDBCDialog dlg(this);
  if (dlg.exec()) {
    dbc()->open("from paste", dlg.dbc_edit->toPlainText());
    dbc_combo->setCurrentText("loaded from paste");
  }
}

void MainWindow::loadDBCFromFingerprint() {
  fingerprint_label->setText(can->carFingerprint());
  auto fingerprint = can->carFingerprint();
  if (!fingerprint.isEmpty() && dbc()->name().isEmpty()) {
    auto dbc_name = fingerprint_to_dbc[fingerprint];
    if (dbc_name != QJsonValue::Undefined) {
      loadDBCFromName(dbc_name.toString());
    }
  }
}

QToolButton *MainWindow::initRouteControl() {
  route_btn = new QToolButton(this);
  route_btn->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
  route_btn->setIcon(style()->standardIcon(QStyle::SP_DirOpenIcon));
  route_btn->setText(can->route());
  QObject::connect(route_btn, &QToolButton::clicked, [=]() {
    LoadRouteDialog dlg(this);
    dlg.exec();
  });
  return route_btn;
}

// void MainWindow::
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
    r_layout->addWidget(charts_widget);
    floating_window->deleteLater();
    floating_window = nullptr;
  } else if (!dock && !floating_window) {
    floating_window = new QWidget(nullptr);
    floating_window->setLayout(new QVBoxLayout());
    floating_window->layout()->addWidget(charts_widget);
    floating_window->installEventFilter(charts_widget);
    floating_window->setMinimumSize(QGuiApplication::primaryScreen()->size() / 2);
    floating_window->showMaximized();
  }
}

void MainWindow::closeEvent(QCloseEvent *event) {
  main_win = nullptr;
  if (floating_window)
    floating_window->deleteLater();
  QWidget::closeEvent(event);
}

void MainWindow::setOption() {
  SettingsDlg dlg(this);
  dlg.exec();
}

// LoadDBCDialog

LoadDBCDialog::LoadDBCDialog(QWidget *parent) : QDialog(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  dbc_edit = new QTextEdit(this);
  dbc_edit->setAcceptRichText(false);
  dbc_edit->setPlaceholderText(tr("paste DBC file here"));
  main_layout->addWidget(dbc_edit);
  auto buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  main_layout->addWidget(buttonBox);

  setFixedWidth(640);
  connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
  connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

// LoadRouteDialog

LoadRouteDialog::LoadRouteDialog(QWidget *parent, const QString &route, const QString &data_dir, bool use_qcam) : QDialog(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  main_layout->addWidget(new QLabel("selec route"));
  QHBoxLayout *h_layout = new QHBoxLayout();
  h_layout->addWidget(new QLabel("Route:"));
  route_edit = new QLineEdit(route, this);
  route_edit->setPlaceholderText(tr("Enter remote route name or select local route"));
  h_layout->addWidget(route_edit);
  QPushButton *file_btn = new QPushButton("Broser", this);
  h_layout->addWidget(file_btn);
  main_layout->addLayout(h_layout);
  main_layout->addStretch();

  auto buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  main_layout->addWidget(buttonBox);

  setFixedWidth(640);

  QObject::connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
  QObject::connect(buttonBox, &QDialogButtonBox::accepted, this, &LoadRouteDialog::loadClicked);
  QObject::connect(file_btn, &QPushButton::clicked, [=]() {
    QString dir = QFileDialog::getExistingDirectory(this, tr("Open Local Route"), "/home");
    route_edit->setText(dir);
  });

  show();
  if (!route.isEmpty()) {
    route_edit->setEnabled(false);
    file_btn->setEnabled(false);
    buttonBox->setEnabled(false);
    loadRoute(route, data_dir, use_qcam);
  }
}

void LoadRouteDialog::loadClicked() {
  QString route_string = route_edit->text();
  QString route = route_string;
  QString data_dir;
  if (route_string.indexOf('/') >= 0) {
    if (int idx = route_string.lastIndexOf('/'); idx != -1) {
      data_dir = route_string.mid(0, idx);
      QString basename = route_string.mid(idx + 1);
      if (int pos = basename.lastIndexOf("--"); pos != -1)
        route = "0000000000000000|" + basename.mid(0, pos);
    }
  }
  loadRoute(route, data_dir, false);
}

void LoadRouteDialog::loadRoute(const QString &route, const QString &data_dir, bool use_qcam) {
  if (can->loadRoute(route, data_dir)) {
    qInfo() << "loading route" << route << (data_dir.isEmpty() ? "from " + data_dir : "");
    accept();
  } else {
    qInfo() << "failed to load route" << route;
    // QMessageBox::warning(this, tr("Warning"), tr("Failed to load route %1\n make sure the route name is correct").arg(route));
  }
}
