#include "tools/cabana/mainwin.h"

#include <QApplication>
#include <QHBoxLayout>
#include <QScreen>
#include <QVBoxLayout>

#include "tools/replay/util.h"

static MainWindow *main_win = nullptr;
void qLogMessageHandler(QtMsgType type, const QMessageLogContext &context, const QString &msg) {
  main_win->showStatusMessage(msg);
}

MainWindow::MainWindow() : QWidget() {
  main_win = this;
  qInstallMessageHandler(qLogMessageHandler);

  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(11, 11, 11, 5);
  main_layout->setSpacing(0);

  QHBoxLayout *h_layout = new QHBoxLayout();
  h_layout->setContentsMargins(0, 0, 0, 0);
  main_layout->addLayout(h_layout);

  splitter = new QSplitter(Qt::Horizontal, this);
  messages_widget = new MessagesWidget(this);
  splitter->addWidget(messages_widget);

  detail_widget = new DetailWidget(this);
  splitter->addWidget(detail_widget);

  splitter->setSizes(settings.h_splitter_sizes);
  h_layout->addWidget(splitter);

  // right widgets
  QWidget *right_container = new QWidget(this);
  right_container->setFixedWidth(640);
  r_layout = new QVBoxLayout(right_container);
  r_layout->setContentsMargins(11, 0, 0, 0);
  QHBoxLayout *right_hlayout = new QHBoxLayout();
  fingerprint_label = new QLabel(this);
  right_hlayout->addWidget(fingerprint_label);

  // TODO: click to select another route.
  right_hlayout->addWidget(new QLabel(can->route()));
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
  QObject::connect(settings_btn, &QPushButton::clicked, this, &MainWindow::openSettingsDlg);
  QObject::connect(can, &CANMessages::eventsMerged, [=]() { fingerprint_label->setText(can->carFingerprint() ); });
  QObject::connect(dbc(), &DBCManager::DBCFileChanged, this, &MainWindow::restoreSession);
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
  if (floating_window)
    floating_window->deleteLater();

  saveSession();
  QWidget::closeEvent(event);
}

void MainWindow::openSettingsDlg() {
  SettingsDlg dlg(this);
  dlg.exec();
}

void MainWindow::saveSession() {
  settings.selected_msgs = detail_widget->selectedMessages();
  settings.charts = charts_widget->allChartIds();
  settings.h_splitter_sizes = splitter->sizes();
  settings.save();
}

void MainWindow::restoreSession() {
  if (!settings.selected_msgs.isEmpty())
    detail_widget->setSelectedMessages(settings.selected_msgs);

  for (const auto &chart_id : settings.charts) {
    if (auto l = chart_id.split(":"); l.size() == 3) {
      auto id = l[0] + ":" + l[1];
      if (auto msg = dbc()->msg(id)) {
        auto it = std::find_if(msg->sigs.begin(), msg->sigs.end(), [&](auto &s) { return l[2] == s.name.c_str(); });
        if (it != msg->sigs.end())
          charts_widget->addChart(id, &(*it));
      }
    }
  }
  QObject::disconnect(dbc(), &DBCManager::DBCFileChanged, this, &MainWindow::restoreSession);
}
