#include "tools/cabana/mainwin.h"

#include <QApplication>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QMenu>
#include <QMessageBox>
#include <QScreen>
#include <QSplitter>
#include <QVBoxLayout>

#include "tools/replay/util.h"

static MainWindow *main_win = nullptr;
void qLogMessageHandler(QtMsgType type, const QMessageLogContext &context, const QString &msg) {
  if (main_win) main_win->showStatusMessage(msg);
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

  QSplitter *splitter = new QSplitter(Qt::Horizontal, this);
  messages_widget = new MessagesWidget(this);
  splitter->addWidget(messages_widget);

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
  QLabel *fingerprint_label = new QLabel(this);
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
  QObject::connect(can, &CANMessages::eventsMerged, [=]() { fingerprint_label->setText(can->carFingerprint()); });
}

QToolButton *MainWindow::initRouteControl() {
  QMenu *menu = new QMenu(this);
  QAction *load_remote_act = new QAction(tr("Open Foute From Remote"), this);
  QAction *load_local_act = new QAction(tr("Open Route From Local"), this);
  menu->addAction(load_remote_act);
  menu->addSeparator();
  menu->addAction(load_local_act);

  QToolButton *route_btn = new QToolButton(this);
  route_btn->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
  route_btn->setIcon(style()->standardIcon(QStyle::SP_DirOpenIcon));
  route_btn->setText(can->route());
  route_btn->setPopupMode(QToolButton::InstantPopup);
  route_btn->setMenu(menu);

  QObject::connect(menu, &QMenu::triggered, [=](QAction *action) {
    QString route, data_dir;
    if (action == load_remote_act) {
      bool ok = false;
      route = QInputDialog::getText(this, tr("Open Remote Route"), tr("Remote route:"), QLineEdit::Normal, "", &ok);
      if (ok == false)
        return;
    } else {
      QString dir = QFileDialog::getExistingDirectory(this, tr("Open Local Route"), "/home");
      if (dir.isEmpty())
        return;

      if (int idx = dir.lastIndexOf('/'); idx != -1) {
        data_dir = dir.mid(0, idx);
        QString basename = dir.mid(idx + 1);
        if (int pos = basename.lastIndexOf("--"); pos != -1)
          route = "000000000000000|" + basename.mid(0, pos);
      }
    }

    if (can->loadRoute(route, data_dir)) {
      qInfo() << "loading route" << route << (data_dir.isEmpty() ? "from " + data_dir : "");
      route_btn->setText(route);
    } else {
      qInfo() << "failed to load route" << route;
      QMessageBox::warning(this, tr("Warning"), tr("Failed to load route %1\n make sure the route name is correct").arg(route));
    }
  });
  return route_btn;
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
  main_win = nullptr;
  if (floating_window)
    floating_window->deleteLater();
  QWidget::closeEvent(event);
}

void MainWindow::setOption() {
  SettingsDlg dlg(this);
  dlg.exec();
}
