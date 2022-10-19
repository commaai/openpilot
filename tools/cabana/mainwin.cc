#include "tools/cabana/mainwin.h"

#include <QApplication>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QScreen>
#include <QVBoxLayout>

MainWindow::MainWindow() : QWidget() {
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  QHBoxLayout *h_layout = new QHBoxLayout();
  main_layout->addLayout(h_layout);

  splitter = new QSplitter(Qt::Horizontal, this);

  messages_widget = new MessagesWidget(this);
  splitter->addWidget(messages_widget);

  detail_widget = new DetailWidget(this);
  splitter->addWidget(detail_widget);

  splitter->setSizes(settings.splitter_sizes);
  h_layout->addWidget(splitter);

  // right widgets
  QWidget *right_container = new QWidget(this);
  right_container->setFixedWidth(640);
  r_layout = new QVBoxLayout(right_container);

  QHBoxLayout *right_hlayout = new QHBoxLayout();
  QLabel *fingerprint_label = new QLabel(this);
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

  QObject::connect(messages_widget, &MessagesWidget::msgSelectionChanged, detail_widget, &DetailWidget::setMessage);
  QObject::connect(detail_widget, &DetailWidget::showChart, charts_widget, &ChartsWidget::addChart);
  QObject::connect(charts_widget, &ChartsWidget::dock, this, &MainWindow::dockCharts);
  QObject::connect(settings_btn, &QPushButton::clicked, this, &MainWindow::openSettingsDlg);
  QObject::connect(can, &CANMessages::eventsMerged, [=]() { fingerprint_label->setText(can->carFingerprint() ); });

  restoreSession();
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
  settings.dbc_name = dbc()->name();
  settings.selected_msg_id = messages_widget->currentMessageId();
  settings.charts = charts_widget->chartIDS();
  settings.splitter_sizes = splitter->sizes();
  settings.save();
}

void MainWindow::restoreSession() {
  if (!settings.dbc_name.isEmpty()) {
    messages_widget->openDBC(settings.dbc_name);
  }
  if (!settings.selected_msg_id.isEmpty()) {
    messages_widget->setCurrentMessageId(settings.selected_msg_id);
  }

  // TODO: move to chart
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
}
