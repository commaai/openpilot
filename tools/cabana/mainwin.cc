#include "tools/cabana/mainwin.h"

#include <QApplication>
#include <QDialogButtonBox>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QScreen>
#include <QSplitter>
#include <QVBoxLayout>

MainWindow::MainWindow() : QWidget() {
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  QHBoxLayout *h_layout = new QHBoxLayout();
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

  QPushButton *settings_btn = new QPushButton("Settings");
  r_layout->addWidget(settings_btn, 0, Qt::AlignRight);

  video_widget = new VideoWidget(this);
  r_layout->addWidget(video_widget, 0, Qt::AlignTop);

  charts_widget = new ChartsWidget(this);
  r_layout->addWidget(charts_widget);

  h_layout->addWidget(right_container);

  QObject::connect(messages_widget, &MessagesWidget::msgSelectionChanged, detail_widget, &DetailWidget::setMessage);
  QObject::connect(detail_widget, &DetailWidget::showChart, charts_widget, &ChartsWidget::addChart);
  QObject::connect(charts_widget, &ChartsWidget::dock, this, &MainWindow::dockCharts);
  QObject::connect(settings_btn, &QPushButton::clicked, this, &MainWindow::setOption);
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
  QWidget::closeEvent(event);
}

void MainWindow::setOption() {
  SettingsDlg dlg(this);
  dlg.exec();
}

// SettingsDlg

SettingsDlg::SettingsDlg(QWidget *parent) : QDialog(parent) {
  setWindowTitle(tr("Settings"));
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  QFormLayout *form_layout = new QFormLayout();

  fps = new QSpinBox(this);
  fps->setRange(10, 100);
  fps->setSingleStep(10);
  fps->setValue(settings.fps);
  form_layout->addRow("FPS", fps);

  log_size = new QSpinBox(this);
  log_size->setRange(50, 500);
  log_size->setSingleStep(10);
  log_size->setValue(settings.can_msg_log_size);
  form_layout->addRow(tr("Log size"), log_size);

  cached_segment = new QSpinBox(this);
  cached_segment->setRange(3, 60);
  cached_segment->setSingleStep(1);
  cached_segment->setValue(settings.cached_segment_limit);
  form_layout->addRow(tr("Cached segments limit"), cached_segment);

  chart_height = new QSpinBox(this);
  chart_height->setRange(100, 500);
  chart_height->setSingleStep(10);
  chart_height->setValue(settings.chart_height);
  form_layout->addRow(tr("Chart height"), chart_height);

  main_layout->addLayout(form_layout);

  auto buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  main_layout->addWidget(buttonBox);

  setFixedWidth(360);
  connect(buttonBox, &QDialogButtonBox::accepted, this, &SettingsDlg::save);
  connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

void SettingsDlg::save() {
  settings.fps = fps->value();
  settings.can_msg_log_size = log_size->value();
  settings.cached_segment_limit = cached_segment->value();
  settings.chart_height = chart_height->value();
  settings.save();
  accept();
}
