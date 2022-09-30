#include "tools/cabana/mainwin.h"

#include <QHBoxLayout>
#include <QVBoxLayout>

MainWindow::MainWindow() : QWidget() {
  assert(parser != nullptr);

  QVBoxLayout *main_layout = new QVBoxLayout(this);
  QHBoxLayout *h_layout = new QHBoxLayout();
  main_layout->addLayout(h_layout);

  can_widget = new CanWidget(this);
  QObject::connect(can_widget, &CanWidget::addressChanged, [=](uint32_t address) {
    detail_widget->setItem(address);
  });
  h_layout->addWidget(can_widget);

  detail_widget = new DetailWidget(this);
  h_layout->addWidget(detail_widget);

  // right widget
  QWidget *right_container = new QWidget(this);
  right_container->setFixedWidth(640);
  QVBoxLayout *r_layout = new QVBoxLayout(right_container);
  video_widget = new VideoWidget(this);
  r_layout->addWidget(video_widget);

  charts_widget = new ChartsWidget(this);
  r_layout->addWidget(charts_widget);
  r_layout->addStretch();

  h_layout->addWidget(right_container);
}

void MainWindow::updated() {
}
