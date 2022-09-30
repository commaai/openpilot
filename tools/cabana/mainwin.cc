#include "tools/cabana/mainwin.h"

#include <QHBoxLayout>
#include <QVBoxLayout>

MainWindow::MainWindow() : QWidget() {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  QHBoxLayout *h_layout = new QHBoxLayout();
  main_layout->addLayout(h_layout);

  can_widget = new CanWidget(this);
  h_layout->addWidget(can_widget);
  video_widget = new VideoWidget(this);
  h_layout->addWidget(video_widget);
}
