#include "tools/cabana/videowidget.h"

#include <QVBoxLayout>

VideoWidget::VideoWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  cam_widget = new CameraViewWidget("camerad", VISION_STREAM_ROAD, false);
  main_layout->addWidget(cam_widget);

  slider = new QSlider(Qt::Horizontal, this);
  main_layout->addWidget(slider);
}
