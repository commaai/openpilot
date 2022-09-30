#include "tools/cabana/videowidget.h"

#include <QVBoxLayout>

VideoWidget::VideoWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  cam_widget = new CameraViewWidget("camerad", VISION_STREAM_ROAD, true, this);
  cam_widget->setFixedSize(640, 480);
  main_layout->addWidget(cam_widget);

  slider = new QSlider(Qt::Horizontal, this);
  slider->setFixedWidth(640);
  main_layout->addWidget(slider);
}
