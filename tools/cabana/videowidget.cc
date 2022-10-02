#include "tools/cabana/videowidget.h"

#include <QHBoxLayout>
#include <QPushButton>
#include <QButtonGroup>
#include <QVBoxLayout>

VideoWidget::VideoWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  cam_widget = new CameraViewWidget("camerad", VISION_STREAM_ROAD, true, this);
  cam_widget->setFixedSize(640, 480);
  main_layout->addWidget(cam_widget);

  slider = new QSlider(Qt::Horizontal, this);
  slider->setFixedWidth(640);
  main_layout->addWidget(slider);

  QHBoxLayout *control_layout = new QHBoxLayout();
  QPushButton *play = new QPushButton("play");
  control_layout->addWidget(play);
  QButtonGroup *group = new QButtonGroup(this);
  group->setExclusive(true);

  QPushButton *speed_1 = new QPushButton(tr("0.1x"), this);
  control_layout->addWidget(speed_1);
  group->addButton(speed_1);
  QPushButton *speed_2 = new QPushButton(tr("0.5x"), this);
  control_layout->addWidget(speed_2);
  group->addButton(speed_2);
  QPushButton *speed_3 = new QPushButton(tr("1x"), this);
  control_layout->addWidget(speed_3);
  group->addButton(speed_3);
  QPushButton *speed_4 = new QPushButton(tr("2x"), this);
  control_layout->addWidget(speed_4);
  group->addButton(speed_4);

  main_layout->addLayout(control_layout);
}
