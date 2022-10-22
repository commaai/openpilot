#pragma once

#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QWidget>

#include "selfdrive/ui/qt/widgets/cameraview.h"
#include "tools/cabana/canmessages.h"

class Slider : public QSlider {
  Q_OBJECT

public:
  Slider(QWidget *parent);
  void mousePressEvent(QMouseEvent *e) override;
  void sliderChange(QAbstractSlider::SliderChange change) override;
  void paintEvent(QPaintEvent *ev) override;

  int slider_x = -1;
  std::vector<std::tuple<int, int, TimelineType>> timeline;
};

class VideoWidget : public QWidget {
  Q_OBJECT

public:
  VideoWidget(QWidget *parnet = nullptr);

protected:
  void rangeChanged(double min, double max);
  void updateState();
  void pause(bool pause);

  CameraWidget *cam_widget;
  QLabel *end_time_label;
  QPushButton *play_btn;
  Slider *slider;
};
