#pragma once

#include <QButtonGroup>
#include <QLabel>
#include <QPushButton>
#include <QSlider>

#include "selfdrive/ui/qt/widgets/cameraview.h"
#include "tools/cabana/canmessages.h"

class Slider : public QSlider {
  Q_OBJECT

public:
  Slider(QWidget *parent);
  void mousePressEvent(QMouseEvent *e) override;
  void sliderChange(QAbstractSlider::SliderChange change) override;
  void paintEvent(QPaintEvent *ev) override;
  void timelineUpdated();
  void streamStarted();

  int slider_x = -1;
  std::vector<std::tuple<int, int, TimelineType>> timeline;
};

class VideoWidget : public QWidget {
  Q_OBJECT

public:
  VideoWidget(QWidget *parnet = nullptr);
  void rangeChanged(double min, double max, bool is_zommed);
  void streamStarted();

protected:
  void updateState();
  void pause(bool pause);

  CameraWidget *cam_widget;
  QButtonGroup *btn_group;
  QLabel *end_time_label;
  QPushButton *play_btn;
  Slider *slider;
};
