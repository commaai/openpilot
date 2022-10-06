#pragma once

#include <QLabel>
#include <QSlider>
#include <QWidget>

#include "selfdrive/ui/qt/widgets/cameraview.h"

class VideoWidget : public QWidget {
  Q_OBJECT

public:
  VideoWidget(QWidget *parnet = nullptr);

protected:
  void rangeChanged(double min, double max);
  void updateState();

  CameraViewWidget *cam_widget;
  QLabel *time_label, *total_time_label;
  QSlider *slider;
};
