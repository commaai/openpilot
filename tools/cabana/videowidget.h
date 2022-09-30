#pragma once

#include <QSlider>
#include <QWidget>

#include "selfdrive/ui/qt/widgets/cameraview.h"

class VideoWidget : public QWidget {
  Q_OBJECT

public:
  VideoWidget(QWidget *parnet);

protected:
  CameraViewWidget *cam_widget;
  QSlider *slider;
};
