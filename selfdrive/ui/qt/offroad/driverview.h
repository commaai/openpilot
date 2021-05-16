#pragma once

#include <memory>

#include "selfdrive/ui/qt/widgets/cameraview.h"
class DriverViewWindow : public CameraViewWidget {
Q_OBJECT
public:
  DriverViewWindow(QWidget *parent);

protected:
  void showEvent(QShowEvent *event);
  void hideEvent(QHideEvent *event);
  void paintGL() override;

private:
  SubMaster sm;
  QImage face_img;
  bool is_rhd;
};
