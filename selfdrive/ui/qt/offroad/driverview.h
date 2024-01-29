#pragma once

#include "selfdrive/ui/qt/widgets/cameraview.h"

class DriverViewWindow : public CameraWidget {
  Q_OBJECT

public:
  explicit DriverViewWindow(QWidget *parent);

signals:
  void done();

protected:
  void showEvent(QShowEvent *event) override;
  void hideEvent(QHideEvent *event) override;
  void paintGL() override;

  Params params;
  QPixmap face_img;
};
