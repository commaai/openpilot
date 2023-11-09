#pragma once

#include "selfdrive/ui/qt/widgets/cameraview.h"

class DriverViewScene : public CameraView {
  Q_OBJECT

public:
  explicit DriverViewScene(QWidget *parent);
  void paintGL() override;

  QPixmap face_img;
};

class DriverViewWindow : public QWidget {
  Q_OBJECT

public:
  explicit DriverViewWindow(QWidget *parent);

signals:
  void done();

protected:
  void mouseReleaseEvent(QMouseEvent* e) override;
  void showEvent(QShowEvent *event);
  void hideEvent(QHideEvent *event);
  void closeView();
  Params params;
};
