#pragma once

#include "selfdrive/ui/qt/widgets/cameraview.h"

class DriverViewScene : public CameraWidget {
  Q_OBJECT

public:
  explicit DriverViewScene(QWidget *parent);
  void showEvent(QShowEvent *event) override;
  void hideEvent(QHideEvent *event) override;
  void paintGL() override;

  Params params;
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
  void closeView();
};
