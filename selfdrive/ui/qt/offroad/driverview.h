#pragma once

#include <QStackedLayout>

#include "selfdrive/ui/qt/widgets/cameraview.h"
#include "selfdrive/ui/qt/widgets/input.h"

class DriverViewScene : public QWidget {
  Q_OBJECT

public:
  explicit DriverViewScene(QWidget *parent);

public slots:
  void frameUpdated();

protected:
  void showEvent(QShowEvent *event) override;
  void hideEvent(QHideEvent *event) override;
  void paintEvent(QPaintEvent *event) override;

  Params params;
  QPixmap face_img;
  bool is_rhd = false;
  bool frame_updated = false;
};

class DriverViewWindow : public DialogBase {
  Q_OBJECT

public:
  explicit DriverViewWindow(QWidget *parent);

protected:
  void mouseReleaseEvent(QMouseEvent* e) override;

  CameraWidget *cameraView;
  DriverViewScene *scene;
  QStackedLayout *layout;
};
