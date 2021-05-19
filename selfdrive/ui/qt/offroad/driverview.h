#pragma once

#include <memory>

#include <QStackedLayout>

#include "selfdrive/common/util.h"
#include "selfdrive/ui/qt/widgets/cameraview.h"

class DriverViewWindow;
class DriverViewScene : public QWidget {
  Q_OBJECT

public:
  DriverViewScene(QWidget *parent);

public slots:
  void update(const UIState &s);
protected:
  void paintEvent(QPaintEvent *event);
  bool face_detected = false;
  bool frame_updated = false;
  bool is_rhd = false;
  float face_x = 0, face_y = 0;
  QImage face_img;
  friend class DriverViewWindow;
};

class DriverViewWindow : public QWidget {
  Q_OBJECT

public:
  DriverViewWindow(QWidget *parent);

signals:
  void update(const UIState &s);

protected:
  void showEvent(QShowEvent *event);
  void hideEvent(QHideEvent *event);

private:
  Params params;
  CameraViewWidget *cameraView;
  DriverViewScene *scene;
  QStackedLayout *layout;
};
