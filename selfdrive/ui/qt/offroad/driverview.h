#pragma once

#include <QStackedLayout>

#include "selfdrive/ui/qt/widgets/cameraview.h"

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

private:
  Params params;
  SubMaster sm;
  QPixmap face_img;
  bool is_rhd = false;
  bool frame_updated = false;
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

  CameraWidget *cameraView;
  DriverViewScene *scene;
  QStackedLayout *layout;
};
