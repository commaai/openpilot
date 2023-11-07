#pragma once

#include <QStackedLayout>

#include "selfdrive/ui/qt/widgets/cameraview.h"

class DriverViewScene : public QWidget {
  Q_OBJECT

public:
  explicit DriverViewScene(QWidget *parent);
  bool frame_updated = false;

public slots:
  void frameUpdated();

protected:
  void paintEvent(QPaintEvent *event) override;

private:
  QPixmap face_img;
  bool is_rhd = false;
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
  void showEvent(QShowEvent *event) override;
  void hideEvent(QHideEvent *event) override;

  CameraWidget *cameraView;
  DriverViewScene *scene;
  QStackedLayout *layout;
  Params params;
};
