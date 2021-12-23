#pragma once

#include <QSlider>
#include <QWidget>

#include "selfdrive/ui/qt/widgets/cameraview.h"
#include "selfdrive/ui/replay/replay.h"

class RouteSelector : public QWidget {
  Q_OBJECT
 public:
  RouteSelector(QWidget *parent = nullptr);
};

class ThumbnailsWidget : public QWidget {
  Q_OBJECT
 public:
  ThumbnailsWidget(QWidget *parent = nullptr);
  void setThumbnail(std::vector<QPixmap> *thumbnails) {
    thumbnails_ = thumbnails;
    update();
  }

  void paintEvent(QPaintEvent *event);
protected:

  std::vector<QPixmap> *thumbnails_ = nullptr;
};

class TimelineWidget : public QWidget {
  Q_OBJECT

 public:
  TimelineWidget(QWidget *parent = nullptr);
  void setThumbnail(std::vector<QPixmap> *thumbnails);

 signals:
  void sliderReleased(int value);

 private:
  ThumbnailsWidget *thumbnails;
  QSlider *slider;
  QTimer timer;
};

class ReplayWidget : public QWidget {
  Q_OBJECT

 public:
  ReplayWidget(QWidget *parent = nullptr);
  void replayRoute(const QString &route);
  void seekTo(int pos);

 private:
  CameraViewWidget *cam_view = nullptr;
  TimelineWidget *timeline = nullptr;
  std::unique_ptr<Replay> replay;
  std::vector<QPixmap> thumbnails;
};
