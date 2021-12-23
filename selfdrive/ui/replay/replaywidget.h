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
  void paintEvent(QPaintEvent *event);
};

class TimelineWidget : public QWidget {
  Q_OBJECT

 public:
  TimelineWidget(QWidget *parent = nullptr);

 private:
  void sliderReleased();
  ThumbnailsWidget *thumbnails;
  QSlider *slider;
};

class ReplayWidget : public QWidget {
  Q_OBJECT

 public:
  ReplayWidget(QWidget *parent = nullptr);
  void replayRoute(const QString &route);

 private:
  CameraViewWidget *cam_view = nullptr;
  TimelineWidget *timeline = nullptr;
  std::unique_ptr<Replay> replay;
  std::vector<QPixmap> thumbnails;
};
