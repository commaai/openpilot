#pragma once

#include <QLabel>
#include <QSlider>
#include <QStackedLayout>
#include <QWidget>
#include <set>

#include "selfdrive/ui/qt/widgets/cameraview.h"
#include "selfdrive/ui/replay/replay.h"

class RouteSelector : public QWidget {
  Q_OBJECT
 public:
  RouteSelector(QWidget *parent = nullptr);

 signals:
  void selectRoute(const QString route);

 protected:
  std::set<QString> route_names;
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
  void replayRoute(const QString &route, const QString &data_dir = {});
  void seekTo(int pos);

 private:
  QStackedLayout *stacked_layout;
  RouteSelector *route_selector = nullptr;
  CameraViewWidget *cam_view = nullptr;
  TimelineWidget *timeline = nullptr;
  std::unique_ptr<Replay> replay;
  std::vector<QPixmap> thumbnails;
};
