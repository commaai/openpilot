#pragma once

#include <atomic>
#include <mutex>

#include <QFuture>
#include <QPixmap>
#include <QLabel>
#include <QList>
#include <QPushButton>
#include <QSlider>

#include "selfdrive/ui/qt/widgets/cameraview.h"
#include "tools/cabana/canmessages.h"

class Slider : public QSlider {
  Q_OBJECT

public:
  Slider(QWidget *parent);
  void mousePressEvent(QMouseEvent *e) override;
  void mouseMoveEvent(QMouseEvent *e) override;
  void leaveEvent(QEvent *event) override;
  void sliderChange(QAbstractSlider::SliderChange change) override;
  void paintEvent(QPaintEvent *ev) override;
  void loadThumbnails();

  int slider_x = -1;
  std::vector<std::tuple<int, int, TimelineType>> timeline;
  QMap<uint64_t, QString> thumbnails;
  std::mutex lock;
  std::atomic<bool> abort_load_thumbnail = false;
  QFuture<void> thumnail_future;
  QSize thumbnail_size = {};
};

class VideoWidget : public QWidget {
  Q_OBJECT

public:
  VideoWidget(QWidget *parnet = nullptr);
  void rangeChanged(double min, double max, bool is_zommed);

protected:
  void updateState();
  void pause(bool pause);

  CameraWidget *cam_widget;
  QLabel *end_time_label;
  QPushButton *play_btn;
  Slider *slider;
};
