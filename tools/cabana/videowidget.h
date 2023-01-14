#pragma once

#include <atomic>
#include <mutex>

#include <QFuture>
#include <QLabel>
#include <QPushButton>
#include <QSlider>

#include "selfdrive/ui/qt/widgets/cameraview.h"
#include "tools/cabana/canmessages.h"

class Slider : public QSlider {
  Q_OBJECT

public:
  Slider(QWidget *parent);

private:
  void mousePressEvent(QMouseEvent *e) override;
  void mouseMoveEvent(QMouseEvent *e) override;
  void sliderChange(QAbstractSlider::SliderChange change) override;
  void paintEvent(QPaintEvent *ev) override;
  void streamStarted();
  void loadThumbnails();
  QString getThumbnailString(const capnp::Data::Reader &data);

  int slider_x = -1;
  std::vector<std::tuple<int, int, TimelineType>> timeline;
  std::mutex thumbnail_lock;
  std::atomic<bool> abort_load_thumbnail = false;
  QMap<uint64_t, QString> thumbnails;
  QFuture<void> thumnail_future;
  QSize thumbnail_size = {};
};

class VideoWidget : public QFrame {
  Q_OBJECT

public:
  VideoWidget(QWidget *parnet = nullptr);
  void rangeChanged(double min, double max, bool is_zommed);

protected:
  void updateState();

  CameraWidget *cam_widget;
  QLabel *end_time_label;
  QPushButton *play_btn;
  Slider *slider;
};
