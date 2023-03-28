#pragma once

#include <atomic>
#include <mutex>

#include <QHBoxLayout>
#include <QFuture>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QTimer>

#include "selfdrive/ui/qt/widgets/cameraview.h"
#include "tools/cabana/streams/abstractstream.h"

class ThumbnailLabel : public QWidget {
public:
  ThumbnailLabel(QWidget *parent);
  void showPixmap(const QPoint &pt, const QString &sec, const QPixmap &pm);
  void paintEvent(QPaintEvent *event) override;
  QPixmap pixmap;
  QString second;
};

class Slider : public QSlider {
  Q_OBJECT

public:
  Slider(QWidget *parent);
  ~Slider();

private:
  void mousePressEvent(QMouseEvent *e) override;
  void mouseMoveEvent(QMouseEvent *e) override;
  void leaveEvent(QEvent *event) override;
  void sliderChange(QAbstractSlider::SliderChange change) override;
  void paintEvent(QPaintEvent *ev) override;
  void streamStarted();
  void loadThumbnails();

  int slider_x = -1;
  std::vector<std::tuple<int, int, TimelineType>> timeline;
  std::mutex thumbnail_lock;
  std::atomic<bool> abort_load_thumbnail = false;
  QMap<uint64_t, QPixmap> thumbnails;
  QFuture<void> thumnail_future;
  ThumbnailLabel thumbnail_label;
  QTimer timer;
};

class VideoWidget : public QFrame {
  Q_OBJECT

public:
  VideoWidget(QWidget *parnet = nullptr);
  void rangeChanged(double min, double max, bool is_zommed);

protected:
  void updateState();
  void updatePlayBtnState();
  QWidget *createCameraWidget();

  CameraWidget *cam_widget;
  QLabel *end_time_label;
  QLabel *time_label;
  QHBoxLayout *slider_layout;
  QPushButton *play_btn;
  Slider *slider;
};
