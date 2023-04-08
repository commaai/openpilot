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

struct AlertInfo {
  cereal::ControlsState::AlertStatus status;
  QString text1;
  QString text2;
};

class InfoLabel : public QWidget {
public:
  InfoLabel(QWidget *parent);
  void showPixmap(const QPoint &pt, const QString &sec, const QPixmap &pm, const AlertInfo &alert);
  void showAlert(const AlertInfo &alert);
  void paintEvent(QPaintEvent *event) override;
  QPixmap pixmap;
  QString second;
  AlertInfo alert_info;
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
  std::map<uint64_t, AlertInfo> alerts;
  QFuture<void> thumnail_future;
  InfoLabel thumbnail_label;
  QTimer timer;
  friend class VideoWidget;
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
  InfoLabel *alert_label;
  Slider *slider;
};
