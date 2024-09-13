#pragma once

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>

#include <QHBoxLayout>
#include <QFrame>
#include <QPropertyAnimation>
#include <QSlider>
#include <QTabBar>

#include "selfdrive/ui/qt/widgets/cameraview.h"
#include "tools/cabana/utils/util.h"
#include "tools/replay/logreader.h"

struct AlertInfo {
  cereal::SelfdriveState::AlertStatus status;
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
  double currentSecond() const { return value() / factor; }
  void setCurrentSecond(double sec) { setValue(sec * factor); }
  void setTimeRange(double min, double max);
  AlertInfo alertInfo(double sec);
  QPixmap thumbnail(double sec);
  void parseQLog(std::shared_ptr<LogReader> qlog);

  const double factor = 1000.0;

private:
  void mousePressEvent(QMouseEvent *e) override;
  void mouseMoveEvent(QMouseEvent *e) override;
  bool event(QEvent *event) override;
  void paintEvent(QPaintEvent *ev) override;

  QMap<uint64_t, QPixmap> thumbnails;
  std::map<uint64_t, AlertInfo> alerts;
  InfoLabel *thumbnail_label;
};

class StreamCameraView : public CameraWidget {
  Q_OBJECT
  Q_PROPERTY(float overlayOpacity READ overlayOpacity WRITE setOverlayOpacity)

public:
  StreamCameraView(std::string stream_name, VisionStreamType stream_type, QWidget *parent = nullptr);
  void paintGL() override;
  void showPausedOverlay() { fade_animation->start(); }
  float overlayOpacity() const { return overlay_opacity; }
  void setOverlayOpacity(float opacity) {
    overlay_opacity = opacity;
    update();
  }

private:
  float overlay_opacity;
  QPropertyAnimation *fade_animation;
};

class VideoWidget : public QFrame {
  Q_OBJECT

public:
  VideoWidget(QWidget *parnet = nullptr);

protected:
  QString formatTime(double sec, bool include_milliseconds = false);
  void timeRangeChanged();
  void updateState();
  void updatePlayBtnState();
  QWidget *createCameraWidget();
  QHBoxLayout *createPlaybackController();
  void loopPlaybackClicked();
  void vipcAvailableStreamsUpdated(std::set<VisionStreamType> streams);

  StreamCameraView *cam_widget;
  QToolButton *time_btn = nullptr;
  ToolButton *seek_backward_btn = nullptr;
  ToolButton *play_btn = nullptr;
  ToolButton *seek_forward_btn = nullptr;
  ToolButton *loop_btn = nullptr;
  QToolButton *speed_btn = nullptr;
  ToolButton *skip_to_end_btn = nullptr;
  InfoLabel *alert_label = nullptr;
  Slider *slider = nullptr;
  QTabBar *camera_tab = nullptr;
};
