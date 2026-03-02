#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>

#include <QFrame>
#include <QImage>
#include <QSlider>
#include <QToolBar>
#include <QTabBar>

#include "tools/cabana/cameraview.h"
#include "tools/cabana/utils/util.h"
#include "tools/replay/logreader.h"
#include "tools/cabana/streams/replaystream.h"

class Slider : public QSlider {
  Q_OBJECT

public:
  Slider(QWidget *parent);
  double currentSecond() const { return value() / factor; }
  void setCurrentSecond(double sec) { setValue(sec * factor); }
  void setTimeRange(double min, double max) { setRange(min * factor, max * factor); }
  void mousePressEvent(QMouseEvent *e) override;
  void paintEvent(QPaintEvent *ev) override;
  const double factor = 1000.0;
  double thumbnail_dispaly_time = -1;
};

class StreamCameraView : public CameraWidget {
  Q_OBJECT

public:
  StreamCameraView(std::string stream_name, VisionStreamType stream_type, QWidget *parent = nullptr);
  ~StreamCameraView();
  void showPausedOverlay() { update(); }
  void parseQLog(std::shared_ptr<LogReader> qlog);

protected:
  void drawImGuiOverlays() override;

private:
  GLuint getOrUploadTexture(uint64_t key, const QImage &img);

  std::map<uint64_t, QImage> big_thumbnails;
  std::map<uint64_t, QImage> thumbnails;
  std::map<uint64_t, GLuint> texture_cache;
  double thumbnail_dispaly_time = -1;
  friend class VideoWidget;
};

class VideoWidget : public QFrame {
  Q_OBJECT

public:
  VideoWidget(QWidget *parnet = nullptr);
  void showThumbnail(double seconds);

protected:
  bool eventFilter(QObject *obj, QEvent *event) override;
  QString formatTime(double sec, bool include_milliseconds = false);
  void timeRangeChanged();
  void updateState();
  void updatePlayBtnState();
  QWidget *createCameraWidget();
  void createPlaybackController();
  void createSpeedDropdown(QToolBar *toolbar);
  void loopPlaybackClicked();
  void vipcAvailableStreamsUpdated(std::set<VisionStreamType> streams);
  void showRouteInfo();

  StreamCameraView *cam_widget;
  QAction *time_display_action = nullptr;
  QAction *play_toggle_action = nullptr;
  QToolButton *speed_btn = nullptr;
  QAction *skip_to_end_action = nullptr;
  Slider *slider = nullptr;
  QTabBar *camera_tab = nullptr;
};
