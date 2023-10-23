#pragma once

#include <map>
#include <memory>

#include <QSlider>
#include <QToolButton>

#include "selfdrive/ui/qt/widgets/cameraview.h"
#include "tools/cabana/streams/replaystream.h"

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
  double mapToSeconds(int pos) const { return (minimum() + pos * ((maximum() - minimum()) / (double)width())) / factor; }
  int mapToPosition(double sec) const { return width() * ((sec * factor - minimum()) / (maximum() - minimum())); }
  double currentSecond() const { return value() / factor; }
  void setCurrentSecond(double sec) { setValue(sec * factor); }
  void setTimeRange(double min, double max) { setRange(min * factor, max * factor); }
  void setTipPosition(int pos);

  const double factor = 1000.0;

private:
  void mousePressEvent(QMouseEvent *e) override;
  void paintEvent(QPaintEvent *ev) override;

  int tip_position = -1;
};

class VideoWidget : public QFrame {
  Q_OBJECT

public:
  VideoWidget(QWidget *parnet = nullptr);
  void zoomChanged(double min, double max, bool is_zommed);
  void setMaximumTime(double sec);
  AlertInfo alertInfo(double sec);
  QPixmap thumbnail(double sec);
  void showTip(double sec);

signals:
  void displayTipAt(double sec);
  void updateMaximumTime(double);

protected:
  bool eventFilter(QObject *obj, QEvent *event) override;
  QString formatTime(double sec, bool include_milliseconds = false);
  void parseQLog(int segnum, std::shared_ptr<LogReader> qlog);
  void updateState();
  void updatePlayBtnState();
  QWidget *createCameraWidget();

  CameraWidget *cam_widget = nullptr;
  double maximum_time = 0;
  QToolButton *time_btn = nullptr;
  ToolButton *play_btn = nullptr;
  ToolButton *skip_to_end_btn = nullptr;
  QToolButton *speed_btn = nullptr;
  Slider *slider = nullptr;
  InfoLabel *alert_label = nullptr;
  InfoLabel *thumbnail_label = nullptr;
  QMap<uint64_t, QPixmap> thumbnails;
  std::map<uint64_t, AlertInfo> alerts;
  bool zoomed = false;
};
