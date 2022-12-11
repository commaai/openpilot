#pragma once

#include <QDateTime>
#include <QLabel>
#include <QPushButton>
#include <QSet>
#include <QSlider>
#include <QTimer>

#include "selfdrive/ui/qt/widgets/controls.h"
#include "tools/replay/replay.h"

class RouteListWidget : public ListWidget {
  Q_OBJECT

public:
  RouteListWidget(QWidget *parent);

public slots:
  void buttonClicked();

protected:
  void showEvent(QShowEvent *event) override;

  QSet<QString> route_names;
  QString current_route;
};

class ReplayControls : public QWidget {
  Q_OBJECT

public:
  ReplayControls(QWidget *parent);
  ~ReplayControls();
  void start(const QString &route, const QString &data_dir);
  void stop();
  void adjustPosition();

protected:
  inline QString formatTime(int seconds) {
    return QDateTime::fromTime_t(seconds).toString(seconds > 60 * 60 ? "hh:mm:ss" : "mm:ss");
  }
  QSlider *slider;
  QLabel *end_time_label;
  QPushButton *play_btn;
  QPushButton *stop_btn;
  QTimer *timer;
  std::unique_ptr<Replay> replay;
};
