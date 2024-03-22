#pragma once

#include <QSlider>
#include <memory>
#include <set>
#include <vector>
#include <QStackedLayout>
#include "selfdrive/ui/qt/offroad/settings.h"
#include "selfdrive/ui/qt/widgets/controls.h"
#include "tools/replay/replay.h"

class ReplayPanel : public QWidget {
  Q_OBJECT
public:
  explicit ReplayPanel(SettingsWindow *parent);

protected:
  void showEvent(QShowEvent *event) override;

  std::set<QString> route_names;
  std::vector<ButtonControl *> routes;
  SettingsWindow *settings_window;
  ListWidget *route_list_widget;
  QLabel *not_available_label;
  QStackedLayout *main_layout;
};

class ReplayControls : public QWidget {
  Q_OBJECT
public:
  ReplayControls(QWidget *parent);
  void adjustPosition();
  void start(const QString &route, const QString &data_dir);
  void stop();

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
