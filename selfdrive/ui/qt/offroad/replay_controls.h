#pragma once

#include <QSlider>
#include <memory>
#include <map>
#include <set>
#include <vector>
#include <QStackedLayout>
#include "selfdrive/ui/qt/offroad/settings.h"
#include "selfdrive/ui/qt/widgets/controls.h"
#include "tools/replay/replay.h"

class RoutesPanel : public QWidget {
  Q_OBJECT
public:
  explicit RoutesPanel(SettingsWindow *parent);

  struct RouteItem {
    QString datetime;
    uint64_t seconds;
  };

protected:
  void showEvent(QShowEvent *event) override;
  void updateRoutes(const std::map<QString, RouteItem> &route_items);

  bool need_refresh = true;
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
  void paintEvent(QPaintEvent *event) override;
  bool route_loaded = false;
  QSlider *slider;
  QLabel *end_time_label;
  QPushButton *play_btn;
  QPushButton *stop_btn;
  QWidget *controls_container;
  QTimer *timer;
  std::unique_ptr<Replay> replay;
};
