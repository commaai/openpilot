#pragma once

#include "selfdrive/ui/qt/offroad/offroad_home.h"

#include "selfdrive/ui/qt/offroad/driverview.h"
#include "selfdrive/ui/qt/body.h"
#include "selfdrive/ui/qt/onroad/onroad_home.h"
#include "selfdrive/ui/qt/sidebar.h"

class HomeWindow : public QWidget {
  Q_OBJECT

public:
  explicit HomeWindow(QWidget* parent = 0);

signals:
  void openSettings(int index = 0, const QString &param = "");
  void closeSettings();

public slots:
  void offroadTransition(bool offroad);
  void showDriverView(bool show);
  void showSidebar(bool show);

protected:
  void mousePressEvent(QMouseEvent* e) override;
  void mouseDoubleClickEvent(QMouseEvent* e) override;

private:
  Sidebar *sidebar;
  OffroadHome *home;
  OnroadWindow *onroad;
  BodyWindow *body;
  DriverViewWindow *driver_view;
  QStackedLayout *slayout;

private slots:
  void updateState(const UIState &s);
};
