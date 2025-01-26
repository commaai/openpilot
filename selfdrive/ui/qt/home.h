#pragma once

#include <QFrame>
#include <QLabel>
#include <QPushButton>
#include <QStackedLayout>
#include <QTimer>
#include <QWidget>

#include "selfdrive/ui/ui.h"
#include "selfdrive/ui/qt/offroad/driverview.h"

#ifdef SUNNYPILOT
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"
#include "selfdrive/ui/sunnypilot/qt/onroad/onroad_home.h"
#include "selfdrive/ui/sunnypilot/qt/offroad/offroad_home.h"
#include "selfdrive/ui/sunnypilot/qt/sidebar.h"
#define OnroadWindow OnroadWindowSP
#define OffroadHome OffroadHomeSP
#define LayoutWidget LayoutWidgetSP
#define Sidebar SidebarSP
#define ElidedLabel ElidedLabelSP
#define SetupWidget SetupWidgetSP
#else
#include "selfdrive/ui/qt/widgets/controls.h"
#include "selfdrive/ui/qt/onroad/onroad_home.h"
#include "selfdrive/ui/qt/sidebar.h"
#endif

#include "selfdrive/ui/qt/offroad/offroad_home.h"

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

  Sidebar *sidebar;
  OffroadHome *home;
  OnroadWindow *onroad;
  BodyWindow *body;
  DriverViewWindow *driver_view;
  QStackedLayout *slayout;

protected slots:
  virtual void updateState(const UIState &s);
};
