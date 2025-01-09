/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include "common/params.h"
#include "selfdrive/ui/qt/body.h"
#include "selfdrive/ui/qt/widgets/offroad_alerts.h"

#ifdef SUNNYPILOT
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"
#include "selfdrive/ui/sunnypilot/qt/onroad/onroad_home.h"
#include "selfdrive/ui/sunnypilot/qt/sidebar.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/prime.h"
#define OnroadWindow OnroadWindowSP
#define LayoutWidget LayoutWidgetSP
#define Sidebar SidebarSP
#define ElidedLabel ElidedLabelSP
#define SetupWidget SetupWidgetSP
#else
#include "selfdrive/ui/qt/widgets/controls.h"
#include "selfdrive/ui/qt/onroad/onroad_home.h"
#include "selfdrive/ui/qt/sidebar.h"
#include "selfdrive/ui/qt/widgets/prime.h"
#endif

class OffroadHome : public QFrame {
  Q_OBJECT

public:
  explicit OffroadHome(QWidget* parent = 0);

  signals:
    void openSettings(int index = 0, const QString &param = "");

protected:
  QHBoxLayout *home_layout;

private:
  void showEvent(QShowEvent *event) override;
  void hideEvent(QHideEvent *event) override;
  void refresh();

  Params params;

  QTimer* timer;
  ElidedLabel* version;
  QStackedLayout* center_layout;
  UpdateAlert *update_widget;
  OffroadAlert* alerts_widget;
  QPushButton* alert_notif;
  QPushButton* update_notif;
};
