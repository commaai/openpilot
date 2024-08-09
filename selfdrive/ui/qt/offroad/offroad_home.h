#pragma once

#include "selfdrive/ui/qt/widgets/controls.h"
#include "selfdrive/ui/qt/offroad/experimental_mode.h"
#include "selfdrive/ui/qt/widgets/offroad_alerts.h"
#include "selfdrive/ui/qt/widgets/prime.h"
#include "selfdrive/ui/ui.h"
#include "selfdrive/ui/qt/util.h"

class OffroadHome : public QFrame {
  Q_OBJECT

public:
  explicit OffroadHome(QWidget* parent = 0);

  signals:
    void openSettings(int index = 0, const QString &param = "");

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
