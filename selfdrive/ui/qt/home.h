#pragma once

#include <QFrame>
#include <QLabel>
#include <QPushButton>
#include <QStackedLayout>
#include <QTimer>
#include <QWidget>

#include "sidebar.h"
#include "onroad.h"
#include "ui/ui.h"
#include "widgets/offroad_alerts.h"

class OffroadHome : public QFrame {
  Q_OBJECT

public:
  explicit OffroadHome(QWidget* parent = 0);

protected:
  void showEvent(QShowEvent *event) override;

private:
  QTimer* timer;

  QLabel* date;
  QStackedLayout* center_layout;
  OffroadAlert* alerts_widget;
  QPushButton* alert_notification;

public slots:
  void closeAlerts();
  void openAlerts();
  void refresh();
};

class HomeWindow : public QWidget {
  Q_OBJECT

public:
  explicit HomeWindow(QWidget* parent = 0);

signals:
  void openSettings();
  void closeSettings();

  // forwarded signals
  void displayPowerChanged(bool on);
  void update(const UIState &s);
  void offroadTransitionSignal(bool offroad);

public slots:
  void offroadTransition(bool offroad);

protected:
  void mousePressEvent(QMouseEvent* e) override;

private:
  Sidebar *sidebar;
  OffroadHome *home;
  OnroadWindow *onroad;
  QStackedLayout *slayout;
};
