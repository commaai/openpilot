#pragma once

#include <QLabel>
#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include <QPushButton>
#include <QStackedLayout>
#include <QStackedWidget>
#include <QTimer>
#include <QWidget>

#include "sound.hpp"
#include "onroad.hpp"
#include "ui/ui.hpp"
#include "common/util.h"
#include "widgets/offroad_alerts.hpp"

class OffroadHome : public QWidget {
  Q_OBJECT

public:
  explicit OffroadHome(QWidget* parent = 0);

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
  void offroadTransition(bool offroad);

  void update(const UIState &s);

protected:
  void mousePressEvent(QMouseEvent* e) override;

private:
  OffroadHome *home;
  OnroadWindow *onroad;
  QStackedLayout *layout;
};
