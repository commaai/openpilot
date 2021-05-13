#pragma once

#include <QFrame>
#include <QLabel>
#include <QPushButton>
#include <QStackedLayout>
#include <QTimer>
#include <QWidget>

#include "selfdrive/ui/qt/onroad.h"
#include "selfdrive/ui/qt/sidebar.h"
#include "selfdrive/ui/qt/widgets/offroad_alerts.h"
#include "selfdrive/ui/ui.h"

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


class DriverViewWindow : public QOpenGLWidget, protected QOpenGLFunctions {
  Q_OBJECT

public:
  using QOpenGLWidget::QOpenGLWidget;
  explicit DriverViewWindow(QWidget* parent = 0);
  ~DriverViewWindow();

protected:
  void paintGL() override;
  void initializeGL() override;

protected slots:
  void onTimeout();

private:
  std::unique_ptr<UIVision> vision;
  QTimer *timer;
  // NVGcontext *vg;
  // int img_driver_face;
  QImage face_img;
  SubMaster sm;
  bool is_rhd;
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
  void previewDriverCam();

public slots:
  void offroadTransition(bool offroad);
  void driverView();

protected:
  void mousePressEvent(QMouseEvent* e) override;

private:
  Sidebar *sidebar;
  OffroadHome *home;
  OnroadWindow *onroad;
  DriverViewWindow *driver_view = nullptr;
  QStackedLayout *slayout;
};
