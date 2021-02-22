#pragma once

#include <QGridLayout>
#include <QLabel>
#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include <QPushButton>
#include <QStackedLayout>
#include <QStackedWidget>
#include <QTimer>
#include <QWidget>

#include "qt_sound.hpp"
#include "ui/ui.hpp"
#include "widgets/offroad_alerts.hpp"

// container window for onroad NVG UI
class GLWindow : public QOpenGLWidget, protected QOpenGLFunctions {
  Q_OBJECT

public:
  using QOpenGLWidget::QOpenGLWidget;
  explicit GLWindow(QWidget* parent = 0);
  void wake();
  ~GLWindow();

  inline static UIState ui_state = {0};

signals:
  void offroadTransition(bool offroad);

protected:
  void initializeGL() override;
  void resizeGL(int w, int h) override;
  void paintGL() override;

private:
  QTimer* timer;
  QTimer* backlight_timer;

  QtSound sound;

  bool onroad = true;
  double prev_draw_t = 0;

  // TODO: this shouldn't be here
  float brightness_b = 0;
  float brightness_m = 0;
  float smooth_brightness = 0;

public slots:
  void timerUpdate();
  void backlightUpdate();
};

// offroad home screen
class OffroadHome : public QWidget {
  Q_OBJECT

public:
  explicit OffroadHome(QWidget* parent = 0);

private:
  QTimer* timer;

  // offroad home screen widgets
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
  GLWindow* glWindow;

signals:
  void offroadTransition(bool offroad);
  void openSettings();

protected:
  void mousePressEvent(QMouseEvent* e) override;

private:
  QGridLayout* layout;
  OffroadHome* home;

private slots:
  void setVisibility(bool offroad);
};
