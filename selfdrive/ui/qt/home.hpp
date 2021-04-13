#pragma once

#include <QLabel>
#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include <QPushButton>
#include <QStackedLayout>
#include <QStackedWidget>
#include <QTimer>
#include <QWidget>
#include <QThread>

#include "sound.hpp"
#include "ui/ui.hpp"
#include "common/util.h"
#include "widgets/offroad_alerts.hpp"

class UIUpdater;

// container window for onroad NVG UI
class GLWindow : public QOpenGLWidget {
  Q_OBJECT

public:
  using QOpenGLWidget::QOpenGLWidget;
  explicit GLWindow(QWidget* parent = 0);
  void wake();

signals:
  void offroadTransition(bool offroad);
  void screen_shutoff();

protected:
  void resizeGL(int w, int h) override;
  void resizeEvent(QResizeEvent* event) override {}
  void paintEvent(QPaintEvent* event) override {}

private:
  QTimer* backlight_timer;

  // TODO: make a nice abstraction to handle embedded device stuff
  float brightness_b = 0;
  float brightness_m = 0;
  float last_brightness = 0;
  FirstOrderFilter brightness_filter;

  UIUpdater *ui_updater;

public slots:
  void backlightUpdate();
};

// offroad home screen
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
  GLWindow* glWindow;

signals:
  void openSettings();
  void closeSettings();
  void offroadTransition(bool offroad);

protected:
  void mousePressEvent(QMouseEvent* e) override;

private:
  OffroadHome* home;
  QStackedLayout* layout;
};


class UIUpdater : public QThread, protected QOpenGLFunctions {
  Q_OBJECT

public:
  UIUpdater(GLWindow* w);
  void pause();
  void resume();

signals:
  void frameSwapped();

private:
  void update();
  void draw();

  bool prev_awake_ = false, inited_ = false, is_updating_ = false, onroad_ = true;
  QTimer asleep_timer_;
  GLWindow* glWindow_;
  inline static UIState ui_state_ = {0};
  Sound sound;
  
  friend UIState *uiState();
};

inline UIState *uiState() {
  return &UIUpdater::ui_state_;
}
