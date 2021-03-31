#pragma once

#include <QLabel>
#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include <QPushButton>
#include <QStackedLayout>
#include <QThread>
#include <QTimer>
#include <QWidget>

#include "sound.hpp"
#include "ui/ui.hpp"
#include "common/util.h"
#include "widgets/offroad_alerts.hpp"

// container window for onroad NVG UI
class UIUpdater;

class GLWindow : public QOpenGLWidget {
  Q_OBJECT

public:
  explicit GLWindow(QWidget* parent = 0);
  void wake();
  ~GLWindow();

signals:
  void offroadTransition(bool offroad);
  void screen_shutoff();

protected:
  void resizeGL(int w, int h) override;
  void resizeEvent(QResizeEvent* event) override {}
  void paintEvent(QPaintEvent* event) override {}

private:
  UIUpdater* ui_updater;
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
  inline static UIState ui_state = {};

signals:
  void offroadTransition(bool);
  void screen_shutoff();
  void frameSwapped();

private:
  void update();
  void draw();
  void backlightUpdate();

  Sound sound;

  bool inited_ = false, is_updating_ = false;
  bool prev_awake_ = false, prev_onroad_ = false;
  QTimer asleep_timer_;
  GLWindow* glWindow_;

  // TODO: make a nice abstraction to handle embedded device stuff
  float brightness_b = 0;
  float brightness_m = 0;
  float last_brightness = 0;
  FirstOrderFilter brightness_filter;
};

inline UIState *uiState() {
  return &UIUpdater::ui_state;
}
