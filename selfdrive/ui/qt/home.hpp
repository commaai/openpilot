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
#include <QMutex>
#include <QWaitCondition>

#include "sound.hpp"
#include "ui/ui.hpp"
#include "common/util.h"
#include "widgets/offroad_alerts.hpp"

class UIUpdater;

// container window for onroad NVG UI
class GLWindow : public QOpenGLWidget {
  Q_OBJECT

public:
  explicit GLWindow(QWidget* parent = 0);
  void wake();
  ~GLWindow();
  std::atomic<bool> frameSwapped_ = false;

signals:
  void offroadTransition(bool offroad);
  void screen_shutoff();

protected:
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
  void moveContextToThread();
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
  
signals:
  void contextWanted();

private:
  void run() override;
  void draw();

  bool inited_ = false, onroad_ = true, exit_ = false;
  GLWindow* glWindow_;

  QMutex renderMutex_;
  QMutex grabMutex_;
  QWaitCondition grabCond_;

  Sound sound;
  inline static UIState ui_state_ = {0};

  friend UIState *uiState();
  friend class GLWindow;
};

inline UIState *uiState() {
  return &UIUpdater::ui_state_;
}
