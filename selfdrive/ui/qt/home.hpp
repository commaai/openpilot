#pragma once

#include <QMutex>
#include <QWaitCondition>
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

class GLWindow;

class UIUpdater : public QObject, protected QOpenGLFunctions {
  Q_OBJECT

public:
  UIUpdater(GLWindow* w);
  void prepareExit() {
    exiting_ = true;
    grabCond_.notify_all();
  }

  QMutex renderMutex_, grabMutex_;
  QWaitCondition grabCond_;

signals:
  void contextWanted();
  void offroadTransition(bool);

public slots:
  void update();

private:
  void draw();
  bool inited_ = false, exiting_ = false, prev_awake_ = false;
  QTimer timer_;
  GLWindow* glWidget_;
};

// container window for onroad NVG UI
class GLWindow : public QOpenGLWidget {
  Q_OBJECT

public:
  using QOpenGLWidget::QOpenGLWidget;
  explicit GLWindow(QWidget* parent = 0);
  void wake();
  ~GLWindow();
  void backlightUpdate();
  inline static UIState ui_state = {0};
  bool onroad = true;

public slots:
  void grabContext();

protected:
  void resizeGL(int w, int h) override;
  void resizeEvent(QResizeEvent* event) override {}
  void paintEvent(QPaintEvent* event) override {}

private:
  QThread *thread;
  UIUpdater* ui_updater;
  Sound sound;
  // TODO: make a nice abstraction to handle embedded device stuff
  float brightness_b = 0;
  float brightness_m = 0;
  float last_brightness = 0;
  FirstOrderFilter brightness_filter;

signals:
  void offroadTransition(bool offroad);
  void screen_shutoff();
  void renderRequested();
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
