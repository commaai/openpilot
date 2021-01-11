#pragma once

#include <QLabel>
#include <QTimer>
#include <QWidget>
#include <QGridLayout>
#include <QStackedWidget>
#include <QStackedLayout>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QPushButton>

#include "qt_sound.hpp"
#include "widgets/offroad_alerts.hpp"
#include "ui/ui.hpp"


// container window for onroad NVG UI
class GLWindow : public QWidget {
  Q_OBJECT

public:
  explicit GLWindow(QWidget *parent = 0);
  void wake();
  ~GLWindow() = default;

  UIState *ui_state = nullptr;

  void initialize();
  void render();

signals:
  void offroadTransition(bool offroad);

protected:
  QPaintEngine *paintEngine() const override { return nullptr; }

  void paintEvent(QPaintEvent *e) override;
  bool event(QEvent *e) override;

private:
  QTimer *timer;
  QTimer *backlight_timer;

  QtSound sound;

  bool initialized = false;
  bool onroad = true;

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
  explicit OffroadHome(QWidget *parent = 0);

private:
  QTimer *timer;

  // offroad home screen widgets
  QLabel *date;
  QStackedLayout *center_layout;
  OffroadAlert *alerts_widget;
  QPushButton *alert_notification;

public slots:
  void closeAlerts();	
  void openAlerts();
  void refresh();
};


class HomeWindow : public QWidget {
  Q_OBJECT

public:
  explicit HomeWindow(QWidget *parent = 0);
  GLWindow *glWindow;

signals:
  void openSettings();

protected:
  void mousePressEvent(QMouseEvent *e) override;

private:
  QGridLayout *layout;
  OffroadHome *home;

private slots:
  void setVisibility(bool offroad);
};

