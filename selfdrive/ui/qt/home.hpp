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

class UIThread;

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
  void mousePressed(int x, int y);
  void openSettings();

public slots:
  void moveContextToThread();

protected:
  void resizeEvent(QResizeEvent* event) override {}
  void paintEvent(QPaintEvent* event) override {}

private:
  UIThread *ui_thread;
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


class UIThread : public QThread, protected QOpenGLFunctions {
  Q_OBJECT

public:
  UIThread(GLWindow* w);
  
private:
  void run() override;
  void draw();
  void handle_display_state(bool user_input);
  void backlightUpdate();

  bool inited_ = false, onroad_ = true, exit_ = false, prev_awake_ = true;
  GLWindow* glWindow_;

  QMutex renderMutex_;
  QMutex grabMutex_;
  QWaitCondition grabCond_;

  Sound sound;
  inline static UIState ui_state_ = {0};

  // TODO: make a nice abstraction to handle embedded device stuff
  float brightness_b_ = 0;
  float brightness_m_ = 0;
  float last_brightness_ = 0;
  FirstOrderFilter brightness_filter_;

signals:
  void contextWanted();

private slots:
  void mousePressed(int x, int y);

  friend UIState *uiState();
  friend class GLWindow;
};

inline UIState *uiState() {
  return &UIThread::ui_state_;
}
