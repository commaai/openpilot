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
#include <mutex>
#include <condition_variable>

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
  ~GLWindow();
  std::atomic<bool> frameSwapped_ = false;

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

signals:
  void mousePressed(int x, int y);

protected:
  void mousePressEvent(QMouseEvent* e) override;

private:
  OffroadHome* home;
  GLWindow* glWindow;
  QStackedLayout* layout;
};


class UIThread : public QThread, protected QOpenGLFunctions {
  Q_OBJECT

public:
  UIThread(GLWindow* w);
  bool onroad() const { return onroad_; }
  bool awake() const { return awake_;}

signals:
  void offroadTransition(bool offroad);
  void openSettings();
  void screen_shutoff();

public slots:
  void driverViewEnabled();
  void mousePressed(int x, int y);
  void wake();
  
private:
  void run() override;
  void draw();
  void backlightUpdate();
  void handle_display_state(bool user_input);

  bool inited_ = false, exit_ = false;
  std::atomic<bool> onroad_ = true, awake_ = true;
  GLWindow* glWindow_;

  std::mutex renderMutex_;
  std::condition_variable grabCond_;
  Sound sound;
  UIState ui_state_ = {0};

  // TODO: make a nice abstraction to handle embedded device stuff
  float brightness_b_ = 0;
  float brightness_m_ = 0;
  float last_brightness_ = 0;
  FirstOrderFilter brightness_filter_;
  inline static UIThread *ui_thread_;
  friend class GLWindow;
  friend UIThread *uiThread(); 
signals:
  void contextWanted();
};

inline UIThread* uiThread() { return UIThread::ui_thread_; }
