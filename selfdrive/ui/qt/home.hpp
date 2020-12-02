#pragma once

#include <QTimer>
#include <QWidget>
#include <QGridLayout>
#include <QStackedWidget>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>

#include "qt_sound.hpp"
#include "ui/ui.hpp"


// container window for onroad NVG UI
class GLWindow : public QOpenGLWidget, protected QOpenGLFunctions {
  Q_OBJECT

public:
  using QOpenGLWidget::QOpenGLWidget;
  explicit GLWindow(QWidget *parent = 0);
  void wake();
  ~GLWindow();

protected:
  void mousePressEvent(QMouseEvent *e) override;
  void initializeGL() override;
  void resizeGL(int w, int h) override;
  void paintGL() override;

private:
  QTimer *timer;
  QTimer *backlight_timer;

  UIState *ui_state = nullptr;
  QtSound sound;

  bool onroad = true;

  // TODO: this shouldn't be here
  float brightness_b = 0;
  float brightness_m = 0;
  float smooth_brightness = 0;

public slots:
  void timerUpdate();
  void backlightUpdate();

signals:
  void openSettings();
};


class HomeWidget : public QStackedWidget {
  Q_OBJECT

public:
  explicit HomeWidget(QWidget *parent = 0);
};


class HomeWindow : public QWidget {
  Q_OBJECT

public:
  explicit HomeWindow(QWidget *parent = 0);
  GLWindow *glWindow;

private:
  QGridLayout *layout;
};

