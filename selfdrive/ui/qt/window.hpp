#pragma once

#include <QWidget>
#include <QTimer>
#include <QLabel>
#include <QGuiApplication>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QStackedLayout>

#include "qt/qt_sound.hpp"
#include "ui/ui.hpp"

class MainWindow : public QWidget
{
  Q_OBJECT

public:
  explicit MainWindow(QWidget *parent = 0);

private:
  QStackedLayout *main_layout;

public slots:
  void openSettings();
  void closeSettings();

};

#ifdef QCOM2
const int vwp_w = 2160;
#else
const int vwp_w = 1920;
#endif
const int vwp_h = 1080;
class GLWindow : public QOpenGLWidget, protected QOpenGLFunctions
{
  Q_OBJECT

public:
  using QOpenGLWidget::QOpenGLWidget;
  explicit GLWindow(QWidget *parent = 0);
  void setLabel(QLabel *l);
  ~GLWindow();

protected:
  void mousePressEvent(QMouseEvent *e) override;
  void initializeGL() override;
  void resizeGL(int w, int h) override;
  void paintGL() override;


private:
  QTimer * timer;
  UIState * ui_state;
  QtSound sound;
  QLabel * label = NULL;
  bool onroad = true;

public slots:
  void timerUpdate();

signals:
  void openSettings();
};
