#pragma once

#include <QGuiApplication>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QBasicTimer>
#include <QWidget>
#include <QVBoxLayout>

#include "ui/ui.hpp"

class GLWindow : public QOpenGLWidget, protected QOpenGLFunctions
{
  Q_OBJECT

public:
  using QOpenGLWidget::QOpenGLWidget;
  ~GLWindow();

protected:
  void timerEvent(QTimerEvent *e) override;

  void initializeGL() override;
  void resizeGL(int w, int h) override;
  void paintGL() override;

private:
  QBasicTimer timer;
  UIState * ui_state;
};

class MainWindow : public QWidget
{
  Q_OBJECT

public:
  explicit MainWindow(QWidget *parent = 0);

};
