#pragma once

#include <QGuiApplication>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QTimer>
#include <QWidget>
#include <QVBoxLayout>
#include <QMouseEvent>

#include "ui/ui.hpp"

class MainWindow : public QWidget
{
  Q_OBJECT

public:
  explicit MainWindow(QWidget *parent = 0);
};

class GLWindow : public QOpenGLWidget, protected QOpenGLFunctions
{
  Q_OBJECT

public:
  using QOpenGLWidget::QOpenGLWidget;
  GLWindow();
  ~GLWindow();

protected:
  void mousePressEvent(QMouseEvent *e) override;
  void initializeGL() override;
  void resizeGL(int w, int h) override;
  void paintGL() override;


private:
  QTimer * timer;
  UIState * ui_state;
  pthread_t connect_thread_handle;

public slots:
  void timerUpdate();
};
