#include <QMouseEvent>
#include <QDebug>

#include <cmath>

#include "glwindow.hpp"
#include "ui/ui.hpp"

#ifndef __APPLE__
#define GLFW_INCLUDE_ES2
#else
#define GLFW_INCLUDE_GLCOREARB
#endif

#define GLFW_INCLUDE_GLEXT
#include <GLFW/glfw3.h>

typedef struct FramebufferState FramebufferState;

GLWindow::~GLWindow()
{
  makeCurrent();
  doneCurrent();
}


void GLWindow::timerEvent(QTimerEvent *)
{
  qDebug() << "Timer";
  update();
}

void GLWindow::initializeGL()
{

  initializeOpenGLFunctions();

  ui_state = new UIState();
  ui_state->fb_w = 1920;
  ui_state->fb_h = 1080;

  ui_nvg_init(ui_state);

  // Use QBasicTimer because its faster than QTimer
  // timer.start(12, this);

  qDebug() << "openGL done init";
}

void GLWindow::resizeGL(int w, int h)
{

}

void GLWindow::paintGL()
{
  qDebug() << "Paint";
  ui_draw(ui_state);
}
