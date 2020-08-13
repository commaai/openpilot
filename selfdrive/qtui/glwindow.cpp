#include <QMouseEvent>

#include <cmath>

#include "glwindow.hpp"
#include "ui/ui.hpp"


GLWindow::~GLWindow()
{
  makeCurrent();
  doneCurrent();
}


void GLWindow::timerEvent(QTimerEvent *)
{
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
}

void GLWindow::resizeGL(int w, int h)
{

}

void GLWindow::paintGL()
{
  ui_draw(ui_state);
}
