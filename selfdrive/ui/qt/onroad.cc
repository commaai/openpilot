#include <iostream>

#include "common/timing.h"
#include "common/swaglog.h"

#include "onroad.h"
#include "paint.h"

OnroadWindow::~OnroadWindow() {
  makeCurrent();
  doneCurrent();
}

void OnroadWindow::initializeGL() {
  initializeOpenGLFunctions();
  std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
  std::cout << "OpenGL vendor: " << glGetString(GL_VENDOR) << std::endl;
  std::cout << "OpenGL renderer: " << glGetString(GL_RENDERER) << std::endl;
  std::cout << "OpenGL language version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

  ui_nvg_init(&QUIState::ui_state);
  prev_draw_t = millis_since_boot();
}

void OnroadWindow::update(const UIState &s) {
  // Connecting to visionIPC requires opengl to be current
  if (s.vipc_client->connected){
    makeCurrent();
  }

  repaint();
}

void OnroadWindow::paintGL() {
  ui_draw(&QUIState::ui_state, width(), height());

  double cur_draw_t = millis_since_boot();
  double dt = cur_draw_t - prev_draw_t;
  // TODO: check if onroad
  if (dt > 66 && QUIState::ui_state.scene.started && !QUIState::ui_state.scene.driver_view) {
    // warn on sub 15fps
    LOGW("slow frame time: %.2f", dt);
  }
  prev_draw_t = cur_draw_t;
}
