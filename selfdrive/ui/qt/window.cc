#include <cassert>
#include <iostream>
#include <cmath>
#include <iostream>
#include <fstream>
#include <future>
#include <signal.h>

#include <QVBoxLayout>
#include <QMouseEvent>
#include <QPushButton>
#include <QGridLayout>

#include "window.hpp"
#include "settings.hpp"

#include "paint.hpp"
#include "common/util.h"

volatile sig_atomic_t do_exit = 0;

MainWindow::MainWindow(QWidget *parent) : QWidget(parent) {
  main_layout = new QStackedLayout;

#ifdef QCOM2
  set_core_affinity(7);
#endif

  GLWindow * glWindow = new GLWindow(this);
  main_layout->addWidget(glWindow);

  SettingsWindow * settingsWindow = new SettingsWindow(this);
  main_layout->addWidget(settingsWindow);


  main_layout->setMargin(0);
  setLayout(main_layout);
  QObject::connect(glWindow, SIGNAL(openSettings()), this, SLOT(openSettings()));
  QObject::connect(settingsWindow, SIGNAL(closeSettings()), this, SLOT(closeSettings()));

  setStyleSheet(R"(
    * {
      color: white;
      background-color: #072339;
    }
  )");
}

void MainWindow::openSettings() {
  main_layout->setCurrentIndex(1);
}

void MainWindow::closeSettings() {
  main_layout->setCurrentIndex(0);
}


GLWindow::GLWindow(QWidget *parent) : QOpenGLWidget(parent) {
  timer = new QTimer(this);
  QObject::connect(timer, SIGNAL(timeout()), this, SLOT(timerUpdate()));

  int result = read_param(&brightness_b, "BRIGHTNESS_B", true);
  result += read_param(&brightness_m, "BRIGHTNESS_M", true);
  if(result != 0) {
    brightness_b = 0.0;
    brightness_m = 5.0;
  }
  smooth_brightness = 512;
}

GLWindow::~GLWindow() {
  makeCurrent();
  doneCurrent();
}

void GLWindow::initializeGL() {
  initializeOpenGLFunctions();
  std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
  std::cout << "OpenGL vendor: " << glGetString(GL_VENDOR) << std::endl;
  std::cout << "OpenGL renderer: " << glGetString(GL_RENDERER) << std::endl;
  std::cout << "OpenGL language version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

  ui_state = new UIState();
  ui_state->sound = &sound;
  ui_state->fb_w = vwp_w;
  ui_state->fb_h = vwp_h;
  ui_init(ui_state);

  timer->start(50);
}

void GLWindow::timerUpdate(){
  // Update brightness
  float clipped_brightness = std::min(1023.0f, (ui_state->light_sensor*brightness_m) + brightness_b);
  smooth_brightness = clipped_brightness * 0.01f + smooth_brightness * 0.99f;
  int brightness = smooth_brightness;


#ifdef QCOM2
  if (ui_state->started != onroad){
    onroad = ui_state->started;
    timer->setInterval(onroad ? 50 : 1000);
  }

  if (!ui_state->started){
    brightness = 0;
  }
#endif

  std::async(std::launch::async,
             [brightness]{
               std::ofstream brightness_control("/sys/class/backlight/panel0-backlight/brightness");
               if (brightness_control.is_open()){
                 brightness_control << brightness << "\n";
                 brightness_control.close();
               }
             });


  ui_update(ui_state);
  update();
}

void GLWindow::resizeGL(int w, int h) {
  std::cout << "resize " << w << "x" << h << std::endl;
}

void GLWindow::paintGL() {
  ui_draw(ui_state);
}

void GLWindow::mousePressEvent(QMouseEvent *e) {
  // Settings button click
  if (!ui_state->scene.uilayout_sidebarcollapsed && settings_btn.ptInRect(e->x(), e->y())) {
    emit openSettings();
  }

  // Vision click
  if (ui_state->started && (e->x() >= ui_state->scene.viz_rect.x - bdr_s)){
    ui_state->scene.uilayout_sidebarcollapsed = !ui_state->scene.uilayout_sidebarcollapsed;
  }
}


GLuint visionimg_to_gl(const VisionImg *img, EGLImageKHR *pkhr, void **pph) {
  unsigned int texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img->width, img->height, 0, GL_RGB, GL_UNSIGNED_BYTE, *pph);
  glGenerateMipmap(GL_TEXTURE_2D);
  *pkhr = (EGLImageKHR)1; // not NULL
  return texture;
}

void visionimg_destroy_gl(EGLImageKHR khr, void *ph) {
  // empty
}

FramebufferState* framebuffer_init(const char* name, int32_t layer, int alpha,
                                   int *out_w, int *out_h) {
  return (FramebufferState*)1; // not null
}
