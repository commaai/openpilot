#include <cassert>
#include <iostream>
#include <cmath>
#include <iostream>
#include <signal.h>

#include "qt_window.hpp"
#include "paint.hpp"
#include "sound.hpp"

volatile sig_atomic_t do_exit = 0;

MainWindow::MainWindow(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout;
  GLWindow * glWindow = new GLWindow;


  main_layout->addWidget(glWindow);
  main_layout->setMargin(0);
  setLayout(main_layout);
}

GLWindow::GLWindow() {
  timer = new QTimer(this);
  QObject::connect(timer, SIGNAL(timeout()), this, SLOT(timerUpdate()));
}

GLWindow::~GLWindow() {
  makeCurrent();
  doneCurrent();
}

void GLWindow::initializeGL() {
  initializeOpenGLFunctions();

  ui_state = new UIState();
  ui_init(ui_state);
  ui_state->fb_w = 2160;
  ui_state->fb_h = 1080;

  int err = pthread_create(&connect_thread_handle, NULL,
                           vision_connect_thread, ui_state);
  assert(err == 0);

  timer->start(50);
}

void GLWindow::timerUpdate(){
  pthread_mutex_lock(&ui_state->lock);

  check_messages(ui_state);
  if (ui_state->vision_connected){
    ui_update(ui_state);
  }
  pthread_mutex_unlock(&ui_state->lock);

  // Paint
  update();
}

void GLWindow::resizeGL(int w, int h) {
  std::cout << "resize " << w << "x" << h << std::endl;
}

void GLWindow::paintGL() {
  pthread_mutex_lock(&ui_state->lock);
  ui_draw(ui_state);
  pthread_mutex_unlock(&ui_state->lock);
}

void GLWindow::mousePressEvent(QMouseEvent *e) {
  std::cout << "Click: " << e->x() << ", " << e->y() << std::endl;
  ui_state->scene.uilayout_sidebarcollapsed = !ui_state->scene.uilayout_sidebarcollapsed;
}


/* HACKS */
bool Sound::init(int volume) { return true; }
bool Sound::play(AudibleAlert alert) { printf("play sound: %d\n", (int)alert); return true; }
void Sound::stop() {}
void Sound::setVolume(int volume) {}
Sound::~Sound() {}

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
