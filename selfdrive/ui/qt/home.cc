#include <cmath>
#include <iostream>
#include <fstream>
#include <thread>

#include <QWidget>
#include <QMouseEvent>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QStackedLayout>
#include <QLayout>
#include <QDateTime>

#include "common/params.h"

#include "home.hpp"
#include "paint.hpp"
#include "qt_window.hpp"

#define BACKLIGHT_DT 0.25
#define BACKLIGHT_TS 2.00


OffroadHome::OffroadHome(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout();
  main_layout->setContentsMargins(sbr_w + 50, 50, 50, 50);

  center_layout = new QStackedLayout();

  // header
  QHBoxLayout *header_layout = new QHBoxLayout();

  date = new QLabel();
  date->setStyleSheet(R"(font-size: 55px;)");
  header_layout->addWidget(date, 0, Qt::AlignTop | Qt::AlignLeft);

  QLabel *version = new QLabel(QString::fromStdString("openpilot v" + Params().get("Version")));
  version->setStyleSheet(R"(font-size: 45px;)");
  header_layout->addWidget(version, 0, Qt::AlignTop | Qt::AlignRight);

  main_layout->addLayout(header_layout);

  alert_notification = new QPushButton();
  QObject::connect(alert_notification, SIGNAL(released()), this, SLOT(openAlerts()));
  main_layout->addWidget(alert_notification, 0, Qt::AlignTop | Qt::AlignRight);

  // center
  QLabel *drive = new QLabel("Drive me");
  drive->setStyleSheet(R"(font-size: 175px;)");
  center_layout->addWidget(drive);

  alerts_widget = new OffroadAlert();
  QObject::connect(alerts_widget, SIGNAL(closeAlerts()), this, SLOT(closeAlerts()));
  center_layout->addWidget(alerts_widget);
  center_layout->setAlignment(alerts_widget, Qt::AlignCenter);

  main_layout->addLayout(center_layout, 1);

  // set up refresh timer
  timer = new QTimer(this);
  QObject::connect(timer, SIGNAL(timeout()), this, SLOT(refresh()));
  refresh();
  timer->start(10 * 1000);

  setLayout(main_layout);
  setStyleSheet(R"(background-color: none;)");
}

void OffroadHome::openAlerts() {
  center_layout->setCurrentIndex(1);
}

void OffroadHome::closeAlerts() {
  center_layout->setCurrentIndex(0);
}

void OffroadHome::refresh() {
  bool first_refresh = !date->text().size();
  if (!isVisible() && !first_refresh) {
    return;
  }

  date->setText(QDateTime::currentDateTime().toString("dddd, MMMM d"));

  // update alerts

  alerts_widget->refresh();
  if (!alerts_widget->alerts.size() && !alerts_widget->updateAvailable) {
    alert_notification->setVisible(false);
    return;
  }

  if (alerts_widget->updateAvailable) {
    // There is a new release
    alert_notification->setText("UPDATE");
  } else {
    int alerts = alerts_widget->alerts.size();
    alert_notification->setText(QString::number(alerts) + " ALERT" + (alerts == 1 ? "" : "S"));
  }

  alert_notification->setVisible(true);

  // Red background for alerts, blue for update available
  QString style = QString(R"(
    padding: 15px;
    padding-left: 30px;
    padding-right: 30px;
    border: 1px solid;
    border-radius: 5px;
    font-size: 40px;
    font-weight: bold;
    background-color: red;
  )");
  if (alerts_widget->updateAvailable){
    style.replace("red", "blue");
  }
  alert_notification->setStyleSheet(style);
}


HomeWindow::HomeWindow(QWidget *parent) : QWidget(parent) {
  layout = new QGridLayout;
  layout->setMargin(0);

  // onroad UI
  glWindow = new GLWindow(this);
  layout->addWidget(glWindow, 0, 0);

  // draw offroad UI on top of onroad UI
  home = new OffroadHome();
  layout->addWidget(home, 0, 0);
  QObject::connect(glWindow, SIGNAL(offroadTransition(bool)), this, SLOT(setVisibility(bool)));
  QObject::connect(this, SIGNAL(openSettings()), home, SLOT(refresh()));
  setLayout(layout);
  setStyleSheet(R"(
    * {
      color: white;
    }
  )");
}

void HomeWindow::setVisibility(bool offroad) {
  home->setVisible(offroad);
}

void HomeWindow::mousePressEvent(QMouseEvent *e) {
  UIState *ui_state = glWindow->ui_state;

  glWindow->wake();

  // Settings button click
  if (!ui_state->scene.uilayout_sidebarcollapsed && settings_btn.ptInRect(e->x(), e->y())) {
    emit openSettings();
  }

  // Vision click
  if (ui_state->started && (e->x() >= ui_state->scene.viz_rect.x - bdr_s)) {
    ui_state->scene.uilayout_sidebarcollapsed = !ui_state->scene.uilayout_sidebarcollapsed;
  }
}


static void handle_display_state(UIState *s, int dt, bool user_input) {
  static int awake_timeout = 0;
  awake_timeout = std::max(awake_timeout-dt, 0);

  if (user_input || s->ignition || s->started) {
    s->awake = true;
    awake_timeout = 30*UI_FREQ;
  } else if (awake_timeout == 0) {
    s->awake = false;
  }
}

static void set_backlight(int brightness) {
  std::ofstream brightness_control("/sys/class/backlight/panel0-backlight/brightness");
  if (brightness_control.is_open()) {
    brightness_control << brightness << "\n";
    brightness_control.close();
  }
}


GLWindow::GLWindow(QWidget *parent) : QOpenGLWidget(parent) {
  timer = new QTimer(this);
  QObject::connect(timer, SIGNAL(timeout()), this, SLOT(timerUpdate()));

  backlight_timer = new QTimer(this);
  QObject::connect(backlight_timer, SIGNAL(timeout()), this, SLOT(backlightUpdate()));

  int result = read_param(&brightness_b, "BRIGHTNESS_B", true);
  result += read_param(&brightness_m, "BRIGHTNESS_M", true);
  if (result != 0) {
    brightness_b = 200.0;
    brightness_m = 10.0;
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

  wake();

  timer->start(0);
  backlight_timer->start(BACKLIGHT_DT * 100);
}

void GLWindow::backlightUpdate() {
  // Update brightness
  float k = (BACKLIGHT_DT / BACKLIGHT_TS) / (1.0f + BACKLIGHT_DT / BACKLIGHT_TS);

  float clipped_brightness = std::min(1023.0f, (ui_state->light_sensor*brightness_m) + brightness_b);
  smooth_brightness = clipped_brightness * k + smooth_brightness * (1.0f - k);
  int brightness = smooth_brightness;

  if (!ui_state->awake) {
    brightness = 0;
  }

  std::thread{set_backlight, brightness}.detach();
}

void GLWindow::timerUpdate() {
  if (ui_state->started != onroad) {
    onroad = ui_state->started;
    emit offroadTransition(!onroad);
#ifdef QCOM2
    timer->setInterval(onroad ? 0 : 1000);
#endif
  }

  // Fix awake timeout if running 1 Hz when offroad
  int dt = timer->interval() == 0 ? 1 : 20;
  handle_display_state(ui_state, dt, false);

  ui_update(ui_state);
  repaint();
}

void GLWindow::resizeGL(int w, int h) {
  std::cout << "resize " << w << "x" << h << std::endl;
}

void GLWindow::paintGL() {
  ui_draw(ui_state);
}

void GLWindow::wake() {
  // UI state might not be initialized yet
  if (ui_state != nullptr) {
    handle_display_state(ui_state, 1, true);
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
