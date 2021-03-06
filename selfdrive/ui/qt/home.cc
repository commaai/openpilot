#include <cmath>
#include <fstream>
#include <iostream>
#include <thread>
#include <exception>


#include <QDateTime>
#include <QHBoxLayout>
#include <QLayout>
#include <QMouseEvent>
#include <QStackedLayout>
#include <QVBoxLayout>
#include <QWidget>

#include "common/params.h"
#include "common/timing.h"
#include "common/swaglog.h"

#include "home.hpp"
#include "paint.hpp"
#include "qt_window.hpp"
#include "widgets/drive_stats.hpp"
#include "widgets/setup.hpp"

#define BACKLIGHT_DT 0.25
#define BACKLIGHT_TS 2.00

OffroadHome::OffroadHome(QWidget* parent) : QWidget(parent) {
  QVBoxLayout* main_layout = new QVBoxLayout();
  main_layout->setContentsMargins(sbr_w + 50, 50, 50, 50);

  // top header
  QHBoxLayout* header_layout = new QHBoxLayout();

  date = new QLabel();
  date->setStyleSheet(R"(font-size: 55px;)");
  header_layout->addWidget(date, 0, Qt::AlignTop | Qt::AlignLeft);

  QLabel* version = new QLabel(QString::fromStdString("openpilot v" + Params().get("Version")));
  version->setStyleSheet(R"(font-size: 45px;)");
  header_layout->addWidget(version, 0, Qt::AlignTop | Qt::AlignRight);

  main_layout->addLayout(header_layout);

  alert_notification = new QPushButton();
  QObject::connect(alert_notification, SIGNAL(released()), this, SLOT(openAlerts()));
  main_layout->addWidget(alert_notification, 0, Qt::AlignTop | Qt::AlignRight);

  // main content
  main_layout->addSpacing(25);
  center_layout = new QStackedLayout();

  QHBoxLayout* statsAndSetup = new QHBoxLayout();

  DriveStats* drive = new DriveStats;
  drive->setFixedSize(1000, 800);
  statsAndSetup->addWidget(drive);

  SetupWidget* setup = new SetupWidget;
  statsAndSetup->addWidget(setup);

  QWidget* statsAndSetupWidget = new QWidget();
  statsAndSetupWidget->setLayout(statsAndSetup);

  center_layout->addWidget(statsAndSetupWidget);

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
    font-weight: 500;
    background-color: #E22C2C;
  )");
  if (alerts_widget->updateAvailable) {
    style.replace("#E22C2C", "#364DEF");
  }
  alert_notification->setStyleSheet(style);
}

HomeWindow::HomeWindow(QWidget* parent) : QWidget(parent) {
  layout = new QGridLayout;
  layout->setMargin(0);

  // onroad UI
  glWindow = new GLWindow(this);
  layout->addWidget(glWindow, 0, 0);

  // draw offroad UI on top of onroad UI
  home = new OffroadHome();
  layout->addWidget(home, 0, 0);
  QObject::connect(glWindow, SIGNAL(offroadTransition(bool)), this, SLOT(setVisibility(bool)));
  QObject::connect(glWindow, SIGNAL(offroadTransition(bool)), this, SIGNAL(offroadTransition(bool)));
  QObject::connect(glWindow, SIGNAL(screen_shutoff()), this, SIGNAL(closeSettings()));
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

void HomeWindow::mousePressEvent(QMouseEvent* e) {
  UIState* ui_state = &glWindow->ui_state;
  if (GLWindow::ui_state.scene.started && GLWindow::ui_state.scene.driver_view) {
    Params().write_db_value("IsDriverViewEnabled", "0", 1);
    return;
  }

  glWindow->wake();

  // Settings button click
  if (!ui_state->sidebar_collapsed && settings_btn.ptInRect(e->x(), e->y())) {
    emit openSettings();
  }

  // Vision click
  if (ui_state->scene.started && (e->x() >= ui_state->viz_rect.x - bdr_s)) {
    ui_state->sidebar_collapsed = !ui_state->sidebar_collapsed;
  }
}

static void handle_display_state(UIState* s, bool user_input) {
  static int awake_timeout = 0; // Somehow this only gets called on program start
  awake_timeout = std::max(awake_timeout - 1, 0);

  if (user_input || s->scene.ignition || s->scene.started) {
    s->awake = true;
    awake_timeout = 30 * UI_FREQ;
  } else if (awake_timeout == 0) {
    s->awake = false;
  }
}

static void set_backlight(int brightness) {
  try {
    std::ofstream brightness_control("/sys/class/backlight/panel0-backlight/brightness");
    if (brightness_control.is_open()) {
      brightness_control << brightness << "\n";
      brightness_control.close();
    }
  } catch (std::exception& e) {
    qDebug() << "Error setting brightness";
  }
}

GLWindow::GLWindow(QWidget* parent) : QOpenGLWidget(parent) {
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

  ui_state.sound = &sound;
  ui_init(&ui_state);

  wake();

  prev_draw_t = millis_since_boot();
  timer->start(1000 / UI_FREQ);
  backlight_timer->start(BACKLIGHT_DT * 1000);
}

void GLWindow::backlightUpdate() {
  // Update brightness
  float k = (BACKLIGHT_DT / BACKLIGHT_TS) / (1.0f + BACKLIGHT_DT / BACKLIGHT_TS);

  float clipped_brightness = std::min(1023.0f, (ui_state.scene.light_sensor * brightness_m) + brightness_b);
  smooth_brightness = clipped_brightness * k + smooth_brightness * (1.0f - k);
  int brightness = smooth_brightness;

  if (!ui_state.awake) {
    brightness = 0;
    emit screen_shutoff();
  }
  std::thread{set_backlight, brightness}.detach();
}

void GLWindow::timerUpdate() {
  // Connecting to visionIPC requires opengl to be current
  if (!ui_state.vipc_client->connected){
    makeCurrent();
  }

  if (ui_state.scene.started != onroad) {
    onroad = ui_state.scene.started;
    emit offroadTransition(!onroad);

    // Change timeout to 0 when onroad, this will call timerUpdate continously.
    // This puts visionIPC in charge of update frequency, reducing video latency
    timer->start(onroad ? 0 : 1000 / UI_FREQ);
  }

  handle_display_state(&ui_state, false);

  ui_update(&ui_state);
  repaint();
}

void GLWindow::resizeGL(int w, int h) {
  std::cout << "resize " << w << "x" << h << std::endl;
}

void GLWindow::paintGL() {
  if(GLWindow::ui_state.awake){
    ui_draw(&ui_state);

    double cur_draw_t = millis_since_boot();
    double dt = cur_draw_t - prev_draw_t;
    if (dt > 66 && onroad){
      // warn on sub 15fps
#ifdef QCOM2
      LOGW("slow frame(%llu) time: %.2f", ui_state.sm->frame, dt);
#endif
    }
    prev_draw_t = cur_draw_t;
  }
}

void GLWindow::wake() {
  handle_display_state(&ui_state, true);
}

FrameBuffer::FrameBuffer(const char *name, uint32_t layer, int alpha, int *out_w, int *out_h) {
  *out_w = vwp_w;
  *out_h = vwp_h;
}
FrameBuffer::~FrameBuffer() {}
