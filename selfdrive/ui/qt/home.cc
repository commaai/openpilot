#include <cmath>
#include <fstream>
#include <iostream>
#include <exception>

#include <QDateTime>
#include <QHBoxLayout>
#include <QMouseEvent>
#include <QVBoxLayout>

#include "common/util.h"
#include "common/params.h"
#include "common/timing.h"
#include "common/swaglog.h"
#include "common/watchdog.h"
#include "selfdrive/hardware/hw.h"

#include "home.hpp"
#include "paint.hpp"
#include "qt_window.hpp"
#include "widgets/drive_stats.hpp"
#include "widgets/setup.hpp"

#define BACKLIGHT_DT 0.25
#define BACKLIGHT_TS 2.00
#define BACKLIGHT_OFFROAD 50

// HomeWindow: the container for the offroad (OffroadHome) and onroad (GLWindow) UIs

HomeWindow::HomeWindow(QWidget* parent) : QWidget(parent) {
  layout = new QStackedLayout();
  layout->setStackingMode(QStackedLayout::StackAll);

  // onroad UI
  glWindow = new GLWindow(this);
  layout->addWidget(glWindow);

  // draw offroad UI on top of onroad UI
  home = new OffroadHome();
  layout->addWidget(home);

  QObject::connect(glWindow, SIGNAL(offroadTransition(bool)), home, SLOT(setVisible(bool)));
  QObject::connect(glWindow, SIGNAL(offroadTransition(bool)), this, SIGNAL(offroadTransition(bool)));
  QObject::connect(glWindow, SIGNAL(screen_shutoff()), this, SIGNAL(closeSettings()));
  QObject::connect(this, SIGNAL(openSettings()), home, SLOT(refresh()));

  setLayout(layout);
}

void HomeWindow::mousePressEvent(QMouseEvent* e) {
  UIState* ui_state = &glWindow->ui_state;
  if (GLWindow::ui_state.scene.driver_view) {
    Params().putBool("IsDriverViewEnabled", false);
    GLWindow::ui_state.scene.driver_view = false;
    return;
  }

  glWindow->wake();

  // Settings button click
  if (!ui_state->sidebar_collapsed && settings_btn.ptInRect(e->x(), e->y())) {
    emit openSettings();
  }

  // Handle sidebar collapsing
  if (ui_state->scene.started && (e->x() >= ui_state->viz_rect.x - bdr_s)) {
    ui_state->sidebar_collapsed = !ui_state->sidebar_collapsed;
  }
}


// OffroadHome: the offroad home page

OffroadHome::OffroadHome(QWidget* parent) : QWidget(parent) {
  QVBoxLayout* main_layout = new QVBoxLayout();
  main_layout->setContentsMargins(sbr_w + 50, 50, 50, 50);

  // top header
  QHBoxLayout* header_layout = new QHBoxLayout();

  date = new QLabel();
  date->setStyleSheet(R"(font-size: 55px;)");
  header_layout->addWidget(date, 0, Qt::AlignHCenter | Qt::AlignLeft);

  alert_notification = new QPushButton();
  alert_notification->setVisible(false);
  QObject::connect(alert_notification, SIGNAL(released()), this, SLOT(openAlerts()));
  header_layout->addWidget(alert_notification, 0, Qt::AlignHCenter | Qt::AlignRight);

  std::string brand = Params().getBool("Passive") ? "dashcam" : "openpilot";
  QLabel* version = new QLabel(QString::fromStdString(brand + " v" + Params().get("Version")));
  version->setStyleSheet(R"(font-size: 55px;)");
  header_layout->addWidget(version, 0, Qt::AlignHCenter | Qt::AlignRight);

  main_layout->addLayout(header_layout);

  // main content
  main_layout->addSpacing(25);
  center_layout = new QStackedLayout();

  QHBoxLayout* statsAndSetup = new QHBoxLayout();
  statsAndSetup->setMargin(0);

  DriveStats* drive = new DriveStats;
  drive->setFixedSize(800, 800);
  statsAndSetup->addWidget(drive);

  SetupWidget* setup = new SetupWidget;
  //setup->setFixedSize(700, 700);
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
  setStyleSheet(R"(
    * {
     color: white;
    }
  )");
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
  if (!alerts_widget->alertCount && !alerts_widget->updateAvailable) {
    emit closeAlerts();
    alert_notification->setVisible(false);
    return;
  }

  if (alerts_widget->updateAvailable) {
    alert_notification->setText("UPDATE");
  } else {
    int alerts = alerts_widget->alertCount;
    alert_notification->setText(QString::number(alerts) + " ALERT" + (alerts == 1 ? "" : "S"));
  }

  if (!alert_notification->isVisible() && !first_refresh) {
    emit openAlerts();
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


// GLWindow: the onroad UI

static void handle_display_state(UIState* s, bool user_input) {
  static int awake_timeout = 0;
  awake_timeout = std::max(awake_timeout - 1, 0);

  constexpr float accel_samples = 5*UI_FREQ;
  static float accel_prev = 0., gyro_prev = 0.;

  bool should_wake = s->scene.started || s->scene.ignition || user_input;
  if (!should_wake) {
    // tap detection while display is off
    bool accel_trigger = abs(s->scene.accel_sensor - accel_prev) > 0.2;
    bool gyro_trigger = abs(s->scene.gyro_sensor - gyro_prev) > 0.15;
    should_wake = accel_trigger && gyro_trigger;
    gyro_prev = s->scene.gyro_sensor;
    accel_prev = (accel_prev * (accel_samples - 1) + s->scene.accel_sensor) / accel_samples;
  }

  if (should_wake) {
    awake_timeout = 30 * UI_FREQ;
  } else if (awake_timeout > 0) {
    should_wake = true;
  }

  // handle state transition
  if (s->awake != should_wake) {
    s->awake = should_wake;
    Hardware::set_display_power(s->awake);
    LOGD("setting display power %d", s->awake.load());
  }
}

GLWindow::GLWindow(QWidget* parent) : brightness_filter(BACKLIGHT_OFFROAD, BACKLIGHT_TS, BACKLIGHT_DT), QOpenGLWidget(parent) {
  brightness_b = Params().get<float>("BRIGHTNESS_B").value_or(10.0);
  brightness_m = Params().get<float>("BRIGHTNESS_M").value_or(0.1);

  ui_state.sound = &sound;
  ui_state.fb_w = vwp_w;
  ui_state.fb_h = vwp_h;
  ui_init(&ui_state);

  wake();

  ui_updater = new UIUpdater(this);
  ui_updater->moveToThread(ui_updater);
  connect(ui_updater, SIGNAL(offroadTransition(bool)), this, SIGNAL(offroadTransition(bool)));

  connect(this, &GLWindow::aboutToCompose, ui_updater, &UIUpdater::pause, Qt::BlockingQueuedConnection);
  connect(this, &GLWindow::frameSwapped, ui_updater, &UIUpdater::resume, Qt::DirectConnection);
  connect(this, &GLWindow::aboutToResize, ui_updater, &UIUpdater::pause, Qt::BlockingQueuedConnection);
  ui_updater->start();
}

GLWindow::~GLWindow() {
  ui_updater->exit();
}

void GLWindow::backlightUpdate() {
  // Update brightness
  float clipped_brightness = std::min(100.0f, (ui_state.scene.light_sensor * brightness_m) + brightness_b);
  if (!ui_state.scene.started) {
    clipped_brightness = BACKLIGHT_OFFROAD;
  }

  int brightness = brightness_filter.update(clipped_brightness);
  if (!ui_state.awake) {
    brightness = 0;
    emit screen_shutoff();
  }

  if (brightness != last_brightness) {
    Hardware::set_brightness(brightness);
  }
  last_brightness = brightness;
}

void GLWindow::resizeGL(int w, int h) {
  std::cout << "resize " << w << "x" << h << std::endl;
}

void GLWindow::wake() {
  handle_display_state(&ui_state, true);
}

// UIUpdater

UIUpdater::UIUpdater(GLWindow* w) : QThread(), glWindow_(w), asleep_timer_(this) {
  connect(&asleep_timer_, &QTimer::timeout, this, &UIUpdater::update);
  connect(this, &UIUpdater::needUpdate, this, &UIUpdater::update, Qt::QueuedConnection);
}

void UIUpdater::pause() {
  if (!is_updating_) return;

  is_updating_ = false;
  glWindow_->context()->moveToThread(glWindow_->thread());
}

void UIUpdater::resume() {
  if (is_updating_) return;

  glWindow_->context()->moveToThread(this);
  is_updating_ = true;
  emit needUpdate();
}

void UIUpdater::update() {
  UIState *s = &glWindow_->ui_state;

  if (!s->scene.started && s->awake) {
    util::sleep_for(50);
  }

  double u1 = millis_since_boot();

  ui_update(s);

  if (s->scene.started != prev_onroad_) {
    prev_onroad_ = s->scene.started;
    emit offroadTransition(!prev_onroad_);
  }

  handle_display_state(s, false);
  if (s->awake != prev_awake_) {
    s->awake ? asleep_timer_.stop() : asleep_timer_.start(50);
    prev_awake_ = s->awake;
  }

  // Don't waste resources on drawing in case screen is off
  if (!s->awake) {
    return;
  }

  glWindow_->backlightUpdate();

  // scale volume with speed
  s->sound->setVolume(util::map_val(s->scene.car_state.getVEgo(), 0.f, 20.f,
                                    Hardware::MIN_VOLUME, Hardware::MAX_VOLUME));

  draw();

  double u2 = millis_since_boot();
  if (!s->scene.driver_view && (u2 - u1 > 66)) {
    // warn on sub 15fps
    LOGW("slow frame(%llu) time: %.2f", s->sm->frame, u2 - u1);
  }
}

void UIUpdater::draw() {
  if (!is_updating_) return;

  // Make the context (and an offscreen surface) current for this thread. The
  // QOpenGLWidget's fbo is bound in the context.
  glWindow_->makeCurrent();

  UIState *s = &glWindow_->ui_state;
  if (!inited_) {
    inited_ = true;
    initializeOpenGLFunctions();
    ui_nvg_init(s);
    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "OpenGL vendor: " << glGetString(GL_VENDOR) << std::endl;
    std::cout << "OpenGL renderer: " << glGetString(GL_RENDERER) << std::endl;
    std::cout << "OpenGL language version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
  }

  ui_update_vision(s);
  ui_draw(s);

  // context back to the gui thread.
  glWindow_->doneCurrent();
  // Schedule composition. Note that this will use QueuedConnection, meaning
  // that update() will be invoked on the gui thread.
  QMetaObject::invokeMethod(glWindow_, "update");
}
