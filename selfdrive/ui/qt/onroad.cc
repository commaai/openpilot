#include <iostream>

#include "common/timing.h"
#include "common/swaglog.h"

#include "onroad.h"
#include "paint.h"


OnroadWindow::OnroadWindow(QWidget *parent) : QWidget(parent) {
  layout.setStackingMode(QStackedLayout::StackAll);

  // old UI on bottom
  nvg = new NvgWindow(this);
  layout.addWidget(nvg);
  QObject::connect(this, &OnroadWindow::update, nvg, &NvgWindow::update);

  // hack to align the onroad alerts, better way to do this?
  QVBoxLayout *alerts_container = new QVBoxLayout(this);
  alerts_container->setMargin(0);
  alerts_container->addStretch(1);
  alerts = new OnroadAlerts(this);
  alerts_container->addWidget(alerts, 0, Qt::AlignBottom);

  QWidget *w = new QWidget(this);
  w->setLayout(alerts_container);
  layout.addWidget(w);
  QObject::connect(this, &OnroadWindow::update, alerts, &OnroadAlerts::update);

  // alerts on top
  layout.setCurrentWidget(w);

  setLayout(&layout);
}

// ***** onroad widgets *****

OnroadAlerts::OnroadAlerts(QWidget *parent) : QFrame(parent) {
  layout = new QVBoxLayout(this);
  layout->setSpacing(0);
  layout->setMargin(10);

  title = new QLabel();
  title->setAlignment(Qt::AlignCenter);
  layout->addWidget(title, 0, Qt::AlignCenter);

  msg = new QLabel();
  msg->setAlignment(Qt::AlignCenter);
  layout->addWidget(msg, 0, Qt::AlignCenter);

  setLayout(layout);
}

void OnroadAlerts::update(const UIState &s) {
  if (s.scene.alert_size == cereal::ControlsState::AlertSize::SMALL) {
    title->setStyleSheet("font-size: 80px; font-weight: 400;");
  } else if (s.scene.alert_size == cereal::ControlsState::AlertSize::MID) {
    msg->setStyleSheet("font-size: 70px; font-weight: 400;");
    title->setStyleSheet("font-size: 80px; font-weight: 500;");
  } else if (s.scene.alert_size == cereal::ControlsState::AlertSize::FULL) {
    msg->setStyleSheet("font-size: 90px; font-weight: 400;");
    title->setStyleSheet("font-size: 120px; font-weight: 500;");
  }
  title->setText(QString::fromStdString(s.scene.alert_text1));
  msg->setText(QString::fromStdString(s.scene.alert_text2));
  msg->setWordWrap(s.scene.alert_size == cereal::ControlsState::AlertSize::FULL);
  title->setWordWrap(s.scene.alert_size == cereal::ControlsState::AlertSize::FULL);

  static std::map<cereal::ControlsState::AlertSize, const int> alert_size_map = {
      {cereal::ControlsState::AlertSize::SMALL, 241},
      {cereal::ControlsState::AlertSize::MID, 390},
      {cereal::ControlsState::AlertSize::FULL, s.fb_h}};
  setFixedHeight(alert_size_map[s.scene.alert_size]);

  setVisible(s.scene.alert_size != cereal::ControlsState::AlertSize::NONE);

  auto c = bg_colors[s.status];
  float alpha = 0.375 * cos((millis_since_boot() / 1000) * 2 * M_PI * s.scene.alert_blinking_rate) + 0.625; 
  const QString style = "OnroadAlerts { background-color: rgba(%1, %2, %3, %4); } * { color: white }";
  setStyleSheet(style.arg(c.r*255).arg(c.g*255).arg(c.b*255).arg(c.a*alpha*255));
}

NvgWindow::~NvgWindow() {
  makeCurrent();
  doneCurrent();
}

void NvgWindow::initializeGL() {
  initializeOpenGLFunctions();
  std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
  std::cout << "OpenGL vendor: " << glGetString(GL_VENDOR) << std::endl;
  std::cout << "OpenGL renderer: " << glGetString(GL_RENDERER) << std::endl;
  std::cout << "OpenGL language version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

  ui_nvg_init(&QUIState::ui_state);
  prev_draw_t = millis_since_boot();
}

void NvgWindow::update(const UIState &s) {
  // Connecting to visionIPC requires opengl to be current
  if (s.vipc_client->connected){
    makeCurrent();
  }
  repaint();
}

void NvgWindow::paintGL() {
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
