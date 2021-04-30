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
  layout.setMargin(50);
  layout.setSpacing(25);

  layout.addStretch(1);

  title.setText("Be ready to take over at any time");
  title.setWordWrap(true);
  title.setStyleSheet("font-size: 60px;");
  layout.addWidget(&title, 0, Qt::AlignHCenter | Qt::AlignBottom);

  msg.setText("Always keep hands on wheel and eyes on road");
  msg.setStyleSheet("font-size: 50px;");
  layout.addWidget(&msg, 0, Qt::AlignHCenter | Qt::AlignBottom);

  setLayout(&layout);
  setStyleSheet(R"(
    color: white;
    background-color: blue;
  )");
}

void OnroadAlerts::update(const UIState &s) {
  /*
  title.setText(QString::fromStdString(s.scene.alert_text1));
  msg.setText(QString::fromStdString(s.scene.alert_text2));
  */

  auto c = bg_colors[s.status];
  const QString style = "OnroadAlerts { background-color: rgba(%1, %2, %3, %4); } * { color: white }";
  setStyleSheet(style.arg(c.r*255).arg(c.g*255).arg(c.b*255).arg(c.a*255));
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
