#include <iostream>

#include "selfdrive/common/timing.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/ui/qt/onroad.h"
#include "selfdrive/ui/paint.h"
#include "selfdrive/ui/qt/util.h"

OnroadWindow::OnroadWindow(QWidget *parent) : QWidget(parent) {
  layout = new QStackedLayout();
  layout->setStackingMode(QStackedLayout::StackAll);

  // old UI on bottom
  nvg = new NvgWindow(this);
  layout->addWidget(nvg);
  QObject::connect(this, &OnroadWindow::update, nvg, &NvgWindow::update);

  alerts = new OnroadAlerts(this);
  QObject::connect(this, &OnroadWindow::update, alerts, &OnroadAlerts::update);
  QObject::connect(this, &OnroadWindow::offroadTransition, alerts, &OnroadAlerts::offroadTransition);
  layout->addWidget(alerts);

  setLayout(layout);
}

// ***** onroad widgets *****

OnroadAlerts::OnroadAlerts(QWidget *parent) : QOpenGLWidget(parent) {
  setAttribute(Qt::WA_AlwaysStackOnTop);
  setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

  QSurfaceFormat format = QSurfaceFormat::defaultFormat();
  format.setSamples(4);
  setFormat(format);

  // setup sounds
  for (auto &kv : sound_map) {
    auto path = QUrl::fromLocalFile(kv.second.first);
    sounds[kv.first].setSource(path);
  }
}

void OnroadAlerts::update(const UIState &s) {
  SubMaster &sm = *(s.sm);
  if (sm.updated("carState")) {
    // scale volume with speed
    volume = util::map_val(sm["carState"].getCarState().getVEgo(), 0.f, 20.f,
                           Hardware::MIN_VOLUME, Hardware::MAX_VOLUME);
  }
  if (sm.updated("controlsState")) {
    const cereal::ControlsState::Reader &cs = sm["controlsState"].getControlsState();
    updateAlert(QString::fromStdString(cs.getAlertText1()), QString::fromStdString(cs.getAlertText2()),
                cs.getAlertBlinkingRate(), cs.getAlertType(), cs.getAlertSize(), cs.getAlertSound());
  } else {
    // Handle controls timeout
    if (s.scene.deviceState.getStarted() && (sm.frame - s.scene.started_frame) > 10 * UI_FREQ) {
      const uint64_t cs_frame = sm.rcv_frame("controlsState");
      if (cs_frame < s.scene.started_frame) {
        // car is started, but controlsState hasn't been seen at all
        updateAlert("openpilot Unavailable", "Waiting for controls to start", 0,
                    "controlsWaiting", cereal::ControlsState::AlertSize::MID, AudibleAlert::NONE);
      } else if ((sm.frame - cs_frame) > 5 * UI_FREQ) {
        // car is started, but controls is lagging or died
        updateAlert("TAKE CONTROL IMMEDIATELY", "Controls Unresponsive", 0,
                    "controlsUnresponsive", cereal::ControlsState::AlertSize::FULL, AudibleAlert::CHIME_WARNING_REPEAT);

        // TODO: clean this up once Qt handles the border
        QUIState::ui_state.status = STATUS_ALERT;
      }
    }
  }

  auto c = bg_colors[s.status];
  float alpha = 0.375 * cos((millis_since_boot() / 1000) * 2 * M_PI * blinking_rate) + 0.625;
  bg.setRgbF(c.r, c.g, c.b, c.a*alpha);
}

void OnroadAlerts::offroadTransition(bool offroad) {
  stopSounds();
  setVisible(false);
  alert_type = "";
}

void OnroadAlerts::updateAlert(const QString &t1, const QString &t2, float blink_rate,
                               const std::string &type, cereal::ControlsState::AlertSize size, AudibleAlert sound) {

  if (alert_type.compare(type) != 0) {
    stopSounds();
    if (sound != AudibleAlert::NONE) {
      playSound(sound);
    }
  }

  text1 = t1;
  text2 = t2;
  alert_size = size;
  alert_type = type;
  blinking_rate = blink_rate;

  setVisible(size != cereal::ControlsState::AlertSize::NONE);
  repaint();
}

void OnroadAlerts::playSound(AudibleAlert alert) {
  int loops = sound_map[alert].second ? QSoundEffect::Infinite : 0;
  sounds[alert].setLoopCount(loops);
  sounds[alert].setVolume(volume);
  sounds[alert].play();
}

void OnroadAlerts::stopSounds() {
  for (auto &kv : sounds) {
    // Only stop repeating sounds
    if (kv.second.loopsRemaining() == QSoundEffect::Infinite) {
      kv.second.stop();
    }
  }
}

void OnroadAlerts::paintEvent(QPaintEvent *event) {
  QPainter p(this);

  // setup transparent widget
  p.beginNativePainting();
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  p.endNativePainting();

  int h = height();
  if (alert_size != cereal::ControlsState::AlertSize::FULL) {
    h = alert_size == cereal::ControlsState::AlertSize::SMALL ? 271 : 420;
  }
  QRect r = QRect(0, height() - h, width(), h);

  // draw background + gradient
  p.setPen(Qt::NoPen);
  p.setBrush(QBrush(bg));
  p.drawRect(r);

  QLinearGradient g(0, 0, 0, height());
  g.setColorAt(0, QColor::fromRgbF(0, 0, 0, 0.05));
  g.setColorAt(1, QColor::fromRgbF(0, 0, 0, 0.35));
  p.setBrush(QBrush(g));
  p.fillRect(r, g);

  // remove sidebar
  r = QRect(0, height() - h, width(), h - 30);

  // text
  const QPoint c = r.center();
  p.setPen(QColor(0xff, 0xff, 0xff));
  p.setRenderHint(QPainter::Antialiasing);
  if (alert_size == cereal::ControlsState::AlertSize::SMALL) {
    configFont(p, "Open Sans SemiBold", 74, 400);
    p.drawText(r, Qt::AlignCenter, text1);
  } else if (alert_size == cereal::ControlsState::AlertSize::MID) {
    configFont(p, "Open Sans", 88, 600); // TODO: bold
    p.drawText(QRect(0, c.y() - 125, width(), 110), Qt::AlignHCenter | Qt::AlignTop, text1);
    configFont(p, "Open Sans", 64, 400);
    p.drawText(QRect(0, c.y() + 21, width(), 90), Qt::AlignHCenter, text2);
  } else if (alert_size == cereal::ControlsState::AlertSize::FULL) {
    bool l = text1.length() > 15;
    configFont(p, "Open Sans", l ? 72 : 96, 700);
    p.drawText(QRect(0, c.y() + (l ? 360 : 420), width(), 90), Qt::AlignHCenter, text1);
    configFont(p, "Open Sans", 60, 400);
    p.drawText(QRect(0, r.height() - (l ? 300 : 360), width(), 90), Qt::AlignHCenter, text2);
  }
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
  if (dt > 66 && !QUIState::ui_state.scene.driver_view) {
    // warn on sub 15fps
    LOGW("slow frame time: %.2f", dt);
  }
  prev_draw_t = cur_draw_t;
}
