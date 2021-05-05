#include <iostream>

#include "selfdrive/common/timing.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/ui/qt/onroad.h"
#include "selfdrive/ui/paint.h"
#include "selfdrive/ui/qt/qt_window.h"

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

  // hack to align the onroad alerts, better way to do this?
  QVBoxLayout *alerts_container = new QVBoxLayout(this);
  alerts_container->setMargin(0);
  alerts_container->addStretch(1);
  alerts_container->addWidget(alerts, 0, Qt::AlignBottom);
  QWidget *w = new QWidget(this);
  w->setLayout(alerts_container);

  layout->addWidget(w);

  // alerts on top
  layout->setCurrentWidget(w);

  setLayout(layout);
}

// ***** onroad widgets *****

OnroadAlerts::OnroadAlerts(QWidget *parent) : QFrame(parent) {
  layout = new QVBoxLayout(this);
  layout->setSpacing(40);
  layout->setMargin(20);

  title = new QLabel();
  title->setWordWrap(true);
  title->setAlignment(Qt::AlignCenter);
  layout->addWidget(title);

  msg = new QLabel();
  msg->setWordWrap(true);
  msg->setAlignment(Qt::AlignCenter);
  layout->addWidget(msg);

  layout->addStretch(1);
  layout->insertStretch(0, 1);

  setLayout(layout);
  setStyleSheet("color: white;");
  setVisible(false);

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

  if (isVisible()) {
    auto c = bg_colors[s.status];
    float alpha = 0.375 * cos((millis_since_boot() / 1000) * 2 * M_PI * blinking_rate) + 0.625;
    bg.setRgb(c.r*255, c.g*255, c.b*255, c.a*alpha*255);
  }
}

void OnroadAlerts::offroadTransition(bool offroad) {
  stopSounds();
  setVisible(false);
  alert_type = "";
}

void OnroadAlerts::updateAlert(const QString &text1, const QString &text2, float blink_rate,
                               const std::string &type, cereal::ControlsState::AlertSize size, AudibleAlert sound) {

  if (alert_type.compare(type) == 0) {
    return;
  }

  stopSounds();
  if (sound != AudibleAlert::NONE) {
    playSound(sound);
  }

  alert_type = type;
  blinking_rate = blink_rate;
  title->setText(text1);
  msg->setText(text2);
  msg->setVisible(!msg->text().isEmpty());

  if (size == cereal::ControlsState::AlertSize::SMALL) {
    setFixedHeight(241);
    title->setStyleSheet("font-size: 70px; font-weight: 500;");
  } else if (size == cereal::ControlsState::AlertSize::MID) {
    setFixedHeight(390);
    msg->setStyleSheet("font-size: 65px; font-weight: 400;");
    title->setStyleSheet("font-size: 80px; font-weight: 500;");
  } else if (size == cereal::ControlsState::AlertSize::FULL) {
    setFixedHeight(vwp_h);
    int title_size = (title->text().size() > 15) ? 130 : 110;
    title->setStyleSheet(QString("font-size: %1px; font-weight: 500;").arg(title_size));
    msg->setStyleSheet("font-size: 90px; font-weight: 400;");
  }

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
  p.setBrush(QBrush(bg));
  p.setPen(Qt::NoPen);
  p.drawRect(rect());
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
