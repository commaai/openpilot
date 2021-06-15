#include "selfdrive/ui/qt/onroad.h"

#include <iostream>

#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/ui/paint.h"
#include "selfdrive/ui/qt/util.h"

#ifdef ENABLE_MAPS
#include "selfdrive/ui/qt/maps/map.h"
#endif

OnroadWindow::OnroadWindow(QWidget *parent) : QWidget(parent) {
  main_layout = new QStackedLayout(this);
  main_layout->setStackingMode(QStackedLayout::StackAll);

  // old UI on bottom
  nvg = new NvgWindow(this);
  QObject::connect(this, &OnroadWindow::update, nvg, &NvgWindow::update);

  QWidget * split_wrapper = new QWidget;
  split = new QHBoxLayout(split_wrapper);
  split->setContentsMargins(0, 0, 0, 0);
  split->setSpacing(0);
  split->addWidget(nvg);

  main_layout->addWidget(split_wrapper);

  alerts = new OnroadAlerts(this);
  alerts->setAttribute(Qt::WA_TransparentForMouseEvents, true);
  QObject::connect(this, &OnroadWindow::update, alerts, &OnroadAlerts::updateState);
  QObject::connect(this, &OnroadWindow::offroadTransitionSignal, alerts, &OnroadAlerts::offroadTransition);
  QObject::connect(this, &OnroadWindow::offroadTransitionSignal, this, &OnroadWindow::offroadTransition);
  main_layout->addWidget(alerts);

  // setup stacking order
  alerts->raise();

  setAttribute(Qt::WA_OpaquePaintEvent);
}


void OnroadWindow::offroadTransition(bool offroad) {
#ifdef ENABLE_MAPS
  if (!offroad) {
    QString token = QString::fromStdString(Params().get("MapboxToken"));
    if (map == nullptr && !token.isEmpty()) {
      QMapboxGLSettings settings;
      if (!Hardware::PC()) {
        settings.setCacheDatabasePath("/data/mbgl-cache.db");
      }
      settings.setCacheDatabaseMaximumSize(20 * 1024 * 1024);
      settings.setAccessToken(token.trimmed());

      MapWindow * m = new MapWindow(settings);
      QObject::connect(this, &OnroadWindow::offroadTransitionSignal, m, &MapWindow::offroadTransition);
      split->addWidget(m);

      map = m;
    }

  }
#endif
}

// ***** onroad widgets *****

OnroadAlerts::OnroadAlerts(QWidget *parent) : QWidget(parent) {
  std::tuple<AudibleAlert, QString, bool> sound_list[] = {
    {AudibleAlert::CHIME_DISENGAGE, "../assets/sounds/disengaged.wav", false},
    {AudibleAlert::CHIME_ENGAGE, "../assets/sounds/engaged.wav", false},
    {AudibleAlert::CHIME_WARNING1, "../assets/sounds/warning_1.wav", false},
    {AudibleAlert::CHIME_WARNING2, "../assets/sounds/warning_2.wav", false},
    {AudibleAlert::CHIME_WARNING2_REPEAT, "../assets/sounds/warning_2.wav", true},
    {AudibleAlert::CHIME_WARNING_REPEAT, "../assets/sounds/warning_repeat.wav", true},
    {AudibleAlert::CHIME_ERROR, "../assets/sounds/error.wav", false},
    {AudibleAlert::CHIME_PROMPT, "../assets/sounds/error.wav", false}};

  for (auto &[alert, fn, loops] : sound_list) {
    sounds[alert].first.setSource(QUrl::fromLocalFile(fn));
    sounds[alert].second = loops ? QSoundEffect::Infinite : 0;
  }
}

void OnroadAlerts::updateState(const UIState &s) {
  SubMaster &sm = *(s.sm);
  if (sm.updated("carState")) {
    // scale volume with speed
    volume = util::map_val(sm["carState"].getCarState().getVEgo(), 0.f, 20.f,
                           Hardware::MIN_VOLUME, Hardware::MAX_VOLUME);
  }
  if (sm["deviceState"].getDeviceState().getStarted()) {
    if (sm.updated("controlsState")) {
      const cereal::ControlsState::Reader &cs = sm["controlsState"].getControlsState();
      updateAlert(QString::fromStdString(cs.getAlertText1()), QString::fromStdString(cs.getAlertText2()),
                  cs.getAlertBlinkingRate(), cs.getAlertType(), cs.getAlertSize(), cs.getAlertSound());
    } else if ((sm.frame - s.scene.started_frame) > 10 * UI_FREQ) {
      // Handle controls timeout
      if (sm.rcv_frame("controlsState") < s.scene.started_frame) {
        // car is started, but controlsState hasn't been seen at all
        updateAlert("openpilot Unavailable", "Waiting for controls to start", 0,
                    "controlsWaiting", cereal::ControlsState::AlertSize::MID, AudibleAlert::NONE);
      } else if ((sm.frame - sm.rcv_frame("controlsState")) > 5 * UI_FREQ) {
        // car is started, but controls is lagging or died
        updateAlert("TAKE CONTROL IMMEDIATELY", "Controls Unresponsive", 0,
                    "controlsUnresponsive", cereal::ControlsState::AlertSize::FULL, AudibleAlert::CHIME_WARNING_REPEAT);

        // TODO: clean this up once Qt handles the border
        QUIState::ui_state.status = STATUS_ALERT;
      }
    }
  }

  // TODO: add blinking back if performant
  //float alpha = 0.375 * cos((millis_since_boot() / 1000) * 2 * M_PI * blinking_rate) + 0.625;
  bg = bg_colors[s.status];
}

void OnroadAlerts::offroadTransition(bool offroad) {
  updateAlert("", "", 0, "", cereal::ControlsState::AlertSize::NONE, AudibleAlert::NONE);
}

void OnroadAlerts::updateAlert(const QString &t1, const QString &t2, float blink_rate,
                               const std::string &type, cereal::ControlsState::AlertSize size, AudibleAlert sound) {
  if (alert_type.compare(type) == 0 && text1.compare(t1) == 0 && text2.compare(t2) == 0) {
    return;
  }

  stopSounds();
  if (sound != AudibleAlert::NONE) {
    playSound(sound);
  }

  text1 = t1;
  text2 = t2;
  alert_type = type;
  alert_size = size;
  blinking_rate = blink_rate;

  update();
}

void OnroadAlerts::playSound(AudibleAlert alert) {
  auto &[sound, loops] = sounds[alert];
  sound.setLoopCount(loops);
  sound.setVolume(volume);
  sound.play();
}

void OnroadAlerts::stopSounds() {
  for (auto &kv : sounds) {
    // Only stop repeating sounds
    auto &[sound, loops] = kv.second;
    if (sound.loopsRemaining() == QSoundEffect::Infinite) {
      sound.stop();
    }
  }
}

void OnroadAlerts::paintEvent(QPaintEvent *event) {
  if (alert_size == cereal::ControlsState::AlertSize::NONE) {
    return;
  }
  static std::map<cereal::ControlsState::AlertSize, const int> alert_sizes = {
    {cereal::ControlsState::AlertSize::SMALL, 271},
    {cereal::ControlsState::AlertSize::MID, 420},
    {cereal::ControlsState::AlertSize::FULL, height()},
  };
  int h = alert_sizes[alert_size];
  QRect r = QRect(0, height() - h, width(), h);

  QPainter p(this);

  // draw background + gradient
  p.setPen(Qt::NoPen);
  p.setCompositionMode(QPainter::CompositionMode_SourceOver);

  p.setBrush(QBrush(bg));
  p.drawRect(r);

  QLinearGradient g(0, r.y(), 0, r.bottom());
  g.setColorAt(0, QColor::fromRgbF(0, 0, 0, 0.05));
  g.setColorAt(1, QColor::fromRgbF(0, 0, 0, 0.35));

  p.setCompositionMode(QPainter::CompositionMode_DestinationOver);
  p.setBrush(QBrush(g));
  p.fillRect(r, g);
  p.setCompositionMode(QPainter::CompositionMode_SourceOver);

  // remove bottom border
  r = QRect(0, height() - h, width(), h - 30);

  // text
  const QPoint c = r.center();
  p.setPen(QColor(0xff, 0xff, 0xff));
  p.setRenderHint(QPainter::TextAntialiasing);
  if (alert_size == cereal::ControlsState::AlertSize::SMALL) {
    configFont(p, "Open Sans", 74, "SemiBold");
    p.drawText(r, Qt::AlignCenter, text1);
  } else if (alert_size == cereal::ControlsState::AlertSize::MID) {
    configFont(p, "Open Sans", 88, "Bold");
    p.drawText(QRect(0, c.y() - 125, width(), 150), Qt::AlignHCenter | Qt::AlignTop, text1);
    configFont(p, "Open Sans", 66, "Regular");
    p.drawText(QRect(0, c.y() + 21, width(), 90), Qt::AlignHCenter, text2);
  } else if (alert_size == cereal::ControlsState::AlertSize::FULL) {
    bool l = text1.length() > 15;
    configFont(p, "Open Sans", l ? 132 : 177, "Bold");
    p.drawText(QRect(0, r.y() + (l ? 240 : 270), width(), 600), Qt::AlignHCenter | Qt::TextWordWrap, text1);
    configFont(p, "Open Sans", 88, "Regular");
    p.drawText(QRect(0, r.height() - (l ? 361 : 420), width(), 300), Qt::AlignHCenter | Qt::TextWordWrap, text2);
  }
}


NvgWindow::NvgWindow(QWidget *parent) : QOpenGLWidget(parent) {
  setAttribute(Qt::WA_OpaquePaintEvent);
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
  if (s.vipc_client->connected) {
    makeCurrent();
  }
  repaint();
}

void NvgWindow::resizeGL(int w, int h) {
  ui_resize(&QUIState::ui_state, w, h);
}

void NvgWindow::paintGL() {
  ui_draw(&QUIState::ui_state, width(), height());

  double cur_draw_t = millis_since_boot();
  double dt = cur_draw_t - prev_draw_t;
  if (dt > 66) {
    // warn on sub 15fps
    LOGW("slow frame time: %.2f", dt);
  }
  prev_draw_t = cur_draw_t;
}
