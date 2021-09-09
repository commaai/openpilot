#include "selfdrive/ui/qt/onroad.h"

#include <QDebug>

#include "selfdrive/common/timing.h"
#include "selfdrive/ui/paint.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/api.h"
#ifdef ENABLE_MAPS
#include "selfdrive/ui/qt/maps/map.h"
#endif

OnroadWindow::OnroadWindow(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout  = new QVBoxLayout(this);
  main_layout->setMargin(bdr_s);
  QStackedLayout *stacked_layout = new QStackedLayout;
  stacked_layout->setStackingMode(QStackedLayout::StackAll);
  main_layout->addLayout(stacked_layout);

  nvg = new NvgWindow(VISION_STREAM_RGB_BACK, this);

  QWidget * split_wrapper = new QWidget;
  split = new QHBoxLayout(split_wrapper);
  split->setContentsMargins(0, 0, 0, 0);
  split->setSpacing(0);
  split->addWidget(nvg);

  stacked_layout->addWidget(split_wrapper);

  hud = new OnroadHud(this);
  QObject::connect(this, &OnroadWindow::updateStateSignal, hud, &OnroadHud::updateState);
  stacked_layout->addWidget(hud);

  // setup stacking order
  hud->raise();

  setAttribute(Qt::WA_OpaquePaintEvent);
  QObject::connect(this, &OnroadWindow::updateStateSignal, this, &OnroadWindow::updateState);
  QObject::connect(this, &OnroadWindow::offroadTransitionSignal, this, &OnroadWindow::offroadTransition);
}

void OnroadWindow::updateState(const UIState &s) {
  SubMaster &sm = *(s.sm);
  QColor bgColor = bg_colors[s.status];
  if (sm.updated("controlsState")) {
    const cereal::ControlsState::Reader &cs = sm["controlsState"].getControlsState();
    hud->updateAlert({QString::fromStdString(cs.getAlertText1()),
                 QString::fromStdString(cs.getAlertText2()),
                 QString::fromStdString(cs.getAlertType()),
                 cs.getAlertSize(), cs.getAlertSound()}, bgColor);
  } else if ((sm.frame - s.scene.started_frame) > 5 * UI_FREQ) {
    // Handle controls timeout
    if (sm.rcv_frame("controlsState") < s.scene.started_frame) {
      // car is started, but controlsState hasn't been seen at all
      hud->updateAlert(CONTROLS_WAITING_ALERT, bgColor);
    } else if ((nanos_since_boot() - sm.rcv_time("controlsState")) / 1e9 > CONTROLS_TIMEOUT) {
      // car is started, but controls is lagging or died
      bgColor = bg_colors[STATUS_ALERT];
      hud->updateAlert(CONTROLS_UNRESPONSIVE_ALERT, bgColor);
    }
  }
  if (bg != bgColor) {
    // repaint border
    bg = bgColor;
    update();
  }
}

void OnroadWindow::mousePressEvent(QMouseEvent* e) {
  if (map != nullptr) {
    bool sidebarVisible = geometry().x() > 0;
    map->setVisible(!sidebarVisible && !map->isVisible());
  }
  // propagation event to parent(HomeWindow)
  QWidget::mousePressEvent(e);
}

void OnroadWindow::offroadTransition(bool offroad) {
#ifdef ENABLE_MAPS
  if (!offroad) {
    if (map == nullptr && (QUIState::ui_state.has_prime || !MAPBOX_TOKEN.isEmpty())) {
      QMapboxGLSettings settings;

      // Valid for 4 weeks since we can't swap tokens on the fly
      QString token = MAPBOX_TOKEN.isEmpty() ? CommaApi::create_jwt({}, 4 * 7 * 24 * 3600) : MAPBOX_TOKEN;

      if (!Hardware::PC()) {
        settings.setCacheDatabasePath("/data/mbgl-cache.db");
      }
      settings.setApiBaseUrl(MAPS_HOST);
      settings.setCacheDatabaseMaximumSize(20 * 1024 * 1024);
      settings.setAccessToken(token.trimmed());

      MapWindow * m = new MapWindow(settings);
      m->setFixedWidth(topWidget(this)->width() / 2);
      QObject::connect(this, &OnroadWindow::offroadTransitionSignal, m, &MapWindow::offroadTransition);
      split->addWidget(m, 0, Qt::AlignRight);
      map = m;
    }
  }
#endif

  hud->updateAlert({}, bg);

  // update stream type
  bool wide_cam = Hardware::TICI() && Params().getBool("EnableWideCamera");
  nvg->setStreamType(wide_cam ? VISION_STREAM_RGB_WIDE : VISION_STREAM_RGB_BACK);
}

void OnroadWindow::paintEvent(QPaintEvent *event) {
  QPainter p(this);
  p.fillRect(rect(), QColor(bg.red(), bg.green(), bg.blue(), 255));
}

// ***** onroad widgets *****

OnroadHud::OnroadHud(QWidget *parent) : QWidget(parent) {
  setAttribute(Qt::WA_TransparentForMouseEvents, true);

  engage_img = QPixmap("../assets/img_chffr_wheel.png").scaled(img_size, img_size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
  dm_img = QPixmap("../assets/img_driver_face.png").scaled(img_size, img_size, Qt::KeepAspectRatio, Qt::SmoothTransformation);

  connect(this, &OnroadHud::valueChanged, [=] { update(); });
}

void OnroadHud::updateState(const UIState &s) {
  SubMaster &sm = *(s.sm);

  auto v = sm["carState"].getCarState().getVEgo() * (metric ? 3.6 : 2.2369363);
  setProperty("speed", QString::number(int(v)));
  setProperty("speedUnit", metric ? "km/h" : "mph");

  auto cs = sm["controlsState"].getControlsState();
  const int SET_SPEED_NA = 255;
  auto vcruise = cs.getVCruise();
  if (vcruise != 0 && vcruise != SET_SPEED_NA) {
    auto max = vcruise * (metric ? 1 : 0.6225);
    setProperty("maxSpeed", QString::number((int)max));
  } else {
    setProperty("maxSpeed", "N/A");
  }

  setProperty("dmActive", sm["driverMonitoringState"].getDriverMonitoringState().getIsActiveMode());
  setProperty("hideDM", cs.getAlertSize() == cereal::ControlsState::AlertSize::NONE);
  setProperty("engageable", cs.getEngageable());
  setProperty("status", s.status);
}

void OnroadHud::updateAlert(const Alert &a, const QColor &color) {
  if (!alert.equal(a) || color != bg) {
    alert = a;
    bg = color;
    update();
  }
}

void OnroadHud::paintEvent(QPaintEvent *event) {
  QPainter p(this);

  // max speed
  QRect rc(bdr_s * 2, bdr_s * 1.5, 184, 202);
  p.setPen(QPen(QColor(0xff, 0xff, 0xff, 100), 10));
  p.setBrush(QColor(0, 0, 0, 100));
  p.drawRoundedRect(rc, 20, 20);

  bool is_cruise_set = maxSpeed_ != "N/A";
  configFont(p, "Open Sans", 48, "Regular");
  drawText(p, rc.center().x(), rc.top() + bdr_s + 10, Qt::AlignTop, "MAX", is_cruise_set ? 200 : 100);
  configFont(p, "Open Sans", 78, is_cruise_set ? "Bold" : "SemiBold");
  drawText(p, rc.center().x(), rc.bottom() - bdr_s, Qt::AlignBottom, maxSpeed_, is_cruise_set ? 255 : 100);

  // current speed
  configFont(p, "Open Sans", 180, "Bold");
  speed_ = "0";
  speedUnit_ = "mph";
  drawText(p, rect().center().x(), rc.center().y(), Qt::AlignVCenter, speed_);
  configFont(p, "Open Sans", 65, "Regular");
  drawText(p, rect().center().x(), rc.bottom() - 20, Qt::AlignTop, speedUnit_, 200);

  // engage-ability icon
  if (engageable_) {
    drawIcon(p, rect().right() - radius / 2 - bdr_s * 2, radius / 2 + int(bdr_s * 1.5), engage_img, bg_colors[status_], 1.0);
  }
  // dm icon
  if (!hideDM_) {
    drawIcon(p, radius / 2 + (bdr_s * 2), rect().bottom() - footer_h / 2, dm_img, QColor(0, 0, 0, 70), dmActive_ ? 1.0 : 0.2);
  }

  drawAlert(p);
}

void OnroadHud::drawText(QPainter &p, int x, int y, Qt::Alignment flag, const QString &text, int alpha) {
  p.setPen(QColor(0xff, 0xff, 0xff, alpha));
  QFontMetrics fm(p.font());
  QRect r = fm.tightBoundingRect(text);
  r.moveCenter({x, y});
  if (flag & Qt::AlignTop) {
    r.moveTop(r.top() + r.height() / 2);
  } else if (flag & Qt::AlignBottom) {
    r.moveTop(r.top() - r.height() / 2);
  }
  p.drawText(r.x(), r.bottom(), text);
}

void OnroadHud::drawIcon(QPainter &p, int x, int y, QPixmap &img, QBrush bg, float opacity) {
  p.setPen(Qt::NoPen);
  p.setBrush(bg);
  p.drawEllipse(x - radius / 2, y - radius / 2, radius, radius);
  p.setOpacity(opacity);
  p.drawPixmap(x - img_size / 2, y - img_size / 2, img);
}

void OnroadHud::drawAlert(QPainter &p) {
  if (alert.size == cereal::ControlsState::AlertSize::NONE) {
    return;
  }
  static std::map<cereal::ControlsState::AlertSize, const int> alert_sizes = {
    {cereal::ControlsState::AlertSize::SMALL, 271},
    {cereal::ControlsState::AlertSize::MID, 420},
    {cereal::ControlsState::AlertSize::FULL, height()},
  };
  int h = alert_sizes[alert.size];
  QRect r = QRect(0, height() - h, width(), h);

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

  // text
  const QPoint c = r.center();
  p.setPen(QColor(0xff, 0xff, 0xff));
  p.setRenderHint(QPainter::TextAntialiasing);
  if (alert.size == cereal::ControlsState::AlertSize::SMALL) {
    configFont(p, "Open Sans", 74, "SemiBold");
    p.drawText(r, Qt::AlignCenter, alert.text1);
  } else if (alert.size == cereal::ControlsState::AlertSize::MID) {
    configFont(p, "Open Sans", 88, "Bold");
    p.drawText(QRect(0, c.y() - 125, width(), 150), Qt::AlignHCenter | Qt::AlignTop, alert.text1);
    configFont(p, "Open Sans", 66, "Regular");
    p.drawText(QRect(0, c.y() + 21, width(), 90), Qt::AlignHCenter, alert.text2);
  } else if (alert.size == cereal::ControlsState::AlertSize::FULL) {
    bool l = alert.text1.length() > 15;
    configFont(p, "Open Sans", l ? 132 : 177, "Bold");
    p.drawText(QRect(0, r.y() + (l ? 240 : 270), width(), 600), Qt::AlignHCenter | Qt::TextWordWrap, alert.text1);
    configFont(p, "Open Sans", 88, "Regular");
    p.drawText(QRect(0, r.height() - (l ? 361 : 420), width(), 300), Qt::AlignHCenter | Qt::TextWordWrap, alert.text2);
  }
}

void NvgWindow::initializeGL() {
  CameraViewWidget::initializeGL();
  qInfo() << "OpenGL version:" << QString((const char*)glGetString(GL_VERSION));
  qInfo() << "OpenGL vendor:" << QString((const char*)glGetString(GL_VENDOR));
  qInfo() << "OpenGL renderer:" << QString((const char*)glGetString(GL_RENDERER));
  qInfo() << "OpenGL language version:" << QString((const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

  ui_nvg_init(&QUIState::ui_state);
  prev_draw_t = millis_since_boot();
  setBackgroundColor(bg_colors[STATUS_DISENGAGED]);
}

void NvgWindow::paintGL() {
  CameraViewWidget::paintGL();
  ui_draw(&QUIState::ui_state, width(), height());

  double cur_draw_t = millis_since_boot();
  double dt = cur_draw_t - prev_draw_t;
  if (dt > 66) {
    // warn on sub 15fps
    LOGW("slow frame time: %.2f", dt);
  }
  prev_draw_t = cur_draw_t;
}
