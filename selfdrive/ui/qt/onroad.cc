#include "selfdrive/ui/qt/onroad.h"

#include <QDebug>

#include "selfdrive/common/timing.h"
#include "selfdrive/ui/paint.h"
#include "selfdrive/ui/qt/util.h"
#ifdef ENABLE_MAPS
#include "selfdrive/ui/qt/maps/map.h"
#include "selfdrive/ui/qt/maps/map_helpers.h"
#endif

OnroadWindow::OnroadWindow(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout  = new QVBoxLayout(this);
  main_layout->setMargin(bdr_s);
  QStackedLayout *stacked_layout = new QStackedLayout;
  stacked_layout->setStackingMode(QStackedLayout::StackAll);
  main_layout->addLayout(stacked_layout);

  QStackedLayout *road_view_layout = new QStackedLayout;
  road_view_layout->setStackingMode(QStackedLayout::StackAll);
  nvg = new NvgWindow(VISION_STREAM_RGB_BACK, this);
  road_view_layout->addWidget(nvg);
  hud = new OnroadHud(this);
  road_view_layout->addWidget(hud);

  QWidget * split_wrapper = new QWidget;
  split = new QHBoxLayout(split_wrapper);
  split->setContentsMargins(0, 0, 0, 0);
  split->setSpacing(0);
  split->addLayout(road_view_layout);

  stacked_layout->addWidget(split_wrapper);

  alerts = new OnroadAlerts(this);
  alerts->setAttribute(Qt::WA_TransparentForMouseEvents, true);
  stacked_layout->addWidget(alerts);

  // setup stacking order
  alerts->raise();

  setAttribute(Qt::WA_OpaquePaintEvent);
  QObject::connect(this, &OnroadWindow::updateStateSignal, this, &OnroadWindow::updateState);
  QObject::connect(this, &OnroadWindow::offroadTransitionSignal, this, &OnroadWindow::offroadTransition);
}

void OnroadWindow::updateState(const UIState &s) {
  QColor bgColor = bg_colors[s.status];
  Alert alert = Alert::get(*(s.sm), s.scene.started_frame);
  if (s.sm->updated("controlsState") || !alert.equal({})) {
    if (alert.type == "controlsUnresponsive") {
      bgColor = bg_colors[STATUS_ALERT];
    }
    alerts->updateAlert(alert, bgColor);
  }

  hud->updateState(s);

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
      MapWindow * m = new MapWindow(get_mapbox_settings());
      m->setFixedWidth(topWidget(this)->width() / 2);
      QObject::connect(this, &OnroadWindow::offroadTransitionSignal, m, &MapWindow::offroadTransition);
      split->addWidget(m, 0, Qt::AlignRight);
      map = m;
    }
  }
#endif

  alerts->updateAlert({}, bg);

  // update stream type
  bool wide_cam = Hardware::TICI() && Params().getBool("EnableWideCamera");
  nvg->setStreamType(wide_cam ? VISION_STREAM_RGB_WIDE : VISION_STREAM_RGB_BACK);
}

void OnroadWindow::paintEvent(QPaintEvent *event) {
  QPainter p(this);
  p.fillRect(rect(), QColor(bg.red(), bg.green(), bg.blue(), 255));
}

// ***** onroad widgets *****

// OnroadAlerts
void OnroadAlerts::updateAlert(const Alert &a, const QColor &color) {
  if (!alert.equal(a) || color != bg) {
    alert = a;
    bg = color;
    update();
  }
}

void OnroadAlerts::paintEvent(QPaintEvent *event) {
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

// OnroadHud
OnroadHud::OnroadHud(QWidget *parent) : QWidget(parent) {
  engage_img = QPixmap("../assets/img_chffr_wheel.png").scaled(img_size, img_size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
  dm_img = QPixmap("../assets/img_driver_face.png").scaled(img_size, img_size, Qt::KeepAspectRatio, Qt::SmoothTransformation);

  connect(this, &OnroadHud::valueChanged, [=] { update(); });
}

void OnroadHud::updateState(const UIState &s) {
  const int SET_SPEED_NA = 255;
  const SubMaster &sm = *(s.sm);
  const auto cs = sm["controlsState"].getControlsState();

  float maxspeed = cs.getVCruise();
  bool cruise_set = maxspeed > 0 && (int)maxspeed != SET_SPEED_NA;
  if (cruise_set && !s.scene.is_metric) {
    maxspeed *= KM_TO_MILE;
  }
  QString maxspeed_str = cruise_set ? QString::number(std::nearbyint(maxspeed)) : "N/A";
  float cur_speed = std::max(0.0, sm["carState"].getCarState().getVEgo() * (s.scene.is_metric ? MS_TO_KPH : MS_TO_MPH));

  setProperty("is_cruise_set", cruise_set);
  setProperty("speed", QString::number(std::nearbyint(cur_speed)));
  setProperty("maxSpeed", maxspeed_str);
  setProperty("speedUnit", s.scene.is_metric ? "km/h" : "mph");
  setProperty("hideDM", cs.getAlertSize() != cereal::ControlsState::AlertSize::NONE);
  setProperty("status", s.status);

  // update engageability and DM icons at 2Hz
  if (sm.frame % (UI_FREQ / 2) == 0) {
    setProperty("engageable", cs.getEngageable() || cs.getEnabled());
    setProperty("dmActive", sm["driverMonitoringState"].getDriverMonitoringState().getIsActiveMode());
  }
}

void OnroadHud::paintEvent(QPaintEvent *event) {
  QPainter p(this);
  p.setRenderHint(QPainter::Antialiasing);

  // max speed
  QRect rc(bdr_s * 2, bdr_s * 1.5, 184, 202);
  p.setPen(QPen(QColor(0xff, 0xff, 0xff, 100), 10));
  p.setBrush(QColor(0, 0, 0, 100));
  p.drawRoundedRect(rc, 20, 20);
  p.setPen(Qt::NoPen);

  configFont(p, "Open Sans", 48, "Regular");
  drawText(p, rc.center().x(), 118, "MAX", is_cruise_set ? 200 : 100);
  if (is_cruise_set) {
    configFont(p, "Open Sans", 88, is_cruise_set ? "Bold" : "SemiBold");
    drawText(p, rc.center().x(), 212, maxSpeed, 255);
  } else {
    configFont(p, "Open Sans", 80, "SemiBold");
    drawText(p, rc.center().x(), 212, maxSpeed, 100);
  }

  // current speed
  configFont(p, "Open Sans", 176, "Bold");
  drawText(p, rect().center().x(), 210, speed);
  configFont(p, "Open Sans", 66, "Regular");
  drawText(p, rect().center().x(), 290, speedUnit, 200);

  // engage-ability icon
  if (engageable) {
    drawIcon(p, rect().right() - radius / 2 - bdr_s * 2, radius / 2 + int(bdr_s * 1.5),
             engage_img, bg_colors[status], 1.0);
  }

  // dm icon
  if (!hideDM) {
    drawIcon(p, radius / 2 + (bdr_s * 2), rect().bottom() - footer_h / 2,
             dm_img, QColor(0, 0, 0, 70), dmActive ? 1.0 : 0.2);
  }
}

void OnroadHud::drawText(QPainter &p, int x, int y, const QString &text, int alpha) {
  QFontMetrics fm(p.font());
  QRect init_rect = fm.boundingRect(text);
  QRect real_rect = fm.boundingRect(init_rect, 0, text);
  real_rect.moveCenter({x, y - real_rect.height() / 2});

  p.setPen(QColor(0xff, 0xff, 0xff, alpha));
  p.drawText(real_rect.x(), real_rect.bottom(), text);
}

void OnroadHud::drawIcon(QPainter &p, int x, int y, QPixmap &img, QBrush bg, float opacity) {
  p.setPen(Qt::NoPen);
  p.setBrush(bg);
  p.drawEllipse(x - radius / 2, y - radius / 2, radius, radius);
  p.setOpacity(opacity);
  p.drawPixmap(x - img_size / 2, y - img_size / 2, img);
}

// NvgWindow
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

void NvgWindow::showEvent(QShowEvent *event) {
  CameraViewWidget::showEvent(event);
  ui_update_params(&QUIState::ui_state);
  prev_draw_t = millis_since_boot();
}
