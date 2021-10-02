#include "selfdrive/ui/qt/onroad.h"

#include <QDebug>

#include "selfdrive/common/timing.h"
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
    hud->setMapWidth(map->isVisible() ? map->width() : 0);
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

  hud->setMetric(Params().getBool("IsMetric"));
}

void OnroadWindow::paintEvent(QPaintEvent *event) {
  QPainter p(this);
  p.fillRect(rect(), QColor(bg.red(), bg.green(), bg.blue(), 255));
}

// ***** onroad widgets *****

OnroadHud::OnroadHud(QWidget *parent) : QWidget(parent) {
  setAttribute(Qt::WA_TransparentForMouseEvents, true);
  connect(this, &OnroadHud::valueChanged, [=] { update(); });

  engage_img = QPixmap("../assets/img_chffr_wheel.png").scaled(img_size, img_size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
  dm_img = QPixmap("../assets/img_driver_face.png").scaled(img_size, img_size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
}

void OnroadHud::updateState(const UIState &s) {
  auto getMaxSpeed = [=](float speed) {
    const int SET_SPEED_NA = 255;
    bool setted = speed > 0 && (int)speed != SET_SPEED_NA;
    return setted ? QString::number((int)(speed * (is_metric ? 1 : 0.6225))) : "N/A";
  };

  SubMaster &sm = *(s.sm);
  auto cs = sm["controlsState"].getControlsState();
  int speed = (int)(sm["carState"].getCarState().getVEgo() * (is_metric ? 3.6 : 2.2369363));
  setProperty("speed", QString::number(speed));
  setProperty("maxSpeed", getMaxSpeed(cs.getVCruise()));
  setProperty("speedUnit", is_metric ? "km/h" : "mph");
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
  QRect hud_rect = rect();
  hud_rect.setRight(hud_rect.right() - map_width);

  // max speed
  QRect rc(bdr_s * 2, bdr_s * 1.5, 184, 202);
  p.setPen(QPen(QColor(0xff, 0xff, 0xff, 100), 10));
  p.setBrush(QColor(0, 0, 0, 100));
  p.drawRoundedRect(rc, 20, 20);

  bool is_cruise_set = maxSpeed_ != "N/A";
  configFont(p, "Open Sans", 48, "Regular");
  drawText(p, rc.center().x(), rc.top() + bdr_s + 10, Qt::AlignTop, "MAX", is_cruise_set ? 200 : 100);
  configFont(p, "Open Sans", 88, is_cruise_set ? "Bold" : "SemiBold");
  drawText(p, rc.center().x(), rc.bottom() - bdr_s, Qt::AlignBottom, maxSpeed_, is_cruise_set ? 255 : 100);
  // current speed
  configFont(p, "Open Sans", 180, "Bold");
  drawText(p, hud_rect.center().x(), rc.center().y(), Qt::AlignVCenter, speed_);
  configFont(p, "Open Sans", 65, "Regular");
  drawText(p, hud_rect.center().x(), rc.bottom() - 20, Qt::AlignTop, speedUnit_, 200);
  // engage-ability icon
  if (engageable_) {
    drawIcon(p, hud_rect.right() - radius / 2 - bdr_s * 2, radius / 2 + int(bdr_s * 1.5), engage_img, bg_colors[status_], 1.0);
  }
  // dm icon
  if (!hideDM_) {
    drawIcon(p, radius / 2 + (bdr_s * 2), hud_rect.bottom() - footer_h / 2, dm_img, QColor(0, 0, 0, 70), dmActive_ ? 1.0 : 0.2);
  }
  // draw alert
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

  prev_draw_t = millis_since_boot();
  setBackgroundColor(bg_colors[STATUS_DISENGAGED]);
}

void NvgWindow::addLanLines(QPainterPath &path, const line_vertices_data &vd) {
  if (vd.cnt == 0) return;

  QPainterPath line_path;
  const vertex_data *v = &vd.v[0];
  line_path.moveTo(v[0].x, v[0].y);
  for (int i = 1; i < vd.cnt; i++) {
    line_path.lineTo(v[i].x, v[i].y);
  }
  line_path.setFillRule(Qt::WindingFill);
  path.addPath(line_path);
}

void NvgWindow::drawVisionLaneLines(QPainter &painter, UIState *s) {
  const UIScene &scene = s->scene;
  QPainterPath path;
  if (!scene.end_to_end) {
    // lanelines
    for (int i = 0; i < std::size(scene.lane_line_vertices); i++) {
      addLanLines(path, scene.lane_line_vertices[i]);
    }
    // road edges
    for (int i = 0; i < std::size(scene.road_edge_vertices); i++) {
      addLanLines(path, scene.road_edge_vertices[i]);
    }
  }
  // paint path
  addLanLines(path, scene.track_vertices);

  painter.setPen(Qt::NoPen);
  QLinearGradient bg(0, height(), 0, height() / 4);
  bg.setColorAt(0, Qt::white);
  bg.setColorAt(1, QColor::fromRgb(255, 255, 255, 0));
  // painter.setBrush(bg);
  painter.fillPath(path, bg);
}

void NvgWindow::drawLead(QPainter &painter, UIState *s, const cereal::ModelDataV2::LeadDataV3::Reader &lead_data, const vertex_data &vd) {
  // Draw lead car indicator
  auto [x, y] = vd;

  float fillAlpha = 0;
  float speedBuff = 10.;
  float leadBuff = 40.;
  float d_rel = lead_data.getX()[0];
  float v_rel = lead_data.getV()[0];
  if (d_rel < leadBuff) {
    fillAlpha = 255*(1.0-(d_rel/leadBuff));
    if (v_rel < 0) {
      fillAlpha += 255*(-1*(v_rel/speedBuff));
    }
    fillAlpha = (int)(fmin(fillAlpha, 255));
  }

  float sz = std::clamp((25 * 30) / (d_rel / 3 + 30), 15.0f, 30.0f) * 2.35;
  x = std::clamp(x, 0.f, s->fb_w - sz / 2);
  y = std::fmin(s->fb_h - sz * .6, y);

  float g_xo = sz/5;
  float g_yo = sz/10;
  QPainterPath path1;
  path1.moveTo(x+(sz*1.35)+g_xo, y+sz+g_yo);
  path1.lineTo(x, y-g_xo);
  path1.lineTo(x-(sz*1.35)-g_xo, y+sz+g_yo);
  painter.fillPath(path1, QColor(218, 202, 37, 255));

  // chevron
  QPainterPath path2;
  path2.moveTo(x+(sz*1.25), y+sz);
  path2.lineTo(x, y);
  path2.lineTo(x-(sz*1.25), y+sz);
  painter.fillPath(path2, QColor(201, 34, 49, fillAlpha));
}

void NvgWindow::resizeGL(int w, int h) {
  CameraViewWidget::resizeGL(w, h);
  UIState *s = &QUIState::ui_state;
  s->fb_w = w;
  s->fb_h = h;
  auto intrinsic_matrix = s->wide_camera ? ecam_intrinsic_matrix : fcam_intrinsic_matrix;
  float zoom = ZOOM / intrinsic_matrix.v[0];
  if (s->wide_camera) {
    zoom *= 0.5;
  }

  // Apply transformation such that video pixel coordinates match video
  // 1) Put (0, 0) in the middle of the video
  // 2) Apply same scaling as video
  // 3) Put (0, 0) in top left corner of video
  s->car_space_transform.reset();
  s->car_space_transform.translate(w / 2, h / 2 + y_offset)
      .scale(zoom, zoom)
      .translate(-intrinsic_matrix.v[2], -intrinsic_matrix.v[5]);
}

void NvgWindow::paintGL() {
  CameraViewWidget::paintGL();
  UIState *s = &QUIState::ui_state;
  
  double t1 = millis_since_boot();
  QPainter painter(this);
 
  drawVisionLaneLines(painter, s);
  if (s->scene.longitudinal_control) {
    auto lead_one = (*s->sm)["modelV2"].getModelV2().getLeadsV3()[0];
    auto lead_two = (*s->sm)["modelV2"].getModelV2().getLeadsV3()[1];
    if (lead_one.getProb() > .5) {
      drawLead(painter, s, lead_one, s->scene.lead_vertices[0]);
    }
    if (lead_two.getProb() > .5 && (std::abs(lead_one.getX()[0] - lead_two.getX()[0]) > 3.0)) {
      drawLead(painter, s, lead_two, s->scene.lead_vertices[1]);
    }
  }
  printf("qt paint %f\n", millis_since_boot() - t1);

  double cur_draw_t = millis_since_boot();
  double dt = cur_draw_t - prev_draw_t;
  if (dt > 66) {
    // warn on sub 15fps
    LOGW("slow frame time: %.2f", dt);
  }
  prev_draw_t = cur_draw_t;
}
