#include "selfdrive/ui/qt/onroad.h"

#include <cmath>

#include <QDebug>

#include "selfdrive/common/timing.h"
#include "selfdrive/ui/qt/util.h"
#ifdef ENABLE_MAPS
#include "selfdrive/ui/qt/maps/map.h"
#include "selfdrive/ui/qt/maps/map_helpers.h"
#endif

static void drawHudText(QPainter &p, int x, int y, const QString &text, int alpha = 255) {
  QFontMetrics fm(p.font());
  QRect init_rect = fm.boundingRect(text);
  QRect real_rect = fm.boundingRect(init_rect, 0, text);
  real_rect.moveCenter({x, y - real_rect.height() / 2});

  p.setPen(QColor(0xff, 0xff, 0xff, alpha));
  p.drawText(real_rect.x(), real_rect.bottom(), text);
}

OnroadWindow::OnroadWindow(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setMargin(bdr_s);
  split = new QHBoxLayout();
  split->setContentsMargins(0, 0, 0, 0);
  split->setSpacing(0);

  onroad_view = new OnroadGraphicsView(this);
  split->addWidget(onroad_view);
  main_layout->addLayout(split);

  setAttribute(Qt::WA_OpaquePaintEvent);
  QObject::connect(uiState(), &UIState::uiUpdate, this, &OnroadWindow::updateState);
  QObject::connect(uiState(), &UIState::offroadTransition, this, &OnroadWindow::offroadTransition);
}

void OnroadWindow::updateState(const UIState &s) {
  QColor bgColor = bg_colors[s.status];
  Alert alert = Alert::get(*(s.sm), s.scene.started_frame);
  if (s.sm->updated("controlsState") || !alert.equal({})) {
    if (alert.type == "controlsUnresponsive") {
      bgColor = bg_colors[STATUS_ALERT];
    } else if (alert.type == "controlsUnresponsivePermanent") {
      bgColor = bg_colors[STATUS_DISENGAGED];
    }
    onroad_view->updateAlert(alert, bgColor);
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
    if (map == nullptr && (uiState()->has_prime || !MAPBOX_TOKEN.isEmpty())) {
      MapWindow * m = new MapWindow(get_mapbox_settings());
      m->setFixedWidth(topWidget(this)->width() / 2);
      m->offroadTransition(offroad);
      QObject::connect(uiState(), &UIState::offroadTransition, m, &MapWindow::offroadTransition);
      split->addWidget(m, 0, Qt::AlignRight);
      map = m;
    }
  }
#endif
  onroad_view->updateAlert({}, bg);
}

void OnroadWindow::paintEvent(QPaintEvent *event) {
  QPainter p(this);
  p.fillRect(rect(), QColor(bg.red(), bg.green(), bg.blue(), 255));
}

// OnroadAlerts
void OnroadAlerts::update(const Alert &a, const QColor &color) {
  setVisible(a.size != cereal::ControlsState::AlertSize::NONE);
  if (!alert.equal(a) || color != bg) {
    alert = a;
    bg = color;
    if (!isVisible()) return;

    int h = 0;
    if (alert.size == cereal::ControlsState::AlertSize::SMALL) h = 271;
    else if (alert.size == cereal::ControlsState::AlertSize::MID) h = 420;
    else if (alert.size == cereal::ControlsState::AlertSize::FULL) h = scene()->sceneRect().height();

    setRect(0, 0, scene()->sceneRect().width(), h);
    setPos(0, scene()->sceneRect().bottom() - h);
    QGraphicsItem::update();
  }
}

void OnroadAlerts::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
  auto &p = *painter;
  QRect r = boundingRect().toRect();
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
    p.drawText(QRect(0, c.y() - 125, rect().width(), 150), Qt::AlignHCenter | Qt::AlignTop, alert.text1);
    configFont(p, "Open Sans", 66, "Regular");
    p.drawText(QRect(0, c.y() + 21, rect().width(), 90), Qt::AlignHCenter, alert.text2);
  } else if (alert.size == cereal::ControlsState::AlertSize::FULL) {
    bool l = alert.text1.length() > 15;
    configFont(p, "Open Sans", l ? 132 : 177, "Bold");
    p.drawText(QRect(0, r.y() + (l ? 240 : 270), rect().width(), 600), Qt::AlignHCenter | Qt::TextWordWrap, alert.text1);
    configFont(p, "Open Sans", 88, "Regular");
    p.drawText(QRect(0, r.height() - (l ? 361 : 420), rect().width(), 300), Qt::AlignHCenter | Qt::TextWordWrap, alert.text2);
  }
}

// OnroadHud
OnroadHud::OnroadHud(QObject *parent) : QGraphicsScene(parent) {
  QLinearGradient bg(0, header_h - (header_h / 2.5), 0, header_h);
  bg.setColorAt(0, QColor::fromRgbF(0, 0, 0, 0.45));
  bg.setColorAt(1, QColor::fromRgbF(0, 0, 0, 0));
  header = addRect({}, Qt::NoPen, bg);

  addItem(max_speed = new MaxSpeedItem);
  addItem(current_speed = new CurrentSpeedItem);
  addItem(wheel = new IconItem("../assets/img_chffr_wheel.png"));
  addItem(dm = new IconItem("../assets/img_driver_face.png"));
  addItem(alerts = new OnroadAlerts);

  for (auto item : items()) {
    item->setFlag(QGraphicsItem::ItemIgnoresTransformations);
    item->setCacheMode(QGraphicsItem::DeviceCoordinateCache);
  }
}

void OnroadHud::setGeometry(const QRectF &rect) {
  setSceneRect(rect);

  max_speed->setPos(bdr_s * 2, bdr_s * 1.5);
  current_speed->setPos((rect.width() / 2 - current_speed->boundingRect().width() / 2), rect.top());
  wheel->setPos(rect.right() - wheel->boundingRect().width() - bdr_s * 2.0, bdr_s * 1.5);
  dm->setPos(bdr_s * 2, rect.bottom() - 200);
  alerts->setPos(0, rect.bottom() - alerts->rect().height());
  alerts->setRect(0, 0, rect.width(), alerts->rect().height());
  header->setRect(0, 0, rect.width(), header_h);
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

  max_speed->update(cruise_set, maxspeed_str);
  current_speed->update(QString::number(std::nearbyint(cur_speed)), s.scene.is_metric ? "km/h" : "mph");
  dm->setVisible(cs.getAlertSize() == cereal::ControlsState::AlertSize::NONE);

  // update engageability and DM icons at 2Hz
  if (sm.frame % (UI_FREQ / 2) == 0) {
    wheel->setVisible(cs.getEngageable() || cs.getEnabled());
    wheel->update(bg_colors[s.status], 1.0);
    const bool dm_active = sm["driverMonitoringState"].getDriverMonitoringState().getIsActiveMode();
    dm->update(QColor(0, 0, 0, 70), dm_active ? 1.0 : 0.2);
  }
}

void MaxSpeedItem::update(bool cruise_set, const QString &speed) {
  if (cruise_set != is_cruise_set || speed != maxSpeed) {
    is_cruise_set = cruise_set;
    maxSpeed = speed;
    QGraphicsItem::update();
  }
}

void MaxSpeedItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
  auto &p = *painter;
  const QRect rc = boundingRect().toRect();
  p.setRenderHint(QPainter::Antialiasing);
  p.setPen(QPen(QColor(0xff, 0xff, 0xff, 100), 10));
  p.setBrush(QColor(0, 0, 0, 100));
  p.drawRoundedRect(rc, 20, 20);
  p.setPen(Qt::NoPen);

  configFont(p, "Open Sans", 48, "Regular");
  drawHudText(p, rc.center().x(), 118 - bdr_s * 1.5, "MAX", is_cruise_set ? 200 : 100);
  if (is_cruise_set) {
    configFont(p, "Open Sans", 88, is_cruise_set ? "Bold" : "SemiBold");
    drawHudText(p, rc.center().x(), 212 - bdr_s * 1.5, maxSpeed, 255);
  } else {
    configFont(p, "Open Sans", 80, "SemiBold");
    drawHudText(p, rc.center().x(), 212 - bdr_s * 1.5, maxSpeed, 100);
  }
}

void CurrentSpeedItem::update(const QString &s, const QString &unit) {
  if (s != speed || unit != speedUnit) {
    speed = s;
    speedUnit = unit;
    QGraphicsItem::update();
  }
}

void CurrentSpeedItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
  auto &p = *painter;
  QRect rc = boundingRect().toRect();
  p.setRenderHint(QPainter::Antialiasing);
  configFont(p, "Open Sans", 176, "Bold");
  drawHudText(p, rc.center().x(), 210, speed);
  configFont(p, "Open Sans", 66, "Regular");
  drawHudText(p, rc.center().x(), 290, speedUnit, 200);
}

void IconItem::update(const QColor color, float alpha) {
  if (bg != color || alpha != alpha) {
    bg = color;
    opacity = alpha;
    QGraphicsItem::update();
  }
}

void IconItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
  painter->setRenderHint(QPainter::Antialiasing);
  painter->setPen(Qt::NoPen);
  painter->setBrush(bg);
  painter->drawEllipse(0, 0, radius, radius);
  painter->setOpacity(opacity);
  painter->drawPixmap((radius - img_size) / 2, (radius - img_size) / 2, pixmap);
};

// OnroadGraphicsView
OnroadGraphicsView::OnroadGraphicsView(QWidget *parent) : QGraphicsView(parent) {
  setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  setFrameStyle(0);

  hud = new OnroadHud(this);
  setScene(hud);

  camera_view = new CameraViewWidget("camerad", VISION_STREAM_RGB_BACK, true, nullptr);
  connect(camera_view, &CameraViewWidget::vipcThreadFrameReceived, [=]() { viewport()->update(); });
  connect(camera_view, &CameraViewWidget::frameMatrixChanged, [=](const mat3 &matrix, float y_offset, float zoom) {
    // Apply transformation such that video pixel coordinates match video
    // 1) Put (0, 0) in the middle of the video
    // 2) Apply same scaling as video
    // 3) Put (0, 0) in top left corner of video
    UIState *s = uiState();
    s->fb_w = rect().width();
    s->fb_h = rect().height();
    s->car_space_transform.reset();
    s->car_space_transform.translate(s->fb_w / 2, s->fb_h / 2 + y_offset)
        .scale(zoom, zoom)
        .translate(-matrix.v[2], -matrix.v[5]);
  });

  setViewport(camera_view);
  setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
  
  QObject::connect(uiState(), &UIState::uiUpdate, hud, &OnroadHud::updateState);
  QObject::connect(uiState(), &UIState::offroadTransition, this, &OnroadGraphicsView::offroadTransition);
}

void OnroadGraphicsView::drawLaneLines(QPainter &painter, const UIScene &scene) {
  if (!scene.end_to_end) {
    // lanelines
    for (int i = 0; i < std::size(scene.lane_line_vertices); ++i) {
      painter.setBrush(QColor::fromRgbF(1.0, 1.0, 1.0, scene.lane_line_probs[i]));
      painter.drawPolygon(scene.lane_line_vertices[i].v, scene.lane_line_vertices[i].cnt);
    }
    // road edges
    for (int i = 0; i < std::size(scene.road_edge_vertices); ++i) {
      painter.setBrush(QColor::fromRgbF(1.0, 0, 0, std::clamp<float>(1.0 - scene.road_edge_stds[i], 0.0, 1.0)));
      painter.drawPolygon(scene.road_edge_vertices[i].v, scene.road_edge_vertices[i].cnt);
    }
  }
  // paint path
  QLinearGradient bg(0, rect().height(), 0, rect().height() / 4);
  bg.setColorAt(0, scene.end_to_end ? redColor() : QColor(255, 255, 255));
  bg.setColorAt(1, scene.end_to_end ? redColor(0) : QColor(255, 255, 255, 0));
  painter.setBrush(bg);
  painter.drawPolygon(scene.track_vertices.v, scene.track_vertices.cnt);
}

void OnroadGraphicsView::drawLead(QPainter &painter, const cereal::ModelDataV2::LeadDataV3::Reader &lead_data, const QPointF &vd) {
  const float speedBuff = 10.;
  const float leadBuff = 40.;
  const float d_rel = lead_data.getX()[0];
  const float v_rel = lead_data.getV()[0];

  float fillAlpha = 0;
  if (d_rel < leadBuff) {
    fillAlpha = 255 * (1.0 - (d_rel / leadBuff));
    if (v_rel < 0) {
      fillAlpha += 255 * (-1 * (v_rel / speedBuff));
    }
    fillAlpha = (int)(fmin(fillAlpha, 255));
  }

  float sz = std::clamp((25 * 30) / (d_rel / 3 + 30), 15.0f, 30.0f) * 2.35;
  float x = std::clamp((float)vd.x(), 0.f, (float)rect().width() - sz / 2);
  float y = std::fmin(rect().height() - sz * .6, (float)vd.y());

  float g_xo = sz / 5;
  float g_yo = sz / 10;

  QPointF glow[] = {{x + (sz * 1.35) + g_xo, y + sz + g_yo}, {x, y - g_xo}, {x - (sz * 1.35) - g_xo, y + sz + g_yo}};
  painter.setBrush(QColor(218, 202, 37, 255));
  painter.drawPolygon(glow, std::size(glow));

  // chevron
  QPointF chevron[] = {{x + (sz * 1.25), y + sz}, {x, y}, {x - (sz * 1.25), y + sz}};
  painter.setBrush(redColor(fillAlpha));
  painter.drawPolygon(chevron, std::size(chevron));
}

void OnroadGraphicsView::offroadTransition(bool offroad) {
  // update stream type
  bool wide_cam = Hardware::TICI() && Params().getBool("EnableWideCamera");
  camera_view->setStreamType(wide_cam ? VISION_STREAM_RGB_WIDE : VISION_STREAM_RGB_BACK);
}

void OnroadGraphicsView::resizeEvent(QResizeEvent *event) {
  QRect rc(QRect(QPoint(0, 0), event->size()));
  hud->setGeometry(rc);
  camera_view->updateFrameMat(event->size().width(), event->size().height());
  QGraphicsView::resizeEvent(event);
}

void OnroadGraphicsView::drawBackground(QPainter *painter, const QRectF &rect) {
  camera_view->doPaint();

  UIState *s = uiState();
  if (s->worldObjectsVisible()) {
    painter->setPen(Qt::NoPen);
    drawLaneLines(*painter, s->scene);

    if (s->scene.longitudinal_control) {
      auto leads = (*s->sm)["modelV2"].getModelV2().getLeadsV3();
      if (leads[0].getProb() > .5) {
        drawLead(*painter, leads[0], s->scene.lead_vertices[0]);
      }
      if (leads[1].getProb() > .5 && (std::abs(leads[1].getX()[0] - leads[0].getX()[0]) > 3.0)) {
        drawLead(*painter, leads[1], s->scene.lead_vertices[1]);
      }
    }
  }
}
