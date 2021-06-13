#include "selfdrive/ui/qt/maps/map.h"

#include <cmath>

#include <QDebug>

#include "selfdrive/common/util.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/ui/ui.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/maps/map_helpers.h"


const int PAN_TIMEOUT = 100;
const bool DRAW_MODEL_PATH = false;
const qreal REROUTE_DISTANCE = 25;
const float MAX_ZOOM = 17;
const float MIN_ZOOM = 14;
const float MAX_PITCH = 50;
const float MIN_PITCH = 0;
const float MAP_SCALE = 2;


MapWindow::MapWindow(const QMapboxGLSettings &settings) : m_settings(settings) {
  if (DRAW_MODEL_PATH) {
    sm = new SubMaster({"liveLocationKalman", "modelV2"});
  } else {
    sm = new SubMaster({"liveLocationKalman"});
  }

  timer = new QTimer(this);
  QObject::connect(timer, SIGNAL(timeout()), this, SLOT(timerUpdate()));

  recompute_timer = new QTimer(this);
  recompute_timer->start(1000);
  QObject::connect(recompute_timer, SIGNAL(timeout()), this, SLOT(recomputeRoute()));

  // Instructions
  map_instructions = new MapInstructions(this);
  connect(this, &MapWindow::instructionsChanged, map_instructions, &MapInstructions::updateInstructions);
  connect(this, &MapWindow::distanceChanged, map_instructions, &MapInstructions::updateDistance);
  map_instructions->setFixedWidth(width());

  map_eta = new MapETA(this);
  connect(this, &MapWindow::ETAChanged, map_eta, &MapETA::updateETA);

  const int h = 180;
  map_eta->setFixedHeight(h);
  map_eta->move(0, 1080 - h);
  map_eta->setVisible(false);

  // Routing
  QVariantMap parameters;
  parameters["mapbox.access_token"] = m_settings.accessToken();

  geoservice_provider = new QGeoServiceProvider("mapbox", parameters);
  routing_manager = geoservice_provider->routingManager();
  if (routing_manager == nullptr) {
    qDebug() << geoservice_provider->errorString();
    assert(routing_manager);
  }
  connect(routing_manager, SIGNAL(finished(QGeoRouteReply*)), this, SLOT(routeCalculated(QGeoRouteReply*)));

  auto last_gps_position = coordinate_from_param("LastGPSPosition");
  if (last_gps_position) {
    last_position = *last_gps_position;
  }


  grabGesture(Qt::GestureType::PinchGesture);
}

MapWindow::~MapWindow() {
  makeCurrent();
}

void MapWindow::initLayers() {
  // This doesn't work from initializeGL
  if (!m_map->layerExists("modelPathLayer")) {
    QVariantMap modelPath;
    modelPath["id"] = "modelPathLayer";
    modelPath["type"] = "line";
    modelPath["source"] = "modelPathSource";
    m_map->addLayer(modelPath);
    m_map->setPaintProperty("modelPathLayer", "line-color", QColor("red"));
    m_map->setPaintProperty("modelPathLayer", "line-width", 5.0);
    m_map->setLayoutProperty("modelPathLayer", "line-cap", "round");
  }
  if (!m_map->layerExists("navLayer")) {
    QVariantMap nav;
    nav["id"] = "navLayer";
    nav["type"] = "line";
    nav["source"] = "navSource";
    m_map->addLayer(nav, "road-intersection");
    m_map->setPaintProperty("navLayer", "line-color", QColor("#8cb3d1"));
    m_map->setPaintProperty("navLayer", "line-width", 7.5);
    m_map->setLayoutProperty("navLayer", "line-cap", "round");
  }
  if (!m_map->layerExists("carPosLayer")) {
    m_map->addImage("label-arrow", QImage("../assets/images/triangle.svg"));

    QVariantMap carPos;
    carPos["id"] = "carPosLayer";
    carPos["type"] = "symbol";
    carPos["source"] = "carPosSource";
    m_map->addLayer(carPos);
    m_map->setLayoutProperty("carPosLayer", "icon-pitch-alignment", "map");
    m_map->setLayoutProperty("carPosLayer", "icon-image", "label-arrow");
    m_map->setLayoutProperty("carPosLayer", "icon-size", 0.5);
    m_map->setLayoutProperty("carPosLayer", "icon-ignore-placement", true);
    m_map->setLayoutProperty("carPosLayer", "icon-allow-overlap", true);
    m_map->setLayoutProperty("carPosLayer", "symbol-sort-key", 0);
  }
}

void MapWindow::timerUpdate() {
  initLayers();

  sm->update(0);
  if (sm->updated("liveLocationKalman")) {
    auto location = (*sm)["liveLocationKalman"].getLiveLocationKalman();
    gps_ok = location.getGpsOK();

    // Update map location, orientation and zoom on valid localizer output
    if (location.getStatus() == cereal::LiveLocationKalman::Status::VALID) {
      auto pos = location.getPositionGeodetic();
      auto orientation = location.getOrientationNED();

      float velocity = location.getVelocityCalibrated().getValue()[0];
      float bearing = RAD2DEG(orientation.getValue()[2]);
      auto coordinate = QMapbox::Coordinate(pos.getValue()[0], pos.getValue()[1]);

      last_position = coordinate;
      last_bearing = bearing;

      if (pan_counter == 0) {
        m_map->setCoordinate(coordinate);
        m_map->setBearing(bearing);
      } else {
        pan_counter--;
      }

      if (zoom_counter == 0) {
        static FirstOrderFilter velocity_filter(velocity, 10, 0.1);
        m_map->setZoom(util::map_val<float>(velocity_filter.update(velocity), 0, 30, MAX_ZOOM, MIN_ZOOM));
      } else {
        zoom_counter--;
      }

      // Update current location marker
      auto point = coordinate_to_collection(coordinate);
      QMapbox::Feature feature1(QMapbox::Feature::PointType, point, {}, {});
      QVariantMap carPosSource;
      carPosSource["type"] = "geojson";
      carPosSource["data"] = QVariant::fromValue<QMapbox::Feature>(feature1);
      m_map->updateSource("carPosSource", carPosSource);

      // Update model path
      if (DRAW_MODEL_PATH) {
        auto model = (*sm)["modelV2"].getModelV2();
        auto path_points = model_to_collection(location.getCalibratedOrientationECEF(), location.getPositionECEF(), model.getPosition());
        QMapbox::Feature feature2(QMapbox::Feature::LineStringType, path_points, {}, {});
        QVariantMap modelPathSource;
        modelPathSource["type"] = "geojson";
        modelPathSource["data"] = QVariant::fromValue<QMapbox::Feature>(feature2);
        m_map->updateSource("modelPathSource", modelPathSource);
      }
    }

    // Show route instructions
    if (segment.isValid()) {
      auto cur_maneuver = segment.maneuver();
      auto attrs = cur_maneuver.extendedAttributes();
      if (cur_maneuver.isValid() && attrs.contains("mapbox.banner_instructions")) {
        float along_geometry = distance_along_geometry(segment.path(), to_QGeoCoordinate(last_position));
        float distance = std::max(0.0f, float(segment.distance()) - along_geometry);
        emit distanceChanged(distance);

        m_map->setPitch(MAX_PITCH); // TODO: smooth pitching based on maneuver distance

        auto banner = attrs["mapbox.banner_instructions"].toList();
        if (banner.size()) {
          map_instructions->setVisible(true);

          auto banner_0 = banner[0].toMap();
          float show_at = banner_0["distance_along_geometry"].toDouble();
          emit instructionsChanged(banner_0, distance < show_at);
        }
      }

      auto next_segment = segment.nextRouteSegment();
      if (next_segment.isValid()) {
        auto next_maneuver = next_segment.maneuver();
        if (next_maneuver.isValid()) {
          float next_maneuver_distance = next_maneuver.position().distanceTo(to_QGeoCoordinate(last_position));
          // Switch to next route segment
          if (next_maneuver_distance < REROUTE_DISTANCE && next_maneuver_distance > last_maneuver_distance) {
            segment = next_segment;

            recompute_backoff = std::max(0, recompute_backoff - 1);
            recompute_countdown = 0;
          }
          last_maneuver_distance = next_maneuver_distance;
        }
      } else {
        Params().remove("NavDestination");

        // Clear route if driving away from destination
        float d = segment.maneuver().position().distanceTo(to_QGeoCoordinate(last_position));
        if (d > REROUTE_DISTANCE) {
          clearRoute();
        }
      }
    }

  }

  update();

  if (!segment.isValid()) {
    map_instructions->setVisible(false);
  }

}

void MapWindow::resizeGL(int w, int h) {
  map_instructions->setFixedWidth(width());
}

void MapWindow::initializeGL() {
  m_map.reset(new QMapboxGL(nullptr, m_settings, size(), 1));

  m_map->setCoordinateZoom(last_position, MAX_ZOOM);
  m_map->setMargins({0, 350, 0, 50});
  m_map->setPitch(MIN_PITCH);
  m_map->setStyleUrl("mapbox://styles/pd0wm/cknuhcgvr0vs817o1akcx6pek"); // Larger fonts

  connect(m_map.data(), SIGNAL(needsRendering()), this, SLOT(update()));
  timer->start(100);
}

void MapWindow::paintGL() {
  if (!isVisible()) return;

  m_map->resize(size() / MAP_SCALE);
  m_map->setFramebufferObject(defaultFramebufferObject(), size());
  m_map->render();
}

static float get_time_typical(const QGeoRouteSegment &segment) {
  auto maneuver = segment.maneuver();
  auto attrs = maneuver.extendedAttributes();
  return attrs.contains("mapbox.duration_typical") ? attrs["mapbox.duration_typical"].toDouble() : segment.travelTime();
}


void MapWindow::recomputeRoute() {
  bool should_recompute = shouldRecompute();
  auto new_destination = coordinate_from_param("NavDestination");

  if (!new_destination) {
    clearRoute();
    return;
  }

  if (*new_destination != nav_destination) {
    setVisible(true); // Show map on destination set/change
    should_recompute = true;
  }

  if (!should_recompute) updateETA(); // ETA is updated after recompute

  if (!gps_ok && segment.isValid()) return; // Don't recompute when gps drifts in tunnels

  // Only do API request when map is loaded
  if (!m_map.isNull()) {
    if (recompute_countdown == 0 && should_recompute) {
      recompute_countdown = std::pow(2, recompute_backoff);
      recompute_backoff = std::min(7, recompute_backoff + 1);
      calculateRoute(*new_destination);
    } else {
      recompute_countdown = std::max(0, recompute_countdown - 1);
    }
  }
}

void MapWindow::updateETA() {
  if (segment.isValid()) {
    float progress = distance_along_geometry(segment.path(), to_QGeoCoordinate(last_position)) / segment.distance();
    float total_distance = segment.distance() * (1.0 - progress);
    float total_time = segment.travelTime() * (1.0 - progress);
    float total_time_typical = get_time_typical(segment) * (1.0 - progress);

    auto s = segment.nextRouteSegment();
    while (s.isValid()) {
      total_distance += s.distance();
      total_time += s.travelTime();
      total_time_typical += get_time_typical(s);

      s = s.nextRouteSegment();
    }

    emit ETAChanged(total_time, total_time_typical, total_distance);
  }
}

void MapWindow::calculateRoute(QMapbox::Coordinate destination) {
  LOGW("calculating route");
  nav_destination = destination;
  QGeoRouteRequest request(to_QGeoCoordinate(last_position), to_QGeoCoordinate(destination));
  request.setFeatureWeight(QGeoRouteRequest::TrafficFeature, QGeoRouteRequest::AvoidFeatureWeight);

  if (last_bearing) {
    QVariantMap params;
    int bearing = ((int)(*last_bearing) + 360) % 360;
    params["bearing"] = bearing;
    request.setWaypointsMetadata({params});
  }

  routing_manager->calculateRoute(request);
}

void MapWindow::routeCalculated(QGeoRouteReply *reply) {
  LOGW("new route calculated");
  if (reply->routes().size() != 0) {
    route = reply->routes().at(0);
    segment = route.firstRouteSegment();

    auto route_points = coordinate_list_to_collection(route.path());
    QMapbox::Feature feature(QMapbox::Feature::LineStringType, route_points, {}, {});
    QVariantMap navSource;
    navSource["type"] = "geojson";
    navSource["data"] = QVariant::fromValue<QMapbox::Feature>(feature);
    m_map->updateSource("navSource", navSource);
    m_map->setLayoutProperty("navLayer", "visibility", "visible");

    updateETA();
  }

  reply->deleteLater();
}

void MapWindow::clearRoute() {
  segment = QGeoRouteSegment();
  nav_destination = QMapbox::Coordinate();

  if (!m_map.isNull()) {
    m_map->setLayoutProperty("navLayer", "visibility", "none");
    m_map->setPitch(MIN_PITCH);
  }

  map_instructions->setVisible(false);
  map_eta->setVisible(false);
}


bool MapWindow::shouldRecompute() {
  if (!segment.isValid()) {
    return true;
  }

  // Compute closest distance to all line segments in the current path
  float min_d = REROUTE_DISTANCE + 1;
  auto path = segment.path();
  auto cur = to_QGeoCoordinate(last_position);
  for (size_t i = 0; i < path.size() - 1; i++) {
    auto a = path[i];
    auto b = path[i+1];
    if (a.distanceTo(b) < 1.0) {
      continue;
    }
    min_d = std::min(min_d, minimum_distance(a, b, cur));
  }
  return min_d > REROUTE_DISTANCE;

  // TODO: Check for going wrong way in segment
}

void MapWindow::mousePressEvent(QMouseEvent *ev) {
  m_lastPos = ev->localPos();
  ev->accept();
}

void MapWindow::mouseMoveEvent(QMouseEvent *ev) {
  QPointF delta = ev->localPos() - m_lastPos;

  if (!delta.isNull()) {
    pan_counter = PAN_TIMEOUT;
    m_map->moveBy(delta / MAP_SCALE);
  }

  m_lastPos = ev->localPos();
  ev->accept();
}

void MapWindow::wheelEvent(QWheelEvent *ev) {
  if (ev->orientation() == Qt::Horizontal) {
      return;
  }

  float factor = ev->delta() / 1200.;
  if (ev->delta() < 0) {
      factor = factor > -1 ? factor : 1 / factor;
  }

  m_map->scaleBy(1 + factor, ev->pos() / MAP_SCALE);
  zoom_counter = PAN_TIMEOUT;
  ev->accept();
}

bool MapWindow::event(QEvent *event) {
  if (event->type() == QEvent::Gesture) {
    return gestureEvent(static_cast<QGestureEvent*>(event));
  }

  return QWidget::event(event);
}

bool MapWindow::gestureEvent(QGestureEvent *event) {
  if (QGesture *pinch = event->gesture(Qt::PinchGesture)) {
    pinchTriggered(static_cast<QPinchGesture *>(pinch));
  }
  return true;
}

void MapWindow::pinchTriggered(QPinchGesture *gesture) {
  QPinchGesture::ChangeFlags changeFlags = gesture->changeFlags();
  if (changeFlags & QPinchGesture::ScaleFactorChanged) {
    // TODO: figure out why gesture centerPoint doesn't work
    m_map->scaleBy(gesture->scaleFactor(), {width() / 2.0 / MAP_SCALE, height() / 2.0 / MAP_SCALE});
    zoom_counter = PAN_TIMEOUT;
  }
}

void MapWindow::offroadTransition(bool offroad) {
  if (!offroad) {
    auto dest = coordinate_from_param("NavDestination");
    setVisible(dest.has_value());
  }
  last_bearing = {};
}

MapInstructions::MapInstructions(QWidget * parent) : QWidget(parent) {
  QHBoxLayout *main_layout = new QHBoxLayout(this);
  main_layout->setContentsMargins(11, 50, 11, 11);
  {
    QVBoxLayout *layout = new QVBoxLayout;
    icon_01 = new QLabel;
    layout->addWidget(icon_01);
    layout->addStretch();
    main_layout->addLayout(layout);
  }

  {
    QWidget *w = new QWidget;
    QVBoxLayout *layout = new QVBoxLayout(w);

    distance = new QLabel;
    distance->setStyleSheet(R"(font-size: 75px; )");
    layout->addWidget(distance);

    primary = new QLabel;
    primary->setStyleSheet(R"(font-size: 50px;)");
    primary->setWordWrap(true);
    layout->addWidget(primary);

    secondary = new QLabel;
    secondary->setStyleSheet(R"(font-size: 40px;)");
    secondary->setWordWrap(true);
    layout->addWidget(secondary);

    lane_layout = new QHBoxLayout;
    layout->addLayout(lane_layout);

    main_layout->addWidget(w);
  }

  setStyleSheet(R"(
    * {
      color: white;
      font-family: "Inter";
    }
  )");

  QPalette pal = palette();
  pal.setColor(QPalette::Background, QColor(0, 0, 0, 150));
  setAutoFillBackground(true);
  setPalette(pal);
}

void MapInstructions::updateDistance(float d) {
  QString distance_str;

  if (QUIState::ui_state.scene.is_metric) {
    if (d > 500) {
      distance_str.setNum(d / 1000, 'f', 1);
      distance_str += " km";
    } else {
      distance_str.setNum(50 * int(d / 50));
      distance_str += " m";
    }
  } else {
    float miles = d * METER_2_MILE;
    float feet = d * METER_2_FOOT;

    if (feet > 500) {
      distance_str.setNum(miles, 'f', 1);
      distance_str += " mi";
    } else {
      distance_str.setNum(50 * int(feet / 50));
      distance_str += " ft";
    }
  }

  distance->setText(distance_str);
}

void MapInstructions::updateInstructions(QMap<QString, QVariant> banner, bool full) {
  // Need multiple calls to adjustSize for it to properly resize
  // seems like it takes a little bit of time for the images to change and
  // the size can only be changed afterwards
  adjustSize();

  if (banner == last_banner) return;
  QString primary_str, secondary_str;

  auto p = banner["primary"].toMap();
  primary_str += p["text"].toString();

  // Show arrow with direction
  if (p.contains("type")) {
    QString fn = "../assets/navigation/direction_" + p["type"].toString();
    if (p.contains("modifier")) {
      fn += "_" + p["modifier"].toString();
    }
    fn +=  + ".png";
    fn = fn.replace(' ', '_');

    QPixmap pix(fn);
    icon_01->setPixmap(pix.scaledToWidth(200, Qt::SmoothTransformation));
    icon_01->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
  }

  // Parse components (e.g. lanes, exit number)
  auto components = p["components"].toList();
  QString icon_fn;
  for (auto &c : components) {
    auto cc = c.toMap();
    if (cc["type"].toString() == "icon") {
      icon_fn = cc["imageBaseURL"].toString() + "@3x.png";
    }
  }

  if (banner.contains("secondary") && full) {
    auto s = banner["secondary"].toMap();
    secondary_str += s["text"].toString();
  }

  clearLayout(lane_layout);
  bool has_lanes = false;

  if (banner.contains("sub") && full) {
    auto s = banner["sub"].toMap();
    auto components = s["components"].toList();
    for (auto &c : components) {
      auto cc = c.toMap();
      if (cc["type"].toString() == "lane") {
        has_lanes = true;

        bool left = false;
        bool straight = false;
        bool right = false;
        bool active = cc["active"].toBool();

        for (auto &dir : cc["directions"].toList()) {
          auto d = dir.toString();
          left |= d.contains("left");
          straight |= d.contains("straight");
          right |= d.contains("right");
        }

        // TODO: Make more images based on active direction and combined directions
        QString fn = "../assets/navigation/direction_";
        if (left) {
          fn += "turn_left";
        } else if (right) {
          fn += "turn_right";
        } else if (straight) {
          fn += "turn_straight";
        }

        QPixmap pix(fn + ".png");
        auto icon = new QLabel;
        icon->setPixmap(pix.scaledToWidth(active ? 125 : 75, Qt::SmoothTransformation));
        icon->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
        lane_layout->addWidget(icon);
      }
    }
  }

  primary->setText(primary_str);
  secondary->setVisible(secondary_str.length() > 0);
  secondary->setText(secondary_str);
  adjustSize();
  last_banner = banner;
}

MapETA::MapETA(QWidget * parent) : QWidget(parent) {
  QHBoxLayout *main_layout = new QHBoxLayout(this);
  main_layout->setContentsMargins(20, 25, 20, 25);

  {
    QVBoxLayout *layout = new QVBoxLayout;
    eta = new QLabel;
    eta->setAlignment(Qt::AlignCenter);

    auto eta_unit = new QLabel("eta");
    eta_unit->setAlignment(Qt::AlignCenter);

    layout->addStretch();
    layout->addWidget(eta);
    layout->addWidget(eta_unit);
    layout->addStretch();
    main_layout->addLayout(layout);
  }
  main_layout->addSpacing(30);
  {
    QVBoxLayout *layout = new QVBoxLayout;
    time = new QLabel;
    time->setAlignment(Qt::AlignCenter);

    time_unit = new QLabel;
    time_unit->setAlignment(Qt::AlignCenter);

    layout->addStretch();
    layout->addWidget(time);
    layout->addWidget(time_unit);
    layout->addStretch();
    main_layout->addLayout(layout);
  }
  main_layout->addSpacing(30);
  {
    QVBoxLayout *layout = new QVBoxLayout;
    distance = new QLabel;
    distance->setAlignment(Qt::AlignCenter);
    distance_unit = new QLabel;
    distance_unit->setAlignment(Qt::AlignCenter);

    layout->addStretch();
    layout->addWidget(distance);
    layout->addWidget(distance_unit);
    layout->addStretch();
    main_layout->addLayout(layout);
  }

  setStyleSheet(R"(
    * {
      color: white;
      font-family: "Inter";
      font-size: 55px;
    }
  )");

  QPalette pal = palette();
  pal.setColor(QPalette::Background, QColor(0, 0, 0, 150));
  setAutoFillBackground(true);
  setPalette(pal);
}


void MapETA::updateETA(float s, float s_typical, float d) {
  setVisible(true);

  // ETA
  auto eta_time = QDateTime::currentDateTime().addSecs(s).time();
  if (params.getBool("NavSettingTime24h")) {
    eta->setText(eta_time.toString("HH:mm"));
  } else {
    eta->setText(eta_time.toString("h:mm a"));
  }

  // Remaining time
  if (s < 3600) {
    time->setText(QString::number(int(s / 60)));
    time_unit->setText("min");
  } else {
    int hours = int(s) / 3600;
    time->setText(QString::number(hours) + ":" + QString::number(int((s - hours * 3600) / 60)).rightJustified(2, '0'));
    time_unit->setText("hr");
  }

  if (s / s_typical > 1.5) {
    time_unit->setStyleSheet(R"(color: #DA3025; )");
    time->setStyleSheet(R"(color: #DA3025; )");
  } else if (s / s_typical > 1.2) {
    time_unit->setStyleSheet(R"(color: #DAA725; )");
    time->setStyleSheet(R"(color: #DAA725; )");
  } else {
    time_unit->setStyleSheet(R"(color: #25DA6E; )");
    time->setStyleSheet(R"(color: #25DA6E; )");
  }

  // Distance
  QString distance_str;
  float num = 0;
  if (QUIState::ui_state.scene.is_metric) {
    num = d / 1000.0;
    distance_unit->setText("km");
  } else {
    num = d * METER_2_MILE;
    distance_unit->setText("mi");
  }

  distance_str.setNum(num, 'f', num < 100 ? 1 : 0);
  distance->setText(distance_str);

  adjustSize();
}
