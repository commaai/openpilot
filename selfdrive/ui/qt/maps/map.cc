#include <cmath>

#include <QDebug>

#include "selfdrive/common/util.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/params.h"
#include "selfdrive/ui/ui.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/maps/map_helpers.h"
#include "selfdrive/ui/qt/maps/map.h"


const int PAN_TIMEOUT = 100;
const bool DRAW_MODEL_PATH = false;
const qreal REROUTE_DISTANCE = 25;
const float MAX_ZOOM = 17;
const float MIN_ZOOM = 14;
const float MAX_PITCH = 50;
const float MIN_PITCH = 0;
const float MAP_SCALE = 2;


MapWindow::MapWindow(const QMapboxGLSettings &settings) : m_settings(settings) {
  if (DRAW_MODEL_PATH){
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
  connect(this, SIGNAL(instructionsChanged(QMap<QString, QVariant>)),
          map_instructions, SLOT(updateInstructions(QMap<QString, QVariant>)));
  connect(this, SIGNAL(distanceChanged(float)),
          map_instructions, SLOT(updateDistance(float)));
  map_instructions->setFixedWidth(width());

  // Routing
  QVariantMap parameters;
  parameters["mapbox.access_token"] = m_settings.accessToken();

  geoservice_provider = new QGeoServiceProvider("mapbox", parameters);
  routing_manager = geoservice_provider->routingManager();
  if (routing_manager == nullptr){
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
  if (!m_map->layerExists("modelPathLayer")){
    QVariantMap modelPath;
    modelPath["id"] = "modelPathLayer";
    modelPath["type"] = "line";
    modelPath["source"] = "modelPathSource";
    m_map->addLayer(modelPath);
    m_map->setPaintProperty("modelPathLayer", "line-color", QColor("red"));
    m_map->setPaintProperty("modelPathLayer", "line-width", 5.0);
    m_map->setLayoutProperty("modelPathLayer", "line-cap", "round");
  }
  if (!m_map->layerExists("navLayer")){
    QVariantMap nav;
    nav["id"] = "navLayer";
    nav["type"] = "line";
    nav["source"] = "navSource";
    m_map->addLayer(nav, "road-intersection");
    m_map->setPaintProperty("navLayer", "line-color", QColor("#8cb3d1"));
    m_map->setPaintProperty("navLayer", "line-width", 7.5);
    m_map->setLayoutProperty("navLayer", "line-cap", "round");
  }
  if (!m_map->layerExists("carPosLayer")){
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
      float velocity = location.getVelocityCalibrated().getValue()[0];
      auto orientation = location.getOrientationNED();
      auto coordinate = QMapbox::Coordinate(pos.getValue()[0], pos.getValue()[1]);
      last_position = coordinate;

      if (pan_counter == 0){
        m_map->setCoordinate(coordinate);
        m_map->setBearing(RAD2DEG(orientation.getValue()[2]));
      } else {
        pan_counter--;
      }

      if (zoom_counter == 0){
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
      if (cur_maneuver.isValid() && attrs.contains("mapbox.banner_instructions")){

        auto banner = attrs["mapbox.banner_instructions"].toList();
        if (banner.size()){
          // TOOD: Only show when traveled distanceAlongGeometry since the start
          map_instructions->setVisible(true);
          emit instructionsChanged(banner[0].toMap());
        }

      }

      auto next_segment = segment.nextRouteSegment();
      if (next_segment.isValid()){
        auto next_maneuver = next_segment.maneuver();
        if (next_maneuver.isValid()){
          float next_maneuver_distance = next_maneuver.position().distanceTo(to_QGeoCoordinate(last_position));
          emit distanceChanged(next_maneuver_distance);
          m_map->setPitch(MAX_PITCH); // TODO: smooth pitching based on maneuver distance

          // Switch to next route segment
          if (next_maneuver_distance < REROUTE_DISTANCE && next_maneuver_distance > last_maneuver_distance){
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

  if (!segment.isValid()){
    map_instructions->setVisible(false);
  }

}

void MapWindow::resizeGL(int w, int h) {
  map_instructions->setFixedWidth(width());
}

void MapWindow::initializeGL() {
  m_map.reset(new QMapboxGL(nullptr, m_settings, size(), 1));

  // TODO: Get from last gps position param
  m_map->setCoordinateZoom(last_position, MAX_ZOOM);
  m_map->setMargins({0, 350, 0, 0});
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


void MapWindow::recomputeRoute() {
  bool should_recompute = shouldRecompute();
  auto new_destination = coordinate_from_param("NavDestination");

  if (!new_destination) {
    clearRoute();
    return;
  }

  if (*new_destination != nav_destination){
    setVisible(true); // Show map on destination set/change
    should_recompute = true;
  }

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

void MapWindow::calculateRoute(QMapbox::Coordinate destination) {
  nav_destination = destination;
  QGeoRouteRequest request(to_QGeoCoordinate(last_position), to_QGeoCoordinate(destination));
  request.setFeatureWeight(QGeoRouteRequest::TrafficFeature, QGeoRouteRequest::AvoidFeatureWeight);
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
}


bool MapWindow::shouldRecompute(){
  if (!segment.isValid()){
    return true;
  }

  // Compute closest distance to all line segments in the current path
  float min_d = REROUTE_DISTANCE + 1;
  auto path = segment.path();
  auto cur = to_QGeoCoordinate(last_position);
  for (size_t i = 0; i < path.size() - 1; i++){
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

void MapWindow::mouseMoveEvent(QMouseEvent *ev){
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
  if (event->type() == QEvent::Gesture){
    return gestureEvent(static_cast<QGestureEvent*>(event));
  }

  return QWidget::event(event);
}

bool MapWindow::gestureEvent(QGestureEvent *event) {
  if (QGesture *pinch = event->gesture(Qt::PinchGesture)){
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
}

MapInstructions::MapInstructions(QWidget * parent) : QWidget(parent){
  QHBoxLayout *layout_outer = new QHBoxLayout;
  layout_outer->setContentsMargins(11, 50, 11, 11);
  {
    QVBoxLayout *layout = new QVBoxLayout;
    icon_01 = new QLabel;
    layout->addWidget(icon_01);
    layout->addStretch();
    layout_outer->addLayout(layout);
  }

  {
    QVBoxLayout *layout = new QVBoxLayout;

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

    QWidget * w = new QWidget;
    w->setLayout(layout);
    layout_outer->addWidget(w);
  }

  setLayout(layout_outer);
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

void MapInstructions::updateDistance(float d){
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
      distance_str += " miles";
    } else {
      distance_str.setNum(50 * int(feet / 50));
      distance_str += " feet";
    }
  }

  distance->setText(distance_str);
}

void MapInstructions::updateInstructions(QMap<QString, QVariant> banner){
  // Need multiple calls to adjustSize for it to properly resize
  // seems like it takes a little bit of time for the images to change and
  // the size can only be changed afterwards
  adjustSize();

  if (banner == last_banner) return;
  QString primary_str, secondary_str;

  auto p = banner["primary"].toMap();
  primary_str += p["text"].toString();

  // Show arrow with direction
  if (p.contains("type")){
    QString fn = "../assets/navigation/direction_" + p["type"].toString();
    if (p.contains("modifier")){
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
    if (cc["type"].toString() == "icon"){
      icon_fn = cc["imageBaseURL"].toString() + "@3x.png";
    }
  }

  if (banner.contains("secondary")){
    auto s = banner["secondary"].toMap();
    secondary_str += s["text"].toString();
  }

  clearLayout(lane_layout);
  bool has_lanes = false;

  if (banner.contains("sub")){
    auto s = banner["sub"].toMap();
    auto components = s["components"].toList();
    for (auto &c : components) {
      auto cc = c.toMap();
      if (cc["type"].toString() == "lane"){
        has_lanes = true;

        bool left = false;
        bool straight = false;
        bool right = false;
        bool active = cc["active"].toBool();

        for (auto &dir : cc["directions"].toList()){
          auto d = dir.toString();
          left |= d.contains("left");
          straight |= d.contains("straight");
          right |= d.contains("right");
        }

        // TODO: Make more images based on active direction and combined directions
        QString fn = "../assets/navigation/direction_";
        if (left) {
          fn += "turn_left";
        } else if (right){
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
