#include <cmath>
#include <eigen3/Eigen/Dense>

#include "map.h"
#include "common/util.h"
#include "common/params.h"
#include "common/transformations/coordinates.hpp"
#include "common/transformations/orientation.hpp"

#include <QDebug>
#include <QJsonDocument>
#include <QJsonObject>

#define RAD2DEG(x) ((x) * 180.0 / M_PI)
const int PAN_TIMEOUT = 100;
const bool DRAW_MODEL_PATH = false;
const qreal REROUTE_DISTANCE = 25;
const float METER_2_MILE = 0.000621371;
const float METER_2_FOOT = 3.28084;

static void clearLayout(QLayout* layout) {
  while (QLayoutItem* item = layout->takeAt(0)) {
    if (QWidget* widget = item->widget()) {
      widget->deleteLater();
    }
    if (QLayout* childLayout = item->layout()) {
      clearLayout(childLayout);
    }
    delete item;
  }
}

static QGeoCoordinate to_QGeoCoordinate(const QMapbox::Coordinate &in) {
  return QGeoCoordinate(in.first, in.second);
}

static QMapbox::CoordinatesCollections model_to_collection(
  const cereal::LiveLocationKalman::Measurement::Reader &calibratedOrientationECEF,
  const cereal::LiveLocationKalman::Measurement::Reader &positionECEF,
  const cereal::ModelDataV2::XYZTData::Reader &line){

  Eigen::Vector3d ecef(positionECEF.getValue()[0], positionECEF.getValue()[1], positionECEF.getValue()[2]);
  Eigen::Vector3d orient(calibratedOrientationECEF.getValue()[0], calibratedOrientationECEF.getValue()[1], calibratedOrientationECEF.getValue()[2]);
  Eigen::Matrix3d ecef_from_local = euler2rot(orient).transpose();

  QMapbox::Coordinates coordinates;
  auto x = line.getX();
  auto y = line.getY();
  auto z = line.getZ();
  for (int i = 0; i < x.size(); i++){
    Eigen::Vector3d point_ecef = ecef_from_local * Eigen::Vector3d(x[i], y[i], z[i]) + ecef;
    Geodetic point_geodetic = ecef2geodetic((ECEF){.x = point_ecef[0], .y = point_ecef[1], .z = point_ecef[2]});
    QMapbox::Coordinate coordinate(point_geodetic.lat, point_geodetic.lon);
    coordinates.push_back(coordinate);
  }

  QMapbox::CoordinatesCollection collection;
  collection.push_back(coordinates);

  QMapbox::CoordinatesCollections collections;
  collections.push_back(collection);
  return collections;
}

static QMapbox::CoordinatesCollections coordinate_to_collection(QMapbox::Coordinate c){
  QMapbox::Coordinates coordinates;
  coordinates.push_back(c);

  QMapbox::CoordinatesCollection collection;
  collection.push_back(coordinates);

  QMapbox::CoordinatesCollections collections;
  collections.push_back(collection);
  return collections;
}

static QMapbox::CoordinatesCollections coordinate_list_to_collection(QList<QGeoCoordinate> coordinate_list) {
  QMapbox::Coordinates coordinates;

  for (auto &c : coordinate_list){
    QMapbox::Coordinate coordinate(c.latitude(), c.longitude());
    coordinates.push_back(coordinate);
  }

  QMapbox::CoordinatesCollection collection;
  collection.push_back(coordinates);

  QMapbox::CoordinatesCollections collections;
  collections.push_back(collection);
  return collections;
}

MapWindow::MapWindow(const QMapboxGLSettings &settings) : m_settings(settings) {
  if (DRAW_MODEL_PATH){
    sm = new SubMaster({"liveLocationKalman", "modelV2"});
  } else {
    sm = new SubMaster({"liveLocationKalman"});
  }

  timer = new QTimer(this);
  QObject::connect(timer, SIGNAL(timeout()), this, SLOT(timerUpdate()));

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

  grabGesture(Qt::GestureType::PinchGesture);
}

MapWindow::~MapWindow() {
  makeCurrent();
}

void MapWindow::timerUpdate() {
  if (!isVisible()) return;

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
    m_map->setLayoutProperty("carPosLayer", "icon-image", "label-arrow");
    m_map->setLayoutProperty("carPosLayer", "icon-ignore-placement", true);
  }

  sm->update(0);
  if (sm->updated("liveLocationKalman")) {
    auto location = (*sm)["liveLocationKalman"].getLiveLocationKalman();
    auto pos = location.getPositionGeodetic();
    auto orientation = location.getOrientationNED();

    float velocity = location.getVelocityCalibrated().getValue()[0];
    static FirstOrderFilter velocity_filter(velocity, 15, 0.1);

    auto coordinate = QMapbox::Coordinate(pos.getValue()[0], pos.getValue()[1]);

    if (location.getStatus() == cereal::LiveLocationKalman::Status::VALID){
      last_position = coordinate;

      if (sm->frame % 10 == 0 && shouldRecompute()){
        calculateRoute(nav_destination);
      }

      if (segment.isValid()) {
        auto cur_maneuver = segment.maneuver();
        auto attrs = cur_maneuver.extendedAttributes();
        if (cur_maneuver.isValid() && attrs.contains("mapbox.banner_instructions")){

          auto banner = attrs["mapbox.banner_instructions"].toList();
          if (banner.size()){
            // TOOD: Only show when traveled distanceAlongGeometry since the start
            emit instructionsChanged(banner[0].toMap());
          }

        }

        auto next_segment = segment.nextRouteSegment();
        if (next_segment.isValid()){
          auto next_maneuver = next_segment.maneuver();
          if (next_maneuver.isValid()){
            float next_maneuver_distance = next_maneuver.position().distanceTo(to_QGeoCoordinate(last_position));
            emit distanceChanged(next_maneuver_distance);

            if (next_maneuver_distance < REROUTE_DISTANCE && next_maneuver_distance > last_maneuver_distance){
              segment = next_segment;
            }
            last_maneuver_distance = next_maneuver_distance;
          }
        }
      }

      if (pan_counter == 0){
        m_map->setCoordinate(coordinate);
        m_map->setBearing(RAD2DEG(orientation.getValue()[2]));
      } else {
        pan_counter--;
      }

      if (zoom_counter == 0){
        // Scale zoom between 16 and 19 based on speed
        m_map->setZoom(19 - std::min(3.0f, velocity_filter.update(velocity) / 10));
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
    update();
  }

}

void MapWindow::resizeGL(int w, int h) {
  qDebug() << "resize map " << w << "x" << h;
  map_instructions->setFixedWidth(width());
}

void MapWindow::initializeGL() {
  m_map.reset(new QMapboxGL(nullptr, m_settings, size(), 1));

  // TODO: Get from last gps position param
  m_map->setCoordinateZoom(last_position, 18);
  m_map->setStyleUrl("mapbox://styles/pd0wm/cknuhcgvr0vs817o1akcx6pek"); // Larger fonts

  connect(m_map.data(), SIGNAL(needsRendering()), this, SLOT(update()));
  timer->start(100);
}

void MapWindow::paintGL() {
  m_map->resize(size());
  m_map->setFramebufferObject(defaultFramebufferObject(), size());
  m_map->render();
}


void MapWindow::calculateRoute(QMapbox::Coordinate destination) {
  QGeoRouteRequest request(to_QGeoCoordinate(last_position), to_QGeoCoordinate(destination));
  routing_manager->calculateRoute(request);
}

void MapWindow::routeCalculated(QGeoRouteReply *reply) {
  qDebug() << "route update";
  if (reply->routes().size() != 0) {
    route = reply->routes().at(0);
    segment = route.firstRouteSegment();

    auto route_points = coordinate_list_to_collection(route.path());
    QMapbox::Feature feature(QMapbox::Feature::LineStringType, route_points, {}, {});
    QVariantMap navSource;
    navSource["type"] = "geojson";
    navSource["data"] = QVariant::fromValue<QMapbox::Feature>(feature);
    m_map->updateSource("navSource", navSource);
    has_route = true;
  }

  reply->deleteLater();
}


// TODO: put in helper file
static QGeoCoordinate sub(QGeoCoordinate v, QGeoCoordinate w){
  return QGeoCoordinate(v.latitude() - w.latitude(), v.longitude() - w.longitude());
}

static QGeoCoordinate add(QGeoCoordinate v, QGeoCoordinate w){
  return QGeoCoordinate(v.latitude() + w.latitude(), v.longitude() + w.longitude());
}

static QGeoCoordinate mul(QGeoCoordinate v, float c){
  return QGeoCoordinate(c * v.latitude(), c * v.longitude());
}

static float dot(QGeoCoordinate v, QGeoCoordinate w){
  return v.latitude() * w.latitude() + v.longitude() * w.longitude();

}

static float minimum_distance(QGeoCoordinate a, QGeoCoordinate b, QGeoCoordinate p) {
  const QGeoCoordinate ap = sub(p, a);
  const QGeoCoordinate ab = sub(b, a);
  const float t = std::clamp(dot(ap, ab) / dot(ab, ab), 0.0f, 1.0f);
  const QGeoCoordinate projection = add(a, mul(ab, t));
  return projection.distanceTo(p);
}

bool MapWindow::shouldRecompute(){
  // Recompute based on some heuristics
  // - Destination changed
  // - Distance to current segment
  // - Wrong direcection in segment
  QString nav_destination_json = QString::fromStdString(Params().get("NavDestination"));
  if (nav_destination_json.isEmpty()) return false;

  QJsonDocument doc = QJsonDocument::fromJson(nav_destination_json.toUtf8());
  if (doc.isNull()) return false;

  QJsonObject json = doc.object();
  if (json["latitude"].isDouble() && json["longitude"].isDouble()){
    QMapbox::Coordinate new_destination(json["latitude"].toDouble(), json["longitude"].toDouble());
    if (new_destination != nav_destination){
      nav_destination = new_destination;
      return true;
    }
  }

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
    min_d = std::min(min_d, minimum_distance(a, b, cur));
  }
  return min_d > REROUTE_DISTANCE;
}


// Events
void MapWindow::mousePressEvent(QMouseEvent *ev) {
  m_lastPos = ev->localPos();
  ev->accept();
}

void MapWindow::mouseMoveEvent(QMouseEvent *ev){
  QPointF delta = ev->localPos() - m_lastPos;

  if (!delta.isNull()) {
    pan_counter = PAN_TIMEOUT;
    m_map->moveBy(delta);
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

  m_map->scaleBy(1 + factor, ev->pos());
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
    m_map->scaleBy(gesture->scaleFactor(), {width() / 2.0, height() / 2.0});
    zoom_counter = PAN_TIMEOUT;
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
    layout_outer->addLayout(layout);
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

  float miles = d * METER_2_MILE;
  float feet = d * METER_2_FOOT;

  if (feet > 500){
    distance_str.setNum(miles, 'f', 1);
    distance_str += " miles";
  } else {
    distance_str.setNum(50 * int(feet / 50));
    distance_str += " feet";
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
