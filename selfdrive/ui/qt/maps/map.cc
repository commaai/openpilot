#include "selfdrive/ui/qt/maps/map.h"

#include <eigen3/Eigen/Dense>
#include <cmath>

#include <QDebug>
#include <QPainter>

#include "common/transformations/coordinates.hpp"
#include "selfdrive/ui/qt/maps/map_helpers.h"
#include "selfdrive/ui/qt/util.h"

const int PAN_TIMEOUT = 100;
const float MANEUVER_TRANSITION_THRESHOLD = 10;

const float MAX_ZOOM = 17;
const float MIN_ZOOM = 14;
const float MAX_PITCH = 50;
const float MIN_PITCH = 0;
const float MAP_SCALE = 2;

const float VALID_POS_STD = 50.0; // m

const QString ICON_SUFFIX = ".png";

MapWindow::MapWindow(const QMapboxGLSettings &settings) : m_settings(settings), velocity_filter(0, 10, 0.05) {
  QObject::connect(uiState(), &UIState::uiUpdate, this, &MapWindow::updateState);

  // Instructions
  map_instructions = new MapInstructions(this);
  QObject::connect(this, &MapWindow::instructionsChanged, map_instructions, &MapInstructions::updateInstructions);
  QObject::connect(this, &MapWindow::distanceChanged, map_instructions, &MapInstructions::updateDistance);
  QObject::connect(this, &MapWindow::ETAChanged, map_instructions, &MapInstructions::updateETA);

  last_position = coordinate_from_param("LastGPSPosition");
  grabGesture(Qt::GestureType::PinchGesture);
  qDebug() << "MapWindow initialized";
}

MapWindow::~MapWindow() {
  makeCurrent();
}

void MapWindow::initLayers() {
  // This doesn't work from initializeGL
  if (!m_map->layerExists("modelPathLayer")) {
    qDebug() << "Initializing modelPathLayer";
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
    qDebug() << "Initializing navLayer";
    QVariantMap nav;
    nav["id"] = "navLayer";
    nav["type"] = "line";
    nav["source"] = "navSource";
    m_map->addLayer(nav, "road-intersection");
    m_map->setPaintProperty("navLayer", "line-color", QColor("#31a1ee"));
    m_map->setPaintProperty("navLayer", "line-width", 7.5);
    m_map->setLayoutProperty("navLayer", "line-cap", "round");
  }
  if (!m_map->layerExists("carPosLayer")) {
    qDebug() << "Initializing carPosLayer";
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

void MapWindow::updateState(const UIState &s) {
  if (!uiState()->scene.started) {
    return;
  }
  const SubMaster &sm = *(s.sm);
  update();

  if (sm.updated("liveLocationKalman")) {
    auto locationd_location = sm["liveLocationKalman"].getLiveLocationKalman();
    auto locationd_pos = locationd_location.getPositionGeodetic();
    auto locationd_orientation = locationd_location.getCalibratedOrientationNED();
    auto locationd_velocity = locationd_location.getVelocityCalibrated();

    locationd_valid = (locationd_location.getStatus() == cereal::LiveLocationKalman::Status::VALID) &&
      locationd_pos.getValid() && locationd_orientation.getValid() && locationd_velocity.getValid();

    if (locationd_valid) {
      last_position = QMapbox::Coordinate(locationd_pos.getValue()[0], locationd_pos.getValue()[1]);
      last_bearing = RAD2DEG(locationd_orientation.getValue()[2]);
      velocity_filter.update(locationd_velocity.getValue()[0]);
    }
  }

  if (sm.updated("gnssMeasurements")) {
    auto laikad_location = sm["gnssMeasurements"].getGnssMeasurements();
    auto laikad_pos = laikad_location.getPositionECEF();
    auto laikad_pos_ecef = laikad_pos.getValue();
    auto laikad_pos_std = laikad_pos.getStd();
    auto laikad_velocity_ecef = laikad_location.getVelocityECEF().getValue();

    laikad_valid = laikad_pos.getValid() && Eigen::Vector3d(laikad_pos_std[0], laikad_pos_std[1], laikad_pos_std[2]).norm() < VALID_POS_STD;

    if (laikad_valid && !locationd_valid) {
      ECEF ecef = {.x = laikad_pos_ecef[0], .y = laikad_pos_ecef[1], .z = laikad_pos_ecef[2]};
      Geodetic laikad_pos_geodetic = ecef2geodetic(ecef);
      last_position = QMapbox::Coordinate(laikad_pos_geodetic.lat, laikad_pos_geodetic.lon);

      // Compute NED velocity
      LocalCoord converter(ecef);
      ECEF next_ecef = {.x = ecef.x + laikad_velocity_ecef[0], .y = ecef.y + laikad_velocity_ecef[1], .z = ecef.z + laikad_velocity_ecef[2]};
      Eigen::VectorXd ned_vel = converter.ecef2ned(next_ecef).to_vector() - converter.ecef2ned(ecef).to_vector();

      float velocity = ned_vel.norm();
      velocity_filter.update(velocity);

      // Convert NED velocity to angle
      if (velocity > 1.0) {
        float new_bearing = fmod(RAD2DEG(atan2(ned_vel[1], ned_vel[0])) + 360.0, 360.0);
        if (last_bearing) {
          float delta = 0.1 * angle_difference(*last_bearing, new_bearing); // Smooth heading
          last_bearing = fmod(*last_bearing + delta + 360.0, 360.0);
        } else {
          last_bearing = new_bearing;
        }
      }
    }
  }

  if (sm.updated("navRoute") && sm["navRoute"].getNavRoute().getCoordinates().size()) {
    qWarning() << "Got new navRoute from navd. Opening map:" << allow_open;
    // Only open the map on setting destination the first time
    if (allow_open) {
      setVisible(true); // Show map on destination set/change
      allow_open = false;
    }
  }

  if (m_map.isNull()) {
    return;
  }

  loaded_once = loaded_once || m_map->isFullyLoaded();
  if (!loaded_once) {
    map_instructions->setError(tr("Map Loading"));
    return;
  }

  initLayers();

  if (locationd_valid || laikad_valid) {
    map_instructions->setError("");
    // Update current location marker
    auto point = coordinate_to_collection(*last_position);
    QMapbox::Feature feature1(QMapbox::Feature::PointType, point, {}, {});
    QVariantMap carPosSource;
    carPosSource["type"] = "geojson";
    carPosSource["data"] = QVariant::fromValue<QMapbox::Feature>(feature1);
    m_map->updateSource("carPosSource", carPosSource);
  } else {
    map_instructions->setError(tr("Waiting for GPS"));
  }

  if (pan_counter == 0) {
    if (last_position) m_map->setCoordinate(*last_position);
    if (last_bearing) m_map->setBearing(*last_bearing);
  } else {
    pan_counter--;
  }

  if (zoom_counter == 0) {
    m_map->setZoom(util::map_val<float>(velocity_filter.x(), 0, 30, MAX_ZOOM, MIN_ZOOM));
  } else {
    zoom_counter--;
  }

  if (sm.updated("navInstruction")) {
    if (sm.valid("navInstruction")) {
      auto i = sm["navInstruction"].getNavInstruction();
      emit ETAChanged(i.getTimeRemaining(), i.getTimeRemainingTypical(), i.getDistanceRemaining());

      if (locationd_valid || laikad_valid) {
        m_map->setPitch(MAX_PITCH); // TODO: smooth pitching based on maneuver distance
        emit distanceChanged(i.getManeuverDistance()); // TODO: combine with instructionsChanged
        emit instructionsChanged(i);
      }
    } else {
      m_map->setPitch(MIN_PITCH);
      clearRoute();
    }
  }

  if (sm.rcv_frame("navRoute") != route_rcv_frame) {
    qWarning() << "Updating navLayer with new route";
    auto route = sm["navRoute"].getNavRoute();
    auto route_points = capnp_coordinate_list_to_collection(route.getCoordinates());
    QMapbox::Feature feature(QMapbox::Feature::LineStringType, route_points, {}, {});
    QVariantMap navSource;
    navSource["type"] = "geojson";
    navSource["data"] = QVariant::fromValue<QMapbox::Feature>(feature);
    m_map->updateSource("navSource", navSource);
    m_map->setLayoutProperty("navLayer", "visibility", "visible");

    route_rcv_frame = sm.rcv_frame("navRoute");

  }
}

void MapWindow::resizeGL(int w, int h) {
  m_map->resize(size() / MAP_SCALE);
  map_instructions->setFixedSize({w, h});
  map_instructions->eta_doc.setTextWidth(w);
}

void MapWindow::initializeGL() {
  m_map.reset(new QMapboxGL(this, m_settings, size(), 1));

  if (last_position) {
    m_map->setCoordinateZoom(*last_position, MAX_ZOOM);
  } else {
    m_map->setCoordinateZoom(QMapbox::Coordinate(64.31990695292795, -149.79038934046247), MIN_ZOOM);
  }

  m_map->setMargins({0, 350, 0, 50});
  m_map->setPitch(MIN_PITCH);
  m_map->setStyleUrl("mapbox://styles/commaai/ckr64tlwp0azb17nqvr9fj13s");

  QObject::connect(m_map.data(), &QMapboxGL::mapChanged, [=](QMapboxGL::MapChange change) {
    if (change == QMapboxGL::MapChange::MapChangeDidFinishLoadingMap) {
      loaded_once = true;
    }
  });
}

void MapWindow::paintGL() {
  if (!isVisible() || m_map.isNull()) return;
  m_map->render();
}

void MapWindow::clearRoute() {
  if (!m_map.isNull()) {
    m_map->setLayoutProperty("navLayer", "visibility", "none");
    m_map->setPitch(MIN_PITCH);
  }

  map_instructions->eta_doc.setHtml("");
  allow_open = true;
}

void MapWindow::mousePressEvent(QMouseEvent *ev) {
  m_lastPos = ev->localPos();
  ev->accept();
}

void MapWindow::mouseDoubleClickEvent(QMouseEvent *ev) {
  if (last_position) m_map->setCoordinate(*last_position);
  if (last_bearing) m_map->setBearing(*last_bearing);
  m_map->setZoom(util::map_val<float>(velocity_filter.x(), 0, 30, MAX_ZOOM, MIN_ZOOM));
  update();

  pan_counter = 0;
  zoom_counter = 0;
}

void MapWindow::mouseMoveEvent(QMouseEvent *ev) {
  QPointF delta = ev->localPos() - m_lastPos;

  if (!delta.isNull()) {
    pan_counter = PAN_TIMEOUT;
    m_map->moveBy(delta / MAP_SCALE);
    update();
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
  update();

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
    update();
    zoom_counter = PAN_TIMEOUT;
  }
}

void MapWindow::offroadTransition(bool offroad) {
  if (offroad) {
    clearRoute();
  } else {
    auto dest = coordinate_from_param("NavDestination");
    setVisible(dest.has_value());
  }
  last_bearing = {};
}

MapInstructions::MapInstructions(QWidget * parent) : QWidget(parent) {
  setAttribute(Qt::WA_NoSystemBackground);
  setAttribute(Qt::WA_TranslucentBackground);
  setAttribute(Qt::WA_TransparentForMouseEvents);
  eta_doc.setUndoRedoEnabled(false);
  eta_doc.setUseDesignMetrics(true);
  eta_doc.setDefaultTextOption(QTextOption(Qt::AlignHCenter));
  is_rhd = Params().getBool("IsRhdDetected");
}

void MapInstructions::paintEvent(QPaintEvent *event) {
  QPainter p(this);
  p.setPen(Qt::white);

  if (!error_str.isEmpty()) {
    QRect rc(0, 0, width(), 200);
    p.fillRect(rc, QColor(0, 0, 0, 150));
    configFont(p, "Inter", 90, "Regular");
    p.drawText(rc, Qt::AlignCenter, error_str);
    return;
  }

  // draw instructions
  int header_height = drawInstructions(p, true);
  if (header_height > 0) {
    p.fillRect(QRect{0, 0, width(), header_height}, QColor(0, 0, 0, 150));
    drawInstructions(p, false);
  }

  // draw ETA
  if (!eta_doc.isEmpty()) {
    p.translate(0, height() - 100);
    p.setPen(Qt::NoPen);
    p.setBrush(QColor(0, 0, 0, 150));
    qreal txt_width = eta_doc.idealWidth();
    p.drawRoundedRect((width() - txt_width) / 2 - 20, 0, txt_width + 40, 90, 16, 16);
    eta_doc.drawContents(&p);
  }
}

static QRect drawText(QPainter &p, const QRect &rect, const QString &font_family, int font_size, const QString &text, bool draw,
                      int flags = Qt::AlignLeft | Qt::AlignTop | Qt::TextWordWrap) {
  configFont(p, "Inter", font_size, font_family);
  QRect text_rect = p.fontMetrics().boundingRect(rect, flags, text);
  if (draw) p.drawText(text_rect, flags, text);
  return text_rect;
}

int MapInstructions::drawInstructions(QPainter &p, bool draw) {
  const int h_margin = 10, v_margin = 50;
  QRect r = rect().adjusted(h_margin, v_margin, -h_margin, -v_margin);

  int icon_height = 0;
  if (!icon.isNull()) {
    if (draw) p.drawPixmap(h_margin, v_margin, icon);
    r.adjust(icon.width(), 0, 0, 0);
    icon_height = icon.height();
  }

  if (!distance_str.isEmpty()) {
    r.setY(drawText(p, r, "Regular", 90, distance_str, draw).bottom());
  }
  if (!primary_str.isEmpty()) {
    r.setY(drawText(p, r, "Regular", 60, primary_str, draw).bottom());
  }
  if (!secondary_str.isEmpty()) {
    r.setY(drawText(p, r, "Regular", 50, secondary_str, draw).bottom());
  }

  if (!lanes.empty()) {
    if (draw) {
      for (int i = 0, x = r.x(); i < lanes.size(); ++i, x += 125) {
        p.drawPixmap(x, r.y(), loadPixmap(lanes[i], {125, 125}, Qt::IgnoreAspectRatio));
      }
    }
    r.setY(r.y() + 125);
  }
  // returns the total height of the instructions
  return std::max(r.y() > v_margin ? r.y() : 0, icon_height);
}

void MapInstructions::updateDistance(float d) {
  d = std::max(d, 0.0f);

  if (uiState()->scene.is_metric) {
    if (d > 500) {
      distance_str = QString::number(d / 1000, 'f', 1) + tr(" km");
    } else {
      distance_str = QString::number(50 * int(d / 50)) + tr(" m");
    }
  } else {
    float miles = d * METER_TO_MILE;
    float feet = d * METER_TO_FOOT;
    if (feet > 500) {
      distance_str = QString::number(miles, 'f', 1) + tr(" mi");
    } else {
      distance_str = QString::number(50 * int(feet / 50)) + tr(" ft");
    }
  }
  update();
}

void MapInstructions::updateInstructions(cereal::NavInstruction::Reader instruction) {
  primary_str = QString::fromStdString(instruction.getManeuverPrimaryText());
  secondary_str = QString::fromStdString(instruction.getManeuverSecondaryText());

  // Show arrow with direction
  QString type = QString::fromStdString(instruction.getManeuverType());
  QString modifier = QString::fromStdString(instruction.getManeuverModifier());
  if (!type.isEmpty()) {
    QString fn = "../assets/navigation/direction_" + type;
    if (!modifier.isEmpty()) {
      fn += "_" + modifier;
    }
    fn += ICON_SUFFIX;
    fn = fn.replace(' ', '_');

    // for rhd, reflect direction and then flip
    if (is_rhd) {
      if (fn.contains("left")) {
        fn.replace("left", "right");
      } else if (fn.contains("right")) {
        fn.replace("right", "left");
      }
    }

    QPixmap pix(fn);
    if (is_rhd) {
      pix = pix.transformed(QTransform().scale(-1, 1));
    }
    icon = pix;
  }

  // Show lanes
  lanes.clear();
  bool has_lanes = false;
  for (auto const &lane: instruction.getLanes()) {
    has_lanes = true;
    bool active = lane.getActive();

    // TODO: only use active direction if active
    bool left = false, straight = false, right = false;
    for (auto const &direction: lane.getDirections()) {
      left |= direction == cereal::NavInstruction::Direction::LEFT;
      right |= direction == cereal::NavInstruction::Direction::RIGHT;
      straight |= direction == cereal::NavInstruction::Direction::STRAIGHT;
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

    if (!active) {
      fn += "_inactive";
    }
    lanes.push_back(fn + ICON_SUFFIX);
  }
  update();
}

void MapInstructions::updateETA(float s, float s_typical, float d) {
  if (d < MANEUVER_TRANSITION_THRESHOLD) {
    eta_doc.clear();
    return;
  }

  QString eta, eta_unit, time;
  auto eta_time = QDateTime::currentDateTime().addSecs(s).time();
  if (params.getBool("NavSettingTime24h")) {
    eta = eta_time.toString("HH:mm");
    eta_unit = tr("eta");
  } else {
    auto t = eta_time.toString("h:mm a").split(' ');
    eta = t[0];
    eta_unit = t[1];
  }

  // Remaining time
  if (s < 3600) {
    time = QString::number(int(s / 60)) + tr("min");
  } else {
    int hours = int(s) / 3600;
    time = QString::number(hours) + ":" + QString::number(int((s - hours * 3600) / 60)).rightJustified(2, '0') + tr("hr");
  }

  QString color;
  if (s / s_typical > 1.5) {
    color = "#DA3025";
  } else if (s / s_typical > 1.2) {
    color = "#DAA725";
  } else {
    color = "#25DA6E";
  }

  // Distance
  float num = uiState()->scene.is_metric ? (d / 1000.0) : (d * METER_TO_MILE);
  QString unit = uiState()->scene.is_metric ? tr("km") : tr("mi");
  QString distance = QString::number(num, 'f', num < 100 ? 1 : 0) + unit;

  eta_doc.setHtml(QString("<font style=\"font-family:Inner;font-size:70px;color:white;\"><b>%1</b>%2 <font color=\"%3\"><b>%4</b></font>  %5</font>")
  .arg(eta).arg(eta_unit).arg(color).arg(time).arg(distance));
  update();
}
