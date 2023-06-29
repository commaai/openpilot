#include "selfdrive/ui/qt/maps/map.h"

#include <eigen3/Eigen/Dense>
#include <cmath>

#include <QDebug>
#include <QFileInfo>
#include <QPainterPath>

#include "common/swaglog.h"
#include "common/transformations/coordinates.hpp"
#include "selfdrive/ui/qt/maps/map_helpers.h"
#include "selfdrive/ui/qt/request_repeater.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/ui.h"


const int PAN_TIMEOUT = 100;
const float MANEUVER_TRANSITION_THRESHOLD = 10;

const float MAX_ZOOM = 17;
const float MIN_ZOOM = 14;
const float MAX_PITCH = 50;
const float MIN_PITCH = 0;
const float MAP_SCALE = 2;

const QString ICON_SUFFIX = ".png";

MapWindow::MapWindow(const QMapboxGLSettings &settings) : m_settings(settings), velocity_filter(0, 10, 0.05) {
  QObject::connect(uiState(), &UIState::uiUpdate, this, &MapWindow::updateState);

  // Instructions
  map_instructions = new MapInstructions(this);
  QObject::connect(this, &MapWindow::instructionsChanged, map_instructions, &MapInstructions::updateInstructions);
  QObject::connect(this, &MapWindow::distanceChanged, map_instructions, &MapInstructions::updateDistance);
  map_instructions->setFixedWidth(width());
  map_instructions->setVisible(false);

  map_eta = new MapETA(this);
  QObject::connect(this, &MapWindow::ETAChanged, map_eta, &MapETA::updateETA);

  const int h = 120;
  map_eta->setFixedHeight(h);
  map_eta->move(25, 1080 - h - bdr_s*2);
  map_eta->setVisible(false);

  // Settings button
  QSize icon_size(120, 120);
  directions_icon = loadPixmap("../assets/navigation/icon_directions_outlined.svg", icon_size);
  settings_icon = loadPixmap("../assets/navigation/icon_settings.svg", icon_size);

  settings_btn = new QPushButton(directions_icon, "", this);
  settings_btn->setIconSize(icon_size);
  settings_btn->setStyleSheet(R"(
    QPushButton {
      background-color: #96000000;
      border-radius: 50px;
      padding: 24px;
    }
    QPushButton:pressed {
      background-color: #D9000000;
    }
  )");
  settings_btn->show();  // force update
  settings_btn->move(bdr_s, 1080 - bdr_s*3 - settings_btn->height());
  QObject::connect(settings_btn, &QPushButton::clicked, [=]() {
    emit openSettings();
  });

  auto last_gps_position = coordinate_from_param("LastGPSPosition");
  if (last_gps_position.has_value()) {
    last_position = *last_gps_position;
  }

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
    m_map->addAnnotationIcon("default_marker", QImage("../assets/navigation/default_marker.svg"));
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

    // Check std norm
    auto pos_ecef_std = locationd_location.getPositionECEF().getStd();
    bool pos_accurate_enough = sqrt(pow(pos_ecef_std[0], 2) + pow(pos_ecef_std[1], 2) + pow(pos_ecef_std[2], 2)) < 100;

    locationd_valid = (locationd_pos.getValid() && locationd_orientation.getValid() && locationd_velocity.getValid() && pos_accurate_enough);

    if (locationd_valid) {
      last_position = QMapbox::Coordinate(locationd_pos.getValue()[0], locationd_pos.getValue()[1]);
      last_bearing = RAD2DEG(locationd_orientation.getValue()[2]);
      velocity_filter.update(locationd_velocity.getValue()[0]);
    }
  }

  if (sm.updated("navRoute") && sm["navRoute"].getNavRoute().getCoordinates().size()) {
    qWarning() << "Got new navRoute from navd. Opening map:" << allow_open;

    // Only open the map on setting destination the first time
    if (allow_open) {
      emit requestVisible(true); // Show map on destination set/change
      allow_open = false;
    }
  }

  if (m_map.isNull()) {
    return;
  }

  loaded_once = loaded_once || m_map->isFullyLoaded();
  if (!loaded_once) {
    map_instructions->showError(tr("Map Loading"));
    return;
  }

  initLayers();

  if (locationd_valid) {
    map_instructions->noError();

    // Update current location marker
    auto point = coordinate_to_collection(*last_position);
    QMapbox::Feature feature1(QMapbox::Feature::PointType, point, {}, {});
    QVariantMap carPosSource;
    carPosSource["type"] = "geojson";
    carPosSource["data"] = QVariant::fromValue<QMapbox::Feature>(feature1);
    m_map->updateSource("carPosSource", carPosSource);
  } else {
    map_instructions->showError(tr("Waiting for GPS"));
  }

  if (pan_counter == 0) {
    if (last_position) m_map->setCoordinate(*last_position);
    if (last_bearing) m_map->setBearing(*last_bearing);
  } else {
    pan_counter--;
  }

  if (zoom_counter == 0) {
    m_map->setZoom(util::map_val<float>(velocity_filter.x(), 0, 30, MAX_ZOOM, MIN_ZOOM));
    zoom_counter = -1;
  } else if (zoom_counter > 0) {
    zoom_counter--;
  }

  if (sm.updated("navInstruction")) {
    if (sm.valid("navInstruction")) {
      auto i = sm["navInstruction"].getNavInstruction();
      emit ETAChanged(i.getTimeRemaining(), i.getTimeRemainingTypical(), i.getDistanceRemaining());

      if (locationd_valid) {
        m_map->setPitch(MAX_PITCH); // TODO: smooth pitching based on maneuver distance
        emit distanceChanged(i.getManeuverDistance()); // TODO: combine with instructionsChanged
        emit instructionsChanged(i);
      }
    } else {
      clearRoute();
    }

    // TODO: only move if position should change
    // don't move while map isn't visible
    if (isVisible()) {
      auto pos = 1080 - bdr_s*2 - settings_btn->height() - bdr_s;
      if (map_eta->isVisible()) {
        settings_btn->move(bdr_s, pos - map_eta->height());
        settings_btn->setIcon(settings_icon);
      } else {
        settings_btn->move(bdr_s, pos);
        settings_btn->setIcon(directions_icon);
      }
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
    updateDestinationMarker();
  }
}

void MapWindow::resizeGL(int w, int h) {
  m_map->resize(size() / MAP_SCALE);
  map_instructions->setFixedWidth(width());
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
  m_map->setStyleUrl("mapbox://styles/commaai/clj7g5vrp007b01qzb5ro0i4j");

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
    updateDestinationMarker();
  }

  map_instructions->hideIfNoError();
  map_eta->setVisible(false);
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
    emit requestVisible(dest.has_value());
  }
  last_bearing = {};
}

void MapWindow::updateDestinationMarker() {
  if (marker_id != -1) {
    m_map->removeAnnotation(marker_id);
    marker_id = -1;
  }

  auto nav_dest = coordinate_from_param("NavDestination");
  if (nav_dest.has_value()) {
    auto ano = QMapbox::SymbolAnnotation {*nav_dest, "default_marker"};
    marker_id = m_map->addAnnotation(QVariant::fromValue<QMapbox::SymbolAnnotation>(ano));
  }
}

MapInstructions::MapInstructions(QWidget * parent) : QWidget(parent) {
  is_rhd = Params().getBool("IsRhdDetected");
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
    QVBoxLayout *layout = new QVBoxLayout;

    distance = new QLabel;
    distance->setStyleSheet(R"(font-size: 90px;)");
    layout->addWidget(distance);

    primary = new QLabel;
    primary->setStyleSheet(R"(font-size: 60px;)");
    primary->setWordWrap(true);
    layout->addWidget(primary);

    secondary = new QLabel;
    secondary->setStyleSheet(R"(font-size: 50px;)");
    secondary->setWordWrap(true);
    layout->addWidget(secondary);

    lane_widget = new QWidget;
    lane_widget->setFixedHeight(125);

    lane_layout = new QHBoxLayout(lane_widget);
    layout->addWidget(lane_widget);

    main_layout->addLayout(layout);
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
  d = std::max(d, 0.0f);
  QString distance_str;

  if (uiState()->scene.is_metric) {
    if (d > 500) {
      distance_str.setNum(d / 1000, 'f', 1);
      distance_str += tr(" km");
    } else {
      distance_str.setNum(50 * int(d / 50));
      distance_str += tr(" m");
    }
  } else {
    float miles = d * METER_TO_MILE;
    float feet = d * METER_TO_FOOT;

    if (feet > 500) {
      distance_str.setNum(miles, 'f', 1);
      distance_str += tr(" mi");
    } else {
      distance_str.setNum(50 * int(feet / 50));
      distance_str += tr(" ft");
    }
  }

  distance->setAlignment(Qt::AlignLeft);
  distance->setText(distance_str);
}

void MapInstructions::showError(QString error_text) {
  primary->setText("");
  distance->setText(error_text);
  distance->setAlignment(Qt::AlignCenter);

  secondary->setVisible(false);
  icon_01->setVisible(false);

  this->error = true;
  lane_widget->setVisible(false);

  setVisible(true);
}

void MapInstructions::noError() {
  error = false;
}

void MapInstructions::updateInstructions(cereal::NavInstruction::Reader instruction) {
  // Word wrap widgets need fixed width
  primary->setFixedWidth(width() - 250);
  secondary->setFixedWidth(width() - 250);


  // Show instruction text
  QString primary_str = QString::fromStdString(instruction.getManeuverPrimaryText());
  QString secondary_str = QString::fromStdString(instruction.getManeuverSecondaryText());

  primary->setText(primary_str);
  secondary->setVisible(secondary_str.length() > 0);
  secondary->setText(secondary_str);

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
    icon_01->setPixmap(pix.scaledToWidth(200, Qt::SmoothTransformation));
    icon_01->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
    icon_01->setVisible(true);
  }

  // Show lanes
  bool has_lanes = false;
  clearLayout(lane_layout);
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

    auto icon = new QLabel;
    icon->setPixmap(loadPixmap(fn + ICON_SUFFIX, {125, 125}, Qt::IgnoreAspectRatio));
    icon->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
    lane_layout->addWidget(icon);
  }
  lane_widget->setVisible(has_lanes);

  show();
  resize(sizeHint());
}


void MapInstructions::hideIfNoError() {
  if (!error) {
    hide();
  }
}

MapETA::MapETA(QWidget * parent) : QWidget(parent) {
  QHBoxLayout *main_layout = new QHBoxLayout(this);
  main_layout->setContentsMargins(40, 25, 40, 25);

  {
    QHBoxLayout *layout = new QHBoxLayout;
    eta = new QLabel;
    eta->setAlignment(Qt::AlignCenter);
    eta->setStyleSheet("font-weight:600");

    eta_unit = new QLabel;
    eta_unit->setAlignment(Qt::AlignCenter);

    layout->addWidget(eta);
    layout->addWidget(eta_unit);
    main_layout->addLayout(layout);
  }
  main_layout->addSpacing(40);
  {
    QHBoxLayout *layout = new QHBoxLayout;
    time = new QLabel;
    time->setAlignment(Qt::AlignCenter);

    time_unit = new QLabel;
    time_unit->setAlignment(Qt::AlignCenter);

    layout->addWidget(time);
    layout->addWidget(time_unit);
    main_layout->addLayout(layout);
  }
  main_layout->addSpacing(40);
  {
    QHBoxLayout *layout = new QHBoxLayout;
    distance = new QLabel;
    distance->setAlignment(Qt::AlignCenter);
    distance->setStyleSheet("font-weight:600");

    distance_unit = new QLabel;
    distance_unit->setAlignment(Qt::AlignCenter);

    layout->addWidget(distance);
    layout->addWidget(distance_unit);
    main_layout->addLayout(layout);
  }

  setStyleSheet(R"(
    * {
      color: white;
      font-family: "Inter";
      font-size: 70px;
    }
  )");

  QPalette pal = palette();
  pal.setColor(QPalette::Background, QColor(0, 0, 0, 150));
  setAutoFillBackground(true);
  setPalette(pal);
}


void MapETA::updateETA(float s, float s_typical, float d) {
  if (d < MANEUVER_TRANSITION_THRESHOLD) {
    hide();
    return;
  }

  // ETA
  auto eta_time = QDateTime::currentDateTime().addSecs(s).time();
  if (params.getBool("NavSettingTime24h")) {
    eta->setText(eta_time.toString("HH:mm"));
    eta_unit->setText(tr("eta"));
  } else {
    auto t = eta_time.toString("h:mm a").split(' ');
    eta->setText(t[0]);
    eta_unit->setText(t[1]);
  }

  // Remaining time
  if (s < 3600) {
    time->setText(QString::number(int(s / 60)));
    time_unit->setText(tr("min"));
  } else {
    int hours = int(s) / 3600;
    time->setText(QString::number(hours) + ":" + QString::number(int((s - hours * 3600) / 60)).rightJustified(2, '0'));
    time_unit->setText(tr("hr"));
  }

  QString color;
  if (s / s_typical > 1.5) {
    color = "#DA3025";
  } else if (s / s_typical > 1.2) {
    color = "#DAA725";
  } else {
    color = "#25DA6E";
  }

  time->setStyleSheet(QString(R"(color: %1; font-weight:600;)").arg(color));
  time_unit->setStyleSheet(QString(R"(color: %1;)").arg(color));

  // Distance
  QString distance_str;
  float num = 0;
  if (uiState()->scene.is_metric) {
    num = d / 1000.0;
    distance_unit->setText(tr("km"));
  } else {
    num = d * METER_TO_MILE;
    distance_unit->setText(tr("mi"));
  }

  distance_str.setNum(num, 'f', num < 100 ? 1 : 0);
  distance->setText(distance_str);

  show();
  adjustSize();
  repaint();
  adjustSize();

  // Rounded corners
  const int radius = 25;
  const auto r = rect();

  // Top corners rounded
  QPainterPath path;
  path.setFillRule(Qt::WindingFill);
  path.addRoundedRect(r, radius, radius);

  // Bottom corners not rounded
  path.addRect(r.marginsRemoved(QMargins(0, radius, 0, 0)));

  // Set clipping mask
  QRegion mask = QRegion(path.simplified().toFillPolygon().toPolygon());
  setMask(mask);

  // Center
  move(static_cast<QWidget*>(parent())->width() / 2 - width() / 2, 1080 - height() - bdr_s*2);
}
