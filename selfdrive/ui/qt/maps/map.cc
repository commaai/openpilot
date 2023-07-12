#include "selfdrive/ui/qt/maps/map.h"

#include <eigen3/Eigen/Dense>

#include <QDebug>

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

  map_overlay = new QWidget (this);
  map_overlay->setAttribute(Qt::WA_TranslucentBackground, true);
  QVBoxLayout *overlay_layout = new QVBoxLayout(map_overlay);
  overlay_layout->setContentsMargins(0, 0, 0, 0);

  // Instructions
  map_instructions = new MapInstructions(this);
  map_instructions->setVisible(false);

  map_eta = new MapETA(this);
  map_eta->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  map_eta->setFixedHeight(120);

  // Settings button
  QSize icon_size(120, 120);
  directions_icon = loadPixmap("../assets/navigation/icon_directions_outlined.svg", icon_size);
  settings_icon = loadPixmap("../assets/navigation/icon_settings.svg", icon_size);

  settings_btn = new QPushButton(directions_icon, "", this);
  settings_btn->setIconSize(icon_size);
  settings_btn->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
  settings_btn->setStyleSheet(R"(
    QPushButton {
      background-color: #96000000;
      border-radius: 50px;
      padding: 24px;
      margin-left: 30px;
    }
    QPushButton:pressed {
      background-color: #D9000000;
    }
  )");
  QObject::connect(settings_btn, &QPushButton::clicked, [=]() {
    emit requestSettings(true);
  });

  overlay_layout->addWidget(map_instructions);
  overlay_layout->addStretch(1);
  overlay_layout->addWidget(settings_btn, Qt::AlignLeft);
  overlay_layout->addSpacing(UI_BORDER_SIZE);
  overlay_layout->addWidget(map_eta);

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

  // update navigate on openpilot status
  if (sm.updated("modelV2")) {
    bool nav_enabled = sm["modelV2"].getModelV2().getNavEnabled();
    if (nav_enabled && !uiState()->scene.navigate_on_openpilot) {
      emit requestVisible(true);  // Show map on rising edge of navigate on openpilot
    }
    uiState()->scene.navigate_on_openpilot = nav_enabled;
  }

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
    emit requestSettings(false);
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
  } else {
    zoom_counter--;
  }

  if (sm.updated("navInstruction")) {
    if (sm.valid("navInstruction")) {
      auto i = sm["navInstruction"].getNavInstruction();
      map_eta->updateETA(i.getTimeRemaining(), i.getTimeRemainingTypical(), i.getDistanceRemaining());

      if (locationd_valid) {
        m_map->setPitch(MAX_PITCH); // TODO: smooth pitching based on maneuver distance
        map_instructions->updateInstructions(i);
      }
    } else {
      clearRoute();
    }

    if (isVisible()) {
      settings_btn->setIcon(map_eta->isVisible() ? settings_icon : directions_icon);
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
  map_overlay->setFixedSize(width(), height());
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
    uiState()->scene.navigate_on_openpilot = false;
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

  setStyleSheet("color:white");

  QPalette pal = palette();
  pal.setColor(QPalette::Background, QColor(0, 0, 0, 150));
  setAutoFillBackground(true);
  setPalette(pal);
}

QString MapInstructions::getDistance(float d) {
  d = std::max(d, 0.0f);
  if (uiState()->scene.is_metric) {
    return (d > 500) ? QString::number(d / 1000, 'f', 1) + tr(" km")
                     : QString::number(50 * int(d / 50)) + tr(" m");
  } else {
    float feet = d * METER_TO_FOOT;
    return (feet > 500) ? QString::number(d * METER_TO_MILE, 'f', 1) + tr(" mi")
                        : QString::number(50 * int(feet / 50)) + tr(" ft");
  }
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
  setUpdatesEnabled(false);

  // Show instruction text
  QString primary_str = QString::fromStdString(instruction.getManeuverPrimaryText());
  QString secondary_str = QString::fromStdString(instruction.getManeuverSecondaryText());

  primary->setText(primary_str);
  secondary->setVisible(secondary_str.length() > 0);
  secondary->setText(secondary_str);
  distance->setAlignment(Qt::AlignLeft);
  distance->setText(getDistance(instruction.getManeuverDistance()));

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
  auto lanes = instruction.getLanes();
  for (int i = 0; i < lanes.size(); ++i) {
    bool active = lanes[i].getActive();

    // TODO: only use active direction if active
    bool left = false, straight = false, right = false;
    for (auto const &direction: lanes[i].getDirections()) {
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

    QLabel *label = (i < lane_labels.size()) ? lane_labels[i] : lane_labels.emplace_back(new QLabel);
    if (!label->parentWidget()) {
      lane_layout->addWidget(label);
    }
    label->setPixmap(loadPixmap(fn + ICON_SUFFIX, {125, 125}, Qt::IgnoreAspectRatio));
    label->setVisible(true);
  }

  for (int i = lanes.size(); i < lane_labels.size(); ++i) {
    lane_labels[i]->setVisible(false);
  }
  lane_widget->setVisible(lanes.size() > 0);

  setUpdatesEnabled(true);
  setVisible(true);
}


void MapInstructions::hideIfNoError() {
  if (!error) {
    hide();
  }
}

MapETA::MapETA(QWidget *parent) : QWidget(parent) {
  setVisible(false);
  setAttribute(Qt::WA_TranslucentBackground);
  eta_doc.setUndoRedoEnabled(false);
  eta_doc.setDefaultStyleSheet("body {font-family:Inter;font-size:60px;color:white;} b{font-size:70px;font-weight:600}");
}

void MapETA::paintEvent(QPaintEvent *event) {
  if (!eta_doc.isEmpty()) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    p.setPen(Qt::NoPen);
    p.setBrush(QColor(0, 0, 0, 150));
    QSizeF txt_size = eta_doc.size();
    p.drawRoundedRect((width() - txt_size.width()) / 2 - UI_BORDER_SIZE, 0, txt_size.width() + UI_BORDER_SIZE * 2, height() + 25, 25, 25);
    p.translate((width() - txt_size.width()) / 2, (height() - txt_size.height()) / 2);
    eta_doc.drawContents(&p);
  }
}

void MapETA::updateETA(float s, float s_typical, float d) {
  // ETA
  auto eta_t = QDateTime::currentDateTime().addSecs(s).time();
  auto eta = format_24h ? std::array{eta_t.toString("HH:mm"), tr("eta")}
                        : std::array{eta_t.toString("h:mm a").split(' ')[0], eta_t.toString("a")};

  // Remaining time
  auto time_t = QDateTime::fromTime_t(s);
  auto remaining = s < 3600 ? std::array{time_t.toString("m"), tr("min")}
                            : std::array{time_t.toString("h:mm"), tr("hr")};
  QString color = "#25DA6E";
  if (s / s_typical > 1.5) color = "#DA3025";
  else if (s / s_typical > 1.2) color = "#DAA725";

  // Distance
  float num = uiState()->scene.is_metric ? (d / 1000.0) : (d * METER_TO_MILE);
  auto distance = std::array{QString::number(num, 'f', num < 100 ? 1 : 0),
                             uiState()->scene.is_metric ? tr("km") : tr("mi")};

  eta_doc.setHtml(QString(R"(<body><b>%1</b>%2 <span style="color:%3"><b>%4</b>%5</span> <b>%6</b>%7</body>)")
                      .arg(eta[0], eta[1], color, remaining[0], remaining[1], distance[0], distance[1]));

  setVisible(d >= MANEUVER_TRANSITION_THRESHOLD);
  update();
}
