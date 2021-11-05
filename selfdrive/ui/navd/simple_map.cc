#include "selfdrive/ui/navd/simple_map.h"


#include "selfdrive/ui/qt/maps/map_helpers.h"

const float ZOOM = 14;

SimpleMap::SimpleMap(const QMapboxGLSettings &settings) : m_settings(settings) {
}


void SimpleMap::updatePosition(QMapbox::Coordinate position, float bearing) {
  if (m_map.isNull()) {
    return;
  }

  loaded_once = loaded_once || m_map->isFullyLoaded();
  m_map->setCoordinate(position);
  m_map->setBearing(bearing);
  update();
}

void SimpleMap::paintGL() {
  if (m_map.isNull()) return;
  m_map->render();
}

void SimpleMap::updateRoute(QList<QGeoCoordinate> coordinates) {
  if (m_map.isNull()) return;
  initLayers();

  auto route_points = coordinate_list_to_collection(coordinates);
  QMapbox::Feature feature(QMapbox::Feature::LineStringType, route_points, {}, {});
  QVariantMap navSource;
  navSource["type"] = "geojson";
  navSource["data"] = QVariant::fromValue<QMapbox::Feature>(feature);
  m_map->updateSource("navSource", navSource);
  m_map->setLayoutProperty("navLayer", "visibility", "visible");
}

void SimpleMap::initLayers() {
  // This doesn't work from initializeGL
  if (!m_map->layerExists("navLayer")) {
    QVariantMap nav;
    nav["id"] = "navLayer";
    nav["type"] = "line";
    nav["source"] = "navSource";
    m_map->addLayer(nav, "road-intersection");
    m_map->setPaintProperty("navLayer", "line-color", QColor("red"));
    m_map->setPaintProperty("navLayer", "line-width", 7.5);
    m_map->setLayoutProperty("navLayer", "line-cap", "round");
  }
}

void SimpleMap::initializeGL() {
  m_map.reset(new QMapboxGL(this, m_settings, size(), 1));

  m_map->setCoordinateZoom(QMapbox::Coordinate(0, 0), ZOOM);
  m_map->setStyleUrl("mapbox://styles/commaai/ckvmksrpd4n0a14pfdo5heqzr");

  QObject::connect(m_map.data(), &QMapboxGL::mapChanged, [=](QMapboxGL::MapChange change) {
    if (change == QMapboxGL::MapChange::MapChangeDidFinishLoadingMap) {
      loaded_once = true;
    }
  });
}

SimpleMap::~SimpleMap() {
  makeCurrent();
}