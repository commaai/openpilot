#include "selfdrive/ui/navd/simple_map.h"

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
  if (!isVisible() || m_map.isNull()) return;
  m_map->render();
}

void SimpleMap::initializeGL() {
  m_map.reset(new QMapboxGL(this, m_settings, size(), 1));

  m_map->setCoordinateZoom(QMapbox::Coordinate(0, 0), ZOOM);
  m_map->setStyleUrl("mapbox://styles/commaai/ckr64tlwp0azb17nqvr9fj13s");

  QObject::connect(m_map.data(), &QMapboxGL::mapChanged, [=](QMapboxGL::MapChange change) {
    if (change == QMapboxGL::MapChange::MapChangeDidFinishLoadingMap) {
      loaded_once = true;
    }
  });
}

SimpleMap::~SimpleMap() {
  makeCurrent();
}