#include "selfdrive/ui/navd/simple_map.h"

#include <QDebug>

#include "selfdrive/ui/qt/maps/map_helpers.h"

const float ZOOM = 14;
const int WIDTH = 512;
const int HEIGHT = 512;

SimpleMap::SimpleMap(const QMapboxGLSettings &settings) : m_settings(settings) {
  QSurfaceFormat fmt;
  fmt.setRenderableType(QSurfaceFormat::OpenGLES);

  ctx = std::make_unique<QOpenGLContext>();
  ctx->setFormat(fmt);
  ctx->create();
  assert(ctx->isValid());

  surface = std::make_unique<QOffscreenSurface>();
  surface->setFormat(ctx->format());
  surface->create();

  ctx->makeCurrent(surface.get());
  assert(QOpenGLContext::currentContext() == ctx.get());

  gl_functions.reset(ctx->functions());
  gl_functions->initializeOpenGLFunctions();

  QOpenGLFramebufferObjectFormat fbo_format;
  fbo.reset(new QOpenGLFramebufferObject(WIDTH, HEIGHT, fbo_format));

  m_map.reset(new QMapboxGL(nullptr, m_settings, fbo->size(), 1));
  m_map->setCoordinateZoom(QMapbox::Coordinate(0, 0), ZOOM);
  m_map->setStyleUrl("mapbox://styles/commaai/ckvmksrpd4n0a14pfdo5heqzr");
  m_map->createRenderer();

  m_map->resize(fbo->size());
  m_map->setFramebufferObject(fbo->handle(), fbo->size());
  gl_functions->glViewport(0, 0, WIDTH, HEIGHT);

  QObject::connect(m_map.data(), &QMapboxGL::mapChanged, [=](QMapboxGL::MapChange change) {
    if (change == QMapboxGL::MapChange::MapChangeDidFinishLoadingMap) {
      loaded_once = true;
    }
  });
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
void SimpleMap::update() {
  gl_functions->glClear(GL_COLOR_BUFFER_BIT);
  m_map->render();

  // Save to png
  static int fn = 0;
  QImage cap = fbo->toImage().convertToFormat(QImage::Format_RGB888, Qt::AutoColor);
  char tmp[100];
  snprintf(tmp, sizeof(tmp)-1, "/tmp/cap/%04d.png", fn++);
  cap.save(tmp);
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

// void SimpleMap::initializeGL() {
// }

SimpleMap::~SimpleMap() {
  // makeCurrent();
}
