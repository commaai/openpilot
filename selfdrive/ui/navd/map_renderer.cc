#include "selfdrive/ui/navd/map_renderer.h"

#include <QDebug>

#include "selfdrive/ui/qt/maps/map_helpers.h"
#include "selfdrive/common/timing.h"

const float ZOOM = 14;
const int WIDTH = 512;
const int HEIGHT = 512;

const int NUM_VIPC_BUFFERS = 4;

MapRenderer::MapRenderer(const QMapboxGLSettings &settings, bool enable_vipc) : m_settings(settings) {
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

  if (enable_vipc) {
    vipc_server.reset(new VisionIpcServer("navd"));
    vipc_server->create_buffers(VisionStreamType::VISION_STREAM_RGB_MAP, NUM_VIPC_BUFFERS, true, WIDTH, HEIGHT);
    vipc_server->start_listener();
  }
}

void MapRenderer::updatePosition(QMapbox::Coordinate position, float bearing) {
  if (m_map.isNull()) {
    return;
  }

  loaded_once = loaded_once || m_map->isFullyLoaded();
  m_map->setCoordinate(position);
  m_map->setBearing(bearing);
  update();
}

void MapRenderer::update() {
  gl_functions->glClear(GL_COLOR_BUFFER_BIT);
  m_map->render();

  if (vipc_server) {
    QImage cap = fbo->toImage().convertToFormat(QImage::Format_RGB888, Qt::AutoColor);
    uint64_t ts = nanos_since_boot();
    VisionBuf* buf = vipc_server->get_buffer(VisionStreamType::VISION_STREAM_RGB_MAP);
    VisionIpcBufExtra extra = {
      .frame_id = frame_id++,
      .timestamp_sof = ts,
      .timestamp_eof = ts,
    };

    assert(cap.sizeInBytes() == buf->len);
    memcpy(buf->addr, cap.bits(), buf->len);
    vipc_server->send(buf, &extra);
  }
}

void MapRenderer::updateRoute(QList<QGeoCoordinate> coordinates) {
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

void MapRenderer::initLayers() {
  if (!m_map->layerExists("navLayer")) {
    QVariantMap nav;
    nav["id"] = "navLayer";
    nav["type"] = "line";
    nav["source"] = "navSource";
    m_map->addLayer(nav, "road-intersection");
    m_map->setPaintProperty("navLayer", "line-color", QColor("blue"));
    m_map->setPaintProperty("navLayer", "line-width", 3);
    m_map->setLayoutProperty("navLayer", "line-cap", "round");
  }
}

MapRenderer::~MapRenderer() {
}
