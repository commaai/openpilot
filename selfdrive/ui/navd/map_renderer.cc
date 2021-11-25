#include "selfdrive/ui/navd/map_renderer.h"

#include <QApplication>
#include <QBuffer>
#include <QDebug>

#include "selfdrive/ui/qt/maps/map_helpers.h"
#include "selfdrive/common/timing.h"

const float ZOOM = 13;
const int WIDTH = 256;
const int HEIGHT = WIDTH;

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

  if (enable_vipc) {
    vipc_server.reset(new VisionIpcServer("navd"));
    vipc_server->create_buffers(VisionStreamType::VISION_STREAM_RGB_MAP, NUM_VIPC_BUFFERS, true, WIDTH, HEIGHT);
    vipc_server->start_listener();

    pm.reset(new PubMaster({"navThumbnail"}));
  }
}

void MapRenderer::updatePosition(QMapbox::Coordinate position, float bearing) {
  if (m_map.isNull()) {
    return;
  }

  m_map->setCoordinate(position);
  m_map->setBearing(bearing);
  update();
}

bool MapRenderer::loaded() {
  return m_map->isFullyLoaded();
}

void MapRenderer::update() {
  gl_functions->glClear(GL_COLOR_BUFFER_BIT);
  m_map->render();
  gl_functions->glFlush();

  sendVipc();
}

void MapRenderer::sendVipc() {
  if (!vipc_server || !loaded()) {
    return;
  }

  QImage cap = fbo->toImage().convertToFormat(QImage::Format_RGB888, Qt::AutoColor);
  uint64_t ts = nanos_since_boot();
  VisionBuf* buf = vipc_server->get_buffer(VisionStreamType::VISION_STREAM_RGB_MAP);
  VisionIpcBufExtra extra = {
    .frame_id = frame_id,
    .timestamp_sof = ts,
    .timestamp_eof = ts,
  };

  assert(cap.sizeInBytes() == buf->len);
  memcpy(buf->addr, cap.bits(), buf->len);
  vipc_server->send(buf, &extra);

  if (frame_id % 100 == 0) {
    // Write jpeg into buffer
    QByteArray buffer_bytes;
    QBuffer buffer(&buffer_bytes);
    buffer.open(QIODevice::WriteOnly);
    cap.save(&buffer, "JPG", 50);

    kj::Array<capnp::byte> buffer_kj = kj::heapArray<capnp::byte>((const capnp::byte*)buffer_bytes.constData(), buffer_bytes.size());

    // Send thumbnail
    MessageBuilder msg;
    auto thumbnaild = msg.initEvent().initNavThumbnail();
    thumbnaild.setFrameId(frame_id);
    thumbnaild.setTimestampEof(ts);
    thumbnaild.setThumbnail(buffer_kj);
    pm->send("navThumbnail", msg);
  }

  frame_id++;
}

uint8_t* MapRenderer::getImage() {
  QImage cap = fbo->toImage().convertToFormat(QImage::Format_RGB888, Qt::AutoColor);
  uint8_t* buf = new uint8_t[cap.sizeInBytes()];
  memcpy(buf, cap.bits(), cap.sizeInBytes());

  return buf;
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

extern "C" {
  MapRenderer* map_renderer_init() {
    char *argv[] = {
      (char*)"navd",
      nullptr
    };
    int argc = 0;
    QApplication *app = new QApplication(argc, argv);
    assert(app);

    QMapboxGLSettings settings;
    settings.setApiBaseUrl(MAPS_HOST);
    settings.setAccessToken(get_mapbox_token());

    return new MapRenderer(settings, false);
  }

  void map_renderer_update_position(MapRenderer *inst, float lat, float lon, float bearing) {
    inst->updatePosition({lat, lon}, bearing);
    QApplication::processEvents();
  }

  void map_renderer_update(MapRenderer *inst) {
    inst->update();
  }

  void map_renderer_process(MapRenderer *inst) {
    QApplication::processEvents();
  }

  bool map_renderer_loaded(MapRenderer *inst) {
    return inst->loaded();
  }

  uint8_t * map_renderer_get_image(MapRenderer *inst) {
    return inst->getImage();
  }

  void map_renderer_free_image(MapRenderer *inst, uint8_t * buf) {
    delete[] buf;
  }
}
