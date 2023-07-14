#include "selfdrive/navd/map_renderer.h"

#include <cmath>
#include <string>
#include <QApplication>
#include <QBuffer>

#include "common/util.h"
#include "common/timing.h"
#include "common/swaglog.h"
#include "selfdrive/ui/qt/maps/map_helpers.h"

const float DEFAULT_ZOOM = 13.5; // Don't go below 13 or features will start to disappear
const int HEIGHT = 256, WIDTH = 256;
const int NUM_VIPC_BUFFERS = 4;

const int EARTH_CIRCUMFERENCE_METERS = 40075000;
const int EARTH_RADIUS_METERS = 6378137;
const int PIXELS_PER_TILE = 256;
const int MAP_OFFSET = 128;

const bool TEST_MODE = getenv("MAP_RENDER_TEST_MODE");
const int LLK_DECIMATION = TEST_MODE ? 1 : 10;

float get_zoom_level_for_scale(float lat, float meters_per_pixel) {
  float meters_per_tile = meters_per_pixel * PIXELS_PER_TILE;
  float num_tiles = cos(DEG2RAD(lat)) * EARTH_CIRCUMFERENCE_METERS / meters_per_tile;
  return log2(num_tiles) - 1;
}

QMapbox::Coordinate get_point_along_line(float lat, float lon, float bearing, float dist) {
  float ang_dist = dist / EARTH_RADIUS_METERS;
  float lat1 = DEG2RAD(lat), lon1 = DEG2RAD(lon), bearing1 = DEG2RAD(bearing);
  float lat2 = asin(sin(lat1)*cos(ang_dist) + cos(lat1)*sin(ang_dist)*cos(bearing1));
  float lon2 = lon1 + atan2(sin(bearing1)*sin(ang_dist)*cos(lat1), cos(ang_dist)-sin(lat1)*sin(lat2));
  return QMapbox::Coordinate(RAD2DEG(lat2), RAD2DEG(lon2));
}


MapRenderer::MapRenderer(const QMapboxGLSettings &settings, bool online) : m_settings(settings) {
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

  std::string style = util::read_file(STYLE_PATH);
  m_map.reset(new QMapboxGL(nullptr, m_settings, fbo->size(), 1));
  m_map->setCoordinateZoom(QMapbox::Coordinate(0, 0), DEFAULT_ZOOM);
  m_map->setStyleJson(style.c_str());
  m_map->createRenderer();

  m_map->resize(fbo->size());
  m_map->setFramebufferObject(fbo->handle(), fbo->size());
  gl_functions->glViewport(0, 0, WIDTH, HEIGHT);

  QObject::connect(m_map.data(), &QMapboxGL::mapChanged, [=](QMapboxGL::MapChange change) {
    // https://github.com/mapbox/mapbox-gl-native/blob/cf734a2fec960025350d8de0d01ad38aeae155a0/platform/qt/include/qmapboxgl.hpp#L116
    //LOGD("new state %d", change);
  });

  QObject::connect(m_map.data(), &QMapboxGL::mapLoadingFailed, [=](QMapboxGL::MapLoadingFailure err_code, const QString &reason) {
    LOGE("Map loading failed with %d: '%s'\n", err_code, reason.toStdString().c_str());
  });

  if (online) {
    vipc_server.reset(new VisionIpcServer("navd"));
    vipc_server->create_buffers(VisionStreamType::VISION_STREAM_MAP, NUM_VIPC_BUFFERS, false, WIDTH, HEIGHT);
    vipc_server->start_listener();

    pm.reset(new PubMaster({"navThumbnail", "mapRenderState"}));
    sm.reset(new SubMaster({"liveLocationKalman", "navRoute"}, {"liveLocationKalman"}));

    timer = new QTimer(this);
    timer->setSingleShot(true);
    QObject::connect(timer, SIGNAL(timeout()), this, SLOT(msgUpdate()));
    timer->start(0);
  }
}

void MapRenderer::msgUpdate() {
  sm->update(1000);

  if (sm->updated("liveLocationKalman")) {
    auto location = (*sm)["liveLocationKalman"].getLiveLocationKalman();
    auto pos = location.getPositionGeodetic();
    auto orientation = location.getCalibratedOrientationNED();

    if ((sm->rcv_frame("liveLocationKalman") % LLK_DECIMATION) == 0) {
      float bearing = RAD2DEG(orientation.getValue()[2]);
      updatePosition(get_point_along_line(pos.getValue()[0], pos.getValue()[1], bearing, MAP_OFFSET), bearing);

      // TODO: use the static rendering mode
      if (!loaded() && frame_id > 0) {
        for (int i = 0; i < 5 && !loaded(); i++) {
          LOGW("map render retry #%d, %d", i+1, m_map.isNull());
          QApplication::processEvents(QEventLoop::AllEvents, 100);
          update();
        }

        if (!loaded()) {
          LOGE("failed to render map after retry");
          publish(0, false);
        }
      }
    }
  }

  if (sm->updated("navRoute")) {
    QList<QGeoCoordinate> route;
    auto coords = (*sm)["navRoute"].getNavRoute().getCoordinates();
    for (auto const &c : coords) {
      route.push_back(QGeoCoordinate(c.getLatitude(), c.getLongitude()));
    }
    updateRoute(route);
  }

  // schedule next update
  timer->start(0);
}

void MapRenderer::updatePosition(QMapbox::Coordinate position, float bearing) {
  if (m_map.isNull()) {
    return;
  }

  // Choose a scale that ensures above 13 zoom level up to and above 75deg of lat
  float meters_per_pixel = 2;
  float zoom = get_zoom_level_for_scale(position.first, meters_per_pixel);

  m_map->setCoordinate(position);
  m_map->setBearing(bearing);
  m_map->setZoom(zoom);
  update();
}

bool MapRenderer::loaded() {
  return m_map->isFullyLoaded();
}

void MapRenderer::update() {
  double start_t = millis_since_boot();
  gl_functions->glClear(GL_COLOR_BUFFER_BIT);
  m_map->render();
  gl_functions->glFlush();
  double end_t = millis_since_boot();

  if ((vipc_server != nullptr) && loaded()) {
    publish((end_t - start_t) / 1000.0, true);
  }
}

void MapRenderer::sendThumbnail(const uint64_t ts, const kj::Array<capnp::byte> &buf) {
  MessageBuilder msg;
  auto thumbnaild = msg.initEvent().initNavThumbnail();
  thumbnaild.setFrameId(frame_id);
  thumbnaild.setTimestampEof(ts);
  thumbnaild.setThumbnail(buf);
  pm->send("navThumbnail", msg);
}

void MapRenderer::publish(const double render_time, const bool loaded) {
  QImage cap = fbo->toImage().convertToFormat(QImage::Format_RGB888, Qt::AutoColor);

  auto location = (*sm)["liveLocationKalman"].getLiveLocationKalman();
  bool valid = loaded && (location.getStatus() == cereal::LiveLocationKalman::Status::VALID) && location.getPositionGeodetic().getValid();
  uint64_t ts = nanos_since_boot();
  VisionBuf* buf = vipc_server->get_buffer(VisionStreamType::VISION_STREAM_MAP);
  VisionIpcBufExtra extra = {
    .frame_id = frame_id,
    .timestamp_sof = (*sm)["liveLocationKalman"].getLogMonoTime(),
    .timestamp_eof = ts,
    .valid = valid,
  };

  assert(cap.sizeInBytes() >= buf->len);
  uint8_t* dst = (uint8_t*)buf->addr;
  uint8_t* src = cap.bits();

  // RGB to greyscale
  memset(dst, 128, buf->len);
  for (int i = 0; i < WIDTH * HEIGHT; i++) {
    dst[i] = src[i * 3];
  }

  vipc_server->send(buf, &extra);

  // Send thumbnail
  if (TEST_MODE) {
    // Full image in thumbnails in test mode
    kj::Array<capnp::byte> buffer_kj = kj::heapArray<capnp::byte>((const capnp::byte*)cap.bits(), cap.sizeInBytes());
    sendThumbnail(ts, buffer_kj);
  } else if (frame_id % 100 == 0) {
    // Write jpeg into buffer
    QByteArray buffer_bytes;
    QBuffer buffer(&buffer_bytes);
    buffer.open(QIODevice::WriteOnly);
    cap.save(&buffer, "JPG", 50);

    kj::Array<capnp::byte> buffer_kj = kj::heapArray<capnp::byte>((const capnp::byte*)buffer_bytes.constData(), buffer_bytes.size());
    sendThumbnail(ts, buffer_kj);
  }

  // Send state msg
  MessageBuilder msg;
  auto evt = msg.initEvent();
  auto state = evt.initMapRenderState();
  evt.setValid(valid);
  state.setLocationMonoTime((*sm)["liveLocationKalman"].getLogMonoTime());
  state.setRenderTime(render_time);
  state.setFrameId(frame_id);
  pm->send("mapRenderState", msg);

  frame_id++;
}

uint8_t* MapRenderer::getImage() {
  QImage cap = fbo->toImage().convertToFormat(QImage::Format_RGB888, Qt::AutoColor);

  uint8_t* src = cap.bits();
  uint8_t* dst = new uint8_t[WIDTH * HEIGHT];

  // RGB to greyscale
  for (int i = 0; i < WIDTH * HEIGHT; i++) {
    dst[i] = src[i * 3];
  }

  return dst;
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
    m_map->setPaintProperty("navLayer", "line-color", QColor("grey"));
    m_map->setPaintProperty("navLayer", "line-width", 5);
    m_map->setLayoutProperty("navLayer", "line-cap", "round");
  }
}

MapRenderer::~MapRenderer() {
}

extern "C" {
  MapRenderer* map_renderer_init(char *maps_host = nullptr, char *token = nullptr) {
    char *argv[] = {
      (char*)"navd",
      nullptr
    };
    int argc = 0;
    QApplication *app = new QApplication(argc, argv);
    assert(app);

    QMapboxGLSettings settings;
    settings.setApiBaseUrl(maps_host == nullptr ? MAPS_HOST : maps_host);
    settings.setAccessToken(token == nullptr ? get_mapbox_token() : token);

    return new MapRenderer(settings, false);
  }

  void map_renderer_update_position(MapRenderer *inst, float lat, float lon, float bearing) {
    inst->updatePosition({lat, lon}, bearing);
    QApplication::processEvents();
  }

  void map_renderer_update_route(MapRenderer *inst, char* polyline) {
    inst->updateRoute(polyline_to_coordinate_list(QString::fromUtf8(polyline)));
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
