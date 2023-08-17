#pragma once

#include <memory>

#include <QOpenGLContext>
#include <QMapboxGL>
#include <QTimer>
#include <QGeoCoordinate>
#include <QOpenGLBuffer>
#include <QOffscreenSurface>
#include <QOpenGLFunctions>
#include <QOpenGLFramebufferObject>

#include "cereal/visionipc/visionipc_server.h"
#include "cereal/messaging/messaging.h"


class MapRenderer : public QObject {
  Q_OBJECT

public:
  MapRenderer(const QMapboxGLSettings &, bool online=true);
  uint8_t* getImage();
  void update();
  bool loaded();
  ~MapRenderer();

private:
  std::unique_ptr<QOpenGLContext> ctx;
  std::unique_ptr<QOffscreenSurface> surface;
  std::unique_ptr<QOpenGLFunctions> gl_functions;
  std::unique_ptr<QOpenGLFramebufferObject> fbo;

  std::unique_ptr<VisionIpcServer> vipc_server;
  std::unique_ptr<PubMaster> pm;
  std::unique_ptr<SubMaster> sm;
  void publish(const double render_time, const bool loaded);
  void sendThumbnail(const uint64_t ts, const kj::Array<capnp::byte> &buf);

  QMapboxGLSettings m_settings;
  QScopedPointer<QMapboxGL> m_map;

  void initLayers();

  uint32_t frame_id = 0;
  uint64_t last_llk_rendered = 0;
  bool rendered() {
    return last_llk_rendered == (*sm)["liveLocationKalman"].getLogMonoTime();
  }

  QTimer* timer;

public slots:
  void updatePosition(QMapbox::Coordinate position, float bearing);
  void updateRoute(QList<QGeoCoordinate> coordinates);
  void msgUpdate();
};
