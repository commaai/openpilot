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


class MapRenderer : public QObject {
  Q_OBJECT

public:
  MapRenderer(const QMapboxGLSettings &, bool enable_vipc=true);
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

  QMapboxGLSettings m_settings;
  QScopedPointer<QMapboxGL> m_map;

  void initLayers();

  uint32_t frame_id = 0;

public slots:
  void updatePosition(QMapbox::Coordinate position, float bearing);
  void updateRoute(QList<QGeoCoordinate> coordinates);
};
