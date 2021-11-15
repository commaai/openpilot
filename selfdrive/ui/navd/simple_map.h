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


class SimpleMap : public QObject {
  Q_OBJECT

public:
  SimpleMap(const QMapboxGLSettings &);
  ~SimpleMap();

private:
  std::unique_ptr<QOpenGLContext> ctx;
  std::unique_ptr<QOffscreenSurface> surface;
  std::unique_ptr<QOpenGLFunctions> gl_functions;
  std::unique_ptr<QOpenGLFramebufferObject> fbo;

  QMapboxGLSettings m_settings;
  QScopedPointer<QMapboxGL> m_map;

  void initLayers();
  void update();

  bool loaded_once = false;

public slots:
  void updatePosition(QMapbox::Coordinate position, float bearing);
  void updateRoute(QList<QGeoCoordinate> coordinates);
};
