#pragma once

#include <QOpenGLWidget>
#include <QMapboxGL>
#include <QTimer>
#include <QGeoCoordinate>

class SimpleMap : public QOpenGLWidget {
  Q_OBJECT

public:
  SimpleMap(const QMapboxGLSettings &);
  ~SimpleMap();

private:
  void initializeGL() final;
  void paintGL() final;

  QMapboxGLSettings m_settings;
  QScopedPointer<QMapboxGL> m_map;

  void initLayers();

  bool loaded_once = false;

public slots:
  void updatePosition(QMapbox::Coordinate position, float bearing);
};