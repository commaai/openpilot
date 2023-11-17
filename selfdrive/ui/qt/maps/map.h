#pragma once

#include <optional>

#include <QGeoCoordinate>
#include <QGestureEvent>
#include <QLabel>
#include <QMap>
#include <QMapboxGL>
#include <QMouseEvent>
#include <QOpenGLWidget>
#include <QPixmap>
#include <QPushButton>
#include <QScopedPointer>
#include <QString>
#include <QVBoxLayout>
#include <QWheelEvent>

#include "cereal/messaging/messaging.h"
#include "common/params.h"
#include "common/util.h"
#include "selfdrive/ui/ui.h"
#include "selfdrive/ui/qt/maps/map_eta.h"
#include "selfdrive/ui/qt/maps/map_instructions.h"

class MapWindow : public QOpenGLWidget {
  Q_OBJECT

public:
  MapWindow(const QMapboxGLSettings &);
  ~MapWindow();

private:
  void initializeGL() final;
  void paintGL() final;
  void resizeGL(int w, int h) override;

  QMapboxGLSettings m_settings;
  QScopedPointer<QMapboxGL> m_map;

  void initLayers();

  void mousePressEvent(QMouseEvent *ev) final;
  void mouseDoubleClickEvent(QMouseEvent *ev) final;
  void mouseMoveEvent(QMouseEvent *ev) final;
  void wheelEvent(QWheelEvent *ev) final;
  bool event(QEvent *event) final;
  bool gestureEvent(QGestureEvent *event);
  void pinchTriggered(QPinchGesture *gesture);
  void setError(const QString &err_str);

  bool loaded_once = false;

  // Panning
  QPointF m_lastPos;
  int interaction_counter = 0;

  // Position
  std::optional<QMapbox::Coordinate> last_valid_nav_dest;
  std::optional<QMapbox::Coordinate> last_position;
  std::optional<float> last_bearing;
  FirstOrderFilter velocity_filter;
  bool locationd_valid = false;
  bool routing_problem = false;

  QWidget *map_overlay;
  QLabel *error;
  MapInstructions* map_instructions;
  MapETA* map_eta;

  // Blue with normal nav, green when nav is input into the model
  QColor getNavPathColor(bool nav_enabled) {
    return nav_enabled ? QColor("#31ee73") : QColor("#31a1ee");
  }

  void clearRoute();
  void updateDestinationMarker();
  uint64_t route_rcv_frame = 0;

private slots:
  void updateState(const UIState &s);

public slots:
  void offroadTransition(bool offroad);

signals:
  void requestVisible(bool visible);
  void requestSettings(bool settings);
};
