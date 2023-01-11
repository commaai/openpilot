#pragma once

#include <optional>

#include <QGeoCoordinate>
#include <QGestureEvent>
#include <QHBoxLayout>
#include <QLabel>
#include <QMap>
#include <QMapboxGL>
#include <QMouseEvent>
#include <QOpenGLWidget>
#include <QPixmap>
#include <QScopedPointer>
#include <QString>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <QtGlobal>

#include "cereal/messaging/messaging.h"
#include "common/params.h"
#include "common/util.h"
#include "selfdrive/ui/ui.h"

class MapInstructions : public QWidget {
  Q_OBJECT

private:
  QLabel *distance;
  QLabel *primary;
  QLabel *secondary;
  QLabel *icon_01;
  QWidget *lane_widget;
  QHBoxLayout *lane_layout;
  bool error = false;
  bool is_rhd = false;

public:
  MapInstructions(QWidget * parent=nullptr);
  void showError(QString error);
  void noError();
  void hideIfNoError();

public slots:
  void updateDistance(float d);
  void updateInstructions(cereal::NavInstruction::Reader instruction);
};

class MapETA : public QWidget {
  Q_OBJECT

private:
  QLabel *eta;
  QLabel *eta_unit;
  QLabel *time;
  QLabel *time_unit;
  QLabel *distance;
  QLabel *distance_unit;
  Params params;

public:
  MapETA(QWidget * parent=nullptr);

public slots:
  void updateETA(float seconds, float seconds_typical, float distance);
};

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
  QMapbox::AnnotationID marker_id = -1;

  void initLayers();

  void mousePressEvent(QMouseEvent *ev) final;
  void mouseDoubleClickEvent(QMouseEvent *ev) final;
  void mouseMoveEvent(QMouseEvent *ev) final;
  void wheelEvent(QWheelEvent *ev) final;
  bool event(QEvent *event) final;
  bool gestureEvent(QGestureEvent *event);
  void pinchTriggered(QPinchGesture *gesture);

  bool m_sourceAdded = false;

  bool loaded_once = false;
  bool allow_open = true;

  // Panning
  QPointF m_lastPos;
  int pan_counter = 0;
  int zoom_counter = -1;

  // Position
  std::optional<QMapbox::Coordinate> last_position;
  std::optional<float> last_bearing;
  FirstOrderFilter velocity_filter;
  bool laikad_valid = false;
  bool locationd_valid = false;

  MapInstructions* map_instructions;
  MapETA* map_eta;

  void clearRoute();
  void updateDestinationMarker();
  uint64_t route_rcv_frame = 0;

private slots:
  void updateState(const UIState &s);

public slots:
  void offroadTransition(bool offroad);

signals:
  void distanceChanged(float distance);
  void instructionsChanged(cereal::NavInstruction::Reader instruction);
  void ETAChanged(float seconds, float seconds_typical, float distance);
};

