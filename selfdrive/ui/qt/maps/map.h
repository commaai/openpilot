#pragma once

#include <optional>

#include <QGeoCoordinate>
#include <QGestureEvent>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QMapboxGL>
#include <QMouseEvent>
#include <QOpenGLWidget>
#include <QScopedPointer>
#include <QString>
#include <QtGlobal>
#include <QTimer>
#include <QWheelEvent>
#include <QMap>
#include <QPixmap>

#include "selfdrive/common/params.h"
#include "selfdrive/common/util.h"
#include "cereal/messaging/messaging.h"

const QString MAPBOX_TOKEN = util::getenv("MAPBOX_TOKEN").c_str();
const QString MAPS_HOST = util::getenv("MAPS_HOST", MAPBOX_TOKEN.isEmpty() ? "https://maps.comma.ai" : "https://api.mapbox.com").c_str();

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

  void initLayers();

  void mousePressEvent(QMouseEvent *ev) final;
  void mouseDoubleClickEvent(QMouseEvent *ev) final;
  void mouseMoveEvent(QMouseEvent *ev) final;
  void wheelEvent(QWheelEvent *ev) final;
  bool event(QEvent *event) final;
  bool gestureEvent(QGestureEvent *event);
  void pinchTriggered(QPinchGesture *gesture);

  bool m_sourceAdded = false;
  SubMaster *sm;
  QTimer* timer;

  bool loaded_once = false;
  bool allow_open = true;

  // Panning
  QPointF m_lastPos;
  int pan_counter = 0;
  int zoom_counter = 0;

  // Position
  std::optional<QMapbox::Coordinate> last_position;
  std::optional<float> last_bearing;
  FirstOrderFilter velocity_filter;
  bool localizer_valid = false;

  MapInstructions* map_instructions;
  MapETA* map_eta;

  void clearRoute();
  uint64_t route_rcv_frame = 0;

private slots:
  void timerUpdate();

public slots:
  void offroadTransition(bool offroad);

signals:
  void distanceChanged(float distance);
  void instructionsChanged(cereal::NavInstruction::Reader instruction);
  void ETAChanged(float seconds, float seconds_typical, float distance);
};

