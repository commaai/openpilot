#pragma once

#include <optional>

#include <QGeoCoordinate>
#include <QGestureEvent>
#include <QMapboxGL>
#include <QMouseEvent>
#include <QOpenGLWidget>
#include <QPixmap>
#include <QScopedPointer>
#include <QString>
#include <QWheelEvent>
#include <QTextDocument>

#include "cereal/messaging/messaging.h"
#include "common/params.h"
#include "selfdrive/ui/ui.h"

class MapInstructions : public QWidget {
  Q_OBJECT

private:
  void paintEvent(QPaintEvent* event) override;
  int drawInstructions(QPainter &p, bool draw);
  bool is_rhd = false;
  QString error_str, primary_str, secondary_str, distance_str;
  QPixmap icon;
  std::vector<QString> lanes;
  Params params;

public:
  QTextDocument eta_doc;
  MapInstructions(QWidget * parent=nullptr);
  inline void setError(QString error) { error_str = error; update(); }
  void updateDistance(float d);
  void updateInstructions(cereal::NavInstruction::Reader instruction);
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

  void clearRoute();
  uint64_t route_rcv_frame = 0;

private slots:
  void updateState(const UIState &s);

public slots:
  void offroadTransition(bool offroad);
};
