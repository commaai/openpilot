#pragma once

#include <optional>

#include <QThread>
#include <QGeoCoordinate>
#include <QGeoManeuver>
#include <QGeoRouteRequest>
#include <QGeoRouteSegment>
#include <QGeoRoutingManager>
#include <QGeoServiceProvider>
#include <QTimer>
#include <QMapboxGL>

#include "cereal/messaging/messaging.h"

class RouteEngine : public QObject {
  Q_OBJECT

public:
  RouteEngine();

  SubMaster *sm;
  PubMaster *pm;

  QTimer* msg_timer;
  QTimer* route_timer;

  std::optional<int> ui_pid;

  // Route
  bool gps_ok = false;
  QGeoServiceProvider *geoservice_provider;
  QGeoRoutingManager *routing_manager;
  QGeoRoute route;
  QGeoRouteSegment segment;
  QMapbox::Coordinate nav_destination;

  // Position
  std::optional<QMapbox::Coordinate> last_position;
  std::optional<float> last_bearing;
  bool localizer_valid = false;

  // Route recompute
  int recompute_backoff = 0;
  int recompute_countdown = 0;
  void calculateRoute(QMapbox::Coordinate destination);
  void clearRoute();
  bool shouldRecompute();

private slots:
  void routeUpdate();
  void msgUpdate();
  void routeCalculated(QGeoRouteReply *reply);
  void recomputeRoute();
  void sendRoute();

signals:
  void positionUpdated(QMapbox::Coordinate position, float bearing);
  void routeUpdated(QList<QGeoCoordinate> coordinates);
};
