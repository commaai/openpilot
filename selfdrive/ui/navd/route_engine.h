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
#include "selfdrive/ui/navd/routing_manager.h"

class QGeoRoutePrivate : public QSharedData
{
public:
  QGeoRoutePrivate();
  QGeoRoutePrivate(const QGeoRoutePrivate &other);
  virtual ~QGeoRoutePrivate();
  virtual QGeoRoutePrivate *clone() = 0;

  bool operator == (const QGeoRoutePrivate &other) const;

  virtual void setId(const QString &id);
  virtual QString id() const;

  virtual void setRequest(const QGeoRouteRequest &request);
  virtual QGeoRouteRequest request() const;

  virtual void setBounds(const QGeoRectangle &bounds);
  virtual QGeoRectangle bounds() const;

  virtual void setTravelTime(int travelTime);
  virtual int travelTime() const;

  virtual void setDistance(qreal distance);
  virtual qreal distance() const;

  virtual void setTravelMode(QGeoRouteRequest::TravelMode mode);
  virtual QGeoRouteRequest::TravelMode travelMode() const;

  virtual void setPath(const QList<QGeoCoordinate> &path);
  virtual QList<QGeoCoordinate> path() const;

  virtual void setFirstSegment(const QGeoRouteSegment &firstSegment);
  virtual QGeoRouteSegment firstSegment() const;

  virtual QVariantMap metadata() const;

  virtual void setRouteLegs(const QList<QGeoRouteLeg> &legs);
  virtual QList<QGeoRouteLeg> routeLegs() const;

  virtual QString engineName() const = 0;
  virtual int segmentsCount() const = 0;

  virtual void setLegIndex(int idx);
  virtual int legIndex() const;
  virtual void setContainingRoute(const QGeoRoute &route);
  virtual QGeoRoute containingRoute() const;

  static const QGeoRoutePrivate *routePrivateData(const QGeoRoute &route);

protected:
  virtual bool equals(const QGeoRoutePrivate &other) const;
};

class QGeoRouteMapbox : public QGeoRoute
{
public:
  QVariantMap metadata() {
    return d()->metadata();
  }
};

struct RouteGeometrySegment {
  double distance;
  double speed_limit;
};

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
  MapboxRoutingManager *routing_manager;
  QGeoRoute route;
  QGeoRouteSegment segment;
  QMapbox::Coordinate nav_destination;
  QList<RouteGeometrySegment> route_geometry_segments = {};

  // Position
  std::optional<QMapbox::Coordinate> last_position;
  std::optional<float> last_bearing;
  bool localizer_valid = false;

  // Route recompute
  bool active = false;
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
