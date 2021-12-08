#pragma once

#include <QGeoCoordinate>
#include <QGeoRouteRequest>
#include <QNetworkReply>

struct RouteManeuverLane {
  bool active;
  QString activeDirection;
  QList<QString> directions;
};

class RouteManeuver {
public:
  RouteManeuver() { }

  bool isValid() const { return m_valid; }
  void setValid(bool valid) { m_valid = valid; }

  QGeoCoordinate position() const { return m_position; }
  void setPosition(const QGeoCoordinate &position) { m_position = position; }

  QString type() const { return m_type; }
  void setType(const QString &type) { m_type = type; }

  QString modifier() const { return m_modifier; }
  void setModifier(const QString &modifier) { m_modifier = modifier; }

  QString primaryText() const { return m_primaryText; }
  void setPrimaryText(const QString &text) { m_primaryText = text; }

  QString secondaryText() const { return m_secondaryText; }
  void setSecondaryText(const QString &text) { m_secondaryText = text; }

  QList<RouteManeuverLane> lanes() const { return m_lanes; }
  void setLanes(const QList<RouteManeuverLane> &lanes) { m_lanes = lanes; }

  float distanceAlongGeometry() const { return m_distanceAlongGeometry; }
  void setDistanceAlongGeometry(float distance) { m_distanceAlongGeometry = distance; }

  float typicalDuration() const { return m_typicalDuration; }
  void setTypicalDuration(float duration) { m_typicalDuration = duration; }

private:
  bool m_valid;
  QGeoCoordinate m_position;
  QString m_primaryText;
  QString m_type;
  QString m_modifier;
  QString m_secondaryText;
  QList<RouteManeuverLane> m_lanes;
  float m_distanceAlongGeometry;
  float m_typicalDuration;
};

class RouteSegment {
public:
  RouteSegment() { }

  bool isValid() const { return m_valid; }
  void setValid(bool valid) { m_valid = valid; }

  RouteSegment nextRouteSegment() const { return *m_nextRouteSegment; }
  void setNextRouteSegment(const RouteSegment &routeSegment) { *m_nextRouteSegment = routeSegment; }

  int travelTime() const { return m_travelTime; }
  void setTravelTime(int secs) { m_travelTime = secs; }

  double distance() const { return m_distance; }
  void setDistance(double distance) { m_distance = distance; }

  QList<QGeoCoordinate> path() const { return m_path; }
  void setPath(const QList<QGeoCoordinate> &path) { m_path = path; }

  RouteManeuver maneuver() const { return m_maneuver; }
  void setManeuver(const RouteManeuver &maneuver) { m_maneuver = maneuver; }

private:
  bool m_valid;
  RouteSegment *m_nextRouteSegment;
  int m_travelTime;
  double m_distance;
  QList<QGeoCoordinate> m_path;
  RouteManeuver m_maneuver;
};

class Route {
public:
  Route() { }

  void setFirstRouteSegment(const RouteSegment &routeSegment) { m_firstRouteSegment = routeSegment; }
  RouteSegment firstRouteSegment() const { return m_firstRouteSegment; }

  void setTravelTime(int secs) { m_travelTime = secs; }
  int travelTime() const { return m_travelTime; }

  void setDistance(double distance) { m_distance = distance; }
  double distance() const { return m_distance; }

  void setPath(const QList<QGeoCoordinate> &path) { m_path = path; }
  QList<QGeoCoordinate> path() const { return m_path; }

private:
  RouteSegment m_firstRouteSegment = RouteSegment();
  int m_travelTime;
  double m_distance;
  QList<QGeoCoordinate> m_path;
};

class RouteReply : public QObject {
  Q_OBJECT

public:
  RouteReply(QNetworkReply *reply, const QGeoRouteRequest &request, QObject *parent = nullptr);

  enum Error {
    NoError,
    CommunicationError,
    ParseError,
    UnknownError,
  };
  Q_ENUM(Error)

  Error error() const { return m_error; }
  QString errorString() const { return m_errorString; }

  QGeoRouteRequest request() const { return m_request; }
  Route route() const { return m_route; }

signals:
  void finished();
  void error(Error error, const QString &errorString = QString());

private slots:
  void networkReplyFinished();
  void networkReplyError(QNetworkReply::NetworkError error);

private:
  QGeoRouteRequest m_request;

  Route m_route;
  void setRoute(const Route &route) { m_route = route; }

  Error m_error = RouteReply::NoError;
  QString m_errorString = QString();

  void setError(Error err, const QString &errorString);
};
