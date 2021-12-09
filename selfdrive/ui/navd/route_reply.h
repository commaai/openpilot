#pragma once

#include <QGeoCoordinate>
#include <QGeoRouteRequest>
#include <QNetworkReply>

struct RouteManeuverLane {
  bool active;
  QString activeDirection;
  QList<QString> directions;
};

struct RouteManeuver {
  QGeoCoordinate position;
  std::optional<QString> primaryText;
  std::optional<QString> type;
  std::optional<QString> modifier;
  std::optional<QString> secondaryText;
  std::optional<QList<RouteManeuverLane>> lanes;
  float distanceAlongGeometry;
  float typicalDuration;
};

struct RouteSegment {
  int travelTime;
  double distance;
  QList<QGeoCoordinate> path = {};
  RouteManeuver maneuver;
};

struct Route {
  double distance;
  int travelTime;
  QList<RouteSegment> segments = {};
  QList<QGeoCoordinate> path = {};
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
