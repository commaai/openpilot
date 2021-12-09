#pragma once

#include <QGeoCoordinate>
#include <QGeoRouteRequest>
#include <QNetworkReply>

struct RouteManeuverLane {
  bool active;
  std::optional<QString> active_direction;
  QList<QString> directions = {};
};

struct RouteManeuver {
  QGeoCoordinate position;
  std::optional<QString> primary_text;
  std::optional<QString> type;
  std::optional<QString> modifier;
  std::optional<QString> secondary_text;
  std::optional<QList<RouteManeuverLane>> lanes;
  float distance_along_geometry;
  float typical_duration;
};

struct RouteSegment {
  int travel_time;
  double distance;
  QList<QGeoCoordinate> path = {};
  RouteManeuver maneuver;
};

struct RouteAnnotation {
  double distance;
  double speed_limit;
};

struct Route {
  double distance;
  int travel_time;
  QList<RouteSegment> segments = {};
  QList<QGeoCoordinate> path = {};
  QList<RouteAnnotation> annotations = {};
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

  QGeoRouteRequest request;
  Route route;
  Error reply_error = RouteReply::NoError;
  QString error_string = QString();

signals:
  void finished();
  void error(RouteReply::Error error, const QString &error_string);

private slots:
  void networkReplyFinished();
  void networkReplyError(QNetworkReply::NetworkError error);

private:
  void setError(Error err, const QString &error_string);
};
