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
  float distance_along_geometry;
  QString primary_text;
  std::optional<QString> type;
  std::optional<QString> modifier;
  std::optional<QString> secondary_text;
  std::optional<QList<RouteManeuverLane>> lanes;
};

struct RouteAnnotation {
  float speed_limit;
};

struct RouteSegment {
  float distance;
  float travel_time;
  float travel_time_typical;
  QList<QGeoCoordinate> path = {};
  QList<RouteAnnotation> annotations = {};
  QList<RouteManeuver> maneuvers = {};
};

struct Route {
  float distance;
  float travel_time;
  QList<QGeoCoordinate> path = {};
  QList<RouteAnnotation> annotations = {};
  QList<RouteSegment> segments = {};
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
