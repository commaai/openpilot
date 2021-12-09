#pragma once

#include "selfdrive/ui/navd/route_parser.h"
#include "selfdrive/ui/navd/route_reply.h"

#include <QNetworkAccessManager>
#include <QUrl>

class RoutingManager : public QObject {
  Q_OBJECT

public:
  RoutingManager();
  RouteParser *route_parser;
  RouteReply *calculateRoute(const QGeoRouteRequest &request);

private slots:
  void replyFinished();
  void replyError(RouteReply::Error errorCode, const QString &error_string);

signals:
  void finished(RouteReply *reply);
  void error(RouteReply *reply, RouteReply::Error error, QString error_string);

private:
  QNetworkAccessManager *network_manager;
};
