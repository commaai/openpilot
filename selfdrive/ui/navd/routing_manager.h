#pragma once

#include "selfdrive/ui/navd/route_parser.h"
#include "selfdrive/ui/navd/route_reply.h"

#include <QNetworkAccessManager>
#include <QUrl>

class RoutingManager : public QObject {
  Q_OBJECT

public:
  RoutingManager();
  RouteReply *calculateRoute(const QGeoRouteRequest &request);
  const RouteParser *routeParser() const;

private slots:
  void replyFinished();
  void replyError(RouteReply::Error errorCode, const QString &errorString);

signals:
  void finished(RouteReply *reply);
  void error(RouteReply *reply, RouteReply::Error error, QString errorString);

private:
  QNetworkAccessManager *networkManager;
  RouteParser *m_routeParser;
};
