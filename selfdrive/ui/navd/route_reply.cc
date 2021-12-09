#include "selfdrive/ui/navd/route_reply.h"

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include "selfdrive/ui/navd/route_parser.h"
#include "selfdrive/ui/navd/routing_manager.h"

RouteReply::RouteReply(QNetworkReply *reply, const QGeoRouteRequest &req, QObject *parent)
    : QObject(parent)
    , request(req) {
  QObject::connect(reply, &QNetworkReply::finished, this, &RouteReply::networkReplyFinished);
  QObject::connect(reply, SIGNAL(error(QNetworkReply::NetworkError)), this, SLOT(networkReplyError(QNetworkReply::NetworkError)));
  QObject::connect(this, &QObject::destroyed, reply, &QObject::deleteLater);
}

void RouteReply::networkReplyFinished() {
  auto *reply = static_cast<QNetworkReply *>(sender());
  reply->deleteLater();

  if (reply->error() != QNetworkReply::NoError)
    return;

  auto *engine = qobject_cast<RoutingManager *>(parent());
  auto const *parser = engine->route_parser;

  auto routeReply = reply->readAll();

  QList<Route> routes;
  QString err_string;
  auto error = parser->parseReply(routes, err_string, routeReply);

  if (error == RouteReply::NoError) {
    route = routes.at(0);
    emit finished();
  } else {
    setError(error, err_string);
  }
}

void RouteReply::networkReplyError(QNetworkReply::NetworkError error) {
  auto *reply = static_cast<QNetworkReply *>(sender());
  reply->deleteLater();
  setError(RouteReply::CommunicationError, reply->errorString());
}

void RouteReply::setError(Error err, const QString &err_string) {
  reply_error = err;
  error_string = err_string;
  emit error(reply_error, error_string);
  emit finished();
}
