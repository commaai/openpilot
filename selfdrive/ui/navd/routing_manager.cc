#include "selfdrive/ui/navd/routing_manager.h"
#include "selfdrive/ui/navd/route_reply.h"
#include "selfdrive/ui/qt/maps/map_helpers.h"
#include "selfdrive/ui/qt/util.h"

#include <QDebug>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkRequest>
#include <QUrlQuery>

RoutingManager::RoutingManager() {
  network_manager = new QNetworkAccessManager();
  route_parser = new RouteParser();
  QObject::connect(this, &QObject::destroyed, network_manager, &QObject::deleteLater);
  QObject::connect(this, &QObject::destroyed, route_parser, &QObject::deleteLater);
}

RouteReply *RoutingManager::calculateRoute(const QGeoRouteRequest &request) {
  QNetworkRequest network_request;
  network_request.setHeader(QNetworkRequest::UserAgentHeader, getUserAgent().toUtf8());

  auto url = MAPS_HOST + "/directions/v5/mapbox/";

  auto traffic_weight = request.featureWeight(QGeoRouteRequest::TrafficFeature);
  if (request.featureTypes().contains(QGeoRouteRequest::TrafficFeature)
      && (traffic_weight == QGeoRouteRequest::AvoidFeatureWeight || traffic_weight == QGeoRouteRequest::DisallowFeatureWeight)) {
    url += QStringLiteral("driving-traffic/");
  } else {
    url += QStringLiteral("driving/");
  }

  network_request.setUrl(route_parser->requestUrl(request, url));
  qWarning() << "Mapbox request: " << network_request.url();

  auto *reply = network_manager->get(network_request);
  auto *route_reply = new RouteReply(reply, request, this);

  QObject::connect(route_reply, &RouteReply::finished, this, &RoutingManager::replyFinished);
  QObject::connect(route_reply, &RouteReply::error, this, &RoutingManager::replyError);

  return route_reply;
}

void RoutingManager::replyFinished() {
  auto *reply = qobject_cast<RouteReply *>(sender());
  if (reply)
    emit finished(reply);
}

void RoutingManager::replyError(RouteReply::Error errorCode, const QString &error_string) {
  auto *reply = qobject_cast<RouteReply *>(sender());
  if (reply)
    emit error(reply, errorCode, error_string);
}
