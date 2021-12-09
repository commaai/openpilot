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
  networkManager = new QNetworkAccessManager();
  m_routeParser = new RouteParser();
}

RouteReply *RoutingManager::calculateRoute(const QGeoRouteRequest &request) {
  QNetworkRequest networkRequest;
  networkRequest.setHeader(QNetworkRequest::UserAgentHeader, getUserAgent().toUtf8());

  QString url = MAPS_HOST + "/directions/v5/mapbox/";

  auto trafficWeight = request.featureWeight(QGeoRouteRequest::TrafficFeature);
  if (
      request.featureTypes().contains(QGeoRouteRequest::TrafficFeature)
      && (trafficWeight == QGeoRouteRequest::AvoidFeatureWeight || trafficWeight == QGeoRouteRequest::DisallowFeatureWeight)) {
    url += QStringLiteral("driving-traffic/");
  } else {
    url += QStringLiteral("driving/");
  }

  networkRequest.setUrl(m_routeParser->requestUrl(request, url));
  qWarning() << "Mapbox request: " << networkRequest.url();

  QNetworkReply *reply = networkManager->get(networkRequest);
  RouteReply *routeReply = new RouteReply(reply, request, this);

  QObject::connect(routeReply, &RouteReply::finished, this, &RoutingManager::replyFinished);
  QObject::connect(routeReply, &RouteReply::error, this, &RoutingManager::replyError);

  return routeReply;
}

const RouteParser *RoutingManager::routeParser() const {
  return m_routeParser;
}

void RoutingManager::replyFinished() {
  RouteReply *reply = qobject_cast<RouteReply *>(sender());
  if (reply)
    emit finished(reply);
}

void RoutingManager::replyError(RouteReply::Error errorCode, const QString &errorString) {
  RouteReply *reply = qobject_cast<RouteReply *>(sender());
  if (reply)
    emit error(reply, errorCode, errorString);
}
