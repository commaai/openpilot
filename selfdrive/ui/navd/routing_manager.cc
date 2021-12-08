#include "selfdrive/ui/navd/route_reply.h"
#include "selfdrive/ui/navd/routing_manager.h"
#include "selfdrive/ui/qt/maps/map_helpers.h"
#include "selfdrive/ui/qt/util.h"

#include <QDebug>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkRequest>
#include <QUrlQuery>

MapboxRoutingManager::MapboxRoutingManager()
{
    networkManager = new QNetworkAccessManager();
    m_routeParser = new MapboxRouteParser();
}

QGeoRouteReplyMapbox* MapboxRoutingManager::calculateRoute(const QGeoRouteRequest &request)
{
    QNetworkRequest networkRequest;
    networkRequest.setHeader(QNetworkRequest::UserAgentHeader, getUserAgent().toUtf8());

    QString url = MAPS_HOST + "/directions/v5/mapbox/";

    auto trafficWeight = request.featureWeight(QGeoRouteRequest::TrafficFeature);
    if (
        request.featureTypes().contains(QGeoRouteRequest::TrafficFeature)
        && (trafficWeight == QGeoRouteRequest::AvoidFeatureWeight || trafficWeight == QGeoRouteRequest::DisallowFeatureWeight)
    ) {
        url += QStringLiteral("driving-traffic/");
    } else {
        url += QStringLiteral("driving/");
    }

    networkRequest.setUrl(m_routeParser->requestUrl(request, url));
    qWarning() << "Mapbox request: " << networkRequest.url();

    QNetworkReply *reply = networkManager->get(networkRequest);
    QGeoRouteReplyMapbox *routeReply = new QGeoRouteReplyMapbox(reply, request, this);

    connect(routeReply, SIGNAL(finished()), this, SLOT(replyFinished()));
    connect(routeReply, SIGNAL(error(QGeoRouteReply::Error, QString)), this, SLOT(replyError(QGeoRouteReply::Error, QString)));

    return routeReply;
}

const MapboxRouteParser *MapboxRoutingManager::routeParser() const
{
    return m_routeParser;
}

void MapboxRoutingManager::replyFinished()
{
    QGeoRouteReplyMapbox *reply = qobject_cast<QGeoRouteReplyMapbox *>(sender());
    if (reply)
        emit finished(reply);
}

void MapboxRoutingManager::replyError(QGeoRouteReply::Error errorCode, const QString &errorString)
{
    QGeoRouteReplyMapbox *reply = qobject_cast<QGeoRouteReplyMapbox *>(sender());
    if (reply)
        emit error(reply, errorCode, errorString);
}
