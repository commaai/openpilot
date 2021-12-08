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

MapboxRoutingManager::MapboxRoutingManager(const QVariantMap &parameters,
                                                         QGeoServiceProvider::Error *error,
                                                         QString *errorString)
    : m_networkManager(new QNetworkAccessManager()),
      m_userAgent(getUserAgent().toUtf8())
{
    if (parameters.contains(QStringLiteral("mapbox.useragent"))) {
        m_userAgent = parameters.value(QStringLiteral("mapbox.useragent")).toString().toLatin1();
    }

    if (parameters.contains(QStringLiteral("mapbox.directions_api_url"))) {
        m_directionsApiUrl = parameters.value(QStringLiteral("mapbox.directions_api_url")).toString().toLatin1();
    }

    if (parameters.contains(QStringLiteral("mapbox.access_token"))) {
        m_accessToken = parameters.value(QStringLiteral("mapbox.access_token")).toString();
    }

    bool use_mapbox_text_instructions = true;
    if (parameters.contains(QStringLiteral("mapbox.routing.use_mapbox_text_instructions"))) {
        use_mapbox_text_instructions = parameters.value(QStringLiteral("mapbox.routing.use_mapbox_text_instructions")).toBool();
    }

    MapboxRouteParser *parser = new MapboxRouteParser(m_accessToken, use_mapbox_text_instructions);
    if (parameters.contains(QStringLiteral("mapbox.routing.traffic_side"))) {
        QString trafficSide = parameters.value(QStringLiteral("mapbox.routing.traffic_side")).toString();
        if (trafficSide == QStringLiteral("right"))
            parser->trafficSide = MapboxRouteParser::RightHandTraffic;
        else if (trafficSide == QStringLiteral("left"))
            parser->trafficSide = MapboxRouteParser::LeftHandTraffic;
    }
    m_routeParser = parser;

    *error = QGeoServiceProvider::NoError;
    errorString->clear();
}

MapboxRoutingManager::~MapboxRoutingManager()
{
}

QGeoRouteReplyMapbox* MapboxRoutingManager::calculateRoute(const QGeoRouteRequest &request)
{
    QNetworkRequest networkRequest;
    networkRequest.setHeader(QNetworkRequest::UserAgentHeader, m_userAgent);

    QString url = m_directionsApiUrl;

    QGeoRouteRequest::TravelModes travelModes = request.travelModes();
    if (travelModes.testFlag(QGeoRouteRequest::PedestrianTravel)) {
        url += QStringLiteral("walking/");
    } else if (travelModes.testFlag(QGeoRouteRequest::BicycleTravel)) {
        url += QStringLiteral("cycling/");
    } else if (travelModes.testFlag(QGeoRouteRequest::CarTravel)) {
        const QList<QGeoRouteRequest::FeatureType> &featureTypes = request.featureTypes();
        int trafficFeatureIdx = featureTypes.indexOf(QGeoRouteRequest::TrafficFeature);
        QGeoRouteRequest::FeatureWeight trafficWeight = request.featureWeight(QGeoRouteRequest::TrafficFeature);
        if (trafficFeatureIdx >= 0 &&
           (trafficWeight == QGeoRouteRequest::AvoidFeatureWeight || trafficWeight == QGeoRouteRequest::DisallowFeatureWeight)) {
            url += QStringLiteral("driving-traffic/");
        } else {
            url += QStringLiteral("driving/");
        }
    }

    networkRequest.setUrl(m_routeParser->requestUrl(request, url));
    qWarning() << "Mapbox request: " << networkRequest.url();

    QNetworkReply *reply = m_networkManager->get(networkRequest);

    QGeoRouteReplyMapbox *routeReply = new QGeoRouteReplyMapbox(reply, request, this);

    connect(routeReply, SIGNAL(finished()), this, SLOT(replyFinished()));
    connect(routeReply, SIGNAL(error(QGeoRouteReply::Error,QString)),
            this, SLOT(replyError(QGeoRouteReply::Error,QString)));

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

void MapboxRoutingManager::replyError(QGeoRouteReply::Error errorCode,
                                             const QString &errorString)
{
    QGeoRouteReplyMapbox *reply = qobject_cast<QGeoRouteReplyMapbox *>(sender());
    if (reply)
        emit error(reply, errorCode, errorString);
}
