#include "selfdrive/ui/navd/route_reply.h"

#include <QGeoManeuver>
#include <QGeoRoute>
#include <QGeoRouteSegment>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

#include "selfdrive/ui/navd/route_parser.h"
#include "selfdrive/ui/navd/routing_manager.h"

QGeoRouteReplyMapbox::QGeoRouteReplyMapbox(QNetworkReply *reply, const QGeoRouteRequest &request, QObject *parent)
    : QGeoRouteReply(request, parent)
{
    if (!reply) {
        setError(UnknownError, QStringLiteral("Null reply"));
        return;
    }
    connect(reply, SIGNAL(finished()), this, SLOT(networkReplyFinished()));
    connect(reply, SIGNAL(error(QNetworkReply::NetworkError)), this, SLOT(networkReplyError(QNetworkReply::NetworkError)));
    connect(this, &QGeoRouteReply::aborted, reply, &QNetworkReply::abort);
    connect(this, &QObject::destroyed, reply, &QObject::deleteLater);
}

QList<QGeoRouteMapbox> QGeoRouteReplyMapbox::routes() const
{
    return m_routes;
}

void QGeoRouteReplyMapbox::setRoutes(const QList<QGeoRouteMapbox> &routes)
{
    m_routes = routes;
}

void QGeoRouteReplyMapbox::addRoutes(const QList<QGeoRouteMapbox> &routes)
{
    m_routes.append(routes);
}

void QGeoRouteReplyMapbox::networkReplyFinished()
{
    QNetworkReply *reply = static_cast<QNetworkReply *>(sender());
    reply->deleteLater();

    if (reply->error() != QNetworkReply::NoError)
        return;

    MapboxRoutingManager *engine = qobject_cast<MapboxRoutingManager *>(parent());
    const MapboxRouteParser *parser = engine->routeParser();

    QList<QGeoRoute> routes;
    QString errorString;

    QByteArray routeReply = reply->readAll();
    qWarning() << "Route reply: " << routeReply;
    QGeoRouteReply::Error error = parser->parseReply(routes, errorString, routeReply);
    qWarning() << "Parsed routes: " << routes.length();
    // Setting the request into the result
    for (QGeoRoute &route : routes) {
        qWarning() << "Route: " << route.path().length();
        route.setRequest(request());
        for (QGeoRoute &leg : route.routeLegs()) {
            leg.setRequest(request());
        }
    }

    QVariantMap metadata;
    metadata["osrm.reply-json"] = routeReply;

    QList<QGeoRouteMapbox> mapboxRoutes;
    for (const QGeoRoute &route : routes.mid(0, request().numberAlternativeRoutes() + 1)) {
        QGeoRouteMapbox mapboxRoute(route, metadata);
        mapboxRoutes.append(mapboxRoute);
    }

    if (error == QGeoRouteReply::NoError) {
        qWarning() << "Setting routes: " << mapboxRoutes.length();
        qWarning() << "First route: " << mapboxRoutes.first().path().length();
        setRoutes(mapboxRoutes);
        setFinished(true);
    } else {
        setError(error, errorString);
    }
}

void QGeoRouteReplyMapbox::networkReplyError(QNetworkReply::NetworkError error)
{
    Q_UNUSED(error)
    QNetworkReply *reply = static_cast<QNetworkReply *>(sender());
    reply->deleteLater();
    setError(QGeoRouteReply::CommunicationError, reply->errorString());
}
