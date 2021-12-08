#include "selfdrive/ui/navd/route_reply.h"

#include <QGeoManeuver>
#include <QGeoRoute>
#include <QGeoRouteSegment>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

#include "selfdrive/ui/navd/route_parser.h"
#include "selfdrive/ui/navd/routing_manager.h"

class QGeoRouteMapbox : public QGeoRoute
{
public:
    QGeoRouteMapbox(const QGeoRoute &other, const QVariantMap &metadata);
    QVariantMap metadata() const;

    QVariantMap m_metadata;

    QString m_id;
    QGeoRouteRequest m_request;

    QGeoRectangle m_bounds;
    mutable QList<QGeoRouteSegment> m_routeSegments;

    int m_travelTime;
    qreal m_distance;

    QGeoRouteRequest::TravelMode m_travelMode;

    QList<QGeoCoordinate> m_path;
    QList<QGeoRouteLeg> m_legs;
    QGeoRouteSegment m_firstSegment;
    mutable int m_numSegments;
    QScopedPointer<QGeoRoute> m_containingRoute;
    int m_legIndex = 0;
};

QGeoRouteMapbox::QGeoRouteMapbox(const QGeoRoute &other, const QVariantMap &metadata)
    : QGeoRoute(other),
      m_metadata(metadata)
{
}

QVariantMap QGeoRouteMapbox::metadata() const
{
    return m_metadata;
}

QGeoRouteReplyMapbox::QGeoRouteReplyMapbox(QNetworkReply *reply, const QGeoRouteRequest &request,
                                           QObject *parent)
    : QGeoRouteReply(request, parent)
{
    if (!reply)
    {
        setError(UnknownError, QStringLiteral("Null reply"));
        return;
    }
    connect(reply, SIGNAL(finished()), this, SLOT(networkReplyFinished()));
    connect(reply, SIGNAL(error(QNetworkReply::NetworkError)),
            this, SLOT(networkReplyError(QNetworkReply::NetworkError)));
    connect(this, &QGeoRouteReply::aborted, reply, &QNetworkReply::abort);
    connect(this, &QObject::destroyed, reply, &QObject::deleteLater);
}

QGeoRouteReplyMapbox::~QGeoRouteReplyMapbox()
{
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
    for (QGeoRoute &route : routes)
    {
        qWarning() << "Route: " << route.path().length();
        route.setRequest(request());
        for (QGeoRoute &leg : route.routeLegs())
        {
            leg.setRequest(request());
        }
    }

    QVariantMap metadata;
    metadata["osrm.reply-json"] = routeReply;

    QList<QGeoRoute> mapboxRoutes;
    for (const QGeoRoute &route : routes.mid(0, request().numberAlternativeRoutes() + 1))
    {
        QGeoRouteMapbox mapboxRoute(route, metadata);
        mapboxRoutes.append(mapboxRoute);
    }

    if (error == QGeoRouteReply::NoError)
    {
        qWarning() << "Setting routes: " << mapboxRoutes.length();
        qWarning() << "First route: " << mapboxRoutes.first().path().length();
        setRoutes(mapboxRoutes);
        // setError(QGeoRouteReply::NoError, status);  // can't do this, or NoError is emitted and does damages
        setFinished(true);
    }
    else
    {
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
