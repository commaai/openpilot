#include "selfdrive/ui/navd/route_reply.h"

#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

#include "selfdrive/ui/navd/route_parser.h"
#include "selfdrive/ui/navd/routing_manager.h"

RouteReply::RouteReply(QNetworkReply *reply, const QGeoRouteRequest &request, QObject *parent)
    : QObject(parent), m_request(request)
{
    connect(reply, SIGNAL(finished()), this, SLOT(networkReplyFinished()));
    connect(reply, SIGNAL(error(QNetworkReply::NetworkError)), this, SLOT(networkReplyError(QNetworkReply::NetworkError)));
    connect(this, &QObject::destroyed, reply, &QObject::deleteLater);
}

void RouteReply::networkReplyFinished()
{
    QNetworkReply *reply = static_cast<QNetworkReply *>(sender());
    reply->deleteLater();

    if (reply->error() != QNetworkReply::NoError)
        return;

    RoutingManager *engine = qobject_cast<RoutingManager *>(parent());
    const RouteParser *parser = engine->routeParser();

    QByteArray routeReply = reply->readAll();
    qWarning() << "Route reply: " << routeReply;

    QList<Route> routes;
    QString errorString;
    RouteReply::Error error = parser->parseReply(routes, errorString, routeReply);
    qWarning() << "Parsed routes: " << routes.length();

    if (error == RouteReply::NoError) {
        setRoute(routes.at(0));
        emit finished();
    } else {
        setError(error, errorString);
    }
}

void RouteReply::networkReplyError(QNetworkReply::NetworkError error)
{
    Q_UNUSED(error)
    QNetworkReply *reply = static_cast<QNetworkReply *>(sender());
    reply->deleteLater();
    setError(RouteReply::CommunicationError, reply->errorString());
}

void RouteReply::setError(Error err, const QString &errorString)
{
    m_error = err;
    m_errorString = errorString;
    emit error(err, errorString);
    emit finished();
}
