#pragma once

#include "selfdrive/ui/navd/route_reply.h"
#include "selfdrive/ui/navd/route_parser.h"

#include <QGeoServiceProvider>
#include <QNetworkAccessManager>
#include <QUrl>

class MapboxRoutingManager : public QObject
{
    Q_OBJECT

public:
    MapboxRoutingManager();
    QGeoRouteReplyMapbox *calculateRoute(const QGeoRouteRequest &request);
    const MapboxRouteParser *routeParser() const;

private slots:
    void replyFinished();
    void replyError(QGeoRouteReply::Error errorCode, const QString &errorString);

signals:
    void finished(QGeoRouteReplyMapbox *reply);
    void error(QGeoRouteReplyMapbox *reply, QGeoRouteReply::Error error, QString errorString = QString());

private:
    QNetworkAccessManager *networkManager;
    MapboxRouteParser *m_routeParser;
};
