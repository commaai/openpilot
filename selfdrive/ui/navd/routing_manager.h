#pragma once

#include "selfdrive/ui/navd/route_parser.h"

#include <QGeoServiceProvider>
#include <QNetworkAccessManager>
#include <QUrl>

class MapboxRoutingManager : public QObject
{
    Q_OBJECT

public:
    MapboxRoutingManager(const QVariantMap &parameters,
                                   QGeoServiceProvider::Error *error,
                                   QString *errorString);
    ~MapboxRoutingManager();

    QGeoRouteReply *calculateRoute(const QGeoRouteRequest &request);
    const MapboxRouteParser *routeParser() const;

private slots:
    void replyFinished();
    void replyError(QGeoRouteReply::Error errorCode, const QString &errorString);

signals:
    void finished(QGeoRouteReply *reply);
    void error(QGeoRouteReply *reply, QGeoRouteReply::Error error, QString errorString = QString());

private:
    QNetworkAccessManager *m_networkManager;
    QByteArray m_userAgent;
    QString m_accessToken;
    QString m_directionsApiUrl;
    // bool m_useMapboxText = false;
    MapboxRouteParser *m_routeParser = nullptr;
};
