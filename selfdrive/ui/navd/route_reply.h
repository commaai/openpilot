#pragma once

#include <QGeoRouteReply>
#include <QNetworkReply>

class QGeoRouteReplyMapbox : public QGeoRouteReply
{
    Q_OBJECT

public:
    explicit QGeoRouteReplyMapbox(QObject *parent = 0);
    QGeoRouteReplyMapbox(QNetworkReply *reply, const QGeoRouteRequest &request, QObject *parent = 0);
    ~QGeoRouteReplyMapbox();

private slots:
    void networkReplyFinished();
    void networkReplyError(QNetworkReply::NetworkError error);
};
