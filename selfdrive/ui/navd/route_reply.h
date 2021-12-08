#pragma once

#include <QGeoRouteReply>
#include <QNetworkReply>

class QGeoRouteMapbox : public QGeoRoute
{
public:
    QGeoRouteMapbox() : QGeoRoute() {}
    QGeoRouteMapbox(const QGeoRoute &other, const QVariantMap &metadata)
        : QGeoRoute(other), m_metadata(metadata) {}

    QVariantMap metadata() const { return m_metadata; };
    QVariantMap m_metadata;
};

class QGeoRouteReplyMapbox : public QGeoRouteReply
{
    Q_OBJECT

public:
    explicit QGeoRouteReplyMapbox(QObject *parent = 0);
    QGeoRouteReplyMapbox(QNetworkReply *reply, const QGeoRouteRequest &request, QObject *parent = 0);
    QList<QGeoRouteMapbox> routes() const;

private slots:
    void networkReplyFinished();
    void networkReplyError(QNetworkReply::NetworkError error);

private:
    void setRoutes(const QList<QGeoRouteMapbox> &routes);
    void addRoutes(const QList<QGeoRouteMapbox> &routes);

    QList<QGeoRouteMapbox> m_routes;
};
