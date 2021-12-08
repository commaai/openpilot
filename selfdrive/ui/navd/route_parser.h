#pragma once

#include <QGeoRouteRequest>
#include <QGeoRouteSegment>
#include <QJsonObject>
#include <QUrl>

#include "selfdrive/ui/navd/route_reply.h"

class RouteParser : public QObject
{
    Q_OBJECT

public:
    RouteParser();

    RouteReply::Error parseReply(QList<Route> &routes, QString &errorString, const QByteArray &reply) const;
    QUrl requestUrl(const QGeoRouteRequest &request, const QString &prefix) const;

private:
    RouteSegment parseStep(const QJsonObject &step, int legIndex, int stepIndex) const;
};
