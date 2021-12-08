#pragma once

#include <QGeoRouteReply>
#include <QGeoRouteRequest>
#include <QGeoRouteSegment>
#include <QJsonObject>
#include <QUrl>

class MapboxRouteSegment : public QGeoRouteSegment
{
public:
    MapboxRouteSegment() {}
    MapboxRouteSegment(const MapboxRouteSegment &other) : QGeoRouteSegment(other) {};
    virtual ~MapboxRouteSegment() {}

    bool isLegLastSegment() const { return m_isLegLastSegment; }
    void setLegLastSegment(bool lastSegment) { m_isLegLastSegment = lastSegment; }

private:
    bool m_isLegLastSegment;
};

class MapboxRouteParser : public QObject
{
    Q_OBJECT

public:
    MapboxRouteParser();

    void updateSegment(QGeoRouteSegment &segment, const QJsonObject &step, const QJsonObject &maneuver) const;
    QGeoRouteReply::Error parseReply(QList<QGeoRoute> &routes, QString &errorString, const QByteArray &reply) const;
    QUrl requestUrl(const QGeoRouteRequest &request, const QString &prefix) const;

private:
    MapboxRouteSegment parseStep(const QJsonObject &step, int legIndex, int stepIndex) const;
};
