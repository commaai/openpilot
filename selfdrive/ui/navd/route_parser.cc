#include "selfdrive/ui/navd/route_parser.h"

#include <math.h>
#include <QDebug>
#include <QGeoPath>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QLocale>
#include <QUrlQuery>

#include "selfdrive/common/params.h"
#include "selfdrive/ui/qt/maps/map_helpers.h"

static QList<QGeoCoordinate> decodePolyline(const QString &polylineString)
{
    QList<QGeoCoordinate> path;
    if (polylineString.isEmpty())
        return path;

    QByteArray data = polylineString.toLatin1();

    bool parsingLatitude = true;

    int shift = 0;
    int value = 0;

    QGeoCoordinate coord(0, 0);

    for (int i = 0; i < data.length(); ++i) {
        unsigned char c = data.at(i) - 63;

        value |= (c & 0x1f) << shift;
        shift += 5;

        // another chunk
        if (c & 0x20)
            continue;

        int diff = (value & 1) ? ~(value >> 1) : (value >> 1);

        if (parsingLatitude) {
            coord.setLatitude(coord.latitude() + (double)diff / 1e6);
        } else {
            coord.setLongitude(coord.longitude() + (double)diff / 1e6);
            path.append(coord);
        }

        parsingLatitude = !parsingLatitude;

        value = 0;
        shift = 0;
    }

    return path;
}

static QVariantMap parseMapboxBannerComponent(const QJsonObject &bannerComponent)
{
    QVariantMap map;

    if (bannerComponent.value("type").isString())
        map.insert("type", bannerComponent.value("type").toString());

    if (bannerComponent.value("text").isString())
        map.insert("text", bannerComponent.value("text").toString());

    if (bannerComponent.value("abbr").isString())
        map.insert("abbr", bannerComponent.value("abbr").toString());

    if (bannerComponent.value("abbr_priority").isDouble())
        map.insert("abbr_priority", bannerComponent.value("abbr_priority").toInt());

    return map;
}

static QVariantList parseMapboxBannerComponents(const QJsonArray &bannerComponents)
{
    QVariantList list;
    for (const QJsonValue &bannerComponentValue : bannerComponents) {
        if (bannerComponentValue.isObject())
            list.append(parseMapboxBannerComponent(bannerComponentValue.toObject()));
    }
    return list;
}

static QVariantMap parseMapboxBanner(const QJsonObject &banner)
{
    QVariantMap map;

    if (banner.value("text").isString())
        map.insert("text", banner.value("text").toString());

    if (banner.value("components").isArray())
        map.insert("components", parseMapboxBannerComponents(banner.value("components").toArray()));

    if (banner.value("type").isString())
        map.insert("type", banner.value("type").toString());

    if (banner.value("modifier").isString())
        map.insert("modifier", banner.value("modifier").toString());

    if (banner.value("degrees").isDouble())
        map.insert("degrees", banner.value("degrees").toDouble());

    if (banner.value("driving_side").isString())
        map.insert("driving_side", banner.value("driving_side").toString());

    return map;
}

static QVariantMap parseMapboxBannerInstruction(const QJsonObject &bannerInstruction)
{
    QVariantMap map;

    if (bannerInstruction.value("distanceAlongGeometry").isDouble())
        map.insert("distance_along_geometry", bannerInstruction.value("distanceAlongGeometry").toDouble());

    if (bannerInstruction.value("primary").isObject())
        map.insert("primary", parseMapboxBanner(bannerInstruction.value("primary").toObject()));

    if (bannerInstruction.value("secondary").isObject())
        map.insert("secondary", parseMapboxBanner(bannerInstruction.value("secondary").toObject()));

    if (bannerInstruction.value("then").isObject())
        map.insert("then", parseMapboxBanner(bannerInstruction.value("then").toObject()));

    return map;
}

static QVariantList parseMapboxBannerInstructions(const QJsonArray &bannerInstructions)
{
    QVariantList list;
    for (const QJsonValue &bannerInstructionValue : bannerInstructions) {
        if (bannerInstructionValue.isObject())
            list.append(parseMapboxBannerInstruction(bannerInstructionValue.toObject()));
    }
    return list;
}


// static void parse_banner(cereal::NavInstruction::Builder &instruction, const QMap<QString, QVariant> &banner, bool full) {
//   QString primary_str, secondary_str;

//   auto p = banner["primary"].toMap();
//   primary_str += p["text"].toString();

//   instruction.setShowFull(full);

//   if (p.contains("type")) {
//     instruction.setManeuverType(p["type"].toString().toStdString());
//   }

//   if (p.contains("modifier")) {
//     instruction.setManeuverModifier(p["modifier"].toString().toStdString());
//   }

//   if (banner.contains("secondary")) {
//     auto s = banner["secondary"].toMap();
//     secondary_str += s["text"].toString();
//   }

//   instruction.setManeuverPrimaryText(primary_str.toStdString());
//   instruction.setManeuverSecondaryText(secondary_str.toStdString());

//   if (banner.contains("sub")) {
//     auto s = banner["sub"].toMap();
//     auto components = s["components"].toList();

//     size_t num_lanes = 0;
//     for (auto &c : components) {
//       auto cc = c.toMap();
//       if (cc["type"].toString() == "lane") {
//         num_lanes += 1;
//       }
//     }

//     auto lanes = instruction.initLanes(num_lanes);

//     size_t i = 0;
//     for (auto &c : components) {
//       auto cc = c.toMap();
//       if (cc["type"].toString() == "lane") {
//         auto lane = lanes[i];
//         lane.setActive(cc["active"].toBool());

//         if (cc.contains("active_direction")) {
//           lane.setActiveDirection(string_to_direction(cc["active_direction"].toString()));
//         }

//         auto directions = lane.initDirections(cc["directions"].toList().size());

//         size_t j = 0;
//         for (auto &dir : cc["directions"].toList()) {
//           directions.set(j, string_to_direction(dir.toString()));
//           j++;
//         }

//         i++;
//       }
//     }
//   }
// }


RouteParser::RouteParser() {}

RouteSegment RouteParser::parseStep(const QJsonObject &step, int legIndex, int stepIndex) const
{
    // OSRM Instructions documentation: https://github.com/Project-OSRM/osrm-text-instructions
    // This goes on top of OSRM: https://github.com/Project-OSRM/osrm-backend/blob/master/docs/http.md
    // Mapbox however, includes this in the reply, under "instruction".
    RouteSegment segment;
    if (!step.value("maneuver").isObject())
        return segment;
    QJsonObject maneuver = step.value("maneuver").toObject();
    if (!step.value("duration").isDouble())
        return segment;
    if (!step.value("distance").isDouble())
        return segment;
    if (!step.value("intersections").isArray())
        return segment;
    if (!maneuver.value("location").isArray())
        return segment;

    double time = step.value("duration").toDouble();
    double distance = step.value("distance").toDouble();

    QJsonArray position = maneuver.value("location").toArray();
    if (position.isEmpty())
        return segment;
    double latitude = position[1].toDouble();
    double longitude = position[0].toDouble();
    QGeoCoordinate coord(latitude, longitude);

    QString geometry = step.value("geometry").toString();
    QList<QGeoCoordinate> path = decodePolyline(geometry);

    RouteManeuver routeManeuver;
    routeManeuver.setPosition(coord);
    // TODO parse banner
    if (step.value("bannerInstructions").isArray())
        parseMapboxBannerInstructions(step.value("bannerInstructions").toArray());

    segment.setDistance(distance);
    segment.setPath(path);
    segment.setTravelTime(time);
    segment.setManeuver(routeManeuver);

    return segment;
}

RouteReply::Error RouteParser::parseReply(QList<Route> &routes, QString &errorString, const QByteArray &reply) const
{
    // OSRM v5 specs: https://github.com/Project-OSRM/osrm-backend/blob/master/docs/http.md
    // Mapbox Directions API spec: https://www.mapbox.com/api-documentation/#directions
    QJsonDocument document = QJsonDocument::fromJson(reply);
    if (document.isObject())
    {
        QJsonObject object = document.object();

        QString status = object.value("code").toString();
        qWarning() << "status: " << status;
        if (status != "Ok") {
            errorString = status;
            return RouteReply::UnknownError;
        }
        if (!object.value("routes").isArray()) {
            errorString = "No routes found";
            return RouteReply::ParseError;
        }

        QJsonArray osrmRoutes = object.value("routes").toArray();
        for (const QJsonValue &r : osrmRoutes) {
            if (!r.isObject())
                continue;
            QJsonObject routeObject = r.toObject();
            if (!routeObject.value("legs").isArray())
                continue;
            if (!routeObject.value("duration").isDouble())
                continue;
            if (!routeObject.value("distance").isDouble())
                continue;

            double distance = routeObject.value("distance").toDouble();
            double travelTime = routeObject.value("duration").toDouble();
            qWarning() << "distance: " << distance;
            qWarning() << "Travel time: " << travelTime;
            bool error = false;
            QList<RouteSegment> segments;

            QJsonArray legs = routeObject.value("legs").toArray();
            Route route;
            for (int legIndex = 0; legIndex < legs.size(); ++legIndex)
            {
                const QJsonValue &l = legs.at(legIndex);
                QList<RouteSegment> legSegments;
                if (!l.isObject()) {
                    error = true;
                    break;
                }
                QJsonObject leg = l.toObject();
                if (!leg.value("steps").isArray()) {
                    error = true;
                    break;
                }
                QJsonArray steps = leg.value("steps").toArray();
                RouteSegment segment;
                for (int stepIndex = 0; stepIndex < steps.size(); ++stepIndex) {
                    const QJsonValue &s = steps.at(stepIndex);
                    if (!s.isObject()) {
                        error = true;
                        break;
                    }
                    segment = parseStep(s.toObject(), legIndex, stepIndex);
                    if (segment.isValid()) {
                        legSegments.append(segment);
                    } else {
                        error = true;
                        break;
                    }
                }
                if (error)
                    break;

                segments.append(legSegments);
            }

            if (!error) {
                QList<QGeoCoordinate> path;
                for (const RouteSegment &s : segments)
                    path.append(s.path());

                for (int i = segments.size() - 1; i > 0; --i)
                    segments[i - 1].setNextRouteSegment(segments[i]);

                qWarning() << "distance: " << distance;
                route.setDistance(distance);
                qWarning() << "Travel time: " << travelTime;
                route.setTravelTime(travelTime);
                if (!path.isEmpty()) {
                    qWarning() << "Path: " << path;
                    route.setPath(path);
                    qWarning() << "First segment: " << segments.first().distance();
                    route.setFirstRouteSegment(segments.first());
                }
                routes.append(route);
            }
        }

        return RouteReply::NoError;
    }
    else {
        errorString = "Couldn't parse json.";
        return RouteReply::ParseError;
    }
}

QUrl RouteParser::requestUrl(const QGeoRouteRequest &request, const QString &prefix) const
{
    QString routingUrl = prefix;
    QString bearings;
    const QList<QVariantMap> metadata = request.waypointsMetadata();
    const QList<QGeoCoordinate> waypoints = request.waypoints();
    for (int i = 0; i < waypoints.size(); i++) {
        const QGeoCoordinate &c = waypoints.at(i);
        if (i > 0) {
            routingUrl.append(';');
            bearings.append(';');
        }
        routingUrl.append(QString::number(c.longitude(), 'f', 7)).append(',').append(QString::number(c.latitude(), 'f', 7));
        if (metadata.size() > i) {
            const QVariantMap &meta = metadata.at(i);
            if (meta.contains("bearing")) {
                qreal bearing = meta.value("bearing").toDouble();
                bearings.append(QString::number(int(bearing))).append(',').append("90"); // 90 is the angle of maneuver allowed.
            }
            else {
                bearings.append("0,180"); // 180 here means anywhere
            }
        }
    }

    QUrl url(routingUrl);
    QUrlQuery query;
    query.addQueryItem("access_token", get_mapbox_token());
    query.addQueryItem("annotations", "distance,maxspeed");
    query.addQueryItem("bearings", bearings);
    query.addQueryItem("geometries", "polyline6");
    query.addQueryItem("overview", "full");
    query.addQueryItem("steps", "true");
    query.addQueryItem("banner_instructions", "true");
    query.addQueryItem("roundabout_exits", "true");
    url.setQuery(query);
    return url;
}
