#include "selfdrive/ui/navd/route_parser.h"

#include <QDebug>
#include <QGeoPath>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLocale>
#include <QUrlQuery>
#include <math.h>

#include "selfdrive/common/params.h"
#include "selfdrive/ui/qt/maps/map_helpers.h"

static QList<QGeoCoordinate> decodePolyline(const QString &polylineString) {
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

static void parse_banner(RouteManeuver &maneuver, const QMap<QString, QVariant> &banner) {
  maneuver.distanceAlongGeometry = banner["distance_along_geometry"].toDouble();

  auto primary = banner["primary"].toMap();
  maneuver.primaryText = primary["text"].toString();

  if (primary.contains("type"))
    maneuver.type = primary["type"].toString();

  if (primary.contains("modifier"))
    maneuver.modifier = primary["modifier"].toString();

  if (banner.contains("secondary")) {
    auto s = banner["secondary"].toMap();
    maneuver.secondaryText = s["text"].toString();
  }

  if (banner.contains("sub")) {
    auto s = banner["sub"].toMap();
    auto components = s["components"].toList();

    size_t num_lanes = 0;
    for (auto &c : components) {
      auto cc = c.toMap();
      if (cc["type"].toString() == "lane") {
        num_lanes += 1;
      }
    }

    auto lanes = QList<RouteManeuverLane>();

    for (auto &c : components) {
      auto component = c.toMap();
      if (component["type"].toString() == "lane") {
        auto lane = RouteManeuverLane();
        lane.active = component["active"].toBool();

        if (component.contains("active_direction")) {
          lane.activeDirection = component["active_direction"].toString();
        }

        auto directions = QList<QString>();
        for (auto &dir : component["directions"].toList()) {
          directions.append(dir.toString());
        }

        lanes.append(lane);
      }
    }
    maneuver.lanes = lanes;
  }
}

RouteParser::RouteParser() { }

std::optional<RouteSegment> RouteParser::parseStep(const QJsonObject &step, int legIndex, int stepIndex) const {
  if (!step.value("maneuver").isObject())
    return {};
  QJsonObject maneuver = step.value("maneuver").toObject();
  if (!step.value("duration").isDouble())
    return {};
  if (!step.value("distance").isDouble())
    return {};
  if (!step.value("intersections").isArray())
    return {};
  if (!maneuver.value("location").isArray())
    return {};

  double time = step.value("duration").toDouble();
  double distance = step.value("distance").toDouble();

  QJsonArray position = maneuver.value("location").toArray();
  if (position.isEmpty())
    return {};
  double latitude = position[1].toDouble();
  double longitude = position[0].toDouble();
  QGeoCoordinate coord(latitude, longitude);

  QString geometry = step.value("geometry").toString();
  QList<QGeoCoordinate> path = decodePolyline(geometry);

  RouteManeuver routeManeuver;
  routeManeuver.position = coord;

  if (step.value("bannerInstructions").isArray() && step.value("bannerInstructions").toArray().size() > 0) {
    auto const &banner = step.value("bannerInstructions").toArray().first();
    if (banner.isObject())
      parse_banner(routeManeuver, banner.toObject().toVariantMap());
  }

  RouteSegment segment;
  segment.distance = distance;
  segment.path = path;
  segment.travelTime = time;
  segment.maneuver = routeManeuver;
  return segment;
}

RouteReply::Error RouteParser::parseReply(QList<Route> &routes, QString &errorString, const QByteArray &reply) const {
  // OSRM v5 specs: https://github.com/Project-OSRM/osrm-backend/blob/master/docs/http.md
  // Mapbox Directions API spec: https://www.mapbox.com/api-documentation/#directions
  QJsonDocument document = QJsonDocument::fromJson(reply);
  if (document.isObject()) {
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
      for (int legIndex = 0; legIndex < legs.size(); ++legIndex) {
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
        for (int stepIndex = 0; stepIndex < steps.size(); ++stepIndex) {
          const QJsonValue &s = steps.at(stepIndex);
          if (!s.isObject()) {
            error = true;
            break;
          }
          auto segment = parseStep(s.toObject(), legIndex, stepIndex);
          if (segment) {
            legSegments.append(segment.value());
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
          path.append(s.path);

        qWarning() << "distance: " << distance;
        route.distance = distance;
        qWarning() << "Travel time: " << travelTime;
        route.travelTime = travelTime;
        if (!path.isEmpty()) {
          qWarning() << "Path: " << path;
          route.path = path;
          qWarning() << "First segment: " << segments.first().distance;
          route.segments = segments;
        }
        routes.append(route);
      }
    }

    return RouteReply::NoError;
  } else {
    errorString = "Couldn't parse json.";
    return RouteReply::ParseError;
  }
}

QUrl RouteParser::requestUrl(const QGeoRouteRequest &request, const QString &prefix) const {
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
      } else {
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
