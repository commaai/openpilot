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

// TODO: this seems to create duplicate coordinates in the path when decoding the polyline, so the corresponding annotations don't match up by index
static QList<QGeoCoordinate> decodePolyline(const QString &polyline) {
  QList<QGeoCoordinate> path;
  if (polyline.isEmpty())
    return path;

  QByteArray data = polyline.toLatin1();

  bool parsing_latitude = true;

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

    if (parsing_latitude) {
      coord.setLatitude(coord.latitude() + (double)diff / 1e6);
    } else {
      coord.setLongitude(coord.longitude() + (double)diff / 1e6);
      path.append(coord);
    }

    parsing_latitude = !parsing_latitude;

    value = 0;
    shift = 0;
  }

  return path;
}

static void parse_banner(RouteManeuver &maneuver, const QMap<QString, QVariant> &banner) {
  maneuver.distance_along_geometry = banner["distance_along_geometry"].toDouble();

  auto primary = banner["primary"].toMap();
  maneuver.primary_text = primary["text"].toString();

  if (primary.contains("type"))
    maneuver.type = primary["type"].toString();

  if (primary.contains("modifier"))
    maneuver.modifier = primary["modifier"].toString();

  if (banner.contains("secondary")) {
    auto s = banner["secondary"].toMap();
    maneuver.secondary_text = s["text"].toString();
  }

  if (banner.contains("sub")) {
    auto s = banner["sub"].toMap();
    auto components = s["components"].toList();

    auto lanes = QList<RouteManeuverLane>();
    for (auto &c : components) {
      auto component = c.toMap();
      if (component["type"].toString() == "lane") {
        auto lane = RouteManeuverLane();
        lane.active = component["active"].toBool();

        if (component.contains("active_direction")) {
          lane.active_direction = component["active_direction"].toString();
        }

        QList<QString> directions;
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

std::optional<RouteSegment> RouteParser::parseStep(const QJsonObject &step, int leg_index, int step_index) const {
  if (!step.value("maneuver").isObject())
    return {};
  auto maneuver = step.value("maneuver").toObject();
  if (!step.value("duration").isDouble())
    return {};
  if (!step.value("distance").isDouble())
    return {};
  if (!step.value("intersections").isArray())
    return {};
  if (!maneuver.value("location").isArray())
    return {};

  auto time = step.value("duration").toDouble();
  auto distance = step.value("distance").toDouble();

  auto position = maneuver.value("location").toArray();
  if (position.isEmpty())
    return {};
  auto latitude = position[1].toDouble();
  auto longitude = position[0].toDouble();
  QGeoCoordinate coord(latitude, longitude);

  auto geometry = step.value("geometry").toString();
  auto path = decodePolyline(geometry);

  RouteManeuver routeManeuver;
  routeManeuver.position = coord;
  routeManeuver.typical_duration = step.value("duration_typical").toDouble();

  if (step.value("bannerInstructions").isArray() && step.value("bannerInstructions").toArray().size() > 0) {
    auto const &banner = step.value("bannerInstructions").toArray().first();
    if (banner.isObject())
      parse_banner(routeManeuver, banner.toObject().toVariantMap());
  }

  RouteSegment segment;
  segment.distance = distance;
  segment.path = path;
  segment.travel_time = time;
  segment.maneuver = routeManeuver;
  return segment;
}

// Mapbox Directions API: https://docs.mapbox.com/api/navigation/directions/
RouteReply::Error RouteParser::parseReply(QList<Route> &routes, QString &error_string, const QByteArray &reply) const {
  auto document = QJsonDocument::fromJson(reply);
  if (document.isObject()) {
    auto object = document.object();

    auto status = object.value("code").toString();
    if (status != "Ok") {
      error_string = status;
      return RouteReply::UnknownError;
    }
    if (!object.value("routes").isArray()) {
      error_string = "No routes found";
      return RouteReply::ParseError;
    }

    auto osrmRoutes = object.value("routes").toArray();
    for (auto const &r : osrmRoutes) {
      if (!r.isObject())
        continue;
      auto routeObject = r.toObject();
      if (!routeObject.value("legs").isArray())
        continue;
      if (!routeObject.value("duration").isDouble())
        continue;
      if (!routeObject.value("distance").isDouble())
        continue;

      auto distance = routeObject.value("distance").toDouble();
      auto travel_time = routeObject.value("duration").toDouble();
      auto error = false;
      QList<RouteSegment> segments;
      QList<RouteAnnotation> annotations;

      auto legs = routeObject.value("legs").toArray();
      Route route;
      for (int leg_index = 0; leg_index < legs.size(); ++leg_index) {
        auto const &l = legs.at(leg_index);
        QList<RouteSegment> leg_segments;
        if (!l.isObject()) {
          error = true;
          break;
        }
        auto leg = l.toObject();
        if (!leg.value("steps").isArray()) {
          error = true;
          break;
        }
        auto steps = leg.value("steps").toArray();
        for (int step_index = 0; step_index < steps.size(); ++step_index) {
          auto const &s = steps.at(step_index);
          if (!s.isObject()) {
            error = true;
            break;
          }
          auto segment = parseStep(s.toObject(), leg_index, step_index);
          if (segment) {
            leg_segments.append(segment.value());
          } else {
            error = true;
            break;
          }
        }
        if (error)
          break;

        segments.append(leg_segments);

        auto annotation = leg.value("annotation").toObject();
        auto distances = annotation.value("distance").toArray();
        auto maxspeeds = annotation.value("maxspeed").toArray();
        auto size = std::min(distances.size(), maxspeeds.size());
        for (int i = 0; i < size; i++) {
          auto max_speed = maxspeeds.at(i).toObject();
          auto unknown = max_speed.value("unknown").toBool();
          auto speed = max_speed.value("speed").toDouble();
          auto unit = max_speed.value("unit").toString();
          auto speed_limit = unknown ? 0
              : unit == "km/h"       ? speed * KPH_TO_MS
              : unit == "mph"        ? speed * MPH_TO_MS
                                     : 0;

          annotations.append({ distances.at(i).toDouble(), speed_limit });
        }
      }

      if (!error) {
        route.distance = distance;
        route.travel_time = travel_time;

        QList<QGeoCoordinate> path;
        for (auto const &s : segments)
          path.append(s.path);

        if (!path.isEmpty()) {
          route.path = path;
          route.segments = segments;
          route.annotations = annotations;
        }

        routes.append(route);
      }
    }

    if (routes.isEmpty()) {
      error_string = "No routes found";
      return RouteReply::ParseError;
    } else {
      qWarning() << "Found" << routes.size() << "route(s)";
    }

    return RouteReply::NoError;
  } else {
    error_string = "Couldn't parse json.";
    return RouteReply::ParseError;
  }
}

QUrl RouteParser::requestUrl(const QGeoRouteRequest &request, const QString &prefix) const {
  auto routing_url = prefix;
  QString bearings;
  auto const metadata = request.waypointsMetadata();
  auto const waypoints = request.waypoints();
  for (int i = 0; i < waypoints.size(); i++) {
    auto const &c = waypoints.at(i);
    if (i > 0) {
      routing_url.append(';');
      bearings.append(';');
    }
    routing_url.append(QString::number(c.longitude(), 'f', 7)).append(',').append(QString::number(c.latitude(), 'f', 7));
    if (metadata.size() > i) {
      auto const &meta = metadata.at(i);
      if (meta.contains("bearing")) {
        auto bearing = meta.value("bearing").toDouble();
        bearings.append(QString::number(int(bearing))).append(',').append("90"); // 90 is the angle of maneuver allowed
      } else {
        bearings.append("0,180"); // 180 here means anywhere
      }
    }
  }

  QUrl url(routing_url);
  QUrlQuery query;
  query.addQueryItem("access_token", get_mapbox_token());
  query.addQueryItem("annotations", "distance,maxspeed");
  query.addQueryItem("bearings", bearings);
  query.addQueryItem("geometries", "polyline6");
  query.addQueryItem("overview", "full");
  query.addQueryItem("steps", "true");
  query.addQueryItem("banner_instructions", "true");
  url.setQuery(query);
  return url;
}
