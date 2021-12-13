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

static QList<QGeoCoordinate> decodeGeojson(QJsonObject const &geojson) {
  QList<QGeoCoordinate> path;
  auto coords = geojson["coordinates"].toArray();
  for (auto const &c : coords) {
    auto cc = c.toArray();
    path.append(QGeoCoordinate(cc[1].toDouble(), cc[0].toDouble()));
  }
  return path;
}

static RouteManeuver parseManeuver(QMap<QString, QVariant> const &banner) {
  RouteManeuver maneuver;
  maneuver.distance_along_geometry = banner["distanceAlongGeometry"].toDouble();

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
    auto lanes = QList<RouteManeuverLane>();
    for (auto &c : s["components"].toList()) {
      auto component = c.toMap();
      if (component["type"].toString() == "lane") {
        auto lane = RouteManeuverLane();
        lane.active = component["active"].toBool();
        if (component.contains("active_direction"))
          lane.active_direction = component["active_direction"].toString();
        for (auto &dir : component["directions"].toList())
          lane.directions.append(dir.toString());
        lanes.append(lane);
      }
    }
    maneuver.lanes = lanes;
  }

  return maneuver;
}

static RouteSegment parseSegment(QJsonObject const &step) {
  RouteSegment segment;
  segment.distance = step.value("distance").toDouble();
  segment.travel_time = step.value("duration").toDouble();
  segment.travel_time_typical = step.contains("duration_typical") ? step.value("duration_typical").toDouble() : segment.travel_time;
  segment.path = decodeGeojson(step.value("geometry").toObject());

  // Mapbox's "banners" are generally more useful than their "maneuver", so prefer the banners, falling back to the maneuver
  auto banners = step.value("bannerInstructions").toArray();
  for (auto const &banner : banners) {
    if (banner.isObject())
      segment.maneuvers.append(parseManeuver(banner.toObject().toVariantMap()));
  }

  if (segment.maneuvers.size() == 0) {
    auto m = step.value("maneuver").toObject();
    RouteManeuver maneuver;
    maneuver.distance_along_geometry = segment.distance;
    maneuver.primary_text = m.value("instruction").toString();
    maneuver.type = m.value("type").toString();
    if (m.contains("modifier"))
      maneuver.modifier = m.value("modifier").toString();
    segment.maneuvers.append(maneuver);
  }

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

      auto travel_time = routeObject.value("duration").toDouble();
      auto distance = routeObject.value("distance").toDouble();
      auto error = false;
      QList<RouteSegment> segments;
      QList<RouteAnnotation> annotations;

      auto legs = routeObject.value("legs").toArray();
      Route route;
      for (auto const &l : legs) {
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
        for (auto const &s : leg.value("steps").toArray()) {
          if (!s.isObject()) {
            error = true;
            break;
          }
          leg_segments.append(parseSegment(s.toObject()));
        }
        if (error)
          break;

        size_t seg_index = 0;
        size_t path_index = 0;
        auto a = leg.value("annotation").toObject();
        auto maxspeeds = a.value("maxspeed").toArray();
        for (size_t i = 0; i < maxspeeds.size(); i++) {
          auto max_speed = maxspeeds.at(i).toObject();
          auto unknown = max_speed.value("unknown").toBool();
          auto speed = max_speed.value("speed").toDouble();
          auto unit = max_speed.value("unit").toString();
          float const speed_limit = unknown ? 0
              : unit == "km/h"       ? speed * KPH_TO_MS
              : unit == "mph"        ? speed * MPH_TO_MS
                                     : 0;

          RouteAnnotation annotation = { speed_limit };
          annotations.append(annotation);

          // add each annotation to the segment it applies to
          if (leg_segments[seg_index].path.size() >= 2)
            leg_segments[seg_index].annotations.append(annotation);
          if (++path_index >= leg_segments[seg_index].path.size() - 1) {
            ++seg_index;
            path_index = 0;
          }
        }

        segments.append(leg_segments);
      }

      if (!error) {
        route.distance = distance;
        route.travel_time = travel_time;

        auto const path = decodeGeojson(routeObject.value("geometry").toObject());
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
  query.addQueryItem("annotations", "maxspeed");
  query.addQueryItem("bearings", bearings);
  query.addQueryItem("geometries", "geojson");
  query.addQueryItem("overview", "full");
  query.addQueryItem("steps", "true");
  query.addQueryItem("banner_instructions", "true");
  url.setQuery(query);
  return url;
}
