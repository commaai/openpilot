#include "selfdrive/ui/navd/route_engine.h"

#include <QDebug>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>

#include "selfdrive/ui/navd/routing_manager.h"
#include "selfdrive/ui/qt/maps/map.h"
#include "selfdrive/ui/qt/maps/map_helpers.h"
#include "selfdrive/ui/qt/api.h"

#include "selfdrive/common/params.h"

const qreal REROUTE_DISTANCE = 25;
const float MANEUVER_TRANSITION_THRESHOLD = 10;

static float get_time_typical(const RouteSegment &segment) {
  return segment.maneuver().typicalDuration() || segment.travelTime();
}

static cereal::NavInstruction::Direction string_to_direction(QString d) {
  if (d.contains("left")) {
    return cereal::NavInstruction::Direction::LEFT;
  } else if (d.contains("right")) {
    return cereal::NavInstruction::Direction::RIGHT;
  } else if (d.contains("straight")) {
    return cereal::NavInstruction::Direction::STRAIGHT;
  }

  return cereal::NavInstruction::Direction::NONE;
}

RouteEngine::RouteEngine() {
  sm = new SubMaster({"liveLocationKalman", "managerState"});
  pm = new PubMaster({"navInstruction", "navRoute"});

  // Timers
  route_timer = new QTimer(this);
  QObject::connect(route_timer, SIGNAL(timeout()), this, SLOT(routeUpdate()));
  route_timer->start(1000);

  msg_timer = new QTimer(this);
  QObject::connect(msg_timer, SIGNAL(timeout()), this, SLOT(msgUpdate()));
  msg_timer->start(50);

  // Build routing engine
  routing_manager = new RoutingManager();
  assert(routing_manager);
  QObject::connect(routing_manager, &RoutingManager::finished, this, &RouteEngine::routeCalculated);

  // Get last gps position from params
  auto last_gps_position = coordinate_from_param("LastGPSPosition");
  if (last_gps_position) {
    last_position = *last_gps_position;
  }
}

void RouteEngine::msgUpdate() {
  sm->update(1000);
  if (!sm->updated("liveLocationKalman")) {
    active = false;
    return;
  }

  if (sm->updated("managerState")) {
    for (auto const &p : (*sm)["managerState"].getManagerState().getProcesses()) {
      if (p.getName() == "ui" && p.getRunning()) {
        if (ui_pid && *ui_pid != p.getPid()){
          qWarning() << "UI restarting, sending route";
          QTimer::singleShot(5000, this, &RouteEngine::sendRoute);
        }
        ui_pid = p.getPid();
      }
    }
  }

  auto location = (*sm)["liveLocationKalman"].getLiveLocationKalman();
  auto pos = location.getPositionGeodetic();
  auto orientation = location.getCalibratedOrientationNED();

  gps_ok = location.getGpsOK();

  localizer_valid = (location.getStatus() == cereal::LiveLocationKalman::Status::VALID) && pos.getValid();

  if (localizer_valid) {
    last_bearing = RAD2DEG(orientation.getValue()[2]);
    last_position = QMapbox::Coordinate(pos.getValue()[0], pos.getValue()[1]);
    emit positionUpdated(*last_position, *last_bearing);
  }

  active = true;
}

void RouteEngine::routeUpdate() {
  if (!active) {
    return;
  }

  recomputeRoute();

  MessageBuilder msg;
  cereal::Event::Builder evt = msg.initEvent(segment.isValid());
  cereal::NavInstruction::Builder instruction = evt.initNavInstruction();

  // Show route instructions
  if (segment.isValid()) {
    auto maneuver = segment.maneuver();
    if (maneuver.isValid()) {
      float along_geometry = distance_along_geometry(segment.path(), to_QGeoCoordinate(*last_position));
      float distance_to_maneuver_along_geometry = segment.distance() - along_geometry;

      instruction.setShowFull(distance_to_maneuver_along_geometry < maneuver.distanceAlongGeometry());
      instruction.setManeuverType(maneuver.type().toStdString());
      instruction.setManeuverModifier(maneuver.modifier().toStdString());
      instruction.setManeuverPrimaryText(maneuver.primaryText().toStdString());
      instruction.setManeuverSecondaryText(maneuver.secondaryText().toStdString());
      instruction.setManeuverDistance(distance_to_maneuver_along_geometry);

      auto lanes = instruction.initLanes(maneuver.lanes().size());
      for (int i = 0; i < maneuver.lanes().size(); i++) {
        auto &l = maneuver.lanes()[i];
        auto lane = lanes[i];
        lane.setActive(l.active);
        lane.setActiveDirection(string_to_direction(l.activeDirection));
        auto directions = lane.initDirections(l.directions.size());
        for (int j = 0; j < l.directions.size(); j++) {
          directions.set(j, string_to_direction(l.directions[j]));
        }
      }

      // ETA
      float progress = distance_along_geometry(segment.path(), to_QGeoCoordinate(*last_position)) / segment.distance();
      float total_distance = segment.distance() * (1.0 - progress);
      float total_time = segment.travelTime() * (1.0 - progress);
      float total_time_typical = get_time_typical(segment) * (1.0 - progress);

      auto s = segment.nextRouteSegment();
      while (s.isValid()) {
        total_distance += s.distance();
        total_time += s.travelTime();
        total_time_typical += get_time_typical(s);

        s = s.nextRouteSegment();
      }
      instruction.setTimeRemaining(total_time);
      instruction.setTimeRemainingTypical(total_time_typical);
      instruction.setDistanceRemaining(total_distance);

      // Transition to next route segment
      if (distance_to_maneuver_along_geometry < -MANEUVER_TRANSITION_THRESHOLD) {
        auto next_segment = segment.nextRouteSegment();
        if (next_segment.isValid()) {
          segment = next_segment;

          recompute_backoff = std::max(0, recompute_backoff - 1);
          recompute_countdown = 0;
        } else {
          qWarning() << "Destination reached";
          Params().remove("NavDestination");

          // Clear route if driving away from destination
          float d = segment.maneuver().position().distanceTo(to_QGeoCoordinate(*last_position));
          if (d > REROUTE_DISTANCE) {
            clearRoute();
          }
        }
      }
    }

    // // would like to just use `distance_along_geometry(route.path(), ...)` but its coordinates can be in reversed order
    // float along_route = 0;
    // auto s = route.firstRouteSegment();
    // while (s.isValid()) {
    //   if (s != segment) {
    //     along_route += s.distance();
    //     s = s.nextRouteSegment();
    //   } else {
    //     along_route += distance_along_geometry(segment.path(), to_QGeoCoordinate(*last_position));
    //     break;
    //   }
    // }

    // float speed_limit = 0;
    // float distance = 0;
    // for (auto const &g : route_geometry_segments) {
    //   distance += g.distance;
    //   if (along_route < distance) {
    //     speed_limit = g.speed_limit;
    //     break;
    //   }
    // }
    // qWarning() << "Along route: " << along_route;
    // qWarning() << "Distance: " << distance;
    // qWarning() << "Speed limit: " << speed_limit;

    // instruction.setSpeedLimit(speed_limit);
  }

  pm->send("navInstruction", msg);
}

void RouteEngine::clearRoute() {
  route = Route();
  segment = RouteSegment();
  nav_destination = QMapbox::Coordinate();
}

bool RouteEngine::shouldRecompute() {
  if (!segment.isValid()) {
    return true;
  }

  // Don't recompute in last segment, assume destination is reached
  if (!segment.nextRouteSegment().isValid()) {
    return false;
  }

  // Compute closest distance to all line segments in the current path
  float min_d = REROUTE_DISTANCE + 1;
  auto path = segment.path();
  auto cur = to_QGeoCoordinate(*last_position);
  for (size_t i = 0; i < path.size() - 1; i++) {
    auto a = path[i];
    auto b = path[i+1];
    if (a.distanceTo(b) < 1.0) {
      continue;
    }
    min_d = std::min(min_d, minimum_distance(a, b, cur));
  }
  return min_d > REROUTE_DISTANCE;

  // TODO: Check for going wrong way in segment
}

void RouteEngine::recomputeRoute() {
  if (!last_position) {
    return;
  }

  auto new_destination = coordinate_from_param("NavDestination");
  if (!new_destination) {
    clearRoute();
    return;
  }

  bool should_recompute = shouldRecompute();
  if (*new_destination != nav_destination) {
    qWarning() << "Got new destination from NavDestination param" << *new_destination;
    should_recompute = true;
  }

  if (!gps_ok && segment.isValid()) return; // Don't recompute when gps drifts in tunnels

  if (recompute_countdown == 0 && should_recompute) {
    recompute_countdown = std::pow(2, recompute_backoff);
    recompute_backoff = std::min(7, recompute_backoff + 1);
    calculateRoute(*new_destination);
  } else {
    recompute_countdown = std::max(0, recompute_countdown - 1);
  }
}

void RouteEngine::calculateRoute(QMapbox::Coordinate destination) {
  qWarning() << "Calculating route" << *last_position << "->" << destination;

  nav_destination = destination;
  QGeoRouteRequest request(to_QGeoCoordinate(*last_position), to_QGeoCoordinate(destination));
  request.setFeatureWeight(QGeoRouteRequest::TrafficFeature, QGeoRouteRequest::AvoidFeatureWeight);

  if (last_bearing) {
    QVariantMap params;
    int bearing = ((int)(*last_bearing) + 360) % 360;
    params["bearing"] = bearing;
    request.setWaypointsMetadata({params});
  }

  routing_manager->calculateRoute(request);
}

void RouteEngine::routeCalculated(RouteReply *reply) {
  if (reply->error() == RouteReply::NoError) {
    route = reply->route();
    qWarning() << "Got route response";
    segment = route.firstRouteSegment();
    auto path = route.path();
    emit routeUpdated(path);
  } else {
    qWarning() << "Got error in route reply" << reply->errorString();
  }

  sendRoute();

  reply->deleteLater();
}

void RouteEngine::sendRoute() {
  MessageBuilder msg;
  cereal::Event::Builder evt = msg.initEvent();
  cereal::NavRoute::Builder nav_route = evt.initNavRoute();

  qWarning() << "Distance: " << route.distance();
  route_geometry_segments.clear();
  // auto metadata = ((Route *) &route)->metadata();
  // const QByteArray &raw_reply = metadata["osrm.reply-json"].toByteArray();
  // QJsonArray osrm_routes = QJsonDocument::fromJson(raw_reply).object().value("routes").toArray();

  // for (const auto &r : osrm_routes) {
  //   if (!r.isObject())
  //       continue;

  //   auto route_object = r.toObject();
  //   auto route_distance = route_object.value("distance").toDouble();
  //   if (route_distance != route.distance()) {
  //       qWarning() << "Skipping route. Distance mismatch" << route_distance << route.distance();
  //       continue;
  //   }

  //   qWarning() << "Route: " << route_object;

  //   auto legs = route_object.value("legs").toArray();
  //   for (const auto &l : legs) {
  //     auto leg = l.toObject();
  //     auto annotation = leg.value("annotation").toObject();
  //     auto distances = annotation.value("distance").toArray();
  //     auto maxspeeds = annotation.value("maxspeed").toArray();
  //     auto size = std::min(distances.size(), maxspeeds.size());
  //     for (int i = 0; i < size; i++) {
  //       auto max_speed = maxspeeds.at(i).toObject();
  //       auto unknown = max_speed.value("unknown").toBool();
  //       auto speed = max_speed.value("speed").toDouble();
  //       auto unit = max_speed.value("unit").toString();
  //       auto speed_limit =
  //         unknown ? 0
  //         : unit == "km/h" ? speed * KPH_TO_MS
  //         : unit == "mph" ? speed * MPH_TO_MS
  //         : 0;

  //       route_geometry_segments.append({ distances.at(i).toDouble(), speed_limit });
  //     }
  //   }

  //   break;
  // }

  // NOTE: Qt seems to create duplicate coordinates in the path when decoding the polyline, so the corresponding annotations don't match up
  auto path = route.path();
  qWarning() << "Path: " << path;
  auto coordinates = nav_route.initCoordinates(path.size());
  size_t i = 0;
  for (auto const &c : route.path()) {
    coordinates[i].setLatitude(c.latitude());
    coordinates[i].setLongitude(c.longitude());
    i++;
  }

  pm->send("navRoute", msg);
}
