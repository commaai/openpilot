#include "selfdrive/ui/navd/route_engine.h"

#include <QDebug>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>

#include "selfdrive/ui/navd/routing_manager.h"
#include "selfdrive/ui/qt/api.h"
#include "selfdrive/ui/qt/maps/map.h"
#include "selfdrive/ui/qt/maps/map_helpers.h"

#include "selfdrive/common/params.h"

const qreal REROUTE_DISTANCE = 25;
const float MANEUVER_TRANSITION_THRESHOLD = 10;

static float get_time_typical(const RouteSegment &segment) {
  return std::fmax(segment.maneuver.typical_duration, segment.travel_time);
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
  sm = new SubMaster({ "liveLocationKalman", "managerState" });
  pm = new PubMaster({ "navInstruction", "navRoute" });

  // Timers
  route_timer = new QTimer(this);
  QObject::connect(route_timer, SIGNAL(timeout()), this, SLOT(routeUpdate()));
  route_timer->start(1000);

  msg_timer = new QTimer(this);
  QObject::connect(msg_timer, SIGNAL(timeout()), this, SLOT(msgUpdate()));
  msg_timer->start(50);

  routing_manager = new RoutingManager();
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
        if (ui_pid && *ui_pid != p.getPid()) {
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
  cereal::Event::Builder evt = msg.initEvent(segment.has_value());
  cereal::NavInstruction::Builder instruction = evt.initNavInstruction();

  if (segment) {
    auto maneuver = segment->maneuver;
    auto along_geometry = distance_along_geometry(segment->path, to_QGeoCoordinate(*last_position));
    float distance_to_maneuver_along_geometry = segment->distance - along_geometry.first;

    // Route instructions
    instruction.setShowFull(distance_to_maneuver_along_geometry < maneuver.distance_along_geometry);
    if (maneuver.type)
      instruction.setManeuverType(maneuver.type->toStdString());
    if (maneuver.modifier)
      instruction.setManeuverModifier(maneuver.modifier->toStdString());
    if (maneuver.primary_text)
      instruction.setManeuverPrimaryText(maneuver.primary_text->toStdString());
    if (maneuver.secondary_text)
      instruction.setManeuverSecondaryText(maneuver.secondary_text->toStdString());
    instruction.setManeuverDistance(distance_to_maneuver_along_geometry);
    if (along_geometry.second < segment->annotations.size())
      instruction.setSpeedLimit(segment->annotations[along_geometry.second].speed_limit);

    // Lanes
    if (maneuver.lanes) {
      auto lanes = instruction.initLanes(maneuver.lanes->size());
      for (int i = 0; i < maneuver.lanes->size(); i++) {
        auto &l = maneuver.lanes->at(i);
        auto lane = lanes[i];
        lane.setActive(l.active);
        if (l.active_direction)
          lane.setActiveDirection(string_to_direction(l.active_direction.value()));
        auto directions = lane.initDirections(l.directions.size());
        for (int j = 0; j < l.directions.size(); j++) {
          directions.set(j, string_to_direction(l.directions[j]));
        }
      }
    }

    // ETA
    float progress = along_geometry.first / segment->distance;
    float total_distance = segment->distance * (1.0 - progress);
    float total_time = segment->travel_time * (1.0 - progress);
    float total_time_typical = get_time_typical(segment.value()) * (1.0 - progress);

    for (int i = segment_index + 1; i < route->segments.size(); i++) {
      auto &s = route->segments.at(i);
      total_distance += s.distance;
      total_time += s.travel_time;
      total_time_typical += get_time_typical(s);
    }
    instruction.setTimeRemaining(total_time);
    instruction.setTimeRemainingTypical(total_time_typical);
    instruction.setDistanceRemaining(total_distance);

    // Transition to next route segment
    if (distance_to_maneuver_along_geometry < -MANEUVER_TRANSITION_THRESHOLD) {
      if (segment_index + 1 < route->segments.size()) {
        segment_index++;
        segment = route->segments.at(segment_index);

        recompute_backoff = std::max(0, recompute_backoff - 1);
        recompute_countdown = 0;
      } else {
        qWarning() << "Destination reached";
        Params().remove("NavDestination");

        // Clear route if driving away from destination
        float d = segment->maneuver.position.distanceTo(to_QGeoCoordinate(*last_position));
        if (d > REROUTE_DISTANCE) {
          clearRoute();
        }
      }
    }
  }

  pm->send("navInstruction", msg);
}

void RouteEngine::clearRoute() {
  route = {};
  segment = {};
  segment_index = 0;
  nav_destination = {};
}

bool RouteEngine::shouldRecompute() {
  if (!segment) {
    return true;
  }

  // Don't recompute in last segment, assume destination is reached
  if (segment_index + 1 >= route->segments.size()) {
    return false;
  }

  // Compute closest distance to all line segments in the current path
  float min_d = REROUTE_DISTANCE + 1;
  auto path = segment->path;
  auto cur = to_QGeoCoordinate(*last_position);
  for (size_t i = 0; i < path.size() - 1; i++) {
    auto a = path[i];
    auto b = path[i + 1];
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

  if (!gps_ok && segment)
    return; // Don't recompute when gps drifts in tunnels

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
    request.setWaypointsMetadata({ params });
  }

  routing_manager->calculateRoute(request);
}

void RouteEngine::routeCalculated(RouteReply *reply) {
  if (reply->reply_error == RouteReply::NoError) {
    route = reply->route;
    segment = route->segments.first();
    segment_index = 0;
    emit routeUpdated(route->path);
    sendRoute();
  } else {
    qWarning() << "Got error in route reply" << reply->error_string;
  }

  reply->deleteLater();
}

void RouteEngine::sendRoute() {
  MessageBuilder msg;
  cereal::Event::Builder evt = msg.initEvent();
  cereal::NavRoute::Builder nav_route = evt.initNavRoute();

  if (route) {
    auto coordinates = nav_route.initCoordinates(route->path.size());
    size_t i = 0;
    for (auto const &c : route->path) {
      coordinates[i].setLatitude(c.latitude());
      coordinates[i].setLongitude(c.longitude());
      i++;
    }
  }

  pm->send("navRoute", msg);
}
