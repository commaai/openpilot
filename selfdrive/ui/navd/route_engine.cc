#include "selfdrive/ui/navd/route_engine.h"

#include <QDebug>

#include "selfdrive/ui/qt/maps/map.h"
#include "selfdrive/ui/qt/maps/map_helpers.h"
#include "selfdrive/ui/qt/api.h"

#include "selfdrive/common/params.h"

const qreal REROUTE_DISTANCE = 25;
const float MANEUVER_TRANSITION_THRESHOLD = 10;
const float UPDATE_FREQ = 20.0;  // Hz

static float get_time_typical(const QGeoRouteSegment &segment) {
  auto maneuver = segment.maneuver();
  auto attrs = maneuver.extendedAttributes();
  return attrs.contains("mapbox.duration_typical") ? attrs["mapbox.duration_typical"].toDouble() : segment.travelTime();
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

static void parse_banner(cereal::NavInstruction::Builder &instruction, const QMap<QString, QVariant> &banner, bool full) {
  QString primary_str, secondary_str;

  auto p = banner["primary"].toMap();
  primary_str += p["text"].toString();

  instruction.setShowFull(full);

  if (p.contains("type")) {
    instruction.setManeuverType(p["type"].toString().toStdString());
  }

  if (p.contains("modifier")) {
    instruction.setManeuverModifier(p["modifier"].toString().toStdString());
  }

  if (banner.contains("secondary")) {
    auto s = banner["secondary"].toMap();
    secondary_str += s["text"].toString();
  }

  instruction.setManeuverPrimaryText(primary_str.toStdString());
  instruction.setManeuverSecondaryText(secondary_str.toStdString());

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

    auto lanes = instruction.initLanes(num_lanes);

    size_t i = 0;
    for (auto &c : components) {
      auto cc = c.toMap();
      if (cc["type"].toString() == "lane") {
        auto lane = lanes[i];
        lane.setActive(cc["active"].toBool());

        if (cc.contains("active_direction")) {
          lane.setActiveDirection(string_to_direction(cc["active_direction"].toString()));
        }

        auto directions = lane.initDirections(cc["directions"].toList().size());

        size_t j = 0;
        for (auto &dir : cc["directions"].toList()) {
          directions.set(j, string_to_direction(dir.toString()));
          j++;
        }


        i++;
      }
    }
  }

}

RouteEngine::RouteEngine() {
  sm = new SubMaster({"liveLocationKalman", "managerState"});
  pm = new PubMaster({"navInstruction", "navRoute"});

  // Timers
  timer = new QTimer(this);
  QObject::connect(timer, SIGNAL(timeout()), this, SLOT(timerUpdate()));
  timer->start(1000 * 1 / UPDATE_FREQ);

  // Build routing engine
  QVariantMap parameters;
  parameters["mapbox.access_token"] = get_mapbox_token();
  parameters["mapbox.directions_api_url"] = MAPS_HOST + "/directions/v5/mapbox/";

  geoservice_provider = new QGeoServiceProvider("mapbox", parameters);
  routing_manager = geoservice_provider->routingManager();
  if (routing_manager == nullptr) {
    qWarning() << geoservice_provider->errorString();
    assert(routing_manager);
  }
  QObject::connect(routing_manager, &QGeoRoutingManager::finished, this, &RouteEngine::routeCalculated);

  // Get last gps position from params
  auto last_gps_position = coordinate_from_param("LastGPSPosition");
  if (last_gps_position) {
    last_position = *last_gps_position;
  }
}

void RouteEngine::timerUpdate() {
  sm->update(0);
  if (!sm->updated("liveLocationKalman")) {
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

  recomputeRoute();

  MessageBuilder msg;
  cereal::Event::Builder evt = msg.initEvent(segment.isValid());
  cereal::NavInstruction::Builder instruction = evt.initNavInstruction();

  // Show route instructions
  if (segment.isValid()) {
    auto cur_maneuver = segment.maneuver();
    auto attrs = cur_maneuver.extendedAttributes();
    if (cur_maneuver.isValid() && attrs.contains("mapbox.banner_instructions")) {
      float along_geometry = distance_along_geometry(segment.path(), to_QGeoCoordinate(*last_position));
      float distance_to_maneuver_along_geometry = segment.distance() - along_geometry;

      auto banners = attrs["mapbox.banner_instructions"].toList();
      if (banners.size()) {
        auto banner = banners[0].toMap();

        for (auto &b : banners) {
          auto bb = b.toMap();
          if (distance_to_maneuver_along_geometry < bb["distance_along_geometry"].toDouble()) {
            banner = bb;
          }
        }

        instruction.setManeuverDistance(distance_to_maneuver_along_geometry);
        parse_banner(instruction, banner, distance_to_maneuver_along_geometry < banner["distance_along_geometry"].toDouble());

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
      }

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
  }

  pm->send("navInstruction", msg);
}

void RouteEngine::clearRoute() {
  route = QGeoRoute();
  segment = QGeoRouteSegment();
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

void RouteEngine::routeCalculated(QGeoRouteReply *reply) {
  if (reply->error() == QGeoRouteReply::NoError) {
    if (reply->routes().size() != 0) {
      qWarning() << "Got route response";

      route = reply->routes().at(0);
      segment = route.firstRouteSegment();

      auto path = route.path();
      emit routeUpdated(path);
    } else {
      qWarning() << "Got empty route response";
    }
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

  auto path = route.path();
  auto coordinates = nav_route.initCoordinates(path.size());

  size_t i = 0;
  for (auto const &c : route.path()) {
    coordinates[i].setLatitude(c.latitude());
    coordinates[i].setLongitude(c.longitude());
    i++;
  }

  pm->send("navRoute", msg);
}
