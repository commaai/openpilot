#!/usr/bin/env python3
import math
import os
import threading

import requests

import cereal.messaging as messaging
from cereal import log
from common.api import Api
from common.params import Params
from common.realtime import Ratekeeper
from selfdrive.navd.helpers import (Coordinate, coordinate_from_param,
                                    distance_along_geometry, maxspeed_to_ms,
                                    minimum_distance,
                                    parse_banner_instructions)
from selfdrive.swaglog import cloudlog

REROUTE_DISTANCE = 25
MANEUVER_TRANSITION_THRESHOLD = 10


class RouteEngine:
  def __init__(self, sm, pm):
    self.sm = sm
    self.pm = pm

    self.params = Params()

    # Get last gps position from params
    self.last_position = coordinate_from_param("LastGPSPosition", self.params)
    self.last_bearing = None

    self.gps_ok = False

    self.nav_destination = None
    self.step_idx = None
    self.route = None
    self.route_geometry = None

    self.recompute_backoff = 0
    self.recompute_countdown = 0

    self.ui_pid = None

    if "MAPBOX_TOKEN" in os.environ:
      self.mapbox_token = os.environ["MAPBOX_TOKEN"]
      self.mapbox_host = "https://api.mapbox.com"
    else:
      try:
        self.mapbox_token = Api(self.params.get("DongleId", encoding='utf8')).get_token(expiry_hours=4 * 7 * 24)
      except FileNotFoundError:
        cloudlog.exception("Failed to generate mapbox token due to missing private key. Ensure device is registered.")
        self.mapbox_token = ""
      self.mapbox_host = "https://maps.comma.ai"

  def update(self):
    self.sm.update(0)

    if self.sm.updated["managerState"]:
      ui_pid = [p.pid for p in self.sm["managerState"].processes if p.name == "ui" and p.running]
      if ui_pid:
        if self.ui_pid and self.ui_pid != ui_pid[0]:
          cloudlog.warning("UI restarting, sending route")
          threading.Timer(5.0, self.send_route).start()
        self.ui_pid = ui_pid[0]

    self.update_location()
    self.recompute_route()
    self.send_instruction()

  def update_location(self):
    location = self.sm['liveLocationKalman']
    self.gps_ok = location.gpsOK

    localizer_valid = (location.status == log.LiveLocationKalman.Status.valid) and location.positionGeodetic.valid

    if localizer_valid:
      self.last_bearing = math.degrees(location.calibratedOrientationNED.value[2])
      self.last_position = Coordinate(location.positionGeodetic.value[0], location.positionGeodetic.value[1])

  def recompute_route(self):
    if self.last_position is None:
      return

    new_destination = coordinate_from_param("NavDestination", self.params)
    if new_destination is None:
      self.clear_route()
      return

    should_recompute = self.should_recompute()
    if new_destination != self.nav_destination:
      cloudlog.warning(f"Got new destination from NavDestination param {new_destination}")
      should_recompute = True

    # Don't recompute when GPS drifts in tunnels
    if not self.gps_ok and self.step_idx is not None:
      return

    if self.recompute_countdown == 0 and should_recompute:
      self.recompute_countdown = 2**self.recompute_backoff
      self.recompute_backoff = min(6, self.recompute_backoff + 1)
      self.calculate_route(new_destination)
    else:
      self.recompute_countdown = max(0, self.recompute_countdown - 1)

  def calculate_route(self, destination):
    cloudlog.warning(f"Calculating route {self.last_position} -> {destination}")
    self.nav_destination = destination

    params = {
      'access_token': self.mapbox_token,
      'annotations': 'maxspeed',
      'geometries': 'geojson',
      'overview': 'full',
      'steps': 'true',
      'banner_instructions': 'true',
      'alternatives': 'false',
    }

    if self.last_bearing is not None:
      params['bearings'] = f"{(self.last_bearing + 360) % 360:.0f},90;"

    url = self.mapbox_host + f'/directions/v5/mapbox/driving-traffic/{self.last_position.longitude},{self.last_position.latitude};{destination.longitude},{destination.latitude}'
    try:
      resp = requests.get(url, params=params)
      resp.raise_for_status()

      r = resp.json()
      if len(r['routes']):
        self.route = r['routes'][0]['legs'][0]['steps']
        self.route_geometry = []

        maxspeed_idx = 0
        maxspeeds = r['routes'][0]['legs'][0]['annotation']['maxspeed']

        # Convert coordinates
        for step in self.route:
          coords = []

          for c in step['geometry']['coordinates']:
            coord = Coordinate.from_mapbox_tuple(c)

            # Last step does not have maxspeed
            if (maxspeed_idx < len(maxspeeds)) and ('unknown' not in maxspeeds[maxspeed_idx]):
              coord.annotations['maxspeed'] = maxspeed_to_ms(maxspeeds[maxspeed_idx])

            coords.append(coord)
            maxspeed_idx += 1

          self.route_geometry.append(coords)
          maxspeed_idx -= 1  # Every segment ends with the same coordinate as the start of the next

        self.step_idx = 0
      else:
        cloudlog.warning("Got empty route response")
        self.clear_route()

    except requests.exceptions.RequestException:
      cloudlog.exception("failed to get route")
      self.clear_route()

    self.send_route()

  def send_instruction(self):
    msg = messaging.new_message('navInstruction')

    if self.step_idx is None:
      msg.valid = False
      self.pm.send('navInstruction', msg)
      return

    step = self.route[self.step_idx]
    geometry = self.route_geometry[self.step_idx]
    along_geometry = distance_along_geometry(geometry, self.last_position)
    distance_to_maneuver_along_geometry = step['distance'] - along_geometry

    # Current instruction
    msg.navInstruction.maneuverDistance = distance_to_maneuver_along_geometry
    parse_banner_instructions(msg.navInstruction, step['bannerInstructions'], distance_to_maneuver_along_geometry)

    # Compute total remaining time and distance
    remaning = 1.0 - along_geometry / max(step['distance'], 1)
    total_distance = step['distance'] * remaning
    total_time = step['duration'] * remaning
    total_time_typical = step['duration_typical'] * remaning

    # Add up totals for future steps
    for i in range(self.step_idx + 1, len(self.route)):
      total_distance += self.route[i]['distance']
      total_time += self.route[i]['duration']
      total_time_typical += self.route[i]['duration_typical']

    msg.navInstruction.distanceRemaining = total_distance
    msg.navInstruction.timeRemaining = total_time
    msg.navInstruction.timeRemainingTypical = total_time_typical

    # Speed limit
    closest_idx, closest = min(enumerate(geometry), key=lambda p: p[1].distance_to(self.last_position))
    if closest_idx > 0:
      # If we are not past the closest point, show previous
      if along_geometry < distance_along_geometry(geometry, geometry[closest_idx]):
        closest = geometry[closest_idx - 1]

    if 'maxspeed' in closest.annotations:
      msg.navInstruction.speedLimit = closest.annotations['maxspeed']

    # Speed limit sign type
    if 'speedLimitSign' in step:
      if step['speedLimitSign'] == 'mutcd':
        msg.navInstruction.speedLimitSign = log.NavInstruction.SpeedLimitSign.mutcd
      elif step['speedLimitSign'] == 'vienna':
        msg.navInstruction.speedLimitSign = log.NavInstruction.SpeedLimitSign.vienna

    self.pm.send('navInstruction', msg)

    # Transition to next route segment
    if distance_to_maneuver_along_geometry < -MANEUVER_TRANSITION_THRESHOLD:
      if self.step_idx + 1 < len(self.route):
        self.step_idx += 1
        self.recompute_backoff = 0
        self.recompute_countdown = 0
      else:
        cloudlog.warning("Destination reached")
        Params().delete("NavDestination")

        # Clear route if driving away from destination
        dist = self.nav_destination.distance_to(self.last_position)
        if dist > REROUTE_DISTANCE:
          self.clear_route()

  def send_route(self):
    coords = []

    if self.route is not None:
      for path in self.route_geometry:
        coords += [c.as_dict() for c in path]

    msg = messaging.new_message('navRoute')
    msg.navRoute.coordinates = coords
    self.pm.send('navRoute', msg)

  def clear_route(self):
    self.route = None
    self.route_geometry = None
    self.step_idx = None
    self.nav_destination = None

  def should_recompute(self):
    if self.step_idx is None or self.route is None:
      return True

    # Don't recompute in last segment, assume destination is reached
    if self.step_idx == len(self.route) - 1:
      return False

    # Compute closest distance to all line segments in the current path
    min_d = REROUTE_DISTANCE + 1
    path = self.route_geometry[self.step_idx]
    for i in range(len(path) - 1):
      a = path[i]
      b = path[i + 1]

      if a.distance_to(b) < 1.0:
        continue

      min_d = min(min_d, minimum_distance(a, b, self.last_position))

    return min_d > REROUTE_DISTANCE

    # TODO: Check for going wrong way in segment


def main(sm=None, pm=None):
  if sm is None:
    sm = messaging.SubMaster(['liveLocationKalman', 'managerState'])
  if pm is None:
    pm = messaging.PubMaster(['navInstruction', 'navRoute'])

  rk = Ratekeeper(1.0)
  route_engine = RouteEngine(sm, pm)
  while True:
    route_engine.update()
    rk.keep_time()


if __name__ == "__main__":
  main()
