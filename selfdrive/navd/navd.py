#!/usr/bin/env python3
import math
import os
import time

import requests

import cereal.messaging as messaging
from cereal import log
from common.api import Api
from common.params import Params
from selfdrive.swaglog import cloudlog
from selfdrive.navd.helpers import (Coordinate, coordinate_from_param,
                                    distance_along_geometry,
                                    minimum_distance,
                                    parse_banner_instructions)

REROUTE_DISTANCE = 25
MANEUVER_TRANSITION_THRESHOLD = 10


class RouteEngine:
  def __init__(self, sm, pm):
    self.sm = sm
    self.pm = pm

    # Get last gps position from params
    self.last_position = coordinate_from_param("LastGPSPosition")
    self.last_bearing = None

    self.gps_ok = False

    self.nav_destination = None
    self.step_idx = None
    self.route = None

    self.recompute_backoff = 0
    self.recompute_countdown = 0

    self.ui_pid = None

    if "MAPBOX_TOKEN" in os.environ:
      self.mapbox_token = os.environ["MAPBOX_TOKEN"]
      self.mapbox_host = "https://api.mapbox.com"
    else:
      self.mapbox_token = Api(Params().get("DongleId", encoding='utf8')).get_token()
      self.mapbox_host = "https://maps.comma.ai"

  def update(self):
    self.sm.update(0)

    if self.sm.updated["managerState"]:
      ui_pid = [p.pid for p in self.sm["managerState"].processes if p.name == "ui"]
      if ui_pid:
        if self.ui_pid and self.ui_pid != ui_pid[0]:
          cloudlog.warning("UI restarting, sending route")
          # TODO: Send new route with delay of 5 seconds
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

    new_destination = coordinate_from_param("NavDestination")
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
      # 'annotations': 'maxspeed',
      'geometries': 'geojson',
      'overview': 'full',
      'steps': 'true',
      'banner_instructions': 'true',
      'alternatives': 'false',
    }

    if self.last_bearing is not None:
      params['bearings'] = f"{(self.last_bearing + 360) % 360:.0f},90;"

    url = self.mapbox_host + f'/directions/v5/mapbox/driving-traffic/{self.last_position.longitude},{self.last_position.latitude};{destination.longitude},{destination.latitude}'
    resp = requests.get(url, params=params)

    if resp.status_code == 200:
      r = resp.json()
      if len(r['routes']):
        self.route = r['routes'][0]['legs'][0]['steps']

        # Convert coordinates
        for step in self.route:
          step['geometry']['coordinates'] = [Coordinate.from_mapbox_tuple(c) for c in step['geometry']['coordinates']]

        self.step_idx = 0
      else:
        cloudlog.warning("Got empty route response")
        self.route = None

    else:
      cloudlog.warning(f"Got error in route reply {resp.status_code}")
      self.route = None

    self.send_route()

  def send_instruction(self):
    if self.step_idx is None:
      return

    step = self.route[self.step_idx]
    along_geometry = distance_along_geometry(step['geometry']['coordinates'], self.last_position)
    distance_to_maneuver_along_geometry = step['distance'] - along_geometry

    msg = messaging.new_message('navInstruction')

    # Current instruction
    msg.navInstruction.maneuverDistance = distance_to_maneuver_along_geometry
    parse_banner_instructions(msg.navInstruction, step['bannerInstructions'], distance_to_maneuver_along_geometry)

    # Compute total remaining time and distance
    remaning = 1.0 - along_geometry / max(step['distance'], 1)
    total_distance = step['distance'] * remaning
    total_time = step['duration'] * remaning
    total_time_typical = step['duration_typical'] * remaning

    for i in range(self.step_idx + 1, len(self.route)):
      total_distance += self.route[i]['distance']
      total_time += self.route[i]['duration']
      total_time_typical += self.route[i]['duration_typical']

    msg.navInstruction.distanceRemaining = total_distance
    msg.navInstruction.timeRemaining = total_time
    msg.navInstruction.timeRemainingTypical = total_time_typical

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
      for step in self.route:
        for c in step['geometry']['coordinates']:
          coords.append(c.as_dict())

    msg = messaging.new_message('navRoute')
    msg.navRoute.coordinates = coords
    self.pm.send('navRoute', msg)

  def clear_route(self):
    self.route = None
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
    path = self.route[self.step_idx]['geometry']['coordinates']
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

  route_engine = RouteEngine(sm, pm)
  while True:
    route_engine.update()

    # TODO: use ratekeeper
    time.sleep(1)


if __name__ == "__main__":
  main()
