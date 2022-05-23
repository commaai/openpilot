#!/usr/bin/env python3
import json
import math
import os
import time

import requests

import cereal.messaging as messaging
from cereal import log
from common.api import Api
from common.params import Params
from common.numpy_fast import clip
from selfdrive.swaglog import cloudlog

EARTH_MEAN_RADIUS = 6371007.2
REROUTE_DISTANCE = 25
MANEUVER_TRANSITION_THRESHOLD = 10

class Coordinate:
  def __init__(self, latitude, longitude):
    self.latitude = latitude
    self.longitude = longitude

  @classmethod
  def from_mapbox_tuple(cls, t):
    return cls(t[1], t[0])

  def as_dict(self):
    return {'latitude': self.latitude, 'longitude': self.longitude}

  def __str__(self):
    return f"({self.latitude}, {self.longitude})"

  def __eq__(self, other):
    if not isinstance(other, Coordinate):
      return False
    return (self.latitude == other.latitude) and (self.longitude == other.longitude)

  def __sub__(self, other):
    return Coordinate(self.latitude - other.latitude, self.longitude - other.longitude)

  def __add__(self, other):
    return Coordinate(self.latitude + other.latitude, self.longitude + other.longitude)

  def __mul__(self, c):
    return Coordinate(self.latitude * c, self.longitude * c)

  def dot(self, other):
    return self.latitude * other.latitude + self.longitude * other.longitude

  def distance_to(self, other):
    # Haversine formula
    dlat = math.radians(other.latitude - self.latitude)
    dlon = math.radians(other.longitude - self.longitude)

    haversine_dlat = math.sin(dlat / 2.0)
    haversine_dlat *= haversine_dlat
    haversine_dlon = math.sin(dlon / 2.0)
    haversine_dlon *= haversine_dlon

    y = haversine_dlat \
             + math.cos(math.radians(self.latitude)) \
             * math.cos(math.radians(other.latitude)) \
             * haversine_dlon
    x = 2 * math.asin(math.sqrt(y))
    return x * EARTH_MEAN_RADIUS


def minimum_distance(a, b, p):
  if a.distance_to(b) < 0.01:
    return a.distanceTo(p)

  ap = p - a
  ab = b - a
  t = clip(ap.dot(ab) / ab.dot(ab), 0.0, 1.0)
  projection = a + ab * t
  return projection.distance_to(p)


def distance_along_geometry(geometry, pos):
  if len(geometry) <= 2:
    return geometry[0].distance_to(pos)

  # 1. Find segment that is closest to current position
  # 2. Total distance is sum of distance to start of closest segment
  #    + all previous segments
  total_distance = 0
  total_distance_closest = 0
  closest_distance = 1e9

  for i in range(len(geometry) - 1):
    d = minimum_distance(geometry[i], geometry[i + 1], pos)

    if d < closest_distance:
      closest_distance = d
      total_distance_closest = total_distance + geometry[i].distance_to(pos)

    total_distance += geometry[i].distance_to(geometry[i + 1])

  return total_distance_closest


def coordinate_from_param(param):
  json_str = Params().get(param)
  if json_str is None:
    return None

  pos = json.loads(json_str)
  if 'latitude' not in pos or 'longitude' not in pos:
    return None

  return Coordinate(pos['latitude'], pos['longitude'])


def string_to_direction(direction):
  for d in ['left', 'right', 'straight']:
    if d in direction:
      return d
  return 'none'


def parse_banner_instructions(instruction, banners, distance_to_maneuver=0):
  current_banner = banners[0]

  # A segment can contain multiple banners, find one that we need to show now
  for banner in banners:
    if distance_to_maneuver < banner['distanceAlongGeometry']:
      current_banner = banner

  # Only show banner when close enough to maneuver
  instruction.showFull = distance_to_maneuver < current_banner['distanceAlongGeometry']

  # Primary
  p = current_banner['primary']
  if 'text' in p:
    instruction.maneuverPrimaryText = p['text']
  if 'type' in p:
    instruction.maneuverType = p['type']
  if 'modifier' in p:
    instruction.maneuverModifier = p['modifier']

  # Secondary
  if 'secondary' in current_banner:
    instruction.maneuverSecondaryText = current_banner['secondary']['text']

  # Lane lines
  if 'sub' in current_banner:
    lanes = []
    for component in current_banner['sub']['components']:
      if component['type'] != 'lane':
        continue

      lane = {
        'active': component['active'],
        'directions': [string_to_direction(d) for d in component['directions']],
      }

      if 'active_direction' in component:
        lane['activeDirection'] = string_to_direction(component['active_direction'])

      lanes.append(lane)
    instruction.lanes = lanes


class RouteEngine:
  def __init__(self, sm, pm) -> None:
    self.sm = sm
    self.pm = pm

    self.last_bearing = None
    self.last_position = None
    self.gps_ok = False

    self.nav_destination = None
    self.step_idx = None
    self.route = None

    self.recompute_backoff = 0
    self.recompute_countdown = 0

    if "MAPBOX_TOKEN" in os.environ:
      self.mapbox_token = os.environ["MAPBOX_TOKEN"]
      self.mapbox_host = "https://api.mapbox.com"
    else:
      self.mapbox_token = Api(Params().get("DongleId", encoding='utf8')).get_token()
      self.mapbox_host = "https://maps.comma.ai"

  def update(self):
    self.sm.update(0)
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
    remaning = 1.0 - along_geometry / step['distance']
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
        Params().remove("NavDestination")

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
    return False


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
