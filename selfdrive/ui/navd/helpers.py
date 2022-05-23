import math
import json

from common.numpy_fast import clip
from common.params import Params

EARTH_MEAN_RADIUS = 6371007.2


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
    return a.distance_to(p)

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
