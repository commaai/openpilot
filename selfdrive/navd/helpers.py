from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from common.conversions import Conversions
from common.numpy_fast import clip
from common.params import Params

EARTH_MEAN_RADIUS = 6371007.2
SPEED_CONVERSIONS = {
    'km/h': Conversions.KPH_TO_MS,
    'mph': Conversions.MPH_TO_MS,
  }


class Coordinate:
  def __init__(self, latitude: float, longitude: float) -> None:
    self.latitude = latitude
    self.longitude = longitude
    self.annotations: Dict[str, float] = {}

  @classmethod
  def from_mapbox_tuple(cls, t: Tuple[float, float]) -> Coordinate:
    return cls(t[1], t[0])

  def as_dict(self) -> Dict[str, float]:
    return {'latitude': self.latitude, 'longitude': self.longitude}

  def __str__(self) -> str:
    return f'Coordinate({self.latitude}, {self.longitude})'

  def __repr__(self) -> str:
    return self.__str__()

  def __eq__(self, other) -> bool:
    if not isinstance(other, Coordinate):
      return False
    return (self.latitude == other.latitude) and (self.longitude == other.longitude)

  def __sub__(self, other: Coordinate) -> Coordinate:
    return Coordinate(self.latitude - other.latitude, self.longitude - other.longitude)

  def __add__(self, other: Coordinate) -> Coordinate:
    return Coordinate(self.latitude + other.latitude, self.longitude + other.longitude)

  def __mul__(self, c: float) -> Coordinate:
    return Coordinate(self.latitude * c, self.longitude * c)

  def dot(self, other: Coordinate) -> float:
    return self.latitude * other.latitude + self.longitude * other.longitude

  def distance_to(self, other: Coordinate) -> float:
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


def minimum_distance(a: Coordinate, b: Coordinate, p: Coordinate):
  if a.distance_to(b) < 0.01:
    return a.distance_to(p)

  ap = p - a
  ab = b - a
  t = clip(ap.dot(ab) / ab.dot(ab), 0.0, 1.0)
  projection = a + ab * t
  return projection.distance_to(p)


def distance_along_geometry(geometry: List[Coordinate], pos: Coordinate) -> float:
  if len(geometry) <= 2:
    return geometry[0].distance_to(pos)

  # 1. Find segment that is closest to current position
  # 2. Total distance is sum of distance to start of closest segment
  #    + all previous segments
  total_distance = 0.0
  total_distance_closest = 0.0
  closest_distance = 1e9

  for i in range(len(geometry) - 1):
    d = minimum_distance(geometry[i], geometry[i + 1], pos)

    if d < closest_distance:
      closest_distance = d
      total_distance_closest = total_distance + geometry[i].distance_to(pos)

    total_distance += geometry[i].distance_to(geometry[i + 1])

  return total_distance_closest


def coordinate_from_param(param: str, params: Optional[Params] = None) -> Optional[Coordinate]:
  if params is None:
    params = Params()

  json_str = params.get(param)
  if json_str is None:
    return None

  pos = json.loads(json_str)
  if 'latitude' not in pos or 'longitude' not in pos:
    return None

  return Coordinate(pos['latitude'], pos['longitude'])


def string_to_direction(direction: str) -> str:
  for d in ['left', 'right', 'straight']:
    if d in direction:
      return d
  return 'none'


def maxspeed_to_ms(maxspeed: Dict[str, Union[str, float]]) -> float:
  unit = cast(str, maxspeed['unit'])
  speed = cast(float, maxspeed['speed'])
  return SPEED_CONVERSIONS[unit] * speed


def field_valid(dat: dict, field: str) -> bool:
  return field in dat and dat[field] is not None


def parse_banner_instructions(instruction: Any, banners: Any, distance_to_maneuver: float = 0.0) -> None:
  if not len(banners):
    return

  current_banner = banners[0]

  # A segment can contain multiple banners, find one that we need to show now
  for banner in banners:
    if distance_to_maneuver < banner['distanceAlongGeometry']:
      current_banner = banner

  # Only show banner when close enough to maneuver
  instruction.showFull = distance_to_maneuver < current_banner['distanceAlongGeometry']

  # Primary
  p = current_banner['primary']
  if field_valid(p, 'text'):
    instruction.maneuverPrimaryText = p['text']
  if field_valid(p, 'type'):
    instruction.maneuverType = p['type']
  if field_valid(p, 'modifier'):
    instruction.maneuverModifier = p['modifier']

  # Secondary
  if field_valid(current_banner, 'secondary'):
    instruction.maneuverSecondaryText = current_banner['secondary']['text']

  # Lane lines
  if field_valid(current_banner, 'sub'):
    lanes = []
    for component in current_banner['sub']['components']:
      if component['type'] != 'lane':
        continue

      lane = {
        'active': component['active'],
        'directions': [string_to_direction(d) for d in component['directions']],
      }

      if field_valid(component, 'active_direction'):
        lane['activeDirection'] = string_to_direction(component['active_direction'])

      lanes.append(lane)
    instruction.lanes = lanes
