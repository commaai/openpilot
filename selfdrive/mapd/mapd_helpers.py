import math
import numpy as np
from datetime import datetime

from selfdrive.config import Conversions as CV
from common.transformations.coordinates import LocalCoord, geodetic2ecef

LOOKAHEAD_TIME = 10.
MAPS_LOOKAHEAD_DISTANCE = 50 * LOOKAHEAD_TIME


def circle_through_points(p1, p2, p3):
  """Fits a circle through three points
  Formulas from: http://www.ambrsoft.com/trigocalc/circle3d.htm"""
  x1, y1, _ = p1
  x2, y2, _ = p2
  x3, y3, _ = p3

  A = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2
  B = (x1**2 + y1**2) * (y3 - y2) + (x2**2 + y2**2) * (y1 - y3) + (x3**2 + y3**2) * (y2 - y1)
  C = (x1**2 + y1**2) * (x2 - x3) + (x2**2 + y2**2) * (x3 - x1) + (x3**2 + y3**2) * (x1 - x2)
  D = (x1**2 + y1**2) * (x3 * y2 - x2 * y3) + (x2**2 + y2**2) * (x1 * y3 - x3 * y1) + (x3**2 + y3**2) * (x2 * y1 - x1 * y2)

  return (-B / (2 * A), - C / (2 * A), np.sqrt((B**2 + C**2 - 4 * A * D) / (4 * A**2)))


def parse_speed_unit(max_speed):
  """Converts a maxspeed string to m/s based on the unit present in the input.
  OpenStreetMap defaults to kph if no unit is present. """

  conversion = CV.KPH_TO_MS
  if 'mph' in max_speed:
    max_speed = max_speed.replace(' mph', '')
    conversion = CV.MPH_TO_MS

  return float(max_speed) * conversion


class Way:
  def __init__(self, way):
    self.id = way.id
    self.way = way

    points = list()

    for node in self.way.get_nodes(resolve_missing=False):
      points.append((float(node.lat), float(node.lon), 0.))

    self.points = np.asarray(points)

  @classmethod
  def closest(cls, query_results, lat, lon, heading):
    results, tree, real_nodes, node_to_way = query_results

    cur_pos = geodetic2ecef((lat, lon, 0))
    nodes = tree.query_ball_point(cur_pos, 500)

    # If no nodes within 500m, choose closest one
    if not nodes:
      nodes = [tree.query(cur_pos)[1]]

    ways = []
    for n in nodes:
      real_node = real_nodes[n]
      ways += node_to_way[real_node.id]
    ways = set(ways)

    closest_way = None
    best_score = None
    for way in ways:
      way = Way(way)
      points = way.points_in_car_frame(lat, lon, heading)

      on_way = way.on_way(lat, lon, heading, points)
      if not on_way:
        continue

      # Create mask of points in front and behind
      x = points[:, 0]
      y = points[:, 1]
      angles = np.arctan2(y, x)
      front = np.logical_and((-np.pi / 2) < angles,
                                angles < (np.pi / 2))
      behind = np.logical_not(front)

      dists = np.linalg.norm(points, axis=1)

      # Get closest point behind the car
      dists_behind = np.copy(dists)
      dists_behind[front] = np.NaN
      closest_behind = points[np.nanargmin(dists_behind)]

      # Get closest point in front of the car
      dists_front = np.copy(dists)
      dists_front[behind] = np.NaN
      closest_front = points[np.nanargmin(dists_front)]

      # fit line: y = a*x + b
      x1, y1, _ = closest_behind
      x2, y2, _ = closest_front
      a = (y2 - y1) / max((x2 - x1), 1e-5)
      b = y1 - a * x1

      # With a factor of 60 a 20m offset causes the same error as a 20 degree heading error
      # (A 20 degree heading offset results in an a of about 1/3)
      score = abs(a) * 60. + abs(b)
      if closest_way is None or score < best_score:
        closest_way = way
        best_score = score

    return closest_way

  def __str__(self):
    return "%s %s" % (self.id, self.way.tags)

  @property
  def max_speed(self):
    """Extracts the (conditional) speed limit from a way"""
    if not self.way:
      return None

    tags = self.way.tags
    max_speed = None

    if 'maxspeed' in tags:
      max_speed = parse_speed_unit(tags['maxspeed'])

    if 'maxspeed:conditional' in tags:
      max_speed_cond, cond = tags['maxspeed:conditional'].split(' @ ')
      cond = cond[1:-1]

      start, end = cond.split('-')
      now = datetime.now()  # TODO: Get time and timezone from gps fix so this will work correctly on replays
      start = datetime.strptime(start, "%H:%M").replace(year=now.year, month=now.month, day=now.day)
      end = datetime.strptime(end, "%H:%M").replace(year=now.year, month=now.month, day=now.day)

      if start <= now <= end:
        max_speed = parse_speed_unit(max_speed_cond)

    return max_speed

  def on_way(self, lat, lon, heading, points=None):
    if points is None:
      points = self.points_in_car_frame(lat, lon, heading)
    x = points[:, 0]
    return np.min(x) < 0. and np.max(x) > 0.

  def closest_point(self, lat, lon, heading, points=None):
    if points is None:
      points = self.points_in_car_frame(lat, lon, heading)
    i = np.argmin(np.linalg.norm(points, axis=1))
    return points[i]

  def distance_to_closest_node(self, lat, lon, heading, points=None):
    if points is None:
      points = self.points_in_car_frame(lat, lon, heading)
    return np.min(np.linalg.norm(points, axis=1))

  def points_in_car_frame(self, lat, lon, heading):
    lc = LocalCoord.from_geodetic([lat, lon, 0.])

    # Build rotation matrix
    heading = math.radians(-heading + 90)
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])

    # Convert to local coordinates
    points_carframe = lc.geodetic2ned(self.points).T

    # Rotate with heading of car
    points_carframe = np.dot(rot, points_carframe[(1, 0, 2), :]).T

    return points_carframe

  def next_way(self, query_results, lat, lon, heading, backwards=False):
    results, tree, real_nodes, node_to_way = query_results

    if backwards:
      node = self.way.nodes[0]
    else:
      node = self.way.nodes[-1]

    ways = node_to_way[node.id]

    way = None
    try:
      # Simple heuristic to find next way
      ways = [w for w in ways if w.id != self.id and w.tags['highway'] == self.way.tags['highway']]
      if len(ways) == 1:
        way = Way(ways[0])
    except KeyError:
      pass

    return way

  def get_lookahead(self, query_results, lat, lon, heading, lookahead):
    pnts = None
    way = self
    valid = False

    for i in range(5):
      # Get new points and append to list
      new_pnts = way.points_in_car_frame(lat, lon, heading)

      if pnts is None:
        pnts = new_pnts
      else:
        pnts = np.vstack([pnts, new_pnts])

      # Check current lookahead distance
      max_dist = np.linalg.norm(pnts[-1, :])
      if max_dist > lookahead:
        valid = True

      if max_dist > 2 * lookahead:
        break

      # Find next way
      way = way.next_way(query_results, lat, lon, heading)
      if not way:
        break

    return pnts, valid
