from selfdrive.mapd.lib.geo import DIRECTION, R, vectors, bearing_to_points, distance_to_points
from selfdrive.mapd.lib.osm import create_way
from common.conversions import Conversions as CV
from selfdrive.mapd.config import LANE_WIDTH
from common.basedir import BASEDIR
from datetime import datetime as dt
import numpy as np
import re
import json


_WAY_BBOX_PADING = 80. / R  # 80 mts of padding to bounding box. (expressed in radians)


with open(BASEDIR + "/selfdrive/mapd/lib/default_speeds.json", "rb") as f:
  _COUNTRY_LIMITS = json.loads(f.read())


_WD = {
  'Mo': 0,
  'Tu': 1,
  'We': 2,
  'Th': 3,
  'Fr': 4,
  'Sa': 5,
  'Su': 6
}

_HIGHWAY_RANK = {
  'motorway': 0,
  'motorway_link': 1,
  'trunk': 10,
  'trunk_link': 11,
  'primary': 20,
  'primary_link': 21,
  'secondary': 30,
  'secondary_link': 31,
  'tertiary': 40,
  'tertiary_link': 41,
  'unclassified': 50,
  'residential': 60,
  'living_street': 61
}


def is_osm_time_condition_active(condition_string):
  """
  Will indicate if a time condition for a restriction as described
  @ https://wiki.openstreetmap.org/wiki/Conditional_restrictions
  is active for the current date and time of day.
  """
  now = dt.now().astimezone()
  today = now.date()
  week_days = []

  # Look for days of week matched and validate if today matches criteria.
  dr = re.findall(r'(Mo|Tu|We|Th|Fr|Sa|Su[-,\s]*?)', condition_string)

  if len(dr) == 1:
    week_days = [_WD[dr[0]]]
  # If two or more matches condider it a range of days between 1st and 2nd element.
  elif len(dr) > 1:
    week_days = list(range(_WD[dr[0]], _WD[dr[1]] + 1))

  # If valid week days list is not empty and today day is not in the list, then the time-date range is not active.
  if len(week_days) > 0 and now.weekday() not in week_days:
    return False

  # Look for time ranges on the day. No time range, means all day
  tr = re.findall(r'([0-9]{1,2}:[0-9]{2})\s*?-\s*?([0-9]{1,2}:[0-9]{2})', condition_string)

  # if no time range but there were week days set, consider it active during the whole day
  if len(tr) == 0:
    return len(dr) > 0

  # Search among time ranges matched, one where now time belongs too. If found range is active.
  for times_tup in tr:
    times = list(map(lambda tt: dt.
                 combine(today, dt.strptime(tt, '%H:%M').time().replace(tzinfo=now.tzinfo)), times_tup))
    if now >= times[0] and now <= times[1]:
      return True

  return False


def speed_limit_value_for_limit_string(limit_string):
  # Look for matches of speed by default in kph, or in mph when explicitly noted.
  v = re.match(r'^\s*([0-9]{1,3})\s*?(mph)?\s*$', limit_string)
  if v is None:
    return None
  conv = CV.MPH_TO_MS if v[2] is not None and v[2] == "mph" else CV.KPH_TO_MS
  return conv * float(v[1])


def speed_limit_for_osm_tag_limit_string(limit_string):
  # https://wiki.openstreetmap.org/wiki/Key:maxspeed
  if limit_string is None:
    # When limit is set to 0. is considered not existing.
    return 0.

  # Attempt to parse limit as simple numeric value considering units.
  limit = speed_limit_value_for_limit_string(limit_string)
  if limit is not None:
    return limit

  # Look for matches of speed with country implicit values.
  v = re.match(r'^\s*([A-Z]{2}):([a-z_]+):?([0-9]{1,3})?(\s+)?(mph)?\s*', limit_string)
  if v is None:
    return 0.

  if v[2] == "zone" and v[3] is not None:
    conv = CV.MPH_TO_MS if v[5] is not None and v[5] == "mph" else CV.KPH_TO_MS
    limit = conv * float(v[3])
  elif f'{v[1]}:{v[2]}' in _COUNTRY_LIMITS:
    limit = speed_limit_value_for_limit_string(_COUNTRY_LIMITS[f'{v[1]}:{v[2]}'])

  return limit if limit is not None else 0.


def conditional_speed_limit_for_osm_tag_limit_string(limit_string):
  if limit_string is None:
    # When limit is set to 0. is considered not existing.
    return 0.

  # Look for matches of the `<restriction-value> @ (<condition>)` format
  v = re.match(r'^(.*)@\s*\((.*)\).*$', limit_string)
  if v is None:
    return 0.  # No valid format match

  value = speed_limit_for_osm_tag_limit_string(v[1])
  if value == 0.:
    return 0.  # Invalid speed limit value

  # Look for date-time conditions separated by semicolon
  v = re.findall(r'(?:;|^)([^;]*)', v[2])
  for datetime_condition in v:
    if is_osm_time_condition_active(datetime_condition):
      return value

  # If we get here, no current date-time condition is active.
  return 0.


class WayRelation():
  """A class that represent the relationship of an OSM way and a given `location` and `bearing` of a driving vehicle.
  """
  def __init__(self, way, parent=None):
    self.way = way
    self.parent = parent
    self.parent_wr_id = parent.id if parent is not None else None  # For WRs created as splits of other WRs
    self.reset_location_variables()
    self.direction = DIRECTION.NONE
    self._speed_limit = None
    self._one_way = way.tags.get("oneway")
    self.name = way.tags.get('name')
    self.ref = way.tags.get('ref')
    self.highway_type = way.tags.get("highway")
    self.highway_rank = _HIGHWAY_RANK.get(self.highway_type, 1000)
    try:
      self.lanes = int(way.tags.get('lanes'))
    except Exception:
      self.lanes = 2

    # Create numpy arrays with nodes data to support calculations.
    self._nodes_np = np.radians(np.array([[node.lat, node.lon] for node in way.nodes], dtype=float))
    self._nodes_ids = np.array([node.id for node in way .nodes], dtype=int)

    # Get the vectors representation of the segments betwheen consecutive nodes. (N-1, 2)
    v = vectors(self._nodes_np) * R

    # Calculate the vector magnitudes (or distance) between nodes. (N-1)
    self._way_distances = np.linalg.norm(v, axis=1)

    # Calculate the bearing (from true north clockwise) for every section of the way (vectors between nodes). (N-1)
    self._way_bearings = np.arctan2(v[:, 0], v[:, 1])

    # Define bounding box to ease the process of locating a node in a way.
    # [[min_lat, min_lon], [max_lat, max_lon]]
    self.bbox = np.row_stack((np.amin(self._nodes_np, 0) - _WAY_BBOX_PADING,
                              np.amax(self._nodes_np, 0) + _WAY_BBOX_PADING))

    # Get the edge nodes ids.
    self.edge_nodes_ids = [way.nodes[0].id, way.nodes[-1].id]

  def __repr__(self):
    return f'(id: {self.id}, between {self.behind_idx} and {self.ahead_idx}, {self.direction}, active: {self.active})'

  def __eq__(self, other):
    if isinstance(other, WayRelation):
      return self.id == other.id
    return False

  def reset_location_variables(self):
    self.distance_to_node_ahead = 0.
    self.location_rad = None
    self.bearing_rad = None
    self.active = False
    self.diverting = False
    self.ahead_idx = None
    self.behind_idx = None
    self._active_bearing_delta = None
    self._distance_to_way = None

  @property
  def id(self):
    return self.way.id

  @property
  def road_name(self):
    if self.name is not None:
      return self.name
    return self.ref

  def update(self, location_rad, bearing_rad, location_stdev):
    """Will update and validate the associated way with a given `location_rad` and `bearing_rad`.
       Specifically it will find the nodes behind and ahead of the current location and bearing.
       If no proper fit to the way geometry, the way relation is marked as invalid.
    """
    self.reset_location_variables()

    # Ignore if location not in way bounding box
    if not self.is_location_in_bbox(location_rad):
      return

    # - Get the distance and bearings from location to all nodes. (N)
    bearings = bearing_to_points(location_rad, self._nodes_np)
    distances = distance_to_points(location_rad, self._nodes_np)

    # - Get absolute bearing delta to current driving bearing. (N)
    delta = np.abs(bearing_rad - bearings)

    # - Nodes are ahead if the cosine of the delta is positive (N)
    is_ahead = np.cos(delta) >= 0.

    # - Possible locations on the way are those where adjacent nodes change from ahead to behind or vice-versa.
    possible_idxs = np.nonzero(np.diff(is_ahead))[0]

    # - when no possible locations found, then the location is not in this way.
    if len(possible_idxs) == 0:
      return

    # - Find then angle formed between the vectors from the current location to consecutive nodes. This is the
    # value of the difference in the bearings of the vectors.
    teta = np.diff(bearings)

    # - When two consecutive nodes will be ahead and behind, they will form a triangle with the current location.
    # We find the closest distance to the way by solving the area of the triangle and finding the height (h).
    # We must use the absolute value of the sin of the angle in the formula, which is equivalent to ensure we
    # are considering the smallest of the two angles formed between the two vectors.
    # https://www.mathsisfun.com/algebra/trig-area-triangle-without-right-angle.html
    h = distances[:-1] * distances[1:] * np.abs(np.sin(teta)) / self._way_distances

    # - Calculate the delta between driving bearing and way bearings. (N-1)
    bw_delta = self._way_bearings - bearing_rad

    # - The absolute value of the sin of `bw_delta` indicates how close the bearings match independent of direction.
    # We will use this value along the distance to the way to aid on way selection. (N-1)
    abs_sin_bw_delta = np.abs(np.sin(bw_delta))

    # - Get the delta to way bearing indicators and the distance to the way for the possible locations.
    abs_sin_bw_delta_possible = abs_sin_bw_delta[possible_idxs]
    h_possible = h[possible_idxs]

    # - Get the index where the distance to the way is minimum. That is the chosen location.
    min_h_possible_idx = np.argmin(h_possible)
    min_delta_idx = possible_idxs[min_h_possible_idx]

    # - If the distance to the way is over 4 standard deviations of the gps accuracy + half the maximum road width
    # estimate, then we are way too far to stick to this way (i.e. we are not on this way anymore)
    half_road_width_estimate = self.lanes * LANE_WIDTH / 2.
    if h_possible[min_h_possible_idx] > 4. * location_stdev + half_road_width_estimate:
      return

    # - If the distance to the road is greater than 2 standard deviations of the gps accuracy + half the maximum road
    # width estimate then we are most likely diverting from this route.
    diverting = h_possible[min_h_possible_idx] > 2. * location_stdev + half_road_width_estimate

    # Populate location variables with result
    if is_ahead[min_delta_idx]:
      self.direction = DIRECTION.BACKWARD
      self.ahead_idx = min_delta_idx
      self.behind_idx = min_delta_idx + 1
    else:
      self.direction = DIRECTION.FORWARD
      self.ahead_idx = min_delta_idx + 1
      self.behind_idx = min_delta_idx

    self._distance_to_way = h[min_delta_idx]
    self._active_bearing_delta = abs_sin_bw_delta_possible[min_h_possible_idx]
    # TODO: The distance to node ahead currently represent the distance from the GPS fix location.
    # It would be perhaps more accurate to use the distance on the projection over the direct line between
    # the two nodes.
    self.distance_to_node_ahead = distances[self.ahead_idx]
    self.active = True
    self.diverting = diverting
    self.location_rad = location_rad
    self.bearing_rad = bearing_rad
    self._speed_limit = None

  def update_direction_from_starting_node(self, start_node_id):
    self._speed_limit = None
    if self.edge_nodes_ids[0] == start_node_id:
      self.direction = DIRECTION.FORWARD
    elif self.edge_nodes_ids[-1] == start_node_id:
      self.direction = DIRECTION.BACKWARD
    else:
      self.direction = DIRECTION.NONE

  def is_location_in_bbox(self, location_rad):
    """Indicates if a given location is contained in the bounding box surrounding the way.
       self.bbox = [[min_lat, min_lon], [max_lat, max_lon]]
    """
    is_g = np.greater_equal(location_rad, self.bbox[0, :])
    is_l = np.less_equal(location_rad, self.bbox[1, :])

    return np.all(np.concatenate((is_g, is_l)))

  @property
  def speed_limit(self):
    if self._speed_limit is not None:
      return self._speed_limit

    # Get string from corresponding tag, consider conditional limits first.
    limit_string = self.way.tags.get("maxspeed:conditional")
    if limit_string is None:
      if self.direction == DIRECTION.FORWARD:
        limit_string = self.way.tags.get("maxspeed:forward:conditional")
      elif self.direction == DIRECTION.BACKWARD:
        limit_string = self.way.tags.get("maxspeed:backward:conditional")

    limit = conditional_speed_limit_for_osm_tag_limit_string(limit_string)

    # When no conditional limit set, attempt to get from regular speed limit tags.
    if limit == 0.:
      limit_string = self.way.tags.get("maxspeed")
      if limit_string is None:
        if self.direction == DIRECTION.FORWARD:
          limit_string = self.way.tags.get("maxspeed:forward")
        elif self.direction == DIRECTION.BACKWARD:
          limit_string = self.way.tags.get("maxspeed:backward")

      limit = speed_limit_for_osm_tag_limit_string(limit_string)

    self._speed_limit = limit
    return self._speed_limit

  @property
  def active_bearing_delta(self):
    """Returns the sine of the delta between the current location bearing and the exact
       bearing of the portion of way we are currentluy located at.
    """
    return self._active_bearing_delta

  @property
  def is_one_way(self):
    return self._one_way in ['yes'] or self.highway_type in ["motorway"]

  @property
  def is_prohibited(self):
    # Direction must be defined to asses this property. Default to `True` if not.
    if self.direction == DIRECTION.NONE:
      return True
    return self.is_one_way and self.direction == DIRECTION.BACKWARD

  @property
  def distance_to_way(self):
    """Returns the perpendicular (i.e. minimum) distance between current location and the way
    """
    return self._distance_to_way

  @property
  def node_ahead(self):
    return self.way.nodes[self.ahead_idx] if self.ahead_idx is not None else None

  @property
  def last_node(self):
    """Returns the last node on the way considering the traveling direction
    """
    if self.direction == DIRECTION.FORWARD:
      return self.way.nodes[-1]
    if self.direction == DIRECTION.BACKWARD:
      return self.way.nodes[0]
    return None

  @property
  def last_node_coordinates(self):
    """Returns the coordinates for the last node on the way considering the traveling direction. (in radians)
    """
    if self.direction == DIRECTION.FORWARD:
      return self._nodes_np[-1]
    if self.direction == DIRECTION.BACKWARD:
      return self._nodes_np[0]
    return None

  def node_before_edge_coordinates(self, node_id):
    """Returns the coordinates of the node before the edge node identifeid with `node_id`. (in radians)
    """
    if self.edge_nodes_ids[0] == node_id:
      return self._nodes_np[1]

    if self.edge_nodes_ids[-1] == node_id:
      return self._nodes_np[-2]

    return np.array([0., 0.])

  def split(self, node_id, way_ids=None):
    """ Returns and array with the way relations resulting from splitting the current way relation at node_id
    """
    idxs = np.nonzero(self._nodes_ids == node_id)[0]
    if len(idxs) == 0:
      return []

    idx = idxs[0]
    if idx == 0 or idx == len(self._nodes_ids) - 1:
      return [self]

    if not isinstance(way_ids, list):
      way_ids = [-1, -2]  # Default id values.

    ways = [create_way(way_ids[0], node_ids=self._nodes_ids[:idx + 1], from_way=self.way),
            create_way(way_ids[1], node_ids=self._nodes_ids[idx:], from_way=self.way)]
    return [WayRelation(way, parent=self) for way in ways]
