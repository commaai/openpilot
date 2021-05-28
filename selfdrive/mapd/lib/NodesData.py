import numpy as np
from opspline import splev, splprep
from enum import Enum
from .geo import DIRECTION, R, vectors

_TURN_CURVATURE_THRESHOLD = 0.002  # 1/mts. A curvature over this value will generate a speed limit section.
_MAX_LAT_ACC = 2.3  # Maximum lateral acceleration in turns.
_SPLINE_EVAL_STEP = 5  # mts for spline evaluation for curvature calculation


def nodes_raw_data_array_for_wr(wr, drop_last=False):
  """Provides an array of raw node data (id, lat, lon, speed_limit) for all nodes in way relation
  """
  sl = wr.speed_limit if wr.speed_limit is not None else 0.
  data = np.array(list(map(lambda n: (n.id, n.lat, n.lon, sl), wr.way.nodes)), dtype=float)

  # reverse the order if way direction is backwards
  if wr.direction == DIRECTION.BACKWARD:
    data = np.flip(data, axis=0)

  # drop last if requested
  return data[:-1] if drop_last else data


def node_calculations(points):
  """Provides node calculations based on an array of (lat, lon) points in radians.
     points is a (N x 1) array where N >= 3
  """
  if len(points) < 3:
    raise(IndexError)

  # Get the vector representation of node points in cartesian plane.
  # (N-1, 2) array. Not including (0., 0.)
  v = vectors(points) * R

  # Calculate the vector magnitudes (or distance)
  # (N-1, 1) array. No distance for v[-1]
  d = np.linalg.norm(v, axis=1)

  # Calculate the bearing (from true north clockwise) for every node.
  # (N-1, 1) array. No bearing for v[-1]
  b = np.arctan2(v[:, 0], v[:, 1])

  # Add origin to vector space. (i.e first node in list)
  v = np.concatenate(([[0., 0.]], v))

  # Provide distance to previous node and distance to next node
  dp = np.concatenate(([0.], d))
  dn = np.concatenate((d, [0.]))

  # Bearing of last node should keep bearing from previous.
  b = np.concatenate((b, [b[-1]]))

  return v, dp, dn, b


def spline_curvature_calculations(vect, dist_prev):
  """Provides an array of curvatures and its distances by applying a spline interpolation
  to the path described by the nodes data.
  """
  # create cumulative arrays for distance traveled and vector (x, y)
  ds = np.cumsum(dist_prev, axis=0)
  vs = np.cumsum(vect, axis=0)

  # spline interpolation
  tck, u = splprep([vs[:, 0], vs[:, 1]])

  # evaluate every _SPLINE_EVAL_STEP mts.
  n = max(int(ds[-1] / _SPLINE_EVAL_STEP), len(u))
  unew = np.arange(0, n + 1) / n

  # get derivatives
  d1 = splev(unew, tck, der=1)
  d2 = splev(unew, tck, der=2)

  # calculate curvatures
  num = d1[0] * d2[1] - d1[1] * d2[0]
  den = (d1[0]**2 + d1[1]**2)**(1.5)
  curv = num / den
  curv_ds = unew * ds[-1]

  return curv, curv_ds


def speed_section(curv_sec):
  """Map curvature section data into turn speed sections data.
    Returns: [section start distance, section end distance, speed limit based on max curvature, sing of curvature]
  """
  max_curv_idx = np.argmax(curv_sec[:, 0])
  start = np.amin(curv_sec[:, 2])
  end = np.amax(curv_sec[:, 2])

  return np.array([start, end, np.sqrt(_MAX_LAT_ACC / curv_sec[max_curv_idx, 0]), curv_sec[max_curv_idx, 1]])


def speed_sections(curv_sec):
  """Map a curvature section data array into turn speed sections arrays.
    It takes care of spliting curvature sections in smaller sections when they contain turns
    in different directions.
  """
  # Find the indexes where the curvatures change signs (if any).
  c_idx = np.nonzero(np.diff(curv_sec[:, 1]))[0] + 1

  # Create independent arrays for each split section base on change of sign.
  splits = np.array(np.split(curv_sec, c_idx), dtype=object)

  # Map all splits into turn speed secion data arrays and return as numpy array.
  return np.array([speed_section(cs) for cs in splits])


def speed_limits_for_curvatures_data(curv, dist):
  """Provides the calculations for the speed limits from the curvatures array and distances,
    by providing distances to curvature sections and correspoinding speed limit values as well as
    curvature direction/sign.
  """
  # Prepare a data array for processing with absolute curvature values, curvature sign and distances.
  curv_abs = np.abs(curv)
  data = np.column_stack((curv_abs, np.sign(curv), dist))

  # Find where curvatures overshoot turn curvature threshold and define as section
  is_section = curv_abs >= _TURN_CURVATURE_THRESHOLD

  # Find the indexes where the sections start and end. i.e. change indexes.
  c_idx = np.nonzero(np.diff(is_section))[0] + 1

  # Create independent arrays for each split section base on change indexes.
  splits = np.array(np.split(data, c_idx), dtype=object)

  # Filter the splits to keep only the curvature section arrays by getting the odd or even split arrays depending
  # on whether the first split is a curvature split or not.
  curv_sec_idxs = np.arange(0 if is_section[0] else 1, len(splits), 2, dtype=int)
  curv_secs = splits[curv_sec_idxs]

  # Return an array where each row represents a turn speed limit section.
  # [start, end, speed_limit, curvature_sign]
  return np.concatenate(tuple([speed_sections(cs) for cs in curv_secs])) if len(curv_secs) > 0 else np.array([])


class SpeedLimitSection():
  """And object representing a speed limited road section ahead.
  provides the start and end distance and the speed limit value
  """
  def __init__(self, start, end, value):
    self.start = start
    self.end = end
    self.value = value

  def __repr__(self):
    return f'from: {self.start}, to: {self.end}, limit: {self.value}'


class TurnSpeedLimitSection(SpeedLimitSection):
  def __init__(self, start, end, value, sign):
      super().__init__(start, end, value)
      self.curv_sign = sign

  def __repr__(self):
    return f'{super().__repr__()}, sign: {self.curv_sign}'


class NodeDataIdx(Enum):
  """Column index for data elements on NodesData underlying data store.
  """
  node_id = 0
  lat = 1
  lon = 2
  speed_limit = 3
  x = 4             # x value of cartesian vector representing the section between last node and this node.
  y = 5             # y value of cartesian vector representing the section between last node and this node.
  dist_prev = 6     # distance to previous node.
  dist_next = 7     # distance to next node
  bearing = 8       # bearing of the vector departing from this node.


class NodesData:
  """Container for the list of node data from a ordered list of way relations to be used in a Route
  """
  def __init__(self, way_relations):
    self._nodes_data = np.array([])
    self._curvature_speed_sections_data = np.array([])

    way_count = len(way_relations)
    if way_count == 0:
      return

    # We want all the nodes from the last way section
    nodes_data = nodes_raw_data_array_for_wr(way_relations[-1])

    # For the ways before the last in the route we want all the nodes but the last, as that one is the first on
    # the next section. Collect them, append last way node data and concatenate the numpy arrays.
    if way_count > 1:
      wrs_data = tuple(map(lambda wr: nodes_raw_data_array_for_wr(wr, True), way_relations[:-1]))
      wrs_data += (nodes_data,)
      nodes_data = np.concatenate(wrs_data)

    # Get a subarray with lat, lon to compute the remaining node values.
    lat_lon_array = nodes_data[:, [1, 2]]
    points = np.radians(lat_lon_array)
    # Ensure we have more than 3 points, if not calculations are not possible.
    if len(points) < 3:
      return
    vect, dist_prev, dist_next, bearing = node_calculations(points)

    # append calculations to nodes_data
    # nodes_data structure: [id, lat, lon, speed_limit, x, y, dist_prev, dist_next, bearing]
    self._nodes_data = np.column_stack((nodes_data, vect, dist_prev, dist_next, bearing))

    # Store calculcations for curvature sections speed limits. We need more than 3 points to be able to process.
    # _curvature_speed_sections_data structure: [dist_start, dist_stop, speed_limits, curv_sign]
    if len(vect) > 3:
      curv, curv_ds = spline_curvature_calculations(vect, dist_prev)
      self._curvature_speed_sections_data = speed_limits_for_curvatures_data(curv, curv_ds)

  @property
  def count(self):
    return len(self._nodes_data)

  def get(self, node_data_idx):
    """Returns the array containing all the elements of a specific NodeDataIdx type.
    """
    if len(self._nodes_data) == 0 or node_data_idx.value >= self._nodes_data.shape[1]:
      return np.array([])

    return self._nodes_data[:, node_data_idx.value]

  def speed_limits_ahead(self, ahead_idx, distance_to_node_ahead):
    """Returns and array of SpeedLimitSection objects for the actual route ahead of current location
    """
    if len(self._nodes_data) == 0 or ahead_idx is None:
      return []

    # Find the cumulative distances where speed limit changes. Build Speed limit sections for those.
    dist = np.concatenate(([distance_to_node_ahead], self.get(NodeDataIdx.dist_next)[ahead_idx:]))
    dist = np.cumsum(dist, axis=0)
    sl = self.get(NodeDataIdx.speed_limit)[ahead_idx - 1:]
    sl_next = np.concatenate((sl[1:], [0.]))

    # Create a boolean mask where speed limit changes and filter values
    sl_change = sl != sl_next
    distances = dist[sl_change]
    speed_limits = sl[sl_change]

    # Create speed limits sections combining all continious nodes that have same speed limit value.
    start = 0.
    limits_ahead = []
    for idx, end in enumerate(distances):
      limits_ahead.append(SpeedLimitSection(start, end, speed_limits[idx]))
      start = end

    return limits_ahead

  def distance_to_end(self, ahead_idx, distance_to_node_ahead):
    if len(self._nodes_data) == 0 or ahead_idx is None:
      return None

    return np.sum(np.concatenate(([distance_to_node_ahead], self.get(NodeDataIdx.dist_next)[ahead_idx:])))

  def curvatures_speed_limit_sections_ahead(self, ahead_idx, distance_to_node_ahead):
    """Returns and array of TurnSpeedLimitSection objects for the actual route ahead of current location for
       speed limit sections due to curvatures in the road.
    """
    if len(self._curvature_speed_sections_data) == 0 or ahead_idx is None:
      return []

    # Find the current distance traveled so far on the route.
    dist_curr = np.cumsum(self.get(NodeDataIdx.dist_next)[:ahead_idx])[-1] - distance_to_node_ahead

    # Filter the sections to get only those where the stop distance is ahead of current.
    sec_filter = self._curvature_speed_sections_data[:, 1] > dist_curr
    data = self._curvature_speed_sections_data[sec_filter]

    # Offset distances to current distance.
    data[:, [0, 1]] -= dist_curr

    # Create speed limits sections
    limits_ahead = [TurnSpeedLimitSection(max(0., d[0]), d[1], d[2], d[3]) for d in data]

    return limits_ahead
