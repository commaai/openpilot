import numpy as np
from enum import Enum
from selfdrive.mapd.lib.geo import DIRECTION, R, vectors

from scipy.interpolate import splev, splprep


_TURN_CURVATURE_THRESHOLD = 0.002  # 1/mts. A curvature over this value will generate a speed limit section.
_MAX_LAT_ACC = 2.3  # Maximum lateral acceleration in turns.
_SPLINE_EVAL_STEP = 5  # mts for spline evaluation for curvature calculation
_MIN_SPEED_SECTION_LENGTH = 100.  # mts. Sections below this value will not be split in smaller sections.
_MAX_CURV_DEVIATION_FOR_SPLIT = 2.  # Split a speed section if the max curvature deviates from mean by this factor.
_MAX_CURV_SPLIT_ARC_ANGLE = 90.  # degrees. Arc section to split into new speed section around max curvature.
_MIN_NODE_DISTANCE = 50.  # mts. Minimum distance between nodes for spline evaluation. Data is enhanced if not met.
_ADDED_NODES_DIST = 15.  # mts. Distance between added nodes when data is enhanced for spline evaluation.
_DIVERTION_SEARCH_RANGE = [-200., 50.]  # mt. Range of distance to current location for diversion search.


def nodes_raw_data_array_for_wr(wr, drop_last=False):
  """Provides an array of raw node data (id, lat, lon, speed_limit) for all nodes in way relation
  """
  sl = wr.speed_limit
  data = np.array([(n.id, n.lat, n.lon, sl) for n in wr.way.nodes], dtype=float)

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

  # Provide cumulative distance on route
  dr = np.cumsum(dp, axis=0)

  # Bearing of last node should keep bearing from previous.
  b = np.concatenate((b, [b[-1]]))

  return v, dp, dn, dr, b


def spline_curvature_calculations(vect, dist_prev):
  """Provides an array of curvatures and its distances by applying a spline interpolation
  to the path described by the nodes data.
  """
  # We need to artificially enhance the data before applying spline interpolation to avoid getting
  # inexistent curvature values close to irregularities on the road when the resolution of nodes data
  # approaching the irregularity is low.

  # - Find indexes where dist_prev is greater than threshold
  too_far_idxs = np.nonzero(dist_prev >= _MIN_NODE_DISTANCE)[0]

  # - Traversing in reverse order, enhance data by adding points at the found indexes.
  for idx in too_far_idxs[::-1]:
    dp = dist_prev[idx]  # distance of vector that needs to be replaced by higher resolution vectors.
    n = int(np.ceil(dp / _ADDED_NODES_DIST))  # number of vectors that need to be added.
    new_v = vect[idx, :] / n  # new relative vector to insert.
    vect = np.delete(vect, idx, axis=0)  # remove the relative vector to be replaced by the insertion of new vectors.
    vect = np.insert(vect, [idx] * n, [new_v] * n, axis=0)  # insert n new relative vectors

  # Data is now enhanced, we can proceed with curvature evaluation.
  # - Create cumulative arrays for distance traveled and vector (x, y)
  ds = np.cumsum(dist_prev, axis=0)
  vs = np.cumsum(vect, axis=0)

  # - spline interpolation
  tck, u = splprep([vs[:, 0], vs[:, 1]])  # pylint: disable=unbalanced-tuple-unpacking

  # - evaluate every _SPLINE_EVAL_STEP mts.
  n = max(int(ds[-1] / _SPLINE_EVAL_STEP), len(u))
  unew = np.arange(0, n + 1) / n

  # - get derivatives
  d1 = splev(unew, tck, der=1)
  d2 = splev(unew, tck, der=2)

  # - calculate curvatures
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


def split_speed_section_by_sign(curv_sec):
  """Will split the given curvature section in subsections if there is a change of sign on the curvature value
  in the section.
  """
  # Find the indexes where the curvatures change signs (if any).
  c_idx = np.nonzero(np.diff(curv_sec[:, 1]))[0] + 1

  # Split section base on change of sign.
  return np.split(curv_sec, c_idx)


def split_speed_section_by_curv_degree(curv_sec):
  """Will split the given curvature section in subsections as to isolate peaks of turn with substantially
  higher curvature values. This will aid on preventing having very long turn sections with low speed limit
  that is only really necessary for a small region of the section.
  """
  # Only consider splitting a section if long enough.
  length = curv_sec[-1, 2] - curv_sec[0, 2]
  if length <= _MIN_SPEED_SECTION_LENGTH:
    return [curv_sec]

  # Only split if max curvature deviates substantially from mean curvature.
  max_curv_idx = np.argmax(curv_sec[:, 0])
  max_curv = curv_sec[max_curv_idx, 0]
  mean_curv = np.mean(curv_sec[:, 0])
  if max_curv / mean_curv <= _MAX_CURV_DEVIATION_FOR_SPLIT:
    return [curv_sec]

  # Calculate where to split as to isolate a curve section around the max curvature peak.
  arc_side = (np.radians(_MAX_CURV_SPLIT_ARC_ANGLE) / max_curv) / 2.
  arc_side_idx_lenght = int(np.ceil(arc_side / _SPLINE_EVAL_STEP))
  split_idxs = [max_curv_idx - arc_side_idx_lenght, max_curv_idx + arc_side_idx_lenght]
  split_idxs = list(filter(lambda idx: idx > 0 and idx < len(curv_sec) - 1, split_idxs))

  # If the arc section to split extendes outside the section, then no need to split.
  if len(split_idxs) == 0:
    return [curv_sec]

  # Create the splits and split the resulting sections recursevly.
  splits = [split_speed_section_by_curv_degree(cs) for cs in np.split(curv_sec, split_idxs)]

  # Flatten the results and return the new list of curvature sections.
  curv_secs = [cs for split in splits for cs in split]
  return curv_secs


def speed_limits_for_curvatures_data(curv, dist):
  """Provides the calculations for the speed limits from the curvatures array and distances,
    by providing distances to curvature sections and corresponding speed limit values as well as
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

  # Further split the curv sections by sign change
  sub_secs = [split_speed_section_by_sign(cs) for cs in curv_secs]
  curv_secs = [cs for sub_sec in sub_secs for cs in sub_sec]

  # Further split the curv sections by degree of curvature
  sub_secs = [split_speed_section_by_curv_degree(cs) for cs in curv_secs]
  curv_secs = [cs for sub_sec in sub_secs for cs in sub_sec]

  # Return an array where each row represents a turn speed limit section.
  # [start, end, speed_limit, curvature_sign]
  return np.array([speed_section(cs) for cs in curv_secs])

def is_wr_a_valid_divertion_from_node(wr, node_id, wr_ids):
  """
  Evaluates if the way relation `wr` is a valid diversion from node with id `node_id`.
  A valid diversion is a way relation with an edge node with the given `node_id` that is not already included
  in the list of way relations in the route (`wr_ids`) and that can be travaled in the direction as if starting
  from node with id `node_id`
  """
  if wr.id in wr_ids:
    return False
  wr.update_direction_from_starting_node(node_id)
  return not wr.is_prohibited


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
  dist_route = 8    # cumulative distance on route
  bearing = 9       # bearing of the vector departing from this node.


class NodesData:
  """Container for the list of node data from a ordered list of way relations to be used in a Route
  """
  def __init__(self, way_relations, wr_index):
    self._nodes_data = np.array([])
    self._divertions = [[]]
    self._curvature_speed_sections_data = np.array([])

    way_count = len(way_relations)
    if way_count == 0:
      return

    # We want all the nodes from the last way section
    nodes_data = nodes_raw_data_array_for_wr(way_relations[-1])

    # For the ways before the last in the route we want all the nodes but the last, as that one is the first on
    # the next section. Collect them, append last way node data and concatenate the numpy arrays.
    if way_count > 1:
      wrs_data = tuple([nodes_raw_data_array_for_wr(wr, drop_last=True) for wr in way_relations[:-1]])
      wrs_data += (nodes_data,)
      nodes_data = np.concatenate(wrs_data)

    # Get a subarray with lat, lon to compute the remaining node values.
    lat_lon_array = nodes_data[:, [1, 2]]
    points = np.radians(lat_lon_array)
    # Ensure we have more than 3 points, if not calculations are not possible.
    if len(points) <= 3:
      return
    vect, dist_prev, dist_next, dist_route, bearing = node_calculations(points)

    # append calculations to nodes_data
    # nodes_data structure: [id, lat, lon, speed_limit, x, y, dist_prev, dist_next, dist_route, bearing]
    self._nodes_data = np.column_stack((nodes_data, vect, dist_prev, dist_next, dist_route, bearing))

    # Build route diversion options data from the wr_index.
    wr_ids = [wr.id for wr in way_relations]
    self._divertions = [[wr for wr in wr_index.way_relations_with_edge_node_id(node_id)
                        if is_wr_a_valid_divertion_from_node(wr, node_id, wr_ids)]
                        for node_id in nodes_data[:, 0]]

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

    # Create speed limits sections combining all continuous nodes that have same speed limit value.
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
    dist_curr = self.get(NodeDataIdx.dist_route)[ahead_idx] - distance_to_node_ahead

    # Filter the sections to get only those where the stop distance is ahead of current.
    sec_filter = self._curvature_speed_sections_data[:, 1] > dist_curr
    data = self._curvature_speed_sections_data[sec_filter]

    # Offset distances to current distance.
    data[:, [0, 1]] -= dist_curr

    # Create speed limits sections
    limits_ahead = [TurnSpeedLimitSection(max(0., d[0]), d[1], d[2], d[3]) for d in data]

    return limits_ahead

  def possible_divertions(self, ahead_idx, distance_to_node_ahead):
    """ Returns and array with the way relations the route could possible divert to by finding
        the alternative way diversions on the nodes in the vicinity of the current location.
    """
    if len(self._nodes_data) == 0 or ahead_idx is None:
      return []

    dist_route = self.get(NodeDataIdx.dist_route)
    rel_dist = dist_route - dist_route[ahead_idx] + distance_to_node_ahead
    valid_idxs = np.nonzero(np.logical_and(rel_dist >= _DIVERTION_SEARCH_RANGE[0],
                            rel_dist <= _DIVERTION_SEARCH_RANGE[1]))[0]
    valid_divertions = [self._divertions[i] for i in valid_idxs]

    return [wr for wrs in valid_divertions for wr in wrs]  # flatten.

  def distance_to_node(self, node_id, ahead_idx, distance_to_node_ahead):
    """
    Provides the distance to a specific node in the route identified by `node_id` in reference to the node ahead
    (`ahead_idx`) and the distance from current location to the node ahead (`distance_to_node_ahead`).
    """
    node_ids = self.get(NodeDataIdx.node_id)
    node_idxs = np.nonzero(node_ids == node_id)[0]
    if len(self._nodes_data) == 0 or ahead_idx is None or len(node_idxs) == 0:
      return None

    return self.get(NodeDataIdx.dist_route)[node_idxs[0]] - self.get(NodeDataIdx.dist_route)[ahead_idx] + \
      distance_to_node_ahead
