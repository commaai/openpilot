from .NodesData import NodesData, NodeDataIdx
from selfdrive.mapd.config import QUERY_RADIUS
from .geo import ref_vectors, R, distance_to_points
from itertools import compress
import numpy as np


_ACCEPTABLE_BEARING_DELTA_COSINE = -0.7  # Continuation paths with a bearing of 180 +/- 45 degrees.
_MAX_ALLOWED_BEARING_DELTA_COSINE_AT_EDGE = -0.3420  # bearing delta at route edge must be 180 +/- 70 degrees.
_MAP_DATA_EDGE_DISTANCE = 50  # mts. Consider edge of map data from this distance to edge of query radius.


class Route():
  """A set of consecutive way relations forming a default driving route.
  """
  def __init__(self, current, wr_index, way_collection_id, query_center):
    """Create a Route object from a given `wr_index` (Way relation index)

    Args:
        current (WayRelation): The Way Relation that is currently located. It must be active.
        wr_index (Dict(NodeId, [WayRelation])): The index of WayRelations by node id of an edge node.
        way_collection_id (UUID): The id of the Way Collection that created this Route.
        query_center (Numpy Array): lat, lon] numpy array in radians indicating the center of the data query.
    """
    self.way_collection_id = way_collection_id
    self._ordered_way_relations = []
    self._nodes_data = None
    self._reset()

    # An active current way is needed to be able to build a route
    if not current.active:
      return

    # Build the route by finding iteratavely the best matching ways continuing after the end of the
    # current (last_wr) way. Use the index to find the continuation posibilities on each iteration.
    last_wr = current
    while True:
      # - Make sure the last_wr is not the already in the ordered list as to prevent circle routes to loop forever.
      if len(self._ordered_way_relations) > 0 and \
         last_wr.id in list(map(lambda wr: wr.id, self._ordered_way_relations)):
        break

      # - Append current element to the route list of ordered way relations.
      self._ordered_way_relations.append(last_wr)

      # - Get the id of the node at the end of the way and the fetch the way relations that share the end node id from
      # the index.
      last_node_id = last_wr.last_node.id
      way_relations = wr_index[last_node_id]

      # - If no more way_relations than last_wr, we got to the end.
      if len(way_relations) == 1:
        break

      # - Get the coordinates for the edge node and build the array of coordinates for the nodes before the edge node
      # on each of the common way relations, then get the vectors in cartesian plane for the end sections of each way.
      ref_point = last_wr.last_node_coordinates
      points = np.array(list(map(lambda wr: wr.node_before_edge_coordinates(last_node_id), way_relations)))
      v = ref_vectors(ref_point, points) * R

      # - Calculate the bearing (from true north clockwise) for every end section of each way.
      b = np.arctan2(v[:, 0], v[:, 1])

      # - Find index of las_wr section and calculate deltas of bearings to the other sections.
      last_wr_idx = way_relations.index(last_wr)
      b_ref = b[last_wr_idx]
      delta = b - b_ref

      # - Update the direction of the possible route continuation ways (excluding the last_wr) as starting
      # from last_node_id
      for idx, wr in enumerate(way_relations):
        if idx != last_wr_idx:
          wr.update_direction_from_starting_node(last_node_id)

      # - Filter the possible route continuation way relations:
      #   - exclude last_wr
      #   - exclude all way relations that are prohibited due to traffic direction.
      mask = [idx != last_wr_idx and not wr.is_prohibited for idx, wr in enumerate(way_relations)]
      way_relations = list(compress(way_relations, mask))
      delta = delta[mask]

      # if no options left, we got to the end.
      if len(way_relations) == 0:
        break

      # - The cosine of the bearing delta will aid us in choosing the way that continues. The cosine is
      # minimum (-1) for a perfect straight continuation as delta would be pi or -pi.
      cos_delta = np.cos(delta)

      def pick_best_idx(cos_delta):
        """Selects the best index on `cos_delta` array for a way that continues the route.
        In principle we want to choose the way that continues as straight as possible.
        Bue we need to make sure that if there are 2 or more ways continuing relatively straight, then we
        need to disambiguate, either by matching the `ref` or `name` value of the continuing way with the
        last way selected.
        This can prevent cases where the chosen route could be for instance an exit ramp of a way due to the fact
        that the ramp has a better match on bearing to previous way. We choose to stay on the road with the same `ref`
        or `name` value if available.
        If there is no ambiguity or there are no `name` or `ref` values to disambiguate, then we pick the one with
        the straightest following direction.
        """
        # Find the indexes of the cosine of the deltas that are considered straight enough to continue.
        idxs = np.nonzero(cos_delta < _ACCEPTABLE_BEARING_DELTA_COSINE)[0]

        # If no amiguity or no way to break it, just return the straightest line.
        if len(idxs) <= 1 or (last_wr.ref is None and last_wr.name is None):
          # The section with the best continuation is the one with a bearing delta closest to pi. This is equivalent
          # to taking the one with the smallest cosine of the bearing delta, as cosine is minimum (-1) on both pi
          # and -pi.
          return np.argmin(cos_delta)

        wrs = [way_relations[idx] for idx in idxs]

        # If we find a continuation way with the same reference we just choose it.
        refs = list(map(lambda wr: wr.ref, wrs))
        if last_wr.ref is not None:
          idx = next((idx for idx, ref in enumerate(refs) if ref == last_wr.ref), None)
          if idx is not None:
            return idxs[idx]

        # If we find a continuation way with the same name we just choose it.
        names = list(map(lambda wr: wr.name, wrs))
        if last_wr.name is not None:
          idx = next((idx for idx, name in enumerate(names) if name == last_wr.name), None)
          if idx is not None:
            return idxs[idx]

        # We did not manage to deambiguate, choose straightest path.
        return np.argmin(cos_delta)

      # Get the index of the continuation way.
      best_idx = pick_best_idx(cos_delta)

      # - Make sure to not select as route continuation a way that turns too much if we are close to the border of
      # map data queried. This is to avoid building a route that takes a sharp turn just because we do not have the
      # data for the way that actually continues straight.
      if cos_delta[best_idx] > _MAX_ALLOWED_BEARING_DELTA_COSINE_AT_EDGE:
        dist_to_center = distance_to_points(query_center, np.array([ref_point]))[0]
        if dist_to_center > QUERY_RADIUS - _MAP_DATA_EDGE_DISTANCE:
          break

      # - Select next way.
      last_wr = way_relations[best_idx]

    # Build the node data from the ordered list of way relations
    self._nodes_data = NodesData(self._ordered_way_relations)

    # Locate where we are in the route node list.
    self._locate()

  def __repr__(self):
    count = self._nodes_data.count if self._nodes_data is not None else None
    return f'Route: {self.way_collection_id}, idx ahead: {self._ahead_idx} of {count}'

  def _reset(self):
    self._limits_ahead = None
    self._cuvature_limits_ahead = None
    self._curvatures_ahead = None
    self._ahead_idx = None
    self._distance_to_node_ahead = None

  @property
  def located(self):
    return self._ahead_idx is not None

  def _locate(self):
    """Will resolve the index in the nodes_data list for the node ahead of the current location.
    It updates as well the distance from the current location to the node ahead.
    """
    current = self.current_wr
    if current is None:
      return

    node_ahead_id = current.node_ahead.id
    self._distance_to_node_ahead = current.distance_to_node_ahead
    start_idx = self._ahead_idx if self._ahead_idx is not None else 1
    self._ahead_idx = None

    ids = self._nodes_data.get(NodeDataIdx.node_id)
    for idx in range(start_idx, len(ids)):
      if ids[idx] == node_ahead_id:
        self._ahead_idx = idx
        break

  @property
  def current_wr(self):
    return self._ordered_way_relations[0] if len(self._ordered_way_relations) else None

  def update(self, location_rad, bearing_rad, accuracy):
    """Will update the route structure based on the given `location_rad` and `bearing_rad` assuming progress on the
    route on the original direction. If direction has changed or active point on the route can not be found, the route
    will become invalid.
    """
    if len(self._ordered_way_relations) == 0 or location_rad is None or bearing_rad is None:
      return

    # Skip if no update on location or bearing.
    if np.array_equal(self.current_wr.location_rad, location_rad) and self.current_wr.bearing_rad == bearing_rad:
      return

    # Transverse the way relations on the actual order until we find an active one. From there, rebuild the route
    # with the way relations remaining ahead.
    for idx, wr in enumerate(self._ordered_way_relations):
      active_direction = wr.direction
      wr.update(location_rad, bearing_rad, accuracy)

      if not wr.active:
        continue

      if wr.direction == active_direction:
        # We have now the current wr. Repopulate from here till the end and locate
        self._ordered_way_relations = self._ordered_way_relations[idx:]
        self._reset()
        self._locate()
        return

      # Driving direction on the route has changed. stop.
      break

    # if we got here, there is no new active way relation or driving direction has changed. Reset.
    self._reset()

  @property
  def speed_limits_ahead(self):
    """Returns and array of SpeedLimitSection objects for the actual route ahead of current location
    """
    if self._limits_ahead is not None:
      return self._limits_ahead

    if self._nodes_data is None or self._ahead_idx is None:
      return []

    self._limits_ahead = self._nodes_data.speed_limits_ahead(self._ahead_idx, self._distance_to_node_ahead)
    return self._limits_ahead

  @property
  def curvature_speed_limits_ahead(self):
    """Returns and array of TurnSpeedLimitSection objects for the actual route ahead of current location due
    to curvatures
    """
    if self._cuvature_limits_ahead is not None:
      return self._cuvature_limits_ahead

    if self._nodes_data is None or self._ahead_idx is None:
      return []

    self._cuvature_limits_ahead = self._nodes_data. \
        curvatures_speed_limit_sections_ahead(self._ahead_idx, self._distance_to_node_ahead)

    return self._cuvature_limits_ahead

  @property
  def current_speed_limit(self):
    if not self.located:
      return None

    limits_ahead = self.speed_limits_ahead
    if not len(limits_ahead) or limits_ahead[0].start != 0:
      return None

    return limits_ahead[0].value

  @property
  def current_curvature_speed_limit_section(self):
    if not self.located:
      return None

    limits_ahead = self.curvature_speed_limits_ahead
    if not len(limits_ahead) or limits_ahead[0].start != 0:
      return None

    return limits_ahead[0]

  @property
  def next_speed_limit_section(self):
    if not self.located:
      return None

    limits_ahead = self.speed_limits_ahead
    if not len(limits_ahead):
      return None

    # Find the first section that does not start in 0. i.e. the next section
    for section in limits_ahead:
      if section.start > 0:
        return section

    return None

  @property
  def next_curvature_speed_limit_section(self):
    if not self.located:
      return None

    limits_ahead = self.curvature_speed_limits_ahead
    if not len(limits_ahead):
      return None

    return limits_ahead[0]

  @property
  def distance_to_end(self):
    if not self.located:
      return None

    return self._nodes_data.distance_to_end(self._ahead_idx, self._distance_to_node_ahead)
