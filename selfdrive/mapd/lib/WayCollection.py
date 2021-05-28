from .WayRelation import WayRelation
from .Route import Route
import uuid


_ACCEPTABLE_BEARING_DELTA_IND = 0.7071067811865475  # sin(pi/4) | 45 degrees acceptable bearing delta


class WayCollection():
  """A collection of WayRelations to use for maps data analysis.
  """
  def __init__(self, ways, query_center):
    """Creates a WayCollection with a set of OSM way objects.

    Args:
        ways (Array): Collection of Way objects fetched from OSM in a radius around `query_center`
        query_center (Numpy Array): [lat, lon] numpy array in radians indicating the center of the data query.
    """
    self.id = uuid.uuid4()
    self.way_relations = list(map(lambda way: WayRelation(way), ways))
    self.query_center = query_center

    # Create the index by edge node ids.
    self.wr_index = {}
    for wr in self.way_relations:
      for node_id in wr.edge_nodes_ids:
        self.wr_index[node_id] = self.wr_index.get(node_id, []) + [wr]

  def get_route(self, location_rad, bearing_rad, accuracy):
    """Provides the best route found in the way collection based on current location and bearing.
    """
    if location_rad is None or bearing_rad is None or accuracy is None:
      return None

    # Update all way relations in collection to the provided location and bearing.
    for wr in self.way_relations:
      wr.update(location_rad, bearing_rad, accuracy)

    # Get the way relations where a match was found. i.e. those now marked as active as long as the direction of
    # travel is valid.
    valid_way_relations = list(filter(lambda wr: wr.active and not wr.is_prohibited, self.way_relations))

    # If no active, then we could not find a current way to build a route.
    if len(valid_way_relations) == 0:
      return None

    # If only one valid, then pick it as current.
    if len(valid_way_relations) == 1:
      current = valid_way_relations[0]

    # If more than one is valid, filter out any valid way relation where the bearing delta indicator is too high.
    else:
      wr_acceptable_bearing = list(filter(lambda wr: wr.active_bearing_delta <= _ACCEPTABLE_BEARING_DELTA_IND,
                                          valid_way_relations))

      # If delta bearing indicator is too high for all, then use as current the one that has the shorter one.
      if len(wr_acceptable_bearing) == 0:
        valid_way_relations.sort(key=lambda wr: wr.active_bearing_delta)
        current = valid_way_relations[0]

      # If only one with acceptable bearing, use it.
      elif len(wr_acceptable_bearing) == 1:
        current = wr_acceptable_bearing[0]

      # If more than one with acceptable bearing, filter the ones with distance to way lower than GPS accuracy.
      else:
        wr_accurate_distance = list(filter(lambda wr: wr.distance_to_way <= accuracy, wr_acceptable_bearing))

        # If none with distance under accuracy, then select the closest to the way
        if len(wr_accurate_distance) == 0:
          wr_acceptable_bearing.sort(key=lambda wr: wr.distance_to_way)
          current = wr_acceptable_bearing[0]

        # If only one with distance under accuracy, select this one.
        elif len(wr_accurate_distance) == 1:
          current = wr_accurate_distance[0]

        # If more than one with distance under accuracy. Then select the one with lowest highway rank.
        # i.e. prefere motorways over other roads and so on. This is to prevent selecting a small paralel
        # road to a main road when the accuracy is poor.
        else:
          wr_accurate_distance.sort(key=lambda wr: wr.highway_rank)
          current = wr_accurate_distance[0]

    return Route(current, self.wr_index, self.id, self.query_center)
