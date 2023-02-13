from selfdrive.mapd.lib.WayCollection import WayCollection
from selfdrive.mapd.lib.geo import vectors, R
from selfdrive.mapd.lib.NodesData import _MIN_NODE_DISTANCE, _ADDED_NODES_DIST, _SPLINE_EVAL_STEP, \
  _MIN_SPEED_SECTION_LENGTH, nodes_raw_data_array_for_wr, node_calculations, is_wr_a_valid_divertion_from_node, \
  spline_curvature_calculations, speed_limits_for_curvatures_data
from scipy.interpolate import splev, splprep
import numpy as np
import overpy


class MockNodesData():
  def __init__(self, way_coords):
    self.degrees = np.array(way_coords)
    self.radians = np.radians(self.degrees)

    # *****************
    # Expected code implementation nodes_data
    self.v = vectors(self.radians) * R
    self.d = np.linalg.norm(self.v, axis=1)
    self.b = np.arctan2(self.v[:, 0], self.v[:, 1])
    self.v = np.concatenate(([[0., 0.]], self.v))
    self.dp = np.concatenate(([0.], self.d))
    self.dn = np.concatenate((self.d, [0.]))
    self.dr = np.cumsum(self.dp, axis=0)
    self.b = np.concatenate((self.b, [self.b[-1]]))

    # Expected code implementation spline_curvature_calculations
    vect = self.v
    dist_prev = self.dp
    too_far_idxs = np.nonzero(self.dp >= _MIN_NODE_DISTANCE)[0]
    for idx in too_far_idxs[::-1]:
      dp = dist_prev[idx]  # distance of vector that needs to be replaced by higher resolution vectors.
      n = int(np.ceil(dp / _ADDED_NODES_DIST))  # number of vectors that need to be added.
      new_v = vect[idx, :] / n  # new relative vector to insert.
      vect = np.delete(vect, idx, axis=0)  # remove the relative vector to be replaced by the insertion of new vectors.
      vect = np.insert(vect, [idx] * n, [new_v] * n, axis=0)  # insert n new relative vectors
    ds = np.cumsum(dist_prev, axis=0)
    vs = np.cumsum(vect, axis=0)
    tck, u = splprep([vs[:, 0], vs[:, 1]])  # pylint: disable=W0632
    n = max(int(ds[-1] / _SPLINE_EVAL_STEP), len(u))
    unew = np.arange(0, n + 1) / n
    d1 = splev(unew, tck, der=1)
    d2 = splev(unew, tck, der=2)
    num = d1[0] * d2[1] - d1[1] * d2[0]
    den = (d1[0]**2 + d1[1]**2)**(1.5)
    self.curv = num / den
    self.curv_ds = unew * ds[-1]
    # *****************


class MockCurveSection():
  def __init__(self, func, di=0., df=1000., step=10.):
    self.di = di
    self.df = df
    self.n = (df - di) // step
    self.u = np.arange(0, self.n + 1) / self.n
    self.curv_ds = self.u * (df - di) + di
    self.curv = func(self.u)
    self.curv_abs = np.abs(self.curv)
    self.curv_sec = np.column_stack((self.curv_abs, np.sign(self.curv), self.curv_ds))


class MockOSMQueryResponse():
  def __init__(self, xml_path, query_center):
    self.api = overpy.Overpass()
    self.query_center = np.radians(np.array(query_center))

    with open(xml_path, 'r') as f:
      overpass_xml = f.read()
      self.ways = self.api.parse_xml(overpass_xml).ways

    self.wayCollection = WayCollection(self.ways, self.query_center)

class MockRouteData():
  def __init__(self, way_ids, way_collection, first_node_id):  # way)ids must be in order forming a route.
    self.wrs = [next(wr for wr in way_collection.way_relations if wr.id == way_id) for way_id in way_ids]
    self.way_collection = way_collection
    self.first_node_id = first_node_id

  def reset(self):
    way_relations = self.wrs
    wr_index = self.way_collection.wr_index

    # Nodes Data processing expects way relations to be updated with direction before running.
    for idx, wr in enumerate(way_relations):
      if idx == 0:
        wr.update_direction_from_starting_node(self.first_node_id)
      else:
        wr.update_direction_from_starting_node(way_relations[idx - 1].last_node.id)

    # ***** Expected calculations
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
      self._curv, self._curv_ds = spline_curvature_calculations(vect, dist_prev)
      self._curvature_speed_sections_data = speed_limits_for_curvatures_data(self._curv, self._curv_ds)
    # *****


# Test data in degrees from this road:
# https://www.google.de/maps/@52.209263,13.8723137,13z
_WAY_NODES_COORDS_01 = [
  [52.1933703, 13.8723799],
  [52.1939477, 13.8711273],
  [52.1942004, 13.8705818],
  [52.1945408, 13.8698496],
  [52.1948447, 13.8691873],
  [52.1950772, 13.8685726],
  [52.1951168, 13.8684641],
  [52.1956681, 13.8670323],
  [52.1958716, 13.8664936],
  [52.1964366, 13.8649875],
  [52.1969283, 13.8636040],
  [52.1970203, 13.8634430],
  [52.1975486, 13.8626307],
  [52.1976354, 13.8624971],
  [52.1977827, 13.8621795],
  [52.1978564, 13.8619220],
  [52.1981843, 13.8604497],
  [52.1982614, 13.8602140],
  [52.1983351, 13.8600595],
  [52.1992768, 13.8579824],
  [52.1995107, 13.8574321],
  [52.1995948, 13.8572604],
  [52.1996818, 13.8571155],
  [52.1998000, 13.8570029],
  [52.2000659, 13.8568236],
  [52.2003868, 13.8566005],
  [52.2007182, 13.8564460],
  [52.2008760, 13.8564117],
  [52.2009865, 13.8564117],
  [52.2011390, 13.8564202],
  [52.2012267, 13.8564496],
  [52.2012544, 13.8564577],
  [52.2013179, 13.8564803],
  [52.2020491, 13.8571756],
  [52.2026014, 13.8576991],
  [52.2027592, 13.8578879],
  [52.2027960, 13.8579309],
  [52.2028960, 13.8580939],
  [52.2030170, 13.8583343],
  [52.2036587, 13.8597076],
  [52.2052946, 13.8633039],
  [52.2064332, 13.8658435],
  [52.2067856, 13.8666332],
  [52.2068961, 13.8668477],
  [52.2070777, 13.8670890],
  [52.2073723, 13.8674409],
  [52.2077457, 13.8679387],
  [52.2083874, 13.8687455],
  [52.2093341, 13.8699214],
  [52.2099652, 13.8707540],
  [52.2102282, 13.8712089],
  [52.2104228, 13.8715694],
  [52.2106122, 13.8718955],
  [52.2107619, 13.8721756],
  [52.2108695, 13.8723771],
  [52.2110747, 13.8727610],
  [52.2111514, 13.8729047],
  [52.2114010, 13.8733718],
  [52.2114694, 13.8735006],
  [52.2115430, 13.8736636],
  [52.2116086, 13.8737571],
  [52.2116770, 13.8738172],
  [52.2117611, 13.8738515],
  [52.2118664, 13.8738566],
  [52.2119322, 13.8738439],
  [52.2121058, 13.8737924],
  [52.2122583, 13.8737495],
  [52.2123265, 13.8737260],
  [52.2124213, 13.8736894],
  [52.2127466, 13.8734888],
  [52.2128263, 13.8734491],
  [52.2131313, 13.8733117],
  [52.2133943, 13.8731830],
  [52.2136625, 13.8731057],
  [52.2139465, 13.8730456],
  [52.2143619, 13.8730113],
  [52.2148773, 13.8729942],
  [52.2152275, 13.8730325],
  [52.2153110, 13.8730398],
  [52.2157442, 13.8730848],
  [52.2158833, 13.8731036]]


mockNodesData01 = MockNodesData(_WAY_NODES_COORDS_01)

# OSM Query around B96 south of Berlin
mockOSMResponse01 = MockOSMQueryResponse('selfdrive/mapd/test/mock_osm_response_01.xml',
                                         [52.31400353586984, 13.447158941786366])

# OSM Query on curvy town area south of Germany.
mockOSMResponse02 = MockOSMQueryResponse('selfdrive/mapd/test/mock_osm_response_02.xml',
                                         [48.16573269276522, 9.81418473659117])

mockWayCollection01 = WayCollection(mockOSMResponse01.ways, mockOSMResponse01.query_center)
mockWayCollection02 = WayCollection(mockOSMResponse02.ways, mockOSMResponse02.query_center)

# Normal curvy Way. way id: 179532213 with 35 Nodes.
mockOSMWay_01_01_LongCurvy = next(way for way in mockOSMResponse01.ways if way.id == 179532213)

# Looped way. way id: 29233907
mockOSMWay_01_02_Loop = next(way for way in mockOSMResponse01.ways if way.id == 29233907)

# Complex curvy road through town with intersections. way id:178450395
mockOSMWay_02_01_CurvyTownWithIntersections = next(way for way in mockOSMResponse02.ways if way.id == 178450395)

# Valid diversion for way 02_01 at node: 34785115. way id: 27955186
mockOSMWay_02_02_Divertion_34785115 = next(way for way in mockOSMResponse02.ways if way.id == 27955186)

# 3 node way. way id: 807781992
mockOSMWay_02_03_Short_3_node_way = next(way for way in mockOSMResponse02.ways if way.id == 807781992)

# data composing route 01 in way collection 02
mockRouteData_02_01 = MockRouteData([60890967, 737120246, 601406617, 60890971, 178450395], mockWayCollection02,
                                    first_node_id=201962346)

# data composing route 02 in way collection 02. Single WR
mockRouteData_02_02_single_wr = MockRouteData([178450395], mockWayCollection02, first_node_id=762086638)

# data composing route 03 in way collection 02. Multiple speed limits
mockRouteData_02_03 = MockRouteData([158799549, 798805532, 28707704, 158797898, 602249535, 602249536, 825823509,
                                     178449088, 916462523, 158796386], mockWayCollection02,
                                    first_node_id=252601829)

# 1000mt section with one full sin cycle as curv values.
mockCurveSectionSin = MockCurveSection(lambda x: np.sin(x * 2 * np.pi))

# 200mt section with changing curvature rate.
mockCurveSteepCurvChange = MockCurveSection(lambda x: 0.05 * x**3 - 0.007 * x**2 + 0.001 * x, df=200)

# _MIN_SPEED_SECTION_LENGTH section with changing curvature rate.
mockCurveSteepCurvChangeShort = MockCurveSection(
  lambda x: 0.05 * x**3 - 0.007 * x**2 + 0.001 * x, df=_MIN_SPEED_SECTION_LENGTH)

# 200mt section with smooth changing curvature rate. no deviation over 2.
mockCurveSmoothCurveChange = MockCurveSection(lambda x: 0.0002 * x**3 - 0.001 * x**2 + 0.6 * x, df=200)
