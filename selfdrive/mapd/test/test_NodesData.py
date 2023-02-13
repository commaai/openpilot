import unittest
import numpy as np
from selfdrive.mapd.lib.geo import DIRECTION
from common.conversions import Conversions as CV
from selfdrive.mapd.lib.WayRelation import WayRelation
from selfdrive.mapd.lib.NodesData import nodes_raw_data_array_for_wr, node_calculations, \
  spline_curvature_calculations, split_speed_section_by_sign, split_speed_section_by_curv_degree, speed_section, \
  speed_limits_for_curvatures_data, is_wr_a_valid_divertion_from_node, SpeedLimitSection, TurnSpeedLimitSection, \
  NodesData, NodeDataIdx
from selfdrive.mapd.test.mock_data import mockOSMWay_01_01_LongCurvy, mockNodesData01, mockCurveSectionSin, \
  mockCurveSteepCurvChange, mockCurveSteepCurvChangeShort, mockCurveSmoothCurveChange, \
  mockOSMWay_02_01_CurvyTownWithIntersections, mockOSMWay_02_02_Divertion_34785115, mockOSMWay_02_03_Short_3_node_way, \
  mockRouteData_02_01, mockRouteData_02_02_single_wr, mockRouteData_02_03
from numpy.testing import assert_array_almost_equal


class TestNodesDataFileFunctions(unittest.TestCase):
  def test_nodes_raw_data_array_for_wr(self):
    wr = WayRelation(mockOSMWay_01_01_LongCurvy)
    data_e = np.array([(n.id, n.lat, n.lon, wr.speed_limit) for n in wr.way.nodes], dtype=float)
    data = nodes_raw_data_array_for_wr(wr)

    assert_array_almost_equal(data, data_e)

  def test_nodes_raw_data_array_for_wr_flips_when_backwards(self):
    wr = WayRelation(mockOSMWay_01_01_LongCurvy)
    wr.direction = DIRECTION.BACKWARD

    data_e = np.array([(n.id, n.lat, n.lon, wr.speed_limit) for n in wr.way.nodes], dtype=float)
    data_e = np.flip(data_e, axis=0)

    data = nodes_raw_data_array_for_wr(wr)

    assert_array_almost_equal(data, data_e)

  def test_nodes_raw_data_array_for_wr_drops_last(self):
    wr = WayRelation(mockOSMWay_01_01_LongCurvy)
    data_e = np.array([(n.id, n.lat, n.lon, wr.speed_limit) for n in wr.way.nodes], dtype=float)[:-1]
    data = nodes_raw_data_array_for_wr(wr, drop_last=True)

    assert_array_almost_equal(data, data_e)

  def test_node_calculations(self):
    points = mockNodesData01.radians

    v, dp, dn, dr, b = node_calculations(points)

    assert_array_almost_equal(v, mockNodesData01.v)
    assert_array_almost_equal(dp, mockNodesData01.dp)
    assert_array_almost_equal(dn, mockNodesData01.dn)
    assert_array_almost_equal(dr, mockNodesData01.dr)
    assert_array_almost_equal(b, mockNodesData01.b)

  def test_node_calculations_index_error(self):
    points = mockNodesData01.radians[:2]

    with self.assertRaises(IndexError):
      node_calculations(points)

  def test_spline_curvature_calculations(self):
    vect = mockNodesData01.v
    dist_prev = mockNodesData01.dp

    curv, curv_ds = spline_curvature_calculations(vect, dist_prev)

    assert_array_almost_equal(curv, mockNodesData01.curv)
    assert_array_almost_equal(curv_ds, mockNodesData01.curv_ds)

  def test_spline_curvature_calculations_with_route_data(self):
    mockRouteData_02_01.reset()
    nodes_data = mockRouteData_02_01._nodes_data
    vect = np.column_stack((nodes_data[:, 4], nodes_data[:, 5]))
    dist_prev = nodes_data[:, 6]

    curv, curv_ds = spline_curvature_calculations(vect, dist_prev)

    assert_array_almost_equal(curv, mockRouteData_02_01._curv)
    assert_array_almost_equal(curv_ds, mockRouteData_02_01._curv_ds)

  def test_split_speed_section_by_sign(self):
    curv_sec = mockCurveSectionSin.curv_sec
    new_secs = split_speed_section_by_sign(curv_sec)

    # 3 sections with matching initial and final distance
    self.assertEqual(len(new_secs), 3)
    self.assertEqual(new_secs[0][0][2], mockCurveSectionSin.di)
    self.assertEqual(new_secs[2][-1][2], mockCurveSectionSin.df)

    # All new sections has same sign internally
    for sec in new_secs:
      self.assertEqual(np.average(sec, axis=0)[1], sec[0][1])

    # Sections change sign
    for idx in range(2):
      self.assertNotEqual(new_secs[idx][0][1], new_secs[idx + 1][0][1])

    # total items consistency
    lengths = [len(sec) for sec in new_secs]
    self.assertEqual(len(curv_sec), sum(lengths))

  def test_split_speed_section_by_curv_degree(self):
    curv_sec = mockCurveSteepCurvChange.curv_sec
    new_secs = split_speed_section_by_curv_degree(curv_sec)

    # 3 sections with matching initial and final distance
    self.assertEqual(len(new_secs), 3)
    self.assertEqual(new_secs[0][0][2], mockCurveSteepCurvChange.di)
    self.assertEqual(new_secs[2][-1][2], mockCurveSteepCurvChange.df)

    # Sections split at the right points
    split_dist = [sec[-1][2] for sec in new_secs]
    self.assertListEqual(split_dist, [50., 150., 200.])

  def test_split_speed_section_by_curv_degree_does_nothing_if_short(self):
    curv_sec = mockCurveSteepCurvChangeShort.curv_sec
    new_secs = split_speed_section_by_curv_degree(curv_sec)

    self.assertEqual(len(new_secs), 1)
    assert_array_almost_equal(curv_sec, new_secs[0])

  def test_split_speed_section_by_curv_degree_does_nothing_if_no_substantial_change(self):
    curv_sec = mockCurveSmoothCurveChange.curv_sec
    new_secs = split_speed_section_by_curv_degree(curv_sec)

    self.assertEqual(len(new_secs), 1)
    assert_array_almost_equal(curv_sec, new_secs[0])

  def test_speed_section(self):
    curv_sec = mockCurveSectionSin.curv_sec

    speed_secs = speed_section(curv_sec)
    expected = np.array([0., 1000., 1.51657509, 1.])

    assert_array_almost_equal(speed_secs, expected)

  def test_speed_limits_for_curvatures_data(self):
    curv = mockCurveSectionSin.curv
    curv_ds = mockCurveSectionSin.curv_ds

    expected = np.array([
      [10., 490., 1.51657509, 1.],
      [510., 990., 1.51657509, -1.]])
    limits = speed_limits_for_curvatures_data(curv, curv_ds)

    assert_array_almost_equal(limits, expected)

  def test_is_wr_a_valid_divertion_from_node(self):
    wr = WayRelation(mockOSMWay_02_01_CurvyTownWithIntersections)
    mockOSMWay_02_02_Divertion_34785115.tags['oneway'] = 'yes'
    wr_div = WayRelation(mockOSMWay_02_02_Divertion_34785115)

    # False if id already in route
    wr_ids = [wr.id, wr_div.id]
    self.assertFalse(is_wr_a_valid_divertion_from_node(wr_div, 34785115, wr_ids))

    # True if id not in route, node_id is edge and not prohibited
    wr_ids = [wr.id, 11111, 22222]
    self.assertTrue(is_wr_a_valid_divertion_from_node(wr_div, 34785115, wr_ids))

    # False if id not in route, node_id is edge but prohibited (wrong direction from node 319503453)
    self.assertFalse(is_wr_a_valid_divertion_from_node(wr_div, 319503453, wr_ids))

    # False if id not in route, node_id is not edge
    self.assertFalse(is_wr_a_valid_divertion_from_node(wr_div, 44444, wr_ids))


class TestSpeedLimitSection(unittest.TestCase):
  def test_speed_limit_section_init(self):
    section = SpeedLimitSection(10., 20., 50.)

    self.assertEqual(section.start, 10.)
    self.assertEqual(section.end, 20.)
    self.assertEqual(section.value, 50.)


class TestTurnSpeedLimitSection(unittest.TestCase):
  def test_turn_speed_limit_section_init(self):
    section = TurnSpeedLimitSection(10., 20., 50., -1.)

    self.assertEqual(section.start, 10.)
    self.assertEqual(section.end, 20.)
    self.assertEqual(section.value, 50.)
    self.assertEqual(section.curv_sign, -1.)


class TestNodesData(unittest.TestCase):
  def test_init_with_empty_list(self):
    nodesData = NodesData([], {})

    self.assertEqual(len(nodesData._nodes_data), 0)
    num_diverstions = sum([len(d) for d in nodesData._divertions])
    self.assertEqual(num_diverstions, 0)
    self.assertEqual(len(nodesData._curvature_speed_sections_data), 0)

  def test_init_with_single_wr_includes_all_wr_nodes(self):
    mockRouteData_02_02_single_wr.reset()
    way_relations = mockRouteData_02_02_single_wr.wrs
    wr_index = mockRouteData_02_02_single_wr.way_collection.wr_index

    nodesData = NodesData(way_relations, wr_index)

    assert_array_almost_equal(nodesData._nodes_data, mockRouteData_02_02_single_wr._nodes_data)
    assert_array_almost_equal(nodesData._curvature_speed_sections_data,
                              mockRouteData_02_02_single_wr._curvature_speed_sections_data)
    self.assertListEqual(nodesData._divertions, mockRouteData_02_02_single_wr._divertions)
    self.assertEqual(len(nodesData._nodes_data), len(way_relations[0].way.nodes))
    self.assertEqual(len(nodesData._curvature_speed_sections_data), 6)
    num_diverstions = sum([len(d) for d in nodesData._divertions])
    self.assertEqual(num_diverstions, 6)

  def test_init_with_less_than_4_nodes(self):
    wr_t = WayRelation(mockOSMWay_02_03_Short_3_node_way)

    nodesData = NodesData([wr_t], {})

    self.assertEqual(len(nodesData._nodes_data), 0)
    num_diverstions = sum([len(d) for d in nodesData._divertions])
    self.assertEqual(num_diverstions, 0)
    self.assertEqual(len(nodesData._curvature_speed_sections_data), 0)

  def test_init_with_multiple_wr(self):
    mockRouteData_02_01.reset()
    way_relations = mockRouteData_02_01.wrs
    wr_index = mockRouteData_02_01.way_collection.wr_index

    nodesData = NodesData(way_relations, wr_index)

    assert_array_almost_equal(nodesData._nodes_data, mockRouteData_02_01._nodes_data)
    assert_array_almost_equal(nodesData._curvature_speed_sections_data, mockRouteData_02_01._curvature_speed_sections_data)
    self.assertListEqual(nodesData._divertions, mockRouteData_02_01._divertions)
    self.assertEqual(len(nodesData._curvature_speed_sections_data), 9)
    num_diverstions = sum([len(d) for d in nodesData._divertions])
    self.assertEqual(num_diverstions, 14)

  def test_count(self):
    mockRouteData_02_01.reset()
    way_relations = mockRouteData_02_01.wrs
    wr_index = mockRouteData_02_01.way_collection.wr_index
    num_n = sum([len(wr.way.nodes) for wr in way_relations]) - len(way_relations) + 1

    nodesData = NodesData(way_relations, wr_index)

    self.assertEqual(nodesData.count, num_n)

  def test_get_on_empty(self):
    wr_t = WayRelation(mockOSMWay_02_03_Short_3_node_way)

    nodesData = NodesData([wr_t], {})
    assert_array_almost_equal(nodesData.get(NodeDataIdx.node_id), np.array([]))

  def test_get_values(self):
    mockRouteData_02_01.reset()
    way_relations = mockRouteData_02_01.wrs
    wr_index = mockRouteData_02_01.way_collection.wr_index

    nodesData = NodesData(way_relations, wr_index)

    assert_array_almost_equal(nodesData.get(NodeDataIdx.node_id), mockRouteData_02_01._nodes_data[:, 0])
    assert_array_almost_equal(nodesData.get(NodeDataIdx.lat), mockRouteData_02_01._nodes_data[:, 1])
    assert_array_almost_equal(nodesData.get(NodeDataIdx.lon), mockRouteData_02_01._nodes_data[:, 2])
    assert_array_almost_equal(nodesData.get(NodeDataIdx.speed_limit), mockRouteData_02_01._nodes_data[:, 3])
    assert_array_almost_equal(nodesData.get(NodeDataIdx.x), mockRouteData_02_01._nodes_data[:, 4])
    assert_array_almost_equal(nodesData.get(NodeDataIdx.y), mockRouteData_02_01._nodes_data[:, 5])
    assert_array_almost_equal(nodesData.get(NodeDataIdx.dist_prev), mockRouteData_02_01._nodes_data[:, 6])
    assert_array_almost_equal(nodesData.get(NodeDataIdx.dist_next), mockRouteData_02_01._nodes_data[:, 7])
    assert_array_almost_equal(nodesData.get(NodeDataIdx.dist_route), mockRouteData_02_01._nodes_data[:, 8])
    assert_array_almost_equal(nodesData.get(NodeDataIdx.bearing), mockRouteData_02_01._nodes_data[:, 9])

  def test_speed_limits_ahead_from_empty(self):
    wr_t = WayRelation(mockOSMWay_02_03_Short_3_node_way)

    nodesData = NodesData([wr_t], {})
    self.assertEqual(len(nodesData.speed_limits_ahead(1, 10.)), 0)

  def test_speed_limits_ahead(self):
    mockRouteData_02_03.reset()
    way_relations = mockRouteData_02_03.wrs
    wr_index = mockRouteData_02_03.way_collection.wr_index

    nodesData = NodesData(way_relations, wr_index)

    # empty when ahead_idx is none.
    self.assertEqual(len(nodesData.speed_limits_ahead(None, 10.)), 0)

    # All limist from 0
    all_limits = nodesData.speed_limits_ahead(1, nodesData.get(NodeDataIdx.dist_next)[0])
    self.assertEqual(len(all_limits), 4)  # 4 limits on this mock road.
    self.assertListEqual([sl.value for sl in all_limits], [v * CV.KPH_TO_MS for v in [50, 100, 50, 100]])
    for idx, sl in enumerate(all_limits):
      self.assertTrue(sl.end > sl.start)
      self.assertTrue(sl.value > 0.)
      if idx == 0:
        self.assertEqual(sl.start, 0.)
      else:
        self.assertEqual(sl.start, all_limits[idx - 1].end)
        self.assertNotEqual(sl.value, all_limits[idx - 1].value)

  def test_distance_to_end_from_empty(self):
    wr_t = WayRelation(mockOSMWay_02_03_Short_3_node_way)

    nodesData = NodesData([wr_t], {})
    self.assertIsNone(nodesData.distance_to_end(1, 10.))

  def test_distance_to_end(self):
    mockRouteData_02_03.reset()
    way_relations = mockRouteData_02_03.wrs
    wr_index = mockRouteData_02_03.way_collection.wr_index

    nodesData = NodesData(way_relations, wr_index)

    # none when ahead_idx is none.
    self.assertIsNone(nodesData.distance_to_end(None, 10.))

    # From the beginning
    expected = np.sum(nodesData.get(NodeDataIdx.dist_next))
    self.assertAlmostEqual(nodesData.distance_to_end(1, nodesData.get(NodeDataIdx.dist_next)[0]), expected)
    self.assertAlmostEqual(nodesData.get(NodeDataIdx.dist_route)[-1], expected)

    # From the node next to last
    expected = nodesData.get(NodeDataIdx.dist_next)[-2]
    self.assertAlmostEqual(nodesData.distance_to_end(nodesData.count - 2, 0.), expected)

  def test_distance_to_node(self):
    mockRouteData_02_03.reset()
    way_relations = mockRouteData_02_03.wrs
    wr_index = mockRouteData_02_03.way_collection.wr_index

    nodesData = NodesData(way_relations, wr_index)
    dist_to_node_ahead = 10.
    node_id = 1887995486  # Some node id in the middle of the way. idx 50
    node_idx = np.nonzero(nodesData.get(NodeDataIdx.node_id) == node_id)[0][0]

    # none when ahead_idx is none.
    self.assertIsNone(nodesData.distance_to_node(node_id, None, dist_to_node_ahead))

    # From the beginning
    expected = nodesData.get(NodeDataIdx.dist_route)[node_idx]
    self.assertAlmostEqual(nodesData.distance_to_node(node_id, 1, nodesData.get(NodeDataIdx.dist_next)[0]), expected)

    # From the end
    expected = -np.sum(nodesData.get(NodeDataIdx.dist_next)[node_idx:])
    self.assertAlmostEqual(nodesData.distance_to_node(node_id, len(nodesData.get(NodeDataIdx.node_id)) - 1, 0.), expected)

    # From some node behind including dist to node ahead
    ahead_idx = node_idx - 10
    expected = np.sum(nodesData.get(NodeDataIdx.dist_next)[ahead_idx:node_idx]) + dist_to_node_ahead
    self.assertAlmostEqual(nodesData.distance_to_node(node_id, ahead_idx, dist_to_node_ahead), expected)

    # From some node ahead including dist to node ahead
    ahead_idx = node_idx + 10
    expected = -np.sum(nodesData.get(NodeDataIdx.dist_next)[node_idx:ahead_idx]) + dist_to_node_ahead
    self.assertAlmostEqual(nodesData.distance_to_node(node_id, ahead_idx, dist_to_node_ahead), expected)

# TODO: Missing tests for curvatures_speed_limit_sections_ahead and possible_divertions
