import copy
import unittest
import numpy as np
from unittest import mock
from numpy.testing import assert_array_almost_equal
from datetime import datetime as dt, timezone, timedelta
from common.conversions import Conversions as CV
from selfdrive.mapd.lib.WayRelation import WayRelation, is_osm_time_condition_active, \
  conditional_speed_limit_for_osm_tag_limit_string, speed_limit_for_osm_tag_limit_string
from selfdrive.mapd.config import LANE_WIDTH
from selfdrive.mapd.lib.geo import DIRECTION, R, vectors
from selfdrive.mapd.test.mock_data import mockOSMWay_01_01_LongCurvy, mockOSMWay_01_02_Loop, \
  mockOSMWay_02_01_CurvyTownWithIntersections


class TestWayRelationFileFunctions(unittest.TestCase):
  def test_speed_limit_for_osm_tag_limit_string(self):
    values = [
      None,  # Invalid
      "1000",  # Invalid
      "60 kph",  # Invalid
      "100",
      "30 mph",
      "DE:zone:40",
      "DE:zone:50 mph",
      "AR:urban",
      "CZ:pedestrian_zone",
      "DK:urban",
      "DK:rural",
      "DK:motorway",
      "DE:living_street",
      "DE:residential",
      "DE:urban",
      "DE:rural",
      "DE:trunk",  # No limit
      "DE:motorway",  # No limit
      "GB:nsl_restricted",
      "GB:nsl_single",
      "GB:nsl_dual",
      "GB:motorway",
      "GB:invalid",  # Invalid
    ]

    expected = [
      0.,
      0.,
      0.,
      100. * CV.KPH_TO_MS,
      30. * CV.MPH_TO_MS,
      40. * CV.KPH_TO_MS,
      50. * CV.MPH_TO_MS,
      40. * CV.KPH_TO_MS,
      20. * CV.KPH_TO_MS,
      50. * CV.KPH_TO_MS,
      80. * CV.KPH_TO_MS,
      130. * CV.KPH_TO_MS,
      7. * CV.KPH_TO_MS,
      30. * CV.KPH_TO_MS,
      50. * CV.KPH_TO_MS,
      100. * CV.KPH_TO_MS,
      0.,
      0.,
      30. * CV.MPH_TO_MS,
      60. * CV.MPH_TO_MS,
      70. * CV.MPH_TO_MS,
      70. * CV.MPH_TO_MS,
      0.,
    ]

    result = [speed_limit_for_osm_tag_limit_string(sls) for sls in values]

    self.assertEqual(result, expected)

  @mock.patch('selfdrive.mapd.lib.WayRelation.dt')
  def test_is_osm_time_condition_active(self, mock_dt):
    tz = timezone(timedelta(hours=1), 'berlin')
    wed_10_10_am = dt(2021, 9, 1, 10, 10, 0)
    mock_dt.now.return_value = wed_10_10_am
    mock_dt.tzinfo = tz
    mock_dt.combine = dt.combine
    mock_dt.strptime = dt.strptime

    values = [
      "WE",  # Invalid
      "We",
      "Mo",
      "Fr",
      "Tu-Th",
      "10:00",  # Invalid
      "10:00-10:30",
      "We 10:00-10:30",
      "SU 10:00-10:30",  # Valid, SU string not considered a day string.
      "Sa 10:00-10:30",
      "Tu-Th 10:00-10:30",
    ]

    expected = [
      False,  # Invalid
      True,
      False,
      False,
      True,
      False,  # Invalid
      True,
      True,
      True,
      False,
      True,
    ]

    result = [is_osm_time_condition_active(cs) for cs in values]

    self.assertEqual(result, expected)

  @mock.patch('selfdrive.mapd.lib.WayRelation.dt')
  def test_conditional_speed_limit_for_osm_tag_limit_string(self, mock_dt):
    tz = timezone(timedelta(hours=1), 'berlin')
    wed_10_10_am = dt(2021, 9, 1, 10, 10, 0)
    mock_dt.now.return_value = wed_10_10_am
    mock_dt.tzinfo = tz
    mock_dt.combine = dt.combine
    mock_dt.strptime = dt.strptime

    values = [
      None,  # Invalid
      "Hola",  # Invalid
      "100 @ (WE)",  # Invalid
      "x @ (We)",  # Invalid
      "100 @ (We)",
      "100 @ (Mo)",
      "100 @ (Fr)",
      "100 @ (Tu-Th)",
      "100 @ (10:00)",  # Invalid
      "100 @ (10:00-10:30)",
      "100 @ (We 10:00-10:30)",
      "100 @ (SU 10:00-10:30)",  # Valid, SU string not considered a day string.
      "100 @ (Sa 10:00-10:30)",
      "100 @ (Tu-Th 10:00-10:30)",
      "100 @ (Mo-Th;Su)",
      "100 @ (Mo Th;Fr-Sa)",
      "100 @ (Fr-Su;Mo-Tu)",
      "100 @ (10:00-10:30;15:00-16:00)",
      "100 @ (We;Mo-Tu)",
      "100 @ (We 10:00-10:30;Th 15:00-16:00)",
      "100 @ (Tu 10:00-10:30;Th 15:00-16:00)",
    ]

    _100 = 100. * CV.KPH_TO_MS

    expected = [
      0.,  # Invalid
      0.,  # Invalid
      0.,  # Invalid
      0.,  # Invalid
      _100,
      0.,
      0.,
      _100,
      0.,  # Invalid
      _100,
      _100,
      _100,
      0.,
      _100,
      _100,
      _100,
      0.,
      _100,
      _100,
      _100,
      0.
    ]

    result = [conditional_speed_limit_for_osm_tag_limit_string(ls) for ls in values]

    self.assertEqual(result, expected)


class TestWayRelation(unittest.TestCase):
  def test_way_relation_init(self):
    wayRelation = WayRelation(mockOSMWay_01_01_LongCurvy)

    nodes_np_expected = np.radians(np.array([[node.lat, node.lon] for node in wayRelation.way.nodes], dtype=float))
    v = vectors(wayRelation._nodes_np)
    way_distances_expected = np.linalg.norm(v * R, axis=1)
    way_bearings_expected = np.arctan2(v[:, 0], v[:, 1])
    bbox_expected = np.array([
      [0.91321784, 0.2346417],
      [0.91344672, 0.23475751]])

    self.assertEqual(wayRelation.way.id, 179532213)
    self.assertIsNone(wayRelation.parent_wr_id)
    self.assertEqual(wayRelation.direction, DIRECTION.NONE)
    self.assertEqual(wayRelation._speed_limit, None)
    self.assertEqual(wayRelation._one_way, 'yes')
    self.assertEqual(wayRelation.name, None)
    self.assertEqual(wayRelation.ref, 'B 96')
    self.assertEqual(wayRelation.highway_type, 'trunk')
    self.assertEqual(wayRelation.highway_rank, 10)
    self.assertEqual(wayRelation.lanes, 2)
    assert_array_almost_equal(wayRelation._nodes_np, nodes_np_expected)
    assert_array_almost_equal(wayRelation._way_distances, way_distances_expected)
    assert_array_almost_equal(wayRelation._way_bearings, way_bearings_expected)
    assert_array_almost_equal(wayRelation.bbox, bbox_expected)
    self.assertEqual(wayRelation.edge_nodes_ids, [wayRelation.way.nodes[0].id, wayRelation.way.nodes[-1].id])

  def test_way_relation_init_with_parent(self):
    wayRelation = WayRelation(mockOSMWay_01_01_LongCurvy, parent=WayRelation(mockOSMWay_01_02_Loop))

    self.assertEqual(wayRelation.way.id, 179532213)
    self.assertEqual(wayRelation.parent_wr_id, 29233907)

  def test_way_relation_equality(self):
    wayRelation1 = WayRelation(mockOSMWay_01_01_LongCurvy)
    wayRelation2 = copy.copy(wayRelation1)
    wayRelation3 = copy.deepcopy(wayRelation1)
    wayRelation3.way.id = 123

    self.assertEqual(wayRelation1, wayRelation2)
    self.assertNotEqual(wayRelation1, wayRelation3)

  def test_way_relation_reset_location_variables(self):
    wayRelation = WayRelation(mockOSMWay_01_01_LongCurvy)
    self.make_wayRelation_location_dirty(wayRelation)

    wayRelation.reset_location_variables()

    self.assert_wayRelation_variables_reset(wayRelation)

  def test_way_relation_id(self):
    wayRelation = WayRelation(mockOSMWay_01_01_LongCurvy)

    self.assertEqual(wayRelation.id, 179532213)

  def test_way_relation_road_name(self):
    # road name when no tag for name or ref
    wayRelation = WayRelation(mockOSMWay_01_02_Loop)
    self.assertIsNone(wayRelation.road_name)
    # road name based on ref tag
    wayRelation = WayRelation(mockOSMWay_01_01_LongCurvy)
    self.assertEqual(wayRelation.road_name, "B 96")
    # road name based on name tag
    wayRelation = WayRelation(mockOSMWay_02_01_CurvyTownWithIntersections)
    self.assertEqual(wayRelation.road_name, "Hauptstraße")

  def test_way_relation_update_resets_on_update(self):
    wayRelation = WayRelation(mockOSMWay_01_01_LongCurvy)
    self.make_wayRelation_location_dirty(wayRelation)
    location_rad = np.array([0., 0.])  # Location outside bbox

    wayRelation.update(location_rad, 0., 10.)

    self.assertFalse(wayRelation.is_location_in_bbox(location_rad))
    self.assert_wayRelation_variables_reset(wayRelation)

  def test_way_relation_update_only_resets_if_no_possible_found(self):
    wayRelation = WayRelation(mockOSMWay_01_01_LongCurvy)
    location_rad = wayRelation.bbox[0]  # Location inside bbox but outside actual way (due to padding)

    wayRelation.update(location_rad, 0., 10.)

    self.assertTrue(wayRelation.is_location_in_bbox(location_rad))
    self.assert_wayRelation_variables_reset(wayRelation)

  def test_way_relation_updates_in_the_correct_direction_with_correct_property_values(self):
    wayRelation = WayRelation(mockOSMWay_01_01_LongCurvy)
    location_rad = np.radians(np.array([52.32855593146639, 13.445320150125069]))
    bearing_rad = 0.

    wayRelation.update(location_rad, bearing_rad, 10.)

    self.assertTrue(wayRelation.is_location_in_bbox(location_rad))
    self.assertEqual(wayRelation.direction, DIRECTION.FORWARD)
    self.assertEqual(wayRelation.ahead_idx, 17)
    self.assertEqual(wayRelation.behind_idx, 16)
    self.assertAlmostEqual(wayRelation._distance_to_way, 3.43290781621360)
    self.assertAlmostEqual(wayRelation._active_bearing_delta, 0.320717420388962)
    self.assertAlmostEqual(wayRelation.distance_to_node_ahead, 25.4998961709014)
    self.assertTrue(wayRelation.active)
    self.assertFalse(wayRelation.diverting)
    assert_array_almost_equal(wayRelation.location_rad, location_rad)
    self.assertEqual(wayRelation.bearing_rad, bearing_rad)
    self.assertIsNone(wayRelation._speed_limit)

    bearing_rad = 180.

    wayRelation.update(location_rad, bearing_rad, 10.)

    self.assertTrue(wayRelation.is_location_in_bbox(location_rad))
    self.assertEqual(wayRelation.direction, DIRECTION.BACKWARD)
    self.assertEqual(wayRelation.ahead_idx, 16)
    self.assertEqual(wayRelation.behind_idx, 17)
    self.assertAlmostEqual(wayRelation._distance_to_way, 3.43290781621360)
    self.assertAlmostEqual(wayRelation._active_bearing_delta, 0.9507682562504284)
    self.assertAlmostEqual(wayRelation.distance_to_node_ahead, 11.11623371145368)
    self.assertTrue(wayRelation.active)
    self.assertFalse(wayRelation.diverting)
    assert_array_almost_equal(wayRelation.location_rad, location_rad)
    self.assertEqual(wayRelation.bearing_rad, bearing_rad)
    self.assertIsNone(wayRelation._speed_limit)

  def test_way_relation_updates_with_location_closest_to_way_when_multiple_possible(self):
    wayRelation = WayRelation(mockOSMWay_01_02_Loop)
    location_rad = np.radians(np.array([52.313303275461564, 13.437729236325788]))
    bearing_rad = np.radians(10.)

    wayRelation.update(location_rad, bearing_rad, 10.)

    self.assertTrue(wayRelation.is_location_in_bbox(location_rad))
    self.assertEqual(wayRelation.direction, DIRECTION.BACKWARD)
    self.assertEqual(wayRelation.ahead_idx, 26)
    self.assertEqual(wayRelation.behind_idx, 27)
    self.assertAlmostEqual(wayRelation._distance_to_way, 10.151775235257011)
    self.assertAlmostEqual(wayRelation._active_bearing_delta, 0.06371131069242782)
    self.assertAlmostEqual(wayRelation.distance_to_node_ahead, 10.174073707120915)
    self.assertTrue(wayRelation.active)
    self.assertFalse(wayRelation.diverting)
    assert_array_almost_equal(wayRelation.location_rad, location_rad)
    self.assertEqual(wayRelation.bearing_rad, bearing_rad)
    self.assertIsNone(wayRelation._speed_limit)

  def test_way_relation_updates_will_become_inactive_if_too_far_from_way(self):
    wayRelation = WayRelation(mockOSMWay_01_01_LongCurvy)
    # Location is 24.9 mts away from the way. There are 2 Lanes in this way.
    location_rad = np.radians(np.array([52.328634560607746, 13.445609877522788]))
    location_stdev = 5.5  # threshold is 4 * location_stdev + LANE_WIDTH
    distance_threshold = 4. * location_stdev + wayRelation.lanes * LANE_WIDTH / 2.

    wayRelation.update(location_rad, 0., location_stdev)
    self.assertTrue(wayRelation.active)
    self.assertLess(wayRelation._distance_to_way, distance_threshold)

    location_stdev = 5.

    wayRelation.update(location_rad, 0., location_stdev)
    self.assertFalse(wayRelation.active)

  def test_way_relation_updates_will_update_diverting_correctly(self):
    wayRelation = WayRelation(mockOSMWay_01_01_LongCurvy)
    # Location is 24.9 mts away from the way. There are 2 Lanes in this way.
    location_rad = np.radians(np.array([52.328634560607746, 13.445609877522788]))
    location_stdev = 11.
    distance_threshold = 2. * location_stdev + wayRelation.lanes * LANE_WIDTH / 2.

    wayRelation.update(location_rad, 0., location_stdev)

    self.assertLess(wayRelation._distance_to_way, distance_threshold)
    self.assertFalse(wayRelation.diverting)

    location_stdev = 10.
    distance_threshold = 2. * location_stdev + wayRelation.lanes * LANE_WIDTH / 2.

    wayRelation.update(location_rad, 0., location_stdev)

    self.assertGreater(wayRelation._distance_to_way, distance_threshold)
    self.assertTrue(wayRelation.diverting)

  def test_way_relation_update_direction_from_starting_node_resets_speed_limit(self):
    wayRelation = WayRelation(mockOSMWay_01_01_LongCurvy)
    wayRelation._speed_limit = 10.

    wayRelation.update_direction_from_starting_node(wayRelation.way.nodes[0].id)

    self.assertIsNone(wayRelation._speed_limit)

  def test_way_relation_update_direction_from_starting_node_updates_correctly(self):
    wayRelation = WayRelation(mockOSMWay_01_01_LongCurvy)
    wayRelation.update_direction_from_starting_node(wayRelation.way.nodes[0].id)
    self.assertEqual(wayRelation.direction, DIRECTION.FORWARD)

    wayRelation.update_direction_from_starting_node(wayRelation.way.nodes[-1].id)
    self.assertEqual(wayRelation.direction, DIRECTION.BACKWARD)

    wayRelation.update_direction_from_starting_node(0)
    self.assertEqual(wayRelation.direction, DIRECTION.NONE)

  def test_way_relation_is_location_in_bbox(self):
    wayRelation = WayRelation(mockOSMWay_01_02_Loop)
    bbox = wayRelation.bbox

    loc_avg = np.average(bbox, axis=0)
    loc_min = np.min(bbox, axis=0)
    loc_max = np.max(bbox, axis=0)

    locations = [
      loc_avg,
      loc_min,
      loc_max,
      [loc_avg[0], loc_min[1]],
      [loc_avg[0], loc_max[1]],
      [loc_min[0], loc_avg[1]],
      [loc_max[0], loc_avg[1]],
      loc_min - 0.1,
      loc_max + 0.1,
      [loc_avg[0], loc_min[1] - 0.1],
      [loc_avg[0], loc_max[1] + 0.1],
      [loc_min[0] - 0.1, loc_avg[1]],
      [loc_max[0] + 0.1, loc_avg[1]],
    ]

    is_in = [wayRelation.is_location_in_bbox(loc) for loc in locations]

    self.assertEqual(is_in, [True, True, True, True, True, True, True, False, False, False, False, False, False])

  def test_way_relation_speed_limit_when_set(self):
    wayRelation = WayRelation(mockOSMWay_01_02_Loop)
    wayRelation._speed_limit = 10.

    self.assertEqual(wayRelation.speed_limit, 10.)

  @mock.patch('selfdrive.mapd.lib.WayRelation.dt')
  def test_way_relation_speed_limit_conditional(self, mock_dt):
    tz = timezone(timedelta(hours=1), 'berlin')
    wed_10_10_am = dt(2021, 9, 1, 10, 10, 0)
    mock_dt.now.return_value = wed_10_10_am
    mock_dt.tzinfo = tz
    mock_dt.combine = dt.combine
    mock_dt.strptime = dt.strptime

    # Reset all tags before teting
    mockOSMWay_01_02_Loop.tags = {}
    wayRelation = WayRelation(mockOSMWay_01_02_Loop)

    # No Value
    self.assertEqual(wayRelation.speed_limit, 0.)

    # Value on both directions
    wayRelation._speed_limit = None
    wayRelation.way.tags["maxspeed:conditional"] = "100 @ (We 10:00-10:30)"
    self.assertEqual(wayRelation.speed_limit, 100. * CV.KPH_TO_MS)

    # Value on forward
    wayRelation.way.tags.pop("maxspeed:conditional")
    wayRelation._speed_limit = None
    wayRelation.direction = DIRECTION.FORWARD
    self.assertEqual(wayRelation.speed_limit, 0.)

    wayRelation._speed_limit = None
    wayRelation.way.tags["maxspeed:forward:conditional"] = "100 @ (We 10:00-10:30)"
    self.assertEqual(wayRelation.speed_limit, 100. * CV.KPH_TO_MS)

    # Value on backward
    wayRelation._speed_limit = None
    wayRelation.direction = DIRECTION.BACKWARD
    self.assertEqual(wayRelation.speed_limit, 0.)

    wayRelation._speed_limit = None
    wayRelation.way.tags["maxspeed:backward:conditional"] = "100 @ (We 10:00-10:30)"
    self.assertEqual(wayRelation.speed_limit, 100. * CV.KPH_TO_MS)

  def test_way_relation_speed_limit_maxspeed(self):
    # Reset all tags before teting
    mockOSMWay_01_02_Loop.tags = {}
    wayRelation = WayRelation(mockOSMWay_01_02_Loop)

    # No Value
    self.assertEqual(wayRelation.speed_limit, 0.)

    # Value on both directions
    wayRelation._speed_limit = None
    wayRelation.way.tags["maxspeed"] = "100"
    self.assertEqual(wayRelation.speed_limit, 100. * CV.KPH_TO_MS)

    # Value on forward
    wayRelation.way.tags.pop("maxspeed")
    wayRelation._speed_limit = None
    wayRelation.direction = DIRECTION.FORWARD
    self.assertEqual(wayRelation.speed_limit, 0.)

    wayRelation._speed_limit = None
    wayRelation.way.tags["maxspeed:forward"] = "100"
    self.assertEqual(wayRelation.speed_limit, 100. * CV.KPH_TO_MS)

    # Value on backward
    wayRelation._speed_limit = None
    wayRelation.direction = DIRECTION.BACKWARD
    self.assertEqual(wayRelation.speed_limit, 0.)

    wayRelation._speed_limit = None
    wayRelation.way.tags["maxspeed:backward"] = "100"
    self.assertEqual(wayRelation.speed_limit, 100. * CV.KPH_TO_MS)

  def test_way_relation_active_bearing_delta_reflects_internal_value(self):
    wayRelation = WayRelation(mockOSMWay_01_02_Loop)
    wayRelation._active_bearing_delta = 10.
    self.assertEqual(wayRelation.active_bearing_delta, 10.)

  def test_way_relation_is_one_way(self):
    # Setup initial tags
    mockOSMWay_01_02_Loop.tags = {
      'oneway': 'yes',
      'highway': 'unclassified'
    }
    wayRelation = WayRelation(mockOSMWay_01_02_Loop)

    # oneway = yes
    self.assertTrue(wayRelation.is_one_way)

    # oneway non existing
    wayRelation._one_way = None
    self.assertFalse(wayRelation.is_one_way)

    # highway = motorway
    wayRelation.highway_type = 'motorway'
    self.assertTrue(wayRelation.is_one_way)

  def test_way_relation_is_prohibited(self):
    # Setup initial tags
    mockOSMWay_01_02_Loop.tags = {
      'oneway': 'yes'
    }
    wayRelation = WayRelation(mockOSMWay_01_02_Loop)

    # Direction undefined
    wayRelation.direction = DIRECTION.NONE
    self.assertTrue(wayRelation.is_prohibited)

    # oneway = yes
    wayRelation.direction = DIRECTION.BACKWARD
    self.assertTrue(wayRelation.is_prohibited)

    wayRelation.direction = DIRECTION.FORWARD
    self.assertFalse(wayRelation.is_prohibited)

    # oneway non existing
    wayRelation._one_way = None
    self.assertFalse(wayRelation.is_one_way)

    wayRelation.direction = DIRECTION.BACKWARD
    self.assertFalse(wayRelation.is_prohibited)

  def test_way_relation_distance_to_way_reflects_internal_value(self):
    wayRelation = WayRelation(mockOSMWay_01_02_Loop)
    wayRelation._distance_to_way = 10.
    self.assertEqual(wayRelation.distance_to_way, 10.)

  def test_way_relation_node_ahead(self):
    wayRelation = WayRelation(mockOSMWay_01_01_LongCurvy)
    # ahead_ids is None on init
    self.assertIsNone(wayRelation.node_ahead)

    wayRelation.ahead_idx = 15
    self.assertEqual(wayRelation.node_ahead, wayRelation.way.nodes[15])

  def test_way_relation_last_node(self):
    wayRelation = WayRelation(mockOSMWay_01_01_LongCurvy)
    # direction is NONE on init
    self.assertIsNone(wayRelation.last_node)

    # forward
    wayRelation.direction = DIRECTION.FORWARD
    self.assertEqual(wayRelation.last_node, wayRelation.way.nodes[-1])

    # backward
    wayRelation.direction = DIRECTION.BACKWARD
    self.assertEqual(wayRelation.last_node, wayRelation.way.nodes[0])

  def test_way_relation_last_node_coordinates(self):
    wayRelation = WayRelation(mockOSMWay_01_01_LongCurvy)
    # direction is NONE on init
    self.assertIsNone(wayRelation.last_node_coordinates)

    # forward
    wayRelation.direction = DIRECTION.FORWARD
    coords = np.radians(np.array([wayRelation.way.nodes[-1].lat, wayRelation.way.nodes[-1].lon], dtype=float))
    assert_array_almost_equal(wayRelation.last_node_coordinates, coords)

    # backward
    wayRelation.direction = DIRECTION.BACKWARD
    coords = np.radians(np.array([wayRelation.way.nodes[0].lat, wayRelation.way.nodes[0].lon], dtype=float))
    assert_array_almost_equal(wayRelation.last_node_coordinates, coords)

  def test_way_relation_node_before_edge_coordinates(self):
    wayRelation = WayRelation(mockOSMWay_01_01_LongCurvy)

    coords = wayRelation.node_before_edge_coordinates(0)
    assert_array_almost_equal(coords, np.array([0., 0.]))

    coords = wayRelation.node_before_edge_coordinates(wayRelation.way.nodes[0].id)
    coords_e = np.radians(np.array([wayRelation.way.nodes[1].lat, wayRelation.way.nodes[1].lon], dtype=float))
    assert_array_almost_equal(coords, coords_e)

    coords = wayRelation.node_before_edge_coordinates(wayRelation.way.nodes[-1].id)
    coords_e = np.radians(np.array([wayRelation.way.nodes[-2].lat, wayRelation.way.nodes[-2].lon], dtype=float))
    assert_array_almost_equal(coords, coords_e)

  def test_way_relation_split_no_matching_node(self):
    wayRelation = WayRelation(mockOSMWay_01_01_LongCurvy)

    wrs = wayRelation.split(0)
    self.assertEqual(len(wrs), 0)

  def test_way_relation_split_use_correct_ids(self):
    wayRelation = WayRelation(mockOSMWay_01_01_LongCurvy)

    wrs = wayRelation.split(wayRelation._nodes_ids[5], [-100, -200])
    self.assertEqual(wrs[0].id, -100)
    self.assertEqual(wrs[1].id, -200)

  def test_way_relation_split_on_edge_node(self):
    wayRelation = WayRelation(mockOSMWay_01_01_LongCurvy)
    edge_node_ids = wayRelation.edge_nodes_ids

    for edge_node_id in edge_node_ids:
      wrs = wayRelation.split(edge_node_id)
      self.assertEqual(len(wrs), 1)
      self.assertEqual(wrs[0], wayRelation)
      self.assertEqual(wrs[0].way.tags, wayRelation.way.tags)

  def test_way_relation_split_on_internal_node(self):
    wayRelation = WayRelation(mockOSMWay_01_01_LongCurvy)
    way_ids = [-10, -20]

    for idx, node_id in enumerate(wayRelation._nodes_ids):
      if idx == 0 or idx == len(wayRelation._nodes_ids) - 1:
        continue
      wrs = wayRelation.split(node_id, way_ids)
      self.assertEqual(len(wrs), 2)
      assert_array_almost_equal(wrs[0]._nodes_ids, wayRelation._nodes_ids[:idx + 1])
      assert_array_almost_equal(wrs[1]._nodes_ids, wayRelation._nodes_ids[idx:])
      self.assertIn(node_id, wrs[0].edge_nodes_ids)
      self.assertIn(node_id, wrs[1].edge_nodes_ids)
      self.assertEqual(wrs[0].way.tags, wayRelation.way.tags)
      self.assertEqual(wrs[1].way.tags, wayRelation.way.tags)
      self.assertEqual(way_ids, [wr.id for wr in wrs])

  # Helpers
  def make_wayRelation_location_dirty(self, wayRelation):
    wayRelation.distance_to_node_ahead = 10.
    wayRelation.location_rad = 0.8
    wayRelation.bearing_rad = 2.
    wayRelation.active = True
    wayRelation.diverting = True
    wayRelation.ahead_idx = 5
    wayRelation.behind_idx = 4
    wayRelation._active_bearing_delta = 3.
    wayRelation._distance_to_way = 20.

  def assert_wayRelation_variables_reset(self, wayRelation):
    self.assertEqual(wayRelation.distance_to_node_ahead, 0.)
    self.assertIsNone(wayRelation.location_rad)
    self.assertIsNone(wayRelation.bearing_rad)
    self.assertFalse(wayRelation.active)
    self.assertFalse(wayRelation.diverting)
    self.assertIsNone(wayRelation.ahead_idx)
    self.assertIsNone(wayRelation.behind_idx)
    self.assertIsNone(wayRelation._active_bearing_delta)
    self.assertIsNone(wayRelation._distance_to_way)

  def wayRelation_mid_point_rad(self, wayRelation):
    return np.average(wayRelation.bbox, axis=0)
