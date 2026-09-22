import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyray as rl

from openpilot.selfdrive.ui.mici.onroad.model_renderer import ModelRenderer, ModelPoints, LeadVehicle


class TestLeadBars(unittest.TestCase):
  def setUp(self):
    self.renderer = ModelRenderer.__new__(ModelRenderer)
    x = np.linspace(0, 100, 101)
    self.renderer._path = ModelPoints(np.column_stack((x, x * 0, x * 0)).astype(np.float32))
    self.renderer._path_offset_z = 1.2
    # Simple forward-facing camera: depth=x, horizontal=y, vertical=z.
    self.renderer._car_space_transform = np.array([[240, 500, 0], [100, 0, 500], [1, 0, 0]], dtype=np.float32)
    self.renderer._rect = rl.Rectangle(13, 17, 480, 240)
    self.renderer._lead_bar_smoothing = [None, None]
    self.renderer._lead_vehicles = [LeadVehicle(), LeadVehicle()]
    self.x = x

  def project(self, distance=20, lateral=0):
    return self.renderer._project_lead_bar(distance, lateral, self.x)

  def test_perspective_size_and_taper(self):
    near, far = self.project(20), self.project(40)
    self.assertEqual(near.shape, (4, 2))
    self.assertGreater(np.ptp(near[:, 0]), np.ptp(far[:, 0]))
    self.assertGreaterEqual(np.ptp(near[:, 1]) + 1e-4, np.ptp(far[:, 1]))
    bottom = near[np.isclose(near[:, 1], near[:, 1].max())]
    top = near[np.isclose(near[:, 1], near[:, 1].min())]
    self.assertGreater(np.ptp(bottom[:, 0]), np.ptp(top[:, 0]))
    # The former top edge is now the bottom anchor; the footprint recedes.
    self.assertAlmostEqual(float(near[:, 1].max()), 100 + 600 / 20, places=4)
    self.assertLess(float(near[:, 1].min()), float(near[:, 1].max()))

  def test_lateral_position_and_road_slope(self):
    center = self.project()
    left = self.project(lateral=3)
    self.assertLess(left[:, 0].mean(), center[:, 0].mean())
    self.renderer._path.raw_points[:, 2] = self.x * 0.03
    np.testing.assert_allclose(self.project()[:, 1] - center[:, 1], 15, atol=1e-4)
    self.renderer._path.raw_points[:, 1] = self.x ** 2 * 0.002
    curved = self.project()
    self.assertFalse(np.allclose(curved, center))
    self.assertFalse(np.isclose(curved[0, 1], curved[1, 1]))

  def test_bounded_extension_and_unlimited_near_height(self):
    for distance in (6, 10, 20, 40, 70, 100):
      points = self.project(distance)
      anchor = distance
      max_height = 600 / anchor - 600 / (anchor + 6.0)
      self.assertGreaterEqual(float(np.ptp(points[:, 1])), min(8, max_height) - 1e-4)
      x = 600 / (points[:, 1] - 100)
      length = float(np.ptp(x))
      self.assertGreaterEqual(length, 4.7 - 1e-3)
      self.assertLessEqual(length, 6.0 + 1e-3)
      if max_height < 8:
        self.assertAlmostEqual(length, 6.0, places=3)
      self.assertAlmostEqual(float(points[:, 1].max()), 100 + 600 / (distance), places=4)
    # Nearby leads retain their full vehicle length, even above 12 pixels.
    for distance in (6, 10, 12):
      points = self.project(distance)
      x = 600 / (points[:, 1] - 100)
      self.assertAlmostEqual(float(np.ptp(x)), 4.7, places=3)
    self.assertGreater(float(np.ptp(self.project(6)[:, 1])), 12)

  def test_footprint_preserves_road_projection(self):
    # Undo the test camera projection onto its flat road. A true road-space
    # rectangle retains its 1.8 m width at both ends at every distance.
    for distance in (6, 10, 40, 100):
      points = self.project(distance, lateral=2)
      x = 600 / (points[:, 1] - 100)
      y = (points[:, 0] - 240) * x / 500
      self.assertAlmostEqual(float(np.ptp(y)), 1.8, places=4)
      self.assertAlmostEqual(float(x.min()), distance, places=3)
      np.testing.assert_allclose(y, [-2.9, -1.1, -1.1, -2.9], atol=1e-4)
      self.assertAlmostEqual(float(x[0]), float(x[1]), places=4)
      self.assertAlmostEqual(float(x[2]), float(x[3]), places=4)

  def test_invalid_and_behind_camera(self):
    for distance, lateral in ((0, 0), (-5, 0), (101, 0), (float('nan'), 0), (20, float('inf'))):
      self.assertEqual(self.project(distance, lateral).size, 0)
    self.renderer._car_space_transform *= -1
    self.assertEqual(self.project().size, 0)

  def test_stopped_trajectory_still_marks_lead(self):
    self.renderer._path.raw_points = np.zeros((33, 3), dtype=np.float32)
    points = self.renderer._project_lead_bar(6, 0, np.zeros(33))
    self.assertEqual(points.shape, (4, 2))
    self.assertTrue(np.isfinite(points).all())

  def test_backward_path_samples_do_not_hide_lead(self):
    expected = self.project(6)
    self.renderer._path.raw_points[-1, 0] = self.x[-2] - 0.001
    points = self.renderer._project_lead_bar(6, 0, self.renderer._path.raw_points[:, 0])
    np.testing.assert_allclose(points, expected)
    # A stopped trajectory can also retreat after a short forward portion.
    self.renderer._path.raw_points = np.array([[0, 0, 0], [1, 0, 0], [0.999, 0.5, 0.5]], dtype=np.float32)
    points = self.renderer._project_lead_bar(6, 0, self.renderer._path.raw_points[:, 0])
    np.testing.assert_allclose(points, expected)

  def test_close_lead_with_tilted_camera(self):
    # The footprint recedes from a visible anchor with a tilted camera.
    self.renderer._car_space_transform[2] = [1, 0, -1]
    points = self.project(6)
    self.assertEqual(points.shape, (4, 2))
    self.assertTrue(np.isfinite(points).all())
    self.assertGreater(float(np.ptp(points[:, 1])), 0)
    self.assertAlmostEqual(float(points[:, 1].max()), (100 * 6 + 600) / (6 - 1.2), places=3)
    # Do not rescue a lead whose anchor is itself behind the camera.
    self.assertEqual(self.project(1).size, 0)

  def test_smoothing_and_lead_change(self):
    lead = SimpleNamespace(present=True, dRel=20, yRel=0, radar=True, radarTrackId=1)
    radar = SimpleNamespace(leadOne=lead, leadTwo=SimpleNamespace(present=False))
    self.renderer._update_leads(radar, self.x)
    np.testing.assert_allclose(self.renderer._lead_vehicles[0].points, self.project(21.52))
    lead.yRel, lead.dRel = 1, 22
    self.renderer._path.raw_points[:, 1] = self.x * 0.1
    self.renderer._update_leads(radar, self.x)
    state = self.renderer._lead_bar_smoothing[0]
    self.assertGreater(state.lateral_filter.x, 0)
    self.assertLess(state.lateral_filter.x, 1)
    self.assertGreater(state.heading_filter.x, 0)
    self.assertLess(state.heading_filter.x, np.arctan(0.1))
    self.assertEqual(self.renderer._lead_vehicles[0].distance, 22)
    for _ in range(100):
      self.renderer._update_leads(radar, self.x)
    np.testing.assert_allclose(self.renderer._lead_vehicles[0].points, self.project(23.52, 1), atol=1e-3)
    lead.radarTrackId, lead.yRel = 2, -1
    self.renderer._update_leads(radar, self.x)
    np.testing.assert_allclose(self.renderer._lead_vehicles[0].points, self.project(23.52, -1))
    lead.present = False
    self.renderer._update_leads(radar, self.x)
    self.assertIsNone(self.renderer._lead_bar_smoothing[0])
    lead.present, lead.yRel = True, 2
    self.renderer._update_leads(radar, self.x)
    np.testing.assert_allclose(self.renderer._lead_vehicles[0].points, self.project(23.52, 2))

  def test_vision_positions_and_source_switch(self):
    lead = SimpleNamespace(present=True, dRel=20, yRel=3, radar=False, radarTrackId=-1)
    radar = SimpleNamespace(leadOne=lead, leadTwo=SimpleNamespace(present=False))
    self.renderer._update_leads(radar, self.x)
    vision = [SimpleNamespace(prob=0.9, x=[31.52], y=[1.0]),
              SimpleNamespace(prob=0.8, x=[51.52], y=[-2.0])]
    self.renderer._update_leads(radar, self.x, vision)
    np.testing.assert_allclose(self.renderer._lead_vehicles[0].points, self.project(31.52, -1))
    np.testing.assert_allclose(self.renderer._lead_vehicles[1].points, self.project(51.52, 2))
    # Missing vision detections disappear instead of falling back to radar.
    vision[0].prob = 0.1
    vision[1].x = []
    self.renderer._update_leads(radar, self.x, vision)
    for _ in range(100):
      self.renderer._update_leads(radar, self.x, vision)
    self.assertTrue(all(lead.points.size == 0 for lead in self.renderer._lead_vehicles))
    self.renderer._update_leads(radar, self.x)
    np.testing.assert_allclose(self.renderer._lead_vehicles[0].points, self.project(21.52, 3))

  def test_radar_and_vision_share_camera_anchor(self):
    radar = SimpleNamespace(leadOne=SimpleNamespace(present=True, dRel=6, yRel=1, radar=True, radarTrackId=5),
                            leadTwo=SimpleNamespace(present=False))
    self.renderer._update_leads(radar, self.x)
    radar_points = self.renderer._lead_vehicles[0].points.copy()
    vision = [SimpleNamespace(prob=0.9, x=[7.52], y=[-1])]
    self.renderer._update_leads(radar, self.x, vision)
    np.testing.assert_allclose(self.renderer._lead_vehicles[0].points, radar_points)
    self.assertAlmostEqual(float(radar_points[:, 1].max()), 100 + 600 / 7.52, places=4)

  def test_visibility_fades_and_recovers(self):
    lead = SimpleNamespace(present=True, dRel=20, yRel=0, radar=True, radarTrackId=1)
    radar = SimpleNamespace(leadOne=lead, leadTwo=SimpleNamespace(present=False))
    self.renderer._update_leads(radar, self.x)
    first = self.renderer._lead_vehicles[0].visibility.x
    self.assertGreater(first, 0)
    self.assertLess(first, 1)
    self.renderer._update_leads(radar, self.x)
    full = self.renderer._lead_vehicles[0].visibility.x
    self.assertGreater(full, first)
    points = self.renderer._lead_vehicles[0].points.copy()
    lead.present = False
    self.renderer._update_leads(radar, self.x)
    faded = self.renderer._lead_vehicles[0].visibility.x
    self.assertGreater(faded, 0)
    self.assertLess(faded, full)
    np.testing.assert_array_equal(self.renderer._lead_vehicles[0].points, points)
    lead.present = True
    self.renderer._update_leads(radar, self.x)
    self.assertGreater(self.renderer._lead_vehicles[0].visibility.x, faded)
    for _ in range(100):
      self.renderer._update_leads(None, self.x)
    self.assertEqual(self.renderer._lead_vehicles[0].visibility.x, 0)
    self.assertEqual(self.renderer._lead_vehicles[0].points.size, 0)

  def test_uniform_opacity_across_policies(self):
    lead = SimpleNamespace(present=True, dRel=20, yRel=0, radar=True, radarTrackId=1)
    radar = SimpleNamespace(leadOne=lead, leadTwo=SimpleNamespace(present=False))
    vision = [SimpleNamespace(prob=0.9, x=[21.52], y=[0])]
    for source in (None, vision, None):
      self.renderer._update_leads(radar, self.x, source)
      with patch('openpilot.selfdrive.ui.mici.onroad.model_renderer.draw_polygon') as draw:
        self.renderer._draw_lead_indicator()
      self.assertEqual(draw.call_args.args[2].a, round(255 * 0.8 * self.renderer._lead_vehicles[0].visibility.x))

  def test_two_leads_duplicates_and_disappearance(self):
    first = SimpleNamespace(present=True, dRel=20, yRel=0, radar=True, radarTrackId=1)
    second = SimpleNamespace(present=True, dRel=20.5, yRel=0.1, radar=True, radarTrackId=2)
    radar = SimpleNamespace(leadOne=first, leadTwo=second)
    self.renderer._update_leads(radar, self.x)
    self.assertEqual(self.renderer._lead_vehicles[1].points.size, 0)
    second.dRel, second.yRel = 40, -3
    self.renderer._update_leads(radar, self.x)
    self.assertEqual(self.renderer._lead_vehicles[1].points.shape, (4, 2))
    with patch('openpilot.selfdrive.ui.mici.onroad.model_renderer.draw_polygon') as draw:
      self.renderer._draw_lead_indicator()
      self.assertEqual(draw.call_count, 2)
      np.testing.assert_allclose(draw.call_args_list[0].args[1], self.renderer._lead_vehicles[1].points + [13, 17])
      self.assertEqual(draw.call_args_list[0].args[2].a, round(255 * 0.8 * self.renderer._lead_vehicles[1].visibility.x))
      self.assertEqual(draw.call_args_list[1].args[2].a, round(255 * 0.8 * self.renderer._lead_vehicles[0].visibility.x))
    first.present = second.present = False
    self.renderer._update_leads(radar, self.x)
    for _ in range(100):
      self.renderer._update_leads(radar, self.x)
    self.assertTrue(all(lead.points.size == 0 for lead in self.renderer._lead_vehicles))


if __name__ == '__main__':
  unittest.main()
