import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyray as rl

from openpilot.selfdrive.ui.mici.onroad.model_renderer import ModelRenderer, ModelPoints


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
    self.x = x

  def project(self, distance=20, lateral=0):
    return self.renderer._project_lead_bar(distance, lateral, self.x)

  def test_perspective_size_and_taper(self):
    near, far = self.project(20), self.project(40)
    self.assertEqual(near.shape, (4, 2))
    self.assertGreater(np.ptp(near[:, 0]), np.ptp(far[:, 0]))
    self.assertGreater(np.ptp(near[:, 1]), np.ptp(far[:, 1]))
    self.assertGreater(abs(near[2, 0] - near[3, 0]), abs(near[1, 0] - near[0, 0]))
    # All corners lie on the road on our side of the vehicle's rear bumper.
    self.assertTrue(np.all(near[:, 1] > 100 + 500 * 1.2 / 20))

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

  def test_screen_height_limits(self):
    for distance in (6, 10, 20, 40, 70, 100):
      points = self.project(distance)
      height = np.ptp(points[:, 1])
      self.assertGreaterEqual(height, 6 - 1e-4)
      self.assertLessEqual(height, 12 + 1e-4)
      # The far edge stays anchored to its calibrated road position.
      self.assertAlmostEqual(float(points[:, 1].min()), 100 + 600 / (distance - 0.2), places=4)
    self.assertAlmostEqual(float(np.ptp(self.project(10)[:, 1])), 12, places=4)
    self.assertAlmostEqual(float(np.ptp(self.project(100)[:, 1])), 6, places=4)

  def test_height_adjustment_preserves_road_projection(self):
    # Undo the test camera projection onto its flat road. A true road-space
    # rectangle retains its 1.8 m width at both ends, even at the height limits.
    for distance in (6, 10, 40, 100):
      points = self.project(distance, lateral=2)
      x = 600 / (points[:, 1] - 100)
      y = (points[:, 0] - 240) * x / 500
      self.assertAlmostEqual(float(y[1] - y[0]), 1.8, places=4)
      self.assertAlmostEqual(float(y[2] - y[3]), 1.8, places=4)
      np.testing.assert_allclose(y[[0, 1]], y[[3, 2]], atol=1e-4)
      self.assertAlmostEqual(float(x[0]), distance - 0.2, places=3)
      self.assertLess(x[2], x[0])

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

  def test_close_lead_shortens_before_camera_plane(self):
    # The lead anchor is in front of the camera, but the initial long bar's
    # near edge is behind it. The shorter projected bar must remain visible.
    self.renderer._car_space_transform[2] = [1, 0, -1]
    points = self.project(6)
    self.assertEqual(points.shape, (4, 2))
    self.assertTrue(np.isfinite(points).all())
    self.assertAlmostEqual(float(np.ptp(points[:, 1])), 12, places=3)
    self.assertAlmostEqual(float(points[:, 1].min()), (100 * 5.8 + 600) / (5.8 - 1.2), places=3)
    # Do not rescue a lead whose anchor is itself behind the camera.
    self.assertEqual(self.project(1).size, 0)

  def test_smoothing_and_lead_change(self):
    lead = SimpleNamespace(present=True, dRel=20, yRel=0, radar=True, radarTrackId=1)
    radar = SimpleNamespace(leadOne=lead, leadTwo=SimpleNamespace(present=False))
    self.renderer._update_leads(radar, self.x)
    np.testing.assert_allclose(self.renderer._lead_vehicles[0].points, self.project())
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
    np.testing.assert_allclose(self.renderer._lead_vehicles[0].points, self.project(22, 1), atol=1e-3)
    lead.radarTrackId, lead.yRel = 2, -1
    self.renderer._update_leads(radar, self.x)
    np.testing.assert_allclose(self.renderer._lead_vehicles[0].points, self.project(22, -1))
    lead.present = False
    self.renderer._update_leads(radar, self.x)
    self.assertIsNone(self.renderer._lead_bar_smoothing[0])
    lead.present, lead.yRel = True, 2
    self.renderer._update_leads(radar, self.x)
    np.testing.assert_allclose(self.renderer._lead_vehicles[0].points, self.project(22, 2))

  def test_vision_positions_and_source_switch(self):
    lead = SimpleNamespace(present=True, dRel=20, yRel=3, radar=False, radarTrackId=-1)
    radar = SimpleNamespace(leadOne=lead, leadTwo=SimpleNamespace(present=False))
    self.renderer._update_leads(radar, self.x)
    vision = [SimpleNamespace(prob=0.9, x=[31.52], y=[1.0]),
              SimpleNamespace(prob=0.8, x=[51.52], y=[-2.0])]
    self.renderer._update_leads(radar, self.x, vision)
    np.testing.assert_allclose(self.renderer._lead_vehicles[0].points, self.project(30, -1))
    np.testing.assert_allclose(self.renderer._lead_vehicles[1].points, self.project(50, 2))
    # Missing vision detections disappear instead of falling back to radar.
    vision[0].prob = 0.1
    vision[1].x = []
    self.renderer._update_leads(radar, self.x, vision)
    self.assertTrue(all(lead.points.size == 0 for lead in self.renderer._lead_vehicles))
    self.renderer._update_leads(radar, self.x)
    np.testing.assert_allclose(self.renderer._lead_vehicles[0].points, self.project(20, 3))

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
      self.assertEqual(draw.call_args_list[0].args[2].a, round(255 * 0.4))
      self.assertEqual(draw.call_args_list[1].args[2].a, round(255 * 0.8))
    first.present = second.present = False
    self.renderer._update_leads(radar, self.x)
    self.assertTrue(all(lead.points.size == 0 for lead in self.renderer._lead_vehicles))


if __name__ == '__main__':
  unittest.main()
