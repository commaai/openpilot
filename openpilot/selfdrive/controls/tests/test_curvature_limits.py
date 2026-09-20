import unittest

from openpilot.selfdrive.controls.lib.drive_helpers import clip_curvature


class TestCurvatureLimits(unittest.TestCase):
  def test_low_speed_curvature_can_exceed_old_cap(self):
    for sign in (-1, 1):
      curvature, limited = clip_curvature(1.0, sign * 0.3, sign * 0.3)
      self.assertAlmostEqual(curvature, sign * 0.3)
      self.assertFalse(limited)

  def test_acceleration_no_longer_caps_curvature(self):
    for sign in (-1, 1):
      curvature, limited = clip_curvature(5.0, sign * 0.3, sign * 0.3)
      self.assertAlmostEqual(curvature, sign * 0.3)
      self.assertFalse(limited)

  def test_jerk_limit_is_three_times_original(self):
    for speed in (0.0, 1.0, 5.0, 20.0):
      for sign in (-1, 1):
        curvature, limited = clip_curvature(speed, 0.0, sign * 0.3)
        self.assertAlmostEqual(curvature, sign * 0.15 / max(speed, 1.0) ** 2)
        self.assertTrue(limited)

  def test_sustained_request_ramps_past_old_acceleration_cap(self):
    curvature = 0.0
    for _ in range(200):
      curvature, _ = clip_curvature(5.0, curvature, 0.3)
    self.assertAlmostEqual(curvature, 0.3)
