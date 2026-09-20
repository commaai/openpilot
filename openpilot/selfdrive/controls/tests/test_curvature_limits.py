import unittest

from openpilot.selfdrive.controls.lib.drive_helpers import clip_curvature


class TestCurvatureLimits(unittest.TestCase):
  def test_low_speed_curvature_can_exceed_old_cap(self):
    for sign in (-1, 1):
      curvature, limited = clip_curvature(1.0, sign * 0.3, sign * 0.3, 0.0)
      self.assertAlmostEqual(curvature, sign * 0.3)
      self.assertFalse(limited)

  def test_acceleration_limit_remains(self):
    for sign in (-1, 1):
      curvature, limited = clip_curvature(5.0, sign * 0.3, sign * 0.3, 0.0)
      self.assertAlmostEqual(curvature, sign * 3.0 / 25)
      self.assertTrue(limited)

  def test_jerk_limit_remains(self):
    curvature, _ = clip_curvature(1.0, 0.0, 0.3, 0.0)
    self.assertAlmostEqual(curvature, 0.05)
