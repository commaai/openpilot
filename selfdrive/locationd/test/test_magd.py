#!/usr/bin/env python3
import math
import unittest

from openpilot.selfdrive.test.openpilotci import get_url
from openpilot.tools.lib.logreader import LogReader
from openpilot.selfdrive.locationd.magd import MagCalibrator, CalibrationParams


TEST_ROUTE, TEST_SEG_NUM = "ff2bd20623fcaeaa|2023-09-05--10-14-54", 4


class TestMagd(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    lr = list(LogReader(get_url(TEST_ROUTE, TEST_SEG_NUM)))
    CP = [msg for msg in lr if msg.which() == 'carParams'][0].carParams.as_builder()
    cls.mag_calibrator = MagCalibrator(CP)
    cls.calibration_params = CalibrationParams()

  def test_get_ellipsoid_center_circle_shape(self):
    """Test center when the ellipse is a circle at center at (0, 0)"""
    self.assertEqual(self.mag_calibrator.get_ellipsoid_center([1, 1, 0, 0, 0, -2]), (0, 0))

  def test_get_ellipsoid_center_unbounded_shape(self):
    """Test center when the ellipse is a unbounded (division by zero)"""
    self.assertRaises(AssertionError, self.mag_calibrator.get_ellipsoid_center, [1, 1, 2, 0, 0, 0])

  def test_get_ellipsoid_rotation_circle_shape(self):
    """Test rotation when the ellipse is a circle at center at (0, 0), with no rotation"""
    self.assertEqual(self.mag_calibrator.get_ellipsoid_rotation([4, 4, 0, 0, 0, -1]), ((0.5, 0.5), 0))

  def test_get_ellipsoid_rotation_rotated_ellipse(self):
    """Test rotation when the ellipse with -45 degree rotation"""
    _, rotation_angle = self.mag_calibrator.get_ellipsoid_rotation([1, 1, 1, 0, 0, -2])
    self.assertAlmostEqual(rotation_angle, math.pi * 0.75, delta=1e-5)

  def test_get_ellipsoid_rotation_no_shape(self):
    """Test rotation when the ellipse is not defined"""
    self.assertRaises(AssertionError, self.mag_calibrator.get_ellipsoid_center, [0, 0, 0, 0, 0, 0])

  def test_get_calibrated_bearing_zero(self):
    """Test bearing given calibration Params"""
    self.assertEqual(self.mag_calibrator.get_calibrated_bearing(0, 0, self.calibration_params), 0)

  def test_get_calibrated_bearing_45deg(self):
    """Test bearing given calibration Params"""
    bearing = self.mag_calibrator.get_calibrated_bearing(1, 1, self.calibration_params)
    self.assertAlmostEqual(bearing, math.pi / 4, delta=1e-5)

  def test_reset_angle_range(self):
    """Test reset angle range, from -pi to pi"""
    self.assertAlmostEqual(self.mag_calibrator.reset_angle_range(0), 0, delta=1e-5)
    self.assertAlmostEqual(self.mag_calibrator.reset_angle_range(math.pi), -math.pi, delta=1e-5)
    self.assertAlmostEqual(self.mag_calibrator.reset_angle_range(-math.pi), -math.pi, delta=1e-5)
    self.assertAlmostEqual(self.mag_calibrator.reset_angle_range(2 * math.pi), 0, delta=1e-5)
    self.assertAlmostEqual(self.mag_calibrator.reset_angle_range(-2 * math.pi), 0, delta=1e-5)
    self.assertAlmostEqual(self.mag_calibrator.reset_angle_range(1.25 * math.pi), -0.75 * math.pi, delta=1e-5)


if __name__ == "__main__":
  unittest.main()
