#!/usr/bin/env python3
import random
import unittest

import numpy as np

import cereal.messaging as messaging
from cereal import log
from openpilot.common.params import Params
from openpilot.selfdrive.locationd.calibrationd import Calibrator, INPUTS_NEEDED, INPUTS_WANTED, BLOCK_SIZE, MIN_SPEED_FILTER, \
                                                         MAX_YAW_RATE_FILTER, SMOOTH_CYCLES, HEIGHT_INIT


class TestCalibrationd(unittest.TestCase):

  def test_read_saved_params(self):
    msg = messaging.new_message('liveCalibration')
    msg.liveCalibration.validBlocks = random.randint(1, 10)
    msg.liveCalibration.rpyCalib = [random.random() for _ in range(3)]
    msg.liveCalibration.height = [random.random() for _ in range(1)]
    Params().put("CalibrationParams", msg.to_bytes())
    c = Calibrator(param_put=True)

    np.testing.assert_allclose(msg.liveCalibration.rpyCalib, c.rpy)
    np.testing.assert_allclose(msg.liveCalibration.height, c.height)
    self.assertEqual(msg.liveCalibration.validBlocks, c.valid_blocks)


  def test_calibration_basics(self):
    c = Calibrator(param_put=False)
    for _ in range(BLOCK_SIZE * INPUTS_WANTED):
      c.handle_v_ego(MIN_SPEED_FILTER + 1)
      c.handle_cam_odom([MIN_SPEED_FILTER + 1, 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [1e-3, 1e-3, 1e-3],
                         [0.0, 0.0, HEIGHT_INIT.item()],
                         [1e-3, 1e-3, 1e-3])
    self.assertEqual(c.valid_blocks, INPUTS_WANTED)
    np.testing.assert_allclose(c.rpy, np.zeros(3))
    np.testing.assert_allclose(c.height, HEIGHT_INIT)
    c.reset()


  def test_calibration_low_speed_reject(self):
    c = Calibrator(param_put=False)
    for _ in range(BLOCK_SIZE * INPUTS_WANTED):
      c.handle_v_ego(MIN_SPEED_FILTER - 1)
      c.handle_cam_odom([MIN_SPEED_FILTER + 1, 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [1e-3, 1e-3, 1e-3],
                         [0.0, 0.0, HEIGHT_INIT.item()],
                         [1e-3, 1e-3, 1e-3])
    for _ in range(BLOCK_SIZE * INPUTS_WANTED):
      c.handle_v_ego(MIN_SPEED_FILTER + 1)
      c.handle_cam_odom([MIN_SPEED_FILTER - 1, 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [1e-3, 1e-3, 1e-3],
                         [0.0, 0.0, HEIGHT_INIT.item()],
                         [1e-3, 1e-3, 1e-3])
    self.assertEqual(c.valid_blocks, 0)
    np.testing.assert_allclose(c.rpy, np.zeros(3))
    np.testing.assert_allclose(c.height, HEIGHT_INIT)


  def test_calibration_yaw_rate_reject(self):
    c = Calibrator(param_put=False)
    for _ in range(BLOCK_SIZE * INPUTS_WANTED):
      c.handle_v_ego(MIN_SPEED_FILTER + 1)
      c.handle_cam_odom([MIN_SPEED_FILTER + 1, 0.0, 0.0],
                         [0.0, 0.0, MAX_YAW_RATE_FILTER ],
                         [0.0, 0.0, 0.0],
                         [1e-3, 1e-3, 1e-3],
                         [0.0, 0.0, HEIGHT_INIT.item()],
                         [1e-3, 1e-3, 1e-3])
    self.assertEqual(c.valid_blocks, 0)
    np.testing.assert_allclose(c.rpy, np.zeros(3))
    np.testing.assert_allclose(c.height, HEIGHT_INIT)


  def test_calibration_speed_std_reject(self):
    c = Calibrator(param_put=False)
    for _ in range(BLOCK_SIZE * INPUTS_WANTED):
      c.handle_v_ego(MIN_SPEED_FILTER + 1)
      c.handle_cam_odom([MIN_SPEED_FILTER + 1, 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [1e3, 1e3, 1e3],
                         [0.0, 0.0, HEIGHT_INIT.item()],
                         [1e-3, 1e-3, 1e-3])
    self.assertEqual(c.valid_blocks, INPUTS_NEEDED)
    np.testing.assert_allclose(c.rpy, np.zeros(3))


  def test_calibration_speed_std_height_reject(self):
    c = Calibrator(param_put=False)
    for _ in range(BLOCK_SIZE * INPUTS_WANTED):
      c.handle_v_ego(MIN_SPEED_FILTER + 1)
      c.handle_cam_odom([MIN_SPEED_FILTER + 1, 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [1e-3, 1e-3, 1e-3],
                         [0.0, 0.0, HEIGHT_INIT.item()],
                         [1e3, 1e3, 1e3])
    self.assertEqual(c.valid_blocks, INPUTS_NEEDED)
    np.testing.assert_allclose(c.rpy, np.zeros(3))


  def test_calibration_auto_reset(self):
    c = Calibrator(param_put=False)
    for _ in range(BLOCK_SIZE * INPUTS_WANTED):
      c.handle_v_ego(MIN_SPEED_FILTER + 1)
      c.handle_cam_odom([MIN_SPEED_FILTER + 1, 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [1e-3, 1e-3, 1e-3],
                         [0.0, 0.0, HEIGHT_INIT.item()],
                         [1e-3, 1e-3, 1e-3])
    self.assertEqual(c.valid_blocks, INPUTS_WANTED)
    np.testing.assert_allclose(c.rpy, [0.0, 0.0, 0.0])
    old_rpy_weight_prev = 0.0
    for _ in range(BLOCK_SIZE + 10):
      self.assertLess(old_rpy_weight_prev - c.old_rpy_weight, 1/SMOOTH_CYCLES + 1e-3)
      old_rpy_weight_prev = c.old_rpy_weight
      c.handle_v_ego(MIN_SPEED_FILTER + 1)
      c.handle_cam_odom([MIN_SPEED_FILTER + 1, -0.05 * MIN_SPEED_FILTER, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [1e-3, 1e-3, 1e-3],
                         [0.0, 0.0, HEIGHT_INIT.item()],
                         [1e-3, 1e-3, 1e-3])
    self.assertEqual(c.valid_blocks, 1)
    self.assertEqual(c.cal_status, log.LiveCalibrationData.Status.recalibrating)
    np.testing.assert_allclose(c.rpy, [0.0, 0.0, -0.05], atol=1e-2)

if __name__ == "__main__":
  unittest.main()
