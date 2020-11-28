#!/usr/bin/env python3
import json
import random
import unittest

import cereal.messaging as messaging
from common.params import Params
from selfdrive.locationd.calibrationd import Calibrator


class TestCalibrationd(unittest.TestCase):

  def test_read_saved_params_json(self):
    r = [random.random() for _ in range(3)]
    b = random.randint(1, 10)
    cal_params = {"calib_radians": r,
                  "valid_blocks": b}
    Params().put("CalibrationParams", json.dumps(cal_params).encode('utf8'))
    c = Calibrator(param_put=True)

    self.assertEqual(r, c.rpy)
    self.assertEqual(b, c.valid_blocks)

  def test_read_saved_params(self):
    msg = messaging.new_message('liveCalibration')
    msg.liveCalibration.validBlocks = random.randint(1, 10)
    msg.liveCalibration.rpyCalib = [random.random() for _ in range(3)]
    Params().put("CalibrationParams", msg.to_bytes())
    c = Calibrator(param_put=True)

    self.assertEqual(list(msg.liveCalibration.rpyCalib), c.rpy)
    self.assertEqual(msg.liveCalibration.validBlocks, c.valid_blocks)


if __name__ == "__main__":
  unittest.main()
