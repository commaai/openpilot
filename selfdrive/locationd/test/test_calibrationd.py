#!/usr/bin/env python3
import json
import random
import unittest

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


if __name__ == "__main__":
  unittest.main()
