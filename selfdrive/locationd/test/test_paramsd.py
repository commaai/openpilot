#!/usr/bin/env python3
import json
import random
import unittest

import cereal.messaging as messaging
from cereal import car
from common.params import Params


class TestPramsd(unittest.TestCase):

  def test_read_saved_params(self):
    cp = car.CarParams.new_message()
    cp.carFingerprint = 'ABC'

    msg = messaging.new_message('liveParameters')
    liveParameters = msg.liveParameters
    liveParameters.steerRatio = random.uniform(0.1, 0.9)
    liveParameters.stiffnessFactor = random.uniform(0.1, 0.9)
    liveParameters.angleOffsetAverageDeg = random.uniform(0.1, 0.9)

    Params().put("LiveParameters", json.dumps(
        {'carFingerprint': cp.carFingerprint, 'parameters': msg.to_dict()}))

    from selfdrive.locationd.paramsd import load_params
    params = load_params(cp, 0, 1)

    self.assertEqual(params.steerRatio, liveParameters.steerRatio)
    self.assertEqual(params.stiffnessFactor, liveParameters.stiffnessFactor)
    self.assertEqual(params.angleOffsetAverageDeg, liveParameters.angleOffsetAverageDeg)

    Params().delete('LiveParameters')


if __name__ == "__main__":
  unittest.main()
