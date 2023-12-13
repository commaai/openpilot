#!/usr/bin/env python3
import unittest

from cereal import car
from cereal import messaging
from cereal.messaging import SubMaster, PubMaster
from openpilot.selfdrive.ui.soundd import CONTROLS_TIMEOUT, check_controls_timeout_alert

import time

AudibleAlert = car.CarControl.HUDControl.AudibleAlert


class TestSoundd(unittest.TestCase):
  def test_check_controls_timeout_alert(self):
    sm = SubMaster(['controlsState'])
    pm = PubMaster(['controlsState'])

    for _ in range(100):
      cs = messaging.new_message('controlsState')
      cs.controlsState.enabled = True

      pm.send("controlsState", cs)

      time.sleep(0.01)

      sm.update(0)

      self.assertFalse(check_controls_timeout_alert(sm))

    for _ in range(CONTROLS_TIMEOUT * 110):
      sm.update(0)
      time.sleep(0.01)

    self.assertTrue(check_controls_timeout_alert(sm))

  # TODO: add test with micd for checking that soundd actually outputs sounds


if __name__ == "__main__":
  unittest.main()
