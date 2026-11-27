#!/usr/bin/env python3

import os
import unittest

from opendbc.car.structs import car
from openpilot.common.test import OpenpilotTestCase
from openpilot.common.params import Params
import openpilot.system.manager.manager as manager
from openpilot.system.manager.process_config import managed_processes, procs, joystick, not_joystick
from openpilot.common.hardware import HARDWARE

os.environ['FAKEUPLOAD'] = "1"


class TestManager(OpenpilotTestCase):
  def setup_method(self):
    HARDWARE.set_power_save(False)

    # ensure clean CarParams
    params = Params()
    params.clear_all()

  def teardown_method(self):
    manager.manager_cleanup()

  def test_duplicate_procs(self):
    assert len(procs) == len(managed_processes), "Duplicate process names"

  def test_joystick_param_latched_offroad(self):
    params = Params()
    CP = car.CarParams.new_message()

    params.put_bool("JoystickDebugMode", False, block=True)
    assert not joystick(False, params, CP)
    assert not joystick(True, params, CP)
    assert not_joystick(True, params, CP)

    # Toggling onroad must not swap controlsd for joystickd until next offroad cycle
    params.put_bool("JoystickDebugMode", True, block=True)
    assert not joystick(True, params, CP)
    assert not_joystick(True, params, CP)
    joystick(False, params, CP)
    assert joystick(True, params, CP)
    assert not not_joystick(True, params, CP)

  def test_set_params_with_default_value(self):
    params = Params()
    params.clear_all()

    os.environ['PREPAREONLY'] = '1'
    manager.main()
    for k in params.all_keys():
      default_value = params.get_default_value(k)
      if default_value is not None:
        assert params.get(k) == default_value
    assert params.get("OpenpilotEnabledToggle")
    assert params.get("RouteCount") == 0


if __name__ == "__main__":
  unittest.main()
