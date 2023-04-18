#!/usr/bin/env python3
import time
import unittest

import cereal.messaging as messaging
from panda import Panda
from common.gpio import gpio_set, gpio_init
from selfdrive.test.helpers import phone_only
from selfdrive.manager.process_config import managed_processes
from system.hardware import HARDWARE
from system.hardware.tici.pins import GPIO


class TestPandad(unittest.TestCase):

  def tearDown(self):
    managed_processes['pandad'].stop()

  def _wait_for_boardd(self, timeout=30):
    sm = messaging.SubMaster(['peripheralState'])
    for _ in range(timeout):
      sm.update(1000)
      if sm.updated['peripheralState']:
        break

    if not sm.updated['peripheralState']:
      raise Exception("boardd failed to start")

  @phone_only
  def test_in_dfu(self):
    HARDWARE.recover_internal_panda()
    time.sleep(1)

    managed_processes['pandad'].start()
    self._wait_for_boardd(60)

  @phone_only
  def test_in_bootstub(self):
    with Panda() as p:
      p.reset(enter_bootstub=True)
      assert p.bootstub
    managed_processes['pandad'].start()
    self._wait_for_boardd()

  @phone_only
  def test_internal_panda_reset(self):
    gpio_init(GPIO.STM_RST_N, True)
    gpio_set(GPIO.STM_RST_N, 1)
    time.sleep(0.5)
    assert all(not Panda(s).is_internal() for s in Panda.list())

    managed_processes['pandad'].start()
    self._wait_for_boardd()

    assert any(Panda(s).is_internal() for s in Panda.list())

  #def test_out_of_date_fw(self):
  #  pass


if __name__ == "__main__":
  unittest.main()
