#!/usr/bin/env python3
import os
import time
import unittest

import cereal.messaging as messaging
from cereal import log
from common.gpio import gpio_set, gpio_init
from common.params import Params
from panda import Panda, PandaDFU, PandaProtocolMismatch
from selfdrive.test.helpers import phone_only
from selfdrive.manager.process_config import managed_processes
from system.hardware import HARDWARE
from system.hardware.tici.pins import GPIO

HERE = os.path.dirname(os.path.realpath(__file__))


class TestPandad(unittest.TestCase):

  def setUp(self):
    self.params = Params()
    self.start_log_state = self.params.get("PandaLogState")

  def tearDown(self):
    managed_processes['pandad'].stop()

  def _wait_for_boardd(self, timeout=30):
    sm = messaging.SubMaster(['peripheralState'])
    for _ in range(timeout*10):
      sm.update(100)
      if sm['peripheralState'].pandaType != log.PandaState.PandaType.unknown:
        break

    if sm['peripheralState'].pandaType == log.PandaState.PandaType.unknown:
      raise Exception("boardd failed to start")

    # simple check that we did something with the panda logs
    cur_log_state = self.params.get("PandaLogState")
    assert cur_log_state != self.start_log_state

  def _go_to_dfu(self):
    HARDWARE.recover_internal_panda()
    assert Panda.wait_for_dfu(None, 10)

  def _flash_and_test(self, fn, expect_mismatch=False):
    self._go_to_dfu()
    pd = PandaDFU(None)
    if fn is None:
      fn = os.path.join(HERE, pd.get_mcu_type().config.bootstub_fn)
    with open(fn, "rb") as f:
      pd.program_bootstub(f.read())
    pd.reset()
    HARDWARE.reset_internal_panda()

    assert Panda.wait_for_panda(None, 10)
    if expect_mismatch:
      with self.assertRaises(PandaProtocolMismatch):
        Panda()
    else:
      with Panda() as p:
        assert p.bootstub

    managed_processes['pandad'].start()
    self._wait_for_boardd(45)

  @phone_only
  def test_in_dfu(self):
    HARDWARE.recover_internal_panda()
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

  @phone_only
  def test_best_case_startup_time(self):
    # run once so we're setup
    managed_processes['pandad'].start()
    self._wait_for_boardd()
    managed_processes['pandad'].stop()

    # should be fast this time
    managed_processes['pandad'].start()
    self._wait_for_boardd(8)

  @phone_only
  def test_protocol_version_check(self):
    if HARDWARE.get_device_type() == 'tici':
      self.skipTest("")

    # flash old fw
    fn = os.path.join(HERE, "bootstub.panda_h7_spiv0.bin")
    self._flash_and_test(fn, expect_mismatch=True)

  @phone_only
  def test_release_to_devel_bootstub(self):
    self._flash_and_test(None)

if __name__ == "__main__":
  unittest.main()
