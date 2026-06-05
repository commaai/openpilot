import os
import pytest
import time

import cereal.messaging as messaging
from cereal import log
from openpilot.common.gpio import gpio_set, gpio_init
from panda import Panda, PandaDFU
from openpilot.system.manager.process_config import managed_processes
from openpilot.system.hardware import HARDWARE
from openpilot.system.hardware.tici.pins import GPIO

HERE = os.path.dirname(os.path.realpath(__file__))


@pytest.mark.tici
class TestPandad:
  def teardown_method(self):
    managed_processes['pandad'].stop()

  def _run_test(self, timeout=30) -> float:
    st = time.monotonic()
    sm = messaging.SubMaster(['pandaStates'])

    managed_processes['pandad'].start()
    while (time.monotonic() - st) < timeout:
      sm.update(10)
      if len(sm['pandaStates']) and sm['pandaStates'][0].pandaType != log.PandaState.PandaType.unknown:
        break
    dt = time.monotonic() - st
    managed_processes['pandad'].stop()

    if len(sm['pandaStates']) == 0 or sm['pandaStates'][0].pandaType == log.PandaState.PandaType.unknown:
      raise Exception("pandad failed to start")

    return dt

  def _go_to_dfu(self):
    HARDWARE.recover_internal_panda()
    assert Panda.wait_for_dfu(None, 10)

  def _flash_bootstub(self, fn):
    self._go_to_dfu()
    pd = PandaDFU(None)
    if fn is None:
      fn = os.path.join(HERE, pd.get_mcu_type().config.bootstub_fn)
    with open(fn, "rb") as f:
      pd.program_bootstub(f.read())
    HARDWARE.reset_internal_panda()

  def test_in_dfu(self):
    HARDWARE.recover_internal_panda()
    self._run_test()

  def test_in_bootstub(self):
    with Panda() as p:
      p.reset(enter_bootstub=True)
      assert p.bootstub
    self._run_test()

  def test_in_reset(self):
    gpio_init(GPIO.STM_RST_N, True)
    gpio_set(GPIO.STM_RST_N, 1)
    assert not Panda.list()
    self._run_test()

  def test_release_to_devel_bootstub(self):
    st = time.monotonic()
    self._flash_bootstub(None)
    print("flash done", time.monotonic() - st)
    self._run_test()

  def test_recover_from_bad_bootstub(self):
    self._go_to_dfu()
    with PandaDFU(None) as pd:
      pd._handle.program(pd.get_mcu_type().config.bootstub_address, b"\x00"*100)
    HARDWARE.reset_internal_panda()
    assert not Panda.list()
    assert not PandaDFU.list()

    self._run_test()
