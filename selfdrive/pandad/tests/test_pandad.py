import os
import pytest
import time

import cereal.messaging as messaging
from cereal import log
from openpilot.common.gpio import gpio_set, gpio_init
from panda import Panda, PandaDFU, PandaProtocolMismatch
from openpilot.common.retry import retry
from openpilot.system.manager.process_config import managed_processes
from openpilot.system.hardware import HARDWARE
from openpilot.system.hardware.tici.pins import GPIO

HERE = os.path.dirname(os.path.realpath(__file__))


@pytest.mark.tici
class TestPandad:

  def setup_method(self):
    # ensure panda is up
    if len(Panda.list()) == 0:
      self._run_test(60)

    self.spi = HARDWARE.get_device_type() != 'tici'

  def teardown_method(self):
    managed_processes['pandad'].stop()

  def _run_test(self, timeout=30) -> float:
    st = time.monotonic()
    sm = messaging.SubMaster(['pandaStates'])

    managed_processes['pandad'].start()
    while (time.monotonic() - st) < timeout:
      sm.update(100)
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

  def _assert_no_panda(self):
    assert not Panda.wait_for_dfu(None, 3)
    assert not Panda.wait_for_panda(None, 3)

  @retry(attempts=3)
  def _flash_bootstub_and_test(self, fn, expect_mismatch=False):
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
      with pytest.raises(PandaProtocolMismatch):
        Panda()
    else:
      with Panda() as p:
        assert p.bootstub

    self._run_test(45)

  def test_in_dfu(self):
    HARDWARE.recover_internal_panda()
    self._run_test(60)

  def test_in_bootstub(self):
    with Panda() as p:
      p.reset(enter_bootstub=True)
      assert p.bootstub
    self._run_test()

  def test_internal_panda_reset(self):
    gpio_init(GPIO.STM_RST_N, True)
    gpio_set(GPIO.STM_RST_N, 1)
    time.sleep(0.5)
    assert all(not Panda(s).is_internal() for s in Panda.list())
    self._run_test()

    assert any(Panda(s).is_internal() for s in Panda.list())

  def test_best_case_startup_time(self):
    # run once so we're up to date
    self._run_test(60)

    ts = []
    for _ in range(10):
      # should be nearly instant this time
      dt = self._run_test(5)
      ts.append(dt)

    # 5s for USB (due to enumeration)
    # - 0.2s pandad -> pandad
    # - plus some buffer
    print("startup times", ts, sum(ts) / len(ts))
    assert 0.1 < (sum(ts)/len(ts)) < (0.7 if self.spi else 5.0)

  def test_protocol_version_check(self):
    if not self.spi:
      pytest.skip("SPI test")
    # flash old fw
    fn = os.path.join(HERE, "bootstub.panda_h7_spiv0.bin")
    self._flash_bootstub_and_test(fn, expect_mismatch=True)

  def test_release_to_devel_bootstub(self):
    self._flash_bootstub_and_test(None)

  def test_recover_from_bad_bootstub(self):
    self._go_to_dfu()
    with PandaDFU(None) as pd:
      pd.program_bootstub(b"\x00"*1024)
      pd.reset()
    HARDWARE.reset_internal_panda()
    self._assert_no_panda()

    self._run_test(60)
