import pytest
import time
import random
import subprocess

from panda import Panda
from openpilot.system.hardware import TICI, HARDWARE
from openpilot.system.hardware.tici.hardware import Tici
from openpilot.system.hardware.tici.amplifier import Amplifier


class TestAmplifier:

  @classmethod
  def setup_class(cls):
    if not TICI:
      pytest.skip()

  def setup_method(self):
    # clear dmesg
    subprocess.check_call("sudo dmesg -C", shell=True)

    HARDWARE.reset_internal_panda()
    Panda.wait_for_panda(None, 30)
    self.panda = Panda()

  def teardown_method(self):
    HARDWARE.reset_internal_panda()

  def _check_for_i2c_errors(self, expected):
    dmesg = subprocess.check_output("dmesg", shell=True, encoding='utf8')
    i2c_lines = [l for l in dmesg.strip().splitlines() if 'i2c_geni a88000.i2c' in l]
    i2c_str = '\n'.join(i2c_lines)

    if not expected:
      return len(i2c_lines) == 0
    else:
      return "i2c error :-107" in i2c_str or "Bus arbitration lost" in i2c_str

  def test_init(self):
    amp = Amplifier(debug=True)
    r = amp.initialize_configuration(Tici().get_device_type())
    assert r
    assert self._check_for_i2c_errors(False)

  def test_shutdown(self):
    amp = Amplifier(debug=True)
    for _ in range(10):
      r = amp.set_global_shutdown(True)
      r = amp.set_global_shutdown(False)
      # amp config should be successful, with no i2c errors
      assert r
      assert self._check_for_i2c_errors(False)

  def test_init_while_siren_play(self):
    for _ in range(10):
      self.panda.set_siren(False)
      time.sleep(0.1)

      self.panda.set_siren(True)
      time.sleep(random.randint(0, 5))

      amp = Amplifier(debug=True)
      r = amp.initialize_configuration(Tici().get_device_type())
      assert r

      if self._check_for_i2c_errors(True):
        break
    else:
      pytest.fail("didn't hit any i2c errors")
