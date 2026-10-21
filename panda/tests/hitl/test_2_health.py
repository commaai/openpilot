import time

from opendbc.car.hyundai.values import HyundaiSafetyFlags
from opendbc.car.structs import CarParams
from panda.tests.hitl.base import PandaTestCase, panda_test
from panda import Panda


class TestHealth(PandaTestCase):
  def test_voltage(self):
    p = self.p
    for _ in range(10):
      voltage = p.health()['voltage']
      assert ((voltage > 11000) and (voltage < 13000))
      time.sleep(0.1)

  def test_hw_type(self):
    """
      hw type should be same in bootstub as application
    """
    p = self.p

    hw_type = p.get_type()

    app_uid =  p.get_uid()
    usb_serial = p.get_usb_serial()
    assert app_uid == usb_serial

    p.reset(enter_bootstub=True, reconnect=True)
    p.close()
    time.sleep(3)
    with Panda(p.get_usb_serial()) as pp:
      assert pp.bootstub
      assert pp.get_type() == hw_type, "Bootstub and app hw type mismatch"
      assert pp.get_uid() == app_uid

  @panda_test(needs_jungle=True)
  def test_heartbeat(self):
    p = self.p
    panda_jungle = self.panda_jungle
    panda_jungle.set_ignition(True)
    # TODO: add more cases here once the tests aren't super slow
    p.set_safety_mode(mode=CarParams.SafetyModel.hyundai, param=HyundaiSafetyFlags.LONG)
    p.send_heartbeat()
    assert p.health()['safety_mode'] == CarParams.SafetyModel.hyundai
    assert p.health()['safety_param'] == HyundaiSafetyFlags.LONG

    # shouldn't do anything once we're in a car safety mode
    p.set_heartbeat_disabled()

    time.sleep(6.)

    h = p.health()
    assert h['heartbeat_lost']
    assert h['safety_mode'] == CarParams.SafetyModel.silent
    assert h['safety_param'] == 0
    assert h['controls_allowed'] == 0

  def test_microsecond_timer(self):
    p = self.p
    start_time = p.get_microsecond_timer()
    time.sleep(1)
    end_time = p.get_microsecond_timer()

    # account for uint32 overflow
    if end_time < start_time:
      end_time += 2**32

    time_diff = (end_time - start_time) / 1e6
    assert 0.98 < time_diff  < 1.02, f"Timer not running at the correct speed! (got {time_diff:.2f}s instead of 1.0s)"
