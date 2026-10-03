import atexit
import os
import signal
import unittest

from panda import Panda, PandaDFU, PandaJungle
from panda.tests.hitl.helpers import clear_can_buffers

SPEED_NORMAL = 500
BUS_SPEEDS = [(0, SPEED_NORMAL), (1, SPEED_NORMAL), (2, SPEED_NORMAL)]

# test options
NO_JUNGLE = os.environ.get("NO_JUNGLE", "0") == "1"

# Find all pandas connected
_panda_jungle = None
_panda_type = None
_panda_serial = None
def init_devices():
  if not NO_JUNGLE:
    global _panda_jungle
    _panda_jungle = PandaJungle()
    _panda_jungle.set_panda_power(True)

  with Panda(serial=None, claim=False) as p:
    global _panda_type
    global _panda_serial
    _panda_serial = p.get_usb_serial()
    _panda_type = bytes(p.get_type())
  assert _panda_serial is not None, "No panda found!"

def init_jungle():
  if _panda_jungle is None:
    return
  clear_can_buffers(_panda_jungle)
  _panda_jungle.set_panda_power(True)
  _panda_jungle.set_can_loopback(False)
  _panda_jungle.set_obd(False)
  _panda_jungle.set_harness_orientation(PandaJungle.HARNESS_ORIENTATION_1)
  for bus, speed in BUS_SPEEDS:
    _panda_jungle.set_can_speed_kbps(bus, speed)


def panda_test(**options):
  def decorate(test):
    test.panda_options = {**getattr(test, "panda_options", {}), **options}
    return test
  return decorate


@unittest.skipUnless(os.environ.get("HITL") == "1", "hardware tests require HITL=1")
class PandaTestCase(unittest.TestCase):
  panda_types = None

  @classmethod
  def setUpClass(cls):
    if _panda_serial is None:
      init_devices()
      if _panda_jungle is not None:
        atexit.register(_panda_jungle.close)
    # init jungle
    init_jungle()

    # init panda
    assert Panda.wait_for_panda(_panda_serial, timeout=10), "panda not found"
    p = Panda(serial=_panda_serial)
    cls.addClassCleanup(p.close)
    cls.p = p
    p.reset(reconnect=True)

    p.set_can_loopback(False)
    p.set_power_save(False)
    for bus, speed in BUS_SPEEDS:
      p.set_can_speed_kbps(bus, speed)
    clear_can_buffers(p)
    p.set_power_save(False)

  def setUp(self):
    self.options = getattr(getattr(self, self._testMethodName), "panda_options", {})
    panda_types = self.options.get("panda_types", self.panda_types)
    if panda_types is not None and _panda_type not in panda_types:
      self.skipTest(f"Not applicable, {panda_types} pandas only")
    if self.options.get("needs_jungle", False) and NO_JUNGLE:
      self.skipTest("skipping tests that require a jungle")

    previous_handler = signal.signal(signal.SIGALRM, self._timeout)
    self.addCleanup(signal.signal, signal.SIGALRM, previous_handler)
    self.addCleanup(signal.setitimer, signal.ITIMER_REAL, 0)
    signal.setitimer(signal.ITIMER_REAL, self.options.get("timeout", 60))

    if self.options.get("needs_jungle", False):
      init_jungle()
    self.panda_jungle = _panda_jungle
    self.p.reset()
    assert self.p.up_to_date()

  def _timeout(self, signum, frame):
    raise TimeoutError(f"{self.id()} exceeded its test timeout")

  def tearDown(self):
    p = self.p
    # reconnect
    if p.get_dfu_serial() in PandaDFU.list():
      PandaDFU(p.get_dfu_serial()).reset()
      p.reconnect()
    if not p.connected:
      p.reconnect()
    if p.bootstub:
      p.reset()

    assert not p.bootstub

    # Check for faults
    assert p.health()['faults'] == 0
    assert p.health()['fault_status'] == 0

    # Check for SPI errors
    #assert p.health()['spi_error_count'] == 0

    # Check health of each CAN core after test, normal to fail for test_gen2_loopback on OBD bus, so skipping
    if not self.options.get("expect_can_error", False):
      for i in range(3):
        can_health = p.can_health(i)
        assert can_health['bus_off_cnt'] == 0
        assert can_health['receive_error_cnt'] < 127
        assert can_health['transmit_error_cnt'] < 255
        assert can_health['error_passive'] == 0
        assert can_health['error_warning'] == 0
        assert can_health['total_rx_lost_cnt'] == 0
        assert can_health['total_tx_lost_cnt'] == 0
        assert can_health['total_error_cnt'] == 0
        assert can_health['total_tx_checksum_error_cnt'] == 0

