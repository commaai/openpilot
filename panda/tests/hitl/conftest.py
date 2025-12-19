import os
import pytest

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
init_devices()

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

  # ensure FW hasn't changed
  assert _panda_jungle.up_to_date()


def pytest_configure(config):
  config.addinivalue_line(
    "markers", "test_panda_types(name): whitelist a test for specific panda types"
  )
  config.addinivalue_line(
    "markers", "skip_panda_types(name): blacklist panda types from a test"
  )
  config.addinivalue_line(
    "markers", "panda_expect_can_error: mark test to ignore CAN health errors"
  )

@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
  for item in items:
    if item.get_closest_marker('timeout') is None:
      item.add_marker(pytest.mark.timeout(60))

    needs_jungle = "panda_jungle" in item.fixturenames
    if needs_jungle and NO_JUNGLE:
      item.add_marker(pytest.mark.skip(reason="skipping tests that requires a jungle"))

@pytest.fixture(name='panda_jungle', scope='function')
def fixture_panda_jungle(request):
  init_jungle()
  return _panda_jungle

@pytest.fixture(name='p', scope='function')
def func_fixture_panda(request, module_panda):
  # *** Setup ***

  p = module_panda

  # Check if test is applicable to this panda
  mark = request.node.get_closest_marker('test_panda_types')
  if mark:
    assert len(mark.args) > 0, "Missing panda types argument in mark"
    test_types = mark.args[0]
    if _panda_type not in test_types:
      pytest.skip(f"Not applicable, {test_types} pandas only")

  mark = request.node.get_closest_marker('skip_panda_types')
  if mark:
    assert len(mark.args) > 0, "Missing panda types argument in mark"
    skip_types = mark.args[0]
    if _panda_type in skip_types:
      pytest.skip(f"Not applicable to {skip_types}")

  # this is 2+ seconds on USB pandas due to slow
  # enumeration on the host side
  p.reset()

  # ensure FW hasn't changed
  assert p.up_to_date()

  # *** Run test ***
  yield p

  # *** Teardown ***

  # reconnect
  if p.get_dfu_serial() in PandaDFU.list():
    PandaDFU(p.get_dfu_serial()).reset()
    p.reconnect()
  if not p.connected:
    p.reconnect()
  if p.bootstub:
    p.reset()

  assert not p.bootstub

  # TODO: would be nice to make these common checks in the teardown
  # show up as failed tests instead of "errors"

  # Check for faults
  assert p.health()['faults'] == 0
  assert p.health()['fault_status'] == 0

  # Check for SPI errors
  #assert p.health()['spi_error_count'] == 0

  # Check health of each CAN core after test, normal to fail for test_gen2_loopback on OBD bus, so skipping
  mark = request.node.get_closest_marker('panda_expect_can_error')
  expect_can_error = mark is not None
  if not expect_can_error:
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

@pytest.fixture(name='module_panda', scope='module')
def fixture_panda_setup(request):
  """
    Clean up panda + jungle and return the panda under test.
  """
  # init jungle
  init_jungle()

  # init panda
  p = Panda(serial=_panda_serial)
  p.reset(reconnect=True)

  p.set_can_loopback(False)
  p.set_power_save(False)
  for bus, speed in BUS_SPEEDS:
    p.set_can_speed_kbps(bus, speed)
  clear_can_buffers(p)
  p.set_power_save(False)

  # run test
  yield p

  # teardown
  p.close()
