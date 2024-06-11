import os
import pytest
import concurrent.futures

from panda import Panda, PandaDFU, PandaJungle
from panda.tests.hitl.helpers import clear_can_buffers

# needed to get output when using xdist
if "DEBUG" in os.environ:
  import sys
  sys.stdout = sys.stderr

SPEED_NORMAL = 500
BUS_SPEEDS = [(0, SPEED_NORMAL), (1, SPEED_NORMAL), (2, SPEED_NORMAL)]


JUNGLE_SERIAL = os.getenv("PANDAS_JUNGLE")
NO_JUNGLE = os.environ.get("NO_JUNGLE", "0") == "1"
PANDAS_EXCLUDE = os.getenv("PANDAS_EXCLUDE", "").strip().split(" ")
HW_TYPES = os.environ.get("HW_TYPES", None)

PARALLEL = "PARALLEL" in os.environ
NON_PARALLEL = "NON_PARALLEL" in os.environ
if PARALLEL:
  NO_JUNGLE = True

class PandaGroup:
  H7 = (Panda.HW_TYPE_RED_PANDA, Panda.HW_TYPE_RED_PANDA_V2, Panda.HW_TYPE_TRES)
  GEN2 = (Panda.HW_TYPE_BLACK_PANDA, Panda.HW_TYPE_UNO, Panda.HW_TYPE_DOS) + H7
  TESTED = (Panda.HW_TYPE_WHITE_PANDA, Panda.HW_TYPE_BLACK_PANDA, Panda.HW_TYPE_RED_PANDA, Panda.HW_TYPE_RED_PANDA_V2, Panda.HW_TYPE_UNO)

if HW_TYPES is not None:
  PandaGroup.TESTED = [bytes([int(x), ]) for x in HW_TYPES.strip().split(",")] # type: ignore


# Find all pandas connected
_all_pandas = {}
_panda_jungle = None
def init_all_pandas():
  if not NO_JUNGLE:
    global _panda_jungle
    _panda_jungle = PandaJungle(JUNGLE_SERIAL)
    _panda_jungle.set_panda_power(True)

  for serial in Panda.list():
    if serial not in PANDAS_EXCLUDE:
      with Panda(serial=serial, claim=False) as p:
        ptype = bytes(p.get_type())
        if ptype in PandaGroup.TESTED:
          _all_pandas[serial] = ptype

  # ensure we have all tested panda types
  missing_types = set(PandaGroup.TESTED) - set(_all_pandas.values())
  assert len(missing_types) == 0, f"Missing panda types: {missing_types}"

  print(f"{len(_all_pandas)} total pandas")
init_all_pandas()
_all_panda_serials = sorted(_all_pandas.keys())


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

    # xdist grouping by panda
    serial = item.name.split("serial=")[1].split(",")[0]
    assert len(serial) == 24
    item.add_marker(pytest.mark.xdist_group(serial))

    needs_jungle = "panda_jungle" in item.fixturenames
    if PARALLEL and needs_jungle:
      item.add_marker(pytest.mark.skip(reason="no jungle tests in PARALLEL mode"))
    elif NON_PARALLEL and not needs_jungle:
      item.add_marker(pytest.mark.skip(reason="only running jungle tests"))

def pytest_make_parametrize_id(config, val, argname):
  if val in _all_pandas:
    # TODO: get nice string instead of int
    hw_type = _all_pandas[val][0]
    return f"serial={val}, hw_type={hw_type}"
  return None


@pytest.fixture(name='panda_jungle', scope='function')
def fixture_panda_jungle(request):
  init_jungle()
  return _panda_jungle

@pytest.fixture(name='p', scope='function')
def func_fixture_panda(request, module_panda):
  p = module_panda

  # Check if test is applicable to this panda
  mark = request.node.get_closest_marker('test_panda_types')
  if mark:
    assert len(mark.args) > 0, "Missing panda types argument in mark"
    test_types = mark.args[0]
    if _all_pandas[p.get_usb_serial()] not in test_types:
      pytest.skip(f"Not applicable, {test_types} pandas only")

  mark = request.node.get_closest_marker('skip_panda_types')
  if mark:
    assert len(mark.args) > 0, "Missing panda types argument in mark"
    skip_types = mark.args[0]
    if _all_pandas[p.get_usb_serial()] in skip_types:
      pytest.skip(f"Not applicable to {skip_types}")

  # this is 2+ seconds on USB pandas due to slow
  # enumeration on the host side
  p.reset()

  # ensure FW hasn't changed
  assert p.up_to_date()

  # Run test
  yield p

  # Teardown

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
  #assert p.health()['spi_checksum_error_count'] == 0

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

@pytest.fixture(name='module_panda', params=_all_panda_serials, scope='module')
def fixture_panda_setup(request):
  """
    Clean up all pandas + jungle and return the panda under test.
  """
  panda_serial = request.param

  # Initialize jungle
  init_jungle()

  # Connect to pandas
  def cnnct(s):
    if s == panda_serial:
      p = Panda(serial=s)
      p.reset(reconnect=True)

      p.set_can_loopback(False)
      p.set_power_save(False)
      for bus, speed in BUS_SPEEDS:
        p.set_can_speed_kbps(bus, speed)
      clear_can_buffers(p)
      p.set_power_save(False)
      return p
    elif not PARALLEL:
      with Panda(serial=s) as p:
        p.reset(reconnect=False)
    return None

  with concurrent.futures.ThreadPoolExecutor() as exc:
    ps = list(exc.map(cnnct, _all_panda_serials, timeout=20))
    pandas = [p for p in ps if p is not None]

  # run test
  yield pandas[0]

  # Teardown
  for p in pandas:
    p.close()
