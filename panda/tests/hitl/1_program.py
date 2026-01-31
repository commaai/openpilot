import os
import time
import pytest

from panda import Panda, PandaDFU, McuType, BASEDIR


def check_signature(p):
  assert not p.bootstub, "Flashed firmware not booting. Stuck in bootstub."
  assert p.up_to_date()


def test_dfu(p):
  app_mcu_type = p.get_mcu_type()
  dfu_serial = p.get_dfu_serial()

  p.reset(enter_bootstub=True)
  p.reset(enter_bootloader=True)
  assert Panda.wait_for_dfu(dfu_serial, timeout=19), "failed to enter DFU"

  dfu = PandaDFU(dfu_serial)
  assert dfu.get_mcu_type() == app_mcu_type

  assert dfu_serial in PandaDFU.list()

  dfu._handle.clear_status()
  dfu.reset()
  p.reconnect()

# TODO: make more comprehensive bootstub tests and run on a few production ones + current
# TODO: also test release-signed app
@pytest.mark.timeout(30)
def test_known_bootstub(p):
  """
  Test that compiled app can work with known production bootstub
  """
  known_bootstubs = {
    McuType.H7: ["bootstub.panda_h7.bin"],
  }

  for kb in known_bootstubs[p.get_mcu_type()]:
    app_ids = (p.get_mcu_type(), p.get_usb_serial())
    assert None not in app_ids

    p.reset(enter_bootstub=True)
    p.reset(enter_bootloader=True)

    dfu_serial = p.get_dfu_serial()
    assert Panda.wait_for_dfu(dfu_serial, timeout=30)

    dfu = PandaDFU(dfu_serial)
    with open(os.path.join(BASEDIR, "tests/hitl/known_bootstub", kb), "rb") as f:
      code = f.read()

    dfu.program_bootstub(code)
    dfu.reset()

    p.connect(claim=False, wait=True)

    # check for MCU or serial mismatch
    with Panda(p._serial, claim=False) as np:
      bootstub_ids = (np.get_mcu_type(), np.get_usb_serial())
      assert app_ids == bootstub_ids

    # ensure we can flash app and it jumps to app
    p.flash()
    check_signature(p)
    assert not p.bootstub

@pytest.mark.timeout(25)
def test_recover(p):
  assert p.recover(timeout=30)
  check_signature(p)

@pytest.mark.timeout(25)
def test_flash(p):
  # test flash from bootstub
  serial = p._serial
  assert serial is not None
  p.reset(enter_bootstub=True)
  p.close()
  time.sleep(2)

  with Panda(serial) as np:
    assert np.bootstub
    assert np._serial == serial
    np.flash()

  p.reconnect()
  p.reset()
  check_signature(p)

  # test flash from app
  p.flash()
  check_signature(p)
