import time

from panda.tests.hitl.base import PandaTestCase, panda_test
from panda import Panda, PandaDFU, McuType


def check_signature(p):
  assert not p.bootstub, "Flashed firmware not booting. Stuck in bootstub."
  assert p.up_to_date()


class TestProgram(PandaTestCase):
  def test_dfu(self):
    p = self.p
    dfu_serial = p.get_dfu_serial()

    p.reset(enter_bootstub=True)
    p.reset(enter_bootloader=True)
    assert Panda.wait_for_dfu(dfu_serial, timeout=19), "failed to enter DFU"

    assert dfu_serial in PandaDFU.list()
    with PandaDFU(dfu_serial) as dfu:
      assert dfu.get_mcu_type() == McuType.H7
      dfu._handle.clear_status()
      dfu.reset()
    p.reconnect()

  def test_recover(self):
    p = self.p
    assert p.recover(timeout=30)
    check_signature(p)

  @panda_test(timeout=25)
  def test_flash(self):
    p = self.p
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
