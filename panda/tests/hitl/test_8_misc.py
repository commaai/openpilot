import time

from panda.tests.hitl.base import PandaTestCase, panda_test
from panda import Panda

class TestMisc(PandaTestCase):
  def test_boot_time(self):
    p = self.p
    # boot time should be instant
    st = time.monotonic()
    p.reset(reconnect=False)
    assert Panda.wait_for_panda(p.get_usb_serial(), timeout=3.0)

    # USB enumeration is slow, so SPI is faster
    assert time.monotonic() - st < (1.0 if p.spi else 5.0)

  @panda_test(expect_can_error=True)
  @panda_test(panda_types=(Panda.HW_TYPE_CUATRO, ))
  def test_stop_mode(self):
    p = self.p
    up = self.up
    serial = p.get_usb_serial()

    for orientation in (Panda.HARNESS_STATUS_FLIPPED, Panda.HARNESS_STATUS_NORMAL):
      up.set_harness_orientation(orientation)
      time.sleep(0.25)  # wait for orientation detection

      for wakeup in ("ign", "0", "1", "2"):
        up.set_ignition(False)
        print(f"orientation={orientation} wakeup={wakeup}")

        # drop any of the last wakeup's frames that nothing ACKed, they'd wake the panda right away
        for bus in range(4):
          up.can_clear(bus)

        p.enter_stop_mode()
        p.close()
        # wait for panda to enter stop mode
        time.sleep(1.5)

        # wake via ignition or CAN activity
        if wakeup == "ign":
          up.set_ignition(True)
        elif wakeup == "1":
          # the panda only wakes up on one of FDCAN2's two pairs, CAN1 or OBD depending on the orientation.
          # those are the UP's buses 1 and 3 in either orientation (see LANES in ultimate_provisioning), so send on both
          for bus in (1, 3):
            up.can_send(0x123, b'\x01\x02', bus)
        else:
          up.can_send(0x123, b'\x01\x02', int(wakeup))

        # panda should reset and come back
        assert Panda.wait_for_panda(serial, timeout=10)

        p.reconnect()
        assert p.health()['uptime'] < 3
