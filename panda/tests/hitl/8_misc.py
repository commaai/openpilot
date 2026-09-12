import time
import pytest

from panda import Panda

def test_boot_time(p):
  # boot time should be instant
  st = time.monotonic()
  p.reset(reconnect=False)
  assert Panda.wait_for_panda(p.get_usb_serial(), timeout=3.0)

  # USB enumeration is slow, so SPI is faster
  assert time.monotonic() - st < (1.0 if p.spi else 5.0)

@pytest.mark.panda_expect_can_error
@pytest.mark.test_panda_types((Panda.HW_TYPE_CUATRO, ))
def test_stop_mode(p, panda_jungle):
  serial = p.get_usb_serial()
  panda_jungle.set_obd(True)

  for orientation in (Panda.HARNESS_STATUS_FLIPPED, Panda.HARNESS_STATUS_NORMAL):
    panda_jungle.set_harness_orientation(orientation)
    time.sleep(0.25)  # wait for orientation detection

    for wakeup in ("ign", "0", "1", "2"):
      panda_jungle.set_ignition(False)
      print(f"orientation={orientation} wakeup={wakeup}")

      p.enter_stop_mode()
      p.close()
      # wait for panda to enter stop mode
      time.sleep(1.5)

      # wake via ignition or CAN activity
      if wakeup == "ign":
        panda_jungle.set_ignition(True)
      else:
        panda_jungle.can_send(0x123, b'\x01\x02', int(wakeup))

      # panda should reset and come back
      assert Panda.wait_for_panda(serial, timeout=10)

      p.reconnect()
      assert p.health()['uptime'] < 3
