import time

from panda import Panda

def test_boot_time(p):
  # boot time should be instant
  st = time.monotonic()
  p.reset(reconnect=False)
  assert Panda.wait_for_panda(p.get_usb_serial(), timeout=3.0)

  # USB enumeration is slow, so SPI is faster
  assert time.monotonic() - st < (1.0 if p.spi else 5.0)

