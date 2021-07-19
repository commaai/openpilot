#!/usr/bin/env python3
import datetime
import os
import struct
import usb1

REQUEST_IN = usb1.ENDPOINT_IN | usb1.TYPE_VENDOR | usb1.RECIPIENT_DEVICE
MIN_DATE = datetime.datetime(year=2021, month=4, day=1)

def set_time(logger):
  sys_time = datetime.datetime.today()
  if sys_time > MIN_DATE:
    logger.info("System time valid")
    return

  try:
    ctx = usb1.USBContext()
    dev = ctx.openByVendorIDAndProductID(0xbbaa, 0xddcc)
    if dev is None:
      logger.info("No panda found")
      return

    # Set system time from panda RTC time
    dat = dev.controlRead(REQUEST_IN, 0xa0, 0, 0, 8)
    a = struct.unpack("HBBBBBB", dat)
    panda_time = datetime.datetime(a[0], a[1], a[2], a[4], a[5], a[6])
    if panda_time > MIN_DATE:
      logger.info(f"adjusting time from '{sys_time}' to '{panda_time}'")
      os.system(f"TZ=UTC date -s '{panda_time}'")
  except Exception:
    logger.warn("Failed to fetch time from panda")

if __name__ == "__main__":
  import logging
  logging.basicConfig(level=logging.INFO)

  set_time(logging)
