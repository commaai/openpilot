#!/usr/bin/env python
import usb1
import time
import traceback
import sys

if __name__ == "__main__":
  context = usb1.USBContext()

  for device in context.getDeviceList(skip_on_error=True):
    if device.getVendorID() == 0xbbaa and device.getProductID()&0xFF00 == 0xdd00:
      print "found device"
      handle = device.open()
      handle.claimInterface(0)

  try:
    handle.controlWrite(0x40, 0xd3, int(sys.argv[1]), 0, '')
  except Exception:
    traceback.print_exc()
    print "expected error, exiting cleanly"
  time.sleep(1)
