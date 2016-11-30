#!/usr/bin/env python
import usb1
import time
import traceback

if __name__ == "__main__":
  context = usb1.USBContext()

  for device in context.getDeviceList(skip_on_error=True):
    if device.getVendorID() == 0xbbaa and device.getProductID()&0xFF00 == 0xdd00:
      print "found device"
      handle = device.open()
      handle.claimInterface(0)

  try:
    handle.controlWrite(usb1.TYPE_VENDOR | usb1.RECIPIENT_DEVICE, 0xd1, 0, 0, '')
  except Exception:
    traceback.print_exc()
    print "expected error, exiting cleanly"
  time.sleep(1)
