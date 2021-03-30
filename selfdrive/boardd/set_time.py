#!/usr/bin/env python3
import datetime
import struct
import sys
import usb1

REQUEST_IN = usb1.ENDPOINT_IN | usb1.TYPE_VENDOR | usb1.RECIPIENT_DEVICE

if __name__ == "__main__":
  ctx = usb1.USBContext()
  dev = ctx.openByVendorIDAndProductID(0xbbaa, 0xddcc)

  if dev is not None:
    dat = dev.controlRead(REQUEST_IN, 0xa0, 0, 0, 8)
    a = struct.unpack("HBBBBBB", dat)
    d = datetime.datetime(a[0], a[1], a[2], a[4], a[5], a[6])
    print("got", d)
  else:
    print("No panda found")
