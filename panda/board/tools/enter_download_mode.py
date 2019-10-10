#!/usr/bin/env python3


import sys
import time
import usb1

def enter_download_mode(device):
  handle = device.open()
  handle.claimInterface(0)

  try:
    handle.controlWrite(usb1.TYPE_VENDOR | usb1.RECIPIENT_DEVICE, 0xd1, 0, 0, b'')
  except (usb1.USBErrorIO, usb1.USBErrorPipe) as e:
    print("Device download mode enabled.")
    time.sleep(1)
  else:
    print("Device failed to enter download mode.")
    sys.exit(1)

def find_first_panda(context=None):
  context = context or usb1.USBContext()
  for device in context.getDeviceList(skip_on_error=True):
    if device.getVendorID() == 0xbbaa and device.getProductID()&0xFF00 == 0xdd00:
      return device

if __name__ == "__main__":
  panda_dev = find_first_panda()
  if panda_dev == None:
    print("no device found")
    sys.exit(0)
  print("found device")
  enter_download_mode(panda_dev)
