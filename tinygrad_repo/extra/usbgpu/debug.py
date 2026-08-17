#!/usr/bin/env python3

import sys
import time
from argparse import ArgumentParser

from pyftdi.ftdi import Ftdi
from pyftdi.eeprom import FtdiEeprom
from pyftdi.misc import hexdump

class USBGPUDebug:
  CBUS_RESET = (1 << 2)
  CBUS_BOOTLOADER = (1 << 1)

  def __init__(self, device_url: str = 'ftdi://ftdi:230x/1'):
    self.device_url = device_url
    self.ftdi = None
    self.eeprom = None
    self.provisioned = False

  def __enter__(self):
    self.open()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()

  def open(self):
    self.ftdi = Ftdi()
    self.ftdi.open_from_url(self.device_url)
    self.ftdi.set_baudrate(921600)
    self.ftdi.set_line_property(8, 1, 'N')

    self.eeprom = FtdiEeprom()
    self.eeprom.connect(self.ftdi)

    self.provisioned = (self.eeprom.cbus_func_1 == "GPIO" and self.eeprom.cbus_func_2 == "GPIO")

    if not self.provisioned:
      print("Warning: Device not provisioned for usbgpu debugging. Use --provision to provision it.")
      return

    # setup gpio for reset control
    self.ftdi.set_cbus_direction(self.CBUS_RESET | self.CBUS_BOOTLOADER, self.CBUS_RESET | self.CBUS_BOOTLOADER)
    self.ftdi.set_cbus_gpio(0x00)

  def close(self):
    self.ftdi.close()

  def provision(self):
    print("Provisioning FTDI device for usbgpu debugging...")
    self.eeprom.set_property('cbus_func_1', 'GPIO')
    self.eeprom.set_property('cbus_func_2', 'GPIO')
    if self.eeprom.commit(dry_run=False):
      self.eeprom.reset_device()
    self.ftdi.reset()
    self.provisioned = True
    print("Provisioning complete.")

  def reset(self, bootloader=False):
    if not self.provisioned:
      raise RuntimeError("Device not provisioned for usbgpu debugging. Use --provision to provision it.")

    self.ftdi.set_cbus_gpio(self.CBUS_RESET | (self.CBUS_BOOTLOADER if bootloader else 0))
    time.sleep(0.5)
    self.ftdi.set_cbus_gpio(self.CBUS_BOOTLOADER if bootloader else 0)
    if bootloader:
      time.sleep(1)
      self.ftdi.set_cbus_gpio(0)
    print("Device reset complete.")

  def read(self) -> bytes:
    return self.ftdi.read_data(256).decode('utf-8', errors='replace')


if __name__ == "__main__":
  args = ArgumentParser()
  args.add_argument('--device', '-d', type=str, default='ftdi://ftdi:230x/1', help="FTDI device URL")
  args.add_argument('--provision', '-p', action='store_true', default=False, help="Provision the connected FTDI for usbgpu debugging")
  args.add_argument('--reset', '-r', action='store_true', default=False, help="Reset the device")
  args.add_argument('--bootloader', '-b', action='store_true', default=False, help="Reset to bootloader")
  args.add_argument('--no-read', '-n', action='store_true', default=False, help="Do not read debug output")

  args = args.parse_args()

  with USBGPUDebug(args.device) as dbg:
    if args.provision:
      dbg.provision()

    if args.reset:
      dbg.reset(bootloader=False)

    if args.bootloader:
      dbg.reset(bootloader=True)

    if not args.no_read:
      print("Starting debug output. Press Ctrl-C to exit.\n------")
      while True:
        sys.stdout.write(dbg.read())
        sys.stdout.flush()
        time.sleep(0.001)

