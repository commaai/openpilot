#!/usr/bin/env python3
# type: ignore
# pylint: skip-file

from selfdrive.locationd.test import ublox
import struct

baudrate = 460800
rate = 100  # send new data every 100ms


def configure_ublox(dev):
  # configure ports  and solution parameters and rate
  dev.configure_port(port=ublox.PORT_USB, inMask=1, outMask=1)  # enable only UBX on USB
  dev.configure_port(port=0, inMask=0, outMask=0)  # disable DDC

  payload = struct.pack('<BBHIIHHHBB', 1, 0, 0, 2240, baudrate, 1, 1, 0, 0, 0)
  dev.configure_poll(ublox.CLASS_CFG, ublox.MSG_CFG_PRT, payload)  # enable UART

  dev.configure_port(port=4, inMask=0, outMask=0)  # disable SPI
  dev.configure_poll_port()
  dev.configure_poll_port(ublox.PORT_SERIAL1)
  dev.configure_poll_port(ublox.PORT_USB)
  dev.configure_solution_rate(rate_ms=rate)

  # Configure solution
  payload = struct.pack('<HBBIIBB4H6BH6B', 5, 4, 3, 0, 0,
                                           0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0,
                                           0, 0, 0, 0)
  dev.configure_poll(ublox.CLASS_CFG, ublox.MSG_CFG_NAV5, payload)
  payload = struct.pack('<B3BBB6BBB2BBB2B', 0, 0, 0, 0, 1,
                                            3, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0)
  dev.configure_poll(ublox.CLASS_CFG, ublox.MSG_CFG_ODO, payload)
  #bits_ITMF_config1 = '10101101011000101010110111111111'
  #bits_ITMF_config2 = '00000000000000000110001100011110'
  ITMF_config1 = 2908925439
  ITMF_config2 = 25374
  payload = struct.pack('<II', ITMF_config1, ITMF_config2)
  dev.configure_poll(ublox.CLASS_CFG, ublox.MSG_CFG_ITMF, payload)
  #payload = struct.pack('<HHIBBBBBBBBBBH6BBB2BH4B3BB', 0, 8192, 0, 0, 0,
  #                                                     0, 0, 0, 0, 0, 0,
  #                                                     0, 0, 0, 0, 0, 0,
  #                                                     0, 0, 0, 0, 0, 0,
  #                                                     0, 0, 0, 0, 0, 0,
  #                                                     0, 0, 0, 0)
  #dev.configure_poll(ublox.CLASS_CFG, ublox.MSG_CFG_NAVX5, payload)

  dev.configure_poll(ublox.CLASS_CFG, ublox.MSG_CFG_NAV5)
  dev.configure_poll(ublox.CLASS_CFG, ublox.MSG_CFG_NAVX5)
  dev.configure_poll(ublox.CLASS_CFG, ublox.MSG_CFG_ODO)
  dev.configure_poll(ublox.CLASS_CFG, ublox.MSG_CFG_ITMF)

  # Configure RAW, PVT and HW messages to be sent every solution cycle
  dev.configure_message_rate(ublox.CLASS_NAV, ublox.MSG_NAV_PVT, 1)
  dev.configure_message_rate(ublox.CLASS_RXM, ublox.MSG_RXM_RAW, 1)
  dev.configure_message_rate(ublox.CLASS_RXM, ublox.MSG_RXM_SFRBX, 1)
  dev.configure_message_rate(ublox.CLASS_MON, ublox.MSG_MON_HW, 1)
  dev.configure_message_rate(ublox.CLASS_MON, ublox.MSG_MON_HW2, 1)

  # Query the backup restore status
  print("backup restore polling message (implement custom response handler!):")
  dev.configure_poll(0x09, 0x14)

  print("if succesfull, send this to clear the flash:")
  dev.send_message(0x09, 0x14, b"\x01\x00\x00\x00")

  print("send on stop:")

  # Save on shutdown
  # Controlled GNSS stop and hot start
  payload = struct.pack('<HBB', 0x0000, 0x08, 0x00)
  dev.send_message(ublox.CLASS_CFG, ublox.MSG_CFG_RST, payload)

  # UBX-UPD-SOS backup
  dev.send_message(0x09, 0x14, b"\x00\x00\x00\x00")


if __name__ == "__main__":
  class Device:
    def write(self, s):
      d = '"{}"s'.format(''.join('\\x{:02X}'.format(b) for b in s))
      print(f"    if (!send_with_ack({d})) continue;")

  dev = ublox.UBlox(Device(), baudrate=baudrate)
  configure_ublox(dev)
