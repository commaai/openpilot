#!/usr/bin/env python3
import os
import time
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from panda import Panda, PandaSerial  # noqa: 402

def add_nmea_checksum(msg):
  d = msg[1:]
  cs = 0
  for i in d:
    cs ^= ord(i)
  return msg + "*%02X" % cs

if __name__ == "__main__":
  panda = Panda()
  ser = PandaSerial(panda, 1, 9600)

  # power cycle by toggling reset
  print("resetting")
  panda.set_esp_power(0)
  time.sleep(0.5)
  panda.set_esp_power(1)
  time.sleep(0.5)
  print("done")
  print(ser.read(1024))

  # upping baud rate
  baudrate = 460800

  print("upping baud rate")
  msg = str.encode(add_nmea_checksum("$PUBX,41,1,0007,0003,%d,0" % baudrate) + "\r\n")
  print(msg)
  ser.write(msg)
  time.sleep(0.1)  # needs a wait for it to actually send

  # new panda serial
  ser = PandaSerial(panda, 1, baudrate)

  while True:
    ret = ser.read(1024)
    if len(ret) > 0:
      sys.stdout.write(ret.decode('ascii', 'ignore'))
      sys.stdout.flush()
