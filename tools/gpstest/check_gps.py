#!/usr/bin/env python3

import sys
import selfdrive.sensord.pigeond as pd

# simple script to distinguish between ublox or quectel GPS module
def main():
  pigeon, pm = pd.create_pigeon()
  pd.init_baudrate(pigeon)

  # send clear config ublox message and check if a response is received
  try:
    pigeon.send_with_ack(b"\xb5\x62\x06\x09\x0d\x00\x00\x00\x1f\x1f\x00\x00\x00\x00\x00\x00\x00\x00\x17\x71\x5b")
  except Exception as _:
    sys.exit(0) # no ublox device
  sys.exit(1)

if __name__ == "__main__":
  main()

