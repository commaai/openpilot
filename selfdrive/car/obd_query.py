#!/usr/bin/env python3
import traceback

import cereal.messaging as messaging
from selfdrive.car.isotp_parallel_query import IsoTpParallelQuery
from selfdrive.swaglog import cloudlog

TX_ADDR = 0x7e0
OBD_DIAG_REQUEST = b'\x01\x1C'
OBD_DIAG_RESPONSE = b'\x41\x1C'

def get_obd_pid28(logcan, sendcan, bus, timeout=0.1, retry=5, debug=False):
  print(f"OBD2 query {hex(TX_ADDR)} ...")
  for i in range(retry):
    try:
      query = IsoTpParallelQuery(sendcan, logcan, bus, [TX_ADDR], [OBD_DIAG_REQUEST], [OBD_DIAG_RESPONSE], debug=debug)
      for addr, dat in query.get_data(timeout).items():
        print("query response")
        print(dat)
        return dat
      print(f"query retry ({i+1}) ...")
    except Exception:
      cloudlog.warning(f"OBD2 query exception: {traceback.format_exc()}")

  return False


if __name__ == "__main__":
  import time
  sendcan = messaging.pub_sock('sendcan')
  logcan = messaging.sub_sock('can')
  time.sleep(1)
  ret = get_obd_pid28(logcan, sendcan, 1, debug=False)
  print(f"result: {ret}")
