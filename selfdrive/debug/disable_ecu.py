#!/usr/bin/env python3
import traceback

import cereal.messaging as messaging
from selfdrive.car.isotp_parallel_query import IsoTpParallelQuery
from selfdrive.swaglog import cloudlog

EXT_DIAG_REQUEST = b'\x10\x03'
EXT_DIAG_RESPONSE = b'\x50\x03'
COM_CONT_REQUEST = b'\x28\x83\x03'
COM_CONT_RESPONSE = b''

def disable_ecu(ecu_addr, logcan, sendcan, bus, timeout=0.5, retry=5, debug=False):
  print(f"ecu disable {hex(ecu_addr)} ...")
  for i in range(retry):
    try:
      # enter extended diagnostic session
      query = IsoTpParallelQuery(sendcan, logcan, bus, [ecu_addr], [EXT_DIAG_REQUEST], [EXT_DIAG_RESPONSE], debug=debug)
      for addr, dat in query.get_data(timeout).items(): # pylint: disable=unused-variable
        print("ecu communication control disable tx/rx ...")
        # communication control disable tx and rx
        query = IsoTpParallelQuery(sendcan, logcan, bus, [ecu_addr], [COM_CONT_REQUEST], [COM_CONT_RESPONSE], debug=debug)
        query.get_data(0)
        return True
      print(f"ecu disable retry ({i+1}) ...")
    except Exception:
      cloudlog.warning(f"ecu disable exception: {traceback.format_exc()}")

  return False


if __name__ == "__main__":
  import time
  sendcan = messaging.pub_sock('sendcan')
  logcan = messaging.sub_sock('can')
  time.sleep(1)

  # honda bosch radar disable
  disabled = disable_ecu(0x18DAB0F1, logcan, sendcan, 1, debug=False)
  print(f"disabled: {disabled}")
