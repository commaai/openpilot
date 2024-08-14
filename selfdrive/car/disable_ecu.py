#!/usr/bin/env python3
from openpilot.selfdrive.car import carlog
from openpilot.selfdrive.car.isotp_parallel_query import IsoTpParallelQuery

EXT_DIAG_REQUEST = b'\x10\x03'
EXT_DIAG_RESPONSE = b'\x50\x03'

COM_CONT_RESPONSE = b''


def disable_ecu(can_recv, can_send, bus=0, addr=0x7d0, sub_addr=None, com_cont_req=b'\x28\x83\x01', timeout=0.1, retry=10, debug=False):
  """Silence an ECU by disabling sending and receiving messages using UDS 0x28.
  The ECU will stay silent as long as openpilot keeps sending Tester Present.

  This is used to disable the radar in some cars. Openpilot will emulate the radar.
  WARNING: THIS DISABLES AEB!"""
  carlog.warning(f"ecu disable {hex(addr), sub_addr} ...")

  for i in range(retry):
    try:
      query = IsoTpParallelQuery(can_send, can_recv, bus, [(addr, sub_addr)], [EXT_DIAG_REQUEST], [EXT_DIAG_RESPONSE], debug=debug)

      for _, _ in query.get_data(timeout).items():
        carlog.warning("communication control disable tx/rx ...")

        query = IsoTpParallelQuery(can_send, can_recv, bus, [(addr, sub_addr)], [com_cont_req], [COM_CONT_RESPONSE], debug=debug)
        query.get_data(0)

        carlog.warning("ecu disabled")
        return True

    except Exception:
      carlog.exception("ecu disable exception")

    carlog.error(f"ecu disable retry ({i + 1}) ...")
  carlog.error("ecu disable failed")
  return False


if __name__ == "__main__":
  import time
  import cereal.messaging as messaging
  from openpilot.selfdrive.car.card import can_comm_callbacks
  sendcan = messaging.pub_sock('sendcan')
  logcan = messaging.sub_sock('can')
  can_callbacks = can_comm_callbacks(logcan, sendcan)
  time.sleep(1)

  # honda bosch radar disable
  disabled = disable_ecu(*can_callbacks, bus=1, addr=0x18DAB0F1, com_cont_req=b'\x28\x83\x03', timeout=0.5, debug=False)
  print(f"disabled: {disabled}")
