#!/usr/bin/env python3
import traceback
from typing import Any, Set

import cereal.messaging as messaging
from panda.python.uds import SERVICE_TYPE
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.swaglog import cloudlog

# TODO: figure out type annotation for CanData
def is_tester_present_response(msg: Any) -> bool:
  # ISO-TP messages are always padded to 8 bytes
  # tester present response is always a single frame
  if len(msg.dat) == 8 and msg.dat[0] >= 1 and msg.dat[0] <= 7:
    # success response
    if msg.dat[1] == (SERVICE_TYPE.TESTER_PRESENT + 0x40):
      return True
    # error response
    if msg.dat[1] == 0x7F and msg.dat[2] == SERVICE_TYPE.TESTER_PRESENT:
      return True
  return False

def get_ecu_addrs(logcan: messaging.SubSocket, sendcan: messaging.PubSocket, bus: int, timeout: float=0.1, debug: bool=False) -> Set[int]:
  ecu_addrs = set()
  try:
    addr_list = [0x700 + i for i in range(256)] + [0x18da00f1 + (i << 8) for i in range(256)]
    msgs = [[addr, 0, bytes([SERVICE_TYPE.TESTER_PRESENT, 0x0]), bus] for addr in addr_list]
    messaging.drain_sock(logcan)
    sendcan.send(can_list_to_can_capnp(msgs, msgtype='sendcan'))
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
      can_packets = messaging.drain_sock(logcan, wait_for_one=True)
      for packet in can_packets:
        for msg in packet.can:
          if msg.src == bus and msg.address in addr_list and is_tester_present_response(msg):
            if debug:
              print(f"CAN-RX: {hex(msg.address)} - 0x{bytes.hex(msg.dat)}")
              if msg.address in ecu_addrs:
                print(f"Duplicate ECU address: {hex(msg.address)}")
            ecu_addrs.add(msg.address)
  except Exception:
    cloudlog.warning(f"ECU addr scan exception: {traceback.format_exc()}")
  return ecu_addrs

if __name__ == "__main__":
  import time
  import argparse

  parser = argparse.ArgumentParser(description='Get addresses of all ECUs')
  parser.add_argument('--debug', action='store_true')
  args = parser.parse_args()

  logcan = messaging.sub_sock('can')
  sendcan = messaging.pub_sock('sendcan')

  time.sleep(1.0)

  print("Getting ECU addresses ...")
  ecu_addrs = get_ecu_addrs(logcan, sendcan, 1, debug=args.debug)

  print()
  print("Found ECUs on addresses:")
  for addr in ecu_addrs:
    print(f"  {hex(addr)}")
