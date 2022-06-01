#!/usr/bin/env python3
import capnp
import time
import traceback
from typing import Dict, Optional, Set

import cereal.messaging as messaging
from panda.python.uds import SERVICE_TYPE
from selfdrive.car import make_can_msg
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.swaglog import cloudlog


def is_tester_present_response(msg: capnp.lib.capnp._DynamicStructReader) -> bool:
  # ISO-TP messages are always padded to 8 bytes
  # tester present response is always a single frame
  if len(msg.dat) == 8 and 1 <= msg.dat[0] <= 7:
    # success response
    if msg.dat[1] == (SERVICE_TYPE.TESTER_PRESENT + 0x40):
      return True
    # error response
    if msg.dat[1] == 0x7F and msg.dat[2] == SERVICE_TYPE.TESTER_PRESENT:
      return True
  return False


def get_all_ecu_addrs(logcan: messaging.SubSocket, sendcan: messaging.PubSocket, bus: int, timeout: float = 1, debug: bool = True) -> Set[int]:
  addr_list = [0x700 + i for i in range(256)] + [0x18da00f1 + (i << 8) for i in range(256)]
  query_addrs = response_addrs = {addr: bus for addr in addr_list}
  return get_ecu_addrs(logcan, sendcan, query_addrs, response_addrs, timeout=timeout, debug=debug)


def get_ecu_addrs(logcan: messaging.SubSocket, sendcan: messaging.PubSocket, query_addrs: Dict[int, int], response_addrs: Dict[int, int], timeout: float = 1, debug: bool = True) -> Set[int]:
  ecu_addrs = set()
  try:
    tester_present = bytes([0x02, SERVICE_TYPE.TESTER_PRESENT, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0])
    msgs = [make_can_msg(addr, tester_present, bus) for addr, bus in query_addrs.items()]

    messaging.drain_sock(logcan)
    sendcan.send(can_list_to_can_capnp(msgs, msgtype='sendcan'))
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
      can_packets = messaging.drain_sock(logcan, wait_for_one=True)
      for packet in can_packets:
        for msg in packet.can:
          if msg.address in response_addrs and msg.src == response_addrs[msg.address] and is_tester_present_response(msg):
            if debug:
              print(f"CAN-RX: {hex(msg.address)} - 0x{bytes.hex(msg.dat)}")
              if msg.address in ecu_addrs:
                print(f"Duplicate ECU address: {hex(msg.address)}")
            ecu_addrs.add(msg.address)
  except Exception:
    cloudlog.warning(f"ECU addr scan exception: {traceback.format_exc()}")
  return ecu_addrs


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description='Get addresses of all ECUs')
  parser.add_argument('--debug', action='store_true')
  args = parser.parse_args()

  logcan = messaging.sub_sock('can')
  sendcan = messaging.pub_sock('sendcan')

  time.sleep(1.0)

  print("Getting ECU addresses ...")
  ecu_addrs = get_all_ecu_addrs(logcan, sendcan, 1, debug=args.debug)

  print()
  print("Found ECUs on addresses:")
  for addr in ecu_addrs:
    print(f"  {hex(addr)}")
