#!/usr/bin/env python3
import capnp
import time
from typing import Optional, Set, Tuple

import cereal.messaging as messaging
from panda.python.uds import SERVICE_TYPE
from selfdrive.car import make_can_msg
from selfdrive.boardd.boardd import can_list_to_can_capnp
from system.swaglog import cloudlog


def make_tester_present_msg(addr, bus, subaddr=None):
  dat = [0x02, SERVICE_TYPE.TESTER_PRESENT, 0x0]
  if subaddr is not None:
    dat.insert(0, subaddr)

  dat.extend([0x0] * (8 - len(dat)))
  return make_can_msg(addr, bytes(dat), bus)


def is_tester_present_response(msg: capnp.lib.capnp._DynamicStructReader, subaddr: Optional[int] = None) -> bool:
  # ISO-TP messages are always padded to 8 bytes
  # tester present response is always a single frame
  dat_offset = 1 if subaddr is not None else 0
  if len(msg.dat) == 8 and 1 <= msg.dat[dat_offset] <= 7:
    # success response
    if msg.dat[dat_offset + 1] == (SERVICE_TYPE.TESTER_PRESENT + 0x40):
      return True
    # error response
    if msg.dat[dat_offset + 1] == 0x7F and msg.dat[dat_offset + 2] == SERVICE_TYPE.TESTER_PRESENT:
      return True
  return False


def get_all_ecu_addrs(logcan: messaging.SubSocket, sendcan: messaging.PubSocket, bus: int, timeout: float = 1, debug: bool = True) -> Set[Tuple[int, Optional[int], int]]:
  addr_list = [0x700 + i for i in range(256)] + [0x18da00f1 + (i << 8) for i in range(256)]
  queries: Set[Tuple[int, Optional[int], int]] = {(addr, None, bus) for addr in addr_list}
  responses = queries
  return get_ecu_addrs(logcan, sendcan, queries, responses, timeout=timeout, debug=debug)


def get_ecu_addrs(logcan: messaging.SubSocket, sendcan: messaging.PubSocket, queries: Set[Tuple[int, Optional[int], int]],
                  responses: Set[Tuple[int, Optional[int], int]], timeout: float = 1, debug: bool = False) -> Set[Tuple[int, Optional[int], int]]:
  ecu_responses: Set[Tuple[int, Optional[int], int]] = set()  # set((addr, subaddr, bus),)
  try:
    msgs = [make_tester_present_msg(addr, bus, subaddr) for addr, subaddr, bus in queries]

    messaging.drain_sock_raw(logcan)
    sendcan.send(can_list_to_can_capnp(msgs, msgtype='sendcan'))
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
      can_packets = messaging.drain_sock(logcan, wait_for_one=True)
      for packet in can_packets:
        for msg in packet.can:
          subaddr = None if (msg.address, None, msg.src) in responses else msg.dat[0]
          if (msg.address, subaddr, msg.src) in responses and is_tester_present_response(msg, subaddr):
            if debug:
              print(f"CAN-RX: {hex(msg.address)} - 0x{bytes.hex(msg.dat)}")
              if (msg.address, subaddr, msg.src) in ecu_responses:
                print(f"Duplicate ECU address: {hex(msg.address)}")
            ecu_responses.add((msg.address, subaddr, msg.src))
  except Exception:
    cloudlog.exception("ECU addr scan exception")
  return ecu_responses


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description='Get addresses of all ECUs')
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--bus', type=int, default=1)
  parser.add_argument('--timeout', type=float, default=1.0)
  args = parser.parse_args()

  logcan = messaging.sub_sock('can')
  sendcan = messaging.pub_sock('sendcan')

  time.sleep(1.0)

  print("Getting ECU addresses ...")
  ecu_addrs = get_all_ecu_addrs(logcan, sendcan, args.bus, args.timeout, debug=args.debug)

  print()
  print("Found ECUs on addresses:")
  for addr, subaddr, bus in ecu_addrs:
    msg = f"  0x{hex(addr)}"
    if subaddr is not None:
      msg += f" (sub-address: {hex(subaddr)})"
    print(msg)
