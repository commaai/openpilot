import time

from opendbc.car import make_tester_present_msg, uds
from opendbc.car.can_definitions import CanData, CanRecvCallable, CanSendCallable
from opendbc.car.carlog import carlog
from opendbc.car.fw_query_definitions import EcuAddrBusType


def _is_tester_present_response(msg: CanData, subaddr: int = None) -> bool:
  # ISO-TP messages may use CAN frame optimization (not always 8 bytes)
  # tester present response is always a single frame
  dat_offset = 1 if subaddr is not None else 0
  min_length = 4 if subaddr is not None else 3  # bytes: frame len, (pos/neg) sid, (optional negative sid)/0x00 sub-function
  if min_length <= len(msg.dat) <= 8 and 1 <= msg.dat[dat_offset] <= 7:
    # success response
    if msg.dat[dat_offset + 1] == (uds.SERVICE_TYPE.TESTER_PRESENT + 0x40):
      return True
    # error response
    if msg.dat[dat_offset + 1] == 0x7F and msg.dat[dat_offset + 2] == uds.SERVICE_TYPE.TESTER_PRESENT:
      return True
  return False


def get_all_ecu_addrs(can_recv: CanRecvCallable, can_send: CanSendCallable, bus: int, timeout: float = 1) -> set[EcuAddrBusType]:
  addr_list = [0x700 + i for i in range(256)] + [0x18da00f1 + (i << 8) for i in range(256)]
  queries: set[EcuAddrBusType] = {(addr, None, bus) for addr in addr_list}
  responses = queries
  return get_ecu_addrs(can_recv, can_send, queries, responses, timeout=timeout)


def get_ecu_addrs(can_recv: CanRecvCallable, can_send: CanSendCallable, queries: set[EcuAddrBusType],
                  responses: set[EcuAddrBusType], timeout: float = 1) -> set[EcuAddrBusType]:
  ecu_responses: set[EcuAddrBusType] = set()  # set((addr, subaddr, bus),)
  try:
    msgs = [make_tester_present_msg(addr, bus, subaddr) for addr, subaddr, bus in queries]

    can_recv()
    can_send(msgs)
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
      can_packets = can_recv(wait_for_one=True)
      for packet in can_packets:
        for msg in packet:
          if not len(msg.dat):
            carlog.warning("ECU addr scan: skipping empty remote frame")
            continue

          subaddr = None if (msg.address, None, msg.src) in responses else msg.dat[0]
          if (msg.address, subaddr, msg.src) in responses and _is_tester_present_response(msg, subaddr):
            carlog.debug(f"CAN-RX: {hex(msg.address)} - 0x{bytes.hex(msg.dat)}")
            if (msg.address, subaddr, msg.src) in ecu_responses:
              carlog.debug(f"Duplicate ECU address: {hex(msg.address)}")
            ecu_responses.add((msg.address, subaddr, msg.src))
  except Exception:
    carlog.exception("ECU addr scan exception")
  return ecu_responses
