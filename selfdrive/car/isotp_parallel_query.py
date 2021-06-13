import time
from collections import defaultdict
from functools import partial
from typing import Optional

import cereal.messaging as messaging
from selfdrive.swaglog import cloudlog
from selfdrive.boardd.boardd import can_list_to_can_capnp
from panda.python.uds import CanClient, IsoTpMessage, FUNCTIONAL_ADDRS, get_rx_addr_for_tx_addr


class IsoTpParallelQuery:
  def __init__(self, sendcan, logcan, bus, addrs, request, response, response_offset=0x8, functional_addr=False, debug=False):
    self.sendcan = sendcan
    self.logcan = logcan
    self.bus = bus
    self.request = request
    self.response = response
    self.debug = debug
    self.functional_addr = functional_addr

    self.real_addrs = []
    for a in addrs:
      if isinstance(a, tuple):
        self.real_addrs.append(a)
      else:
        self.real_addrs.append((a, None))

    self.msg_addrs = {tx_addr: get_rx_addr_for_tx_addr(tx_addr[0], rx_offset=response_offset) for tx_addr in self.real_addrs}
    self.msg_buffer = defaultdict(list)

  def rx(self):
    """Drain can socket and sort messages into buffers based on address"""
    can_packets = messaging.drain_sock(self.logcan, wait_for_one=True)

    for packet in can_packets:
      for msg in packet.can:
        if msg.src == self.bus:
          if self.functional_addr:
            if (0x7E8 <= msg.address <= 0x7EF) or (0x18DAF100 <= msg.address <= 0x18DAF1FF):
              fn_addr = next(a for a in FUNCTIONAL_ADDRS if msg.address - a <= 32)
              self.msg_buffer[fn_addr].append((msg.address, msg.busTime, msg.dat, msg.src))
          elif msg.address in self.msg_addrs.values():
            self.msg_buffer[msg.address].append((msg.address, msg.busTime, msg.dat, msg.src))

  def _can_tx(self, tx_addr, dat, bus):
    """Helper function to send single message"""
    msg = [tx_addr, 0, dat, bus]
    self.sendcan.send(can_list_to_can_capnp([msg], msgtype='sendcan'))

  def _can_rx(self, addr, sub_addr=None):
    """Helper function to retrieve message with specified address and subadress from buffer"""
    keep_msgs = []

    if sub_addr is None:
      msgs = self.msg_buffer[addr]
    else:
      # Filter based on subadress
      msgs = []
      for m in self.msg_buffer[addr]:
        first_byte = m[2][0]
        if first_byte == sub_addr:
          msgs.append(m)
        else:
          keep_msgs.append(m)

    self.msg_buffer[addr] = keep_msgs
    return msgs

  def _drain_rx(self):
    messaging.drain_sock(self.logcan)
    self.msg_buffer = defaultdict(list)

  def get_data(self, timeout):
    self._drain_rx()

    # Create message objects
    msgs = {}
    request_counter = {}
    request_done = {}
    for tx_addr, rx_addr in self.msg_addrs.items():
      # rx_addr not set when using functional tx addr
      id_addr = rx_addr or tx_addr[0]
      sub_addr = tx_addr[1]

      can_client = CanClient(self._can_tx, partial(self._can_rx, id_addr, sub_addr=sub_addr), tx_addr[0], rx_addr,
                             self.bus, sub_addr=sub_addr, debug=self.debug)

      max_len = 8 if sub_addr is None else 7

      msg = IsoTpMessage(can_client, timeout=0, max_len=max_len, debug=self.debug)
      msg.send(self.request[0])

      msgs[tx_addr] = msg
      request_counter[tx_addr] = 0
      request_done[tx_addr] = False

    results = {}
    start_time = time.time()
    while True:
      self.rx()

      if all(request_done.values()):
        break

      for tx_addr, msg in msgs.items():
        dat: Optional[bytes] = msg.recv()

        if not dat:
          continue

        counter = request_counter[tx_addr]
        expected_response = self.response[counter]
        response_valid = dat[:len(expected_response)] == expected_response

        if response_valid:
          if counter + 1 < len(self.request):
            msg.send(self.request[counter + 1])
            request_counter[tx_addr] += 1
          else:
            results[tx_addr] = dat[len(expected_response):]
            request_done[tx_addr] = True
        else:
          request_done[tx_addr] = True
          cloudlog.warning(f"iso-tp query bad response: 0x{dat.hex()}")

      if time.time() - start_time > timeout:
        break

    return results
