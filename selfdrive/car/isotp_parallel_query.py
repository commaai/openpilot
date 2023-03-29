import time
from collections import defaultdict
from functools import partial

import cereal.messaging as messaging
from system.swaglog import cloudlog
from selfdrive.boardd.boardd import can_list_to_can_capnp
from panda.python.uds import CanClient, IsoTpMessage, FUNCTIONAL_ADDRS, get_rx_addr_for_tx_addr


class IsoTpParallelQuery:
  def __init__(self, sendcan, logcan, bus, addrs, request, response, response_offset=0x8, functional_addrs=None, debug=False, response_pending_timeout=10):
    self.sendcan = sendcan
    self.logcan = logcan
    self.bus = bus
    self.request = request
    self.response = response
    self.functional_addrs = functional_addrs or []
    self.debug = debug
    self.response_pending_timeout = response_pending_timeout

    real_addrs = [a if isinstance(a, tuple) else (a, None) for a in addrs]
    for tx_addr, _ in real_addrs:
      assert tx_addr not in FUNCTIONAL_ADDRS, f"Functional address should be defined in functional_addrs: {hex(tx_addr)}"

    self.msg_addrs = {tx_addr: get_rx_addr_for_tx_addr(tx_addr[0], rx_offset=response_offset) for tx_addr in real_addrs}
    self.msg_buffer = defaultdict(list)

  def rx(self):
    """Drain can socket and sort messages into buffers based on address"""
    can_packets = messaging.drain_sock(self.logcan, wait_for_one=True)

    for packet in can_packets:
      for msg in packet.can:
        if msg.src == self.bus and msg.address in self.msg_addrs.values():
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
    messaging.drain_sock_raw(self.logcan)
    self.msg_buffer = defaultdict(list)

  def _create_isotp_msg(self, tx_addr, sub_addr, rx_addr):
    can_client = CanClient(self._can_tx, partial(self._can_rx, rx_addr, sub_addr=sub_addr), tx_addr, rx_addr,
                           self.bus, sub_addr=sub_addr, debug=self.debug)

    max_len = 8 if sub_addr is None else 7
    # uses iso-tp frame separation time of 10 ms
    # TODO: use single_frame_mode so ECUs can send as fast as they want,
    # as well as reduces chances we process messages from previous queries
    return IsoTpMessage(can_client, timeout=0, separation_time=0.01, debug=self.debug, max_len=max_len)

  def get_data(self, timeout, total_timeout=60.):
    self._drain_rx()

    # Create message objects
    msgs = {}
    request_counter = {}
    request_done = {}
    for tx_addr, rx_addr in self.msg_addrs.items():
      msgs[tx_addr] = self._create_isotp_msg(*tx_addr, rx_addr)
      request_counter[tx_addr] = 0
      request_done[tx_addr] = False

    # Send first request to functional addrs, subsequent responses are handled on physical addrs
    if len(self.functional_addrs):
      for addr in self.functional_addrs:
        self._create_isotp_msg(addr, None, -1).send(self.request[0])

    # If querying functional addrs, set up physical IsoTpMessages to send consecutive frames
    for msg in msgs.values():
      msg.send(self.request[0], setup_only=len(self.functional_addrs) > 0)

    results = {}
    start_time = time.monotonic()
    response_timeouts = {tx_addr: start_time + timeout for tx_addr in self.msg_addrs}
    while True:
      self.rx()

      if all(request_done.values()):
        break

      for tx_addr, msg in msgs.items():
        try:
          dat, updated = msg.recv()
        except Exception:
          cloudlog.exception(f"Error processing UDS response: {tx_addr}")
          request_done[tx_addr] = True
          continue

        if updated:
          response_timeouts[tx_addr] = time.monotonic() + timeout

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
          error_code = dat[2] if len(dat) > 2 else -1
          if error_code == 0x78:
            response_timeouts[tx_addr] = time.monotonic() + self.response_pending_timeout
            if self.debug:
              cloudlog.warning(f"iso-tp query response pending: {tx_addr}")
          else:
            response_timeouts[tx_addr] = 0
            request_done[tx_addr] = True
            cloudlog.error(f"iso-tp query bad response: {tx_addr} - 0x{dat.hex()}")

      cur_time = time.monotonic()
      if cur_time - max(response_timeouts.values()) > 0:
        for tx_addr in msgs:
          if request_counter[tx_addr] > 0 and not request_done[tx_addr]:
            cloudlog.error(f"iso-tp query timeout after receiving response: {tx_addr}")
        break

      if cur_time - start_time > total_timeout:
        cloudlog.error("iso-tp query timeout while receiving data")
        break

    return results
