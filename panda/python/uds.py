#!/usr/bin/env python3
import time
import struct
from collections import deque
from typing import Callable, NamedTuple, Tuple, List, Deque, Generator, Optional, cast
from enum import IntEnum
from functools import partial

class SERVICE_TYPE(IntEnum):
  DIAGNOSTIC_SESSION_CONTROL = 0x10
  ECU_RESET = 0x11
  SECURITY_ACCESS = 0x27
  COMMUNICATION_CONTROL = 0x28
  TESTER_PRESENT = 0x3E
  ACCESS_TIMING_PARAMETER = 0x83
  SECURED_DATA_TRANSMISSION = 0x84
  CONTROL_DTC_SETTING = 0x85
  RESPONSE_ON_EVENT = 0x86
  LINK_CONTROL = 0x87
  READ_DATA_BY_IDENTIFIER = 0x22
  READ_MEMORY_BY_ADDRESS = 0x23
  READ_SCALING_DATA_BY_IDENTIFIER = 0x24
  READ_DATA_BY_PERIODIC_IDENTIFIER = 0x2A
  DYNAMICALLY_DEFINE_DATA_IDENTIFIER = 0x2C
  WRITE_DATA_BY_IDENTIFIER = 0x2E
  WRITE_MEMORY_BY_ADDRESS = 0x3D
  CLEAR_DIAGNOSTIC_INFORMATION = 0x14
  READ_DTC_INFORMATION = 0x19
  INPUT_OUTPUT_CONTROL_BY_IDENTIFIER = 0x2F
  ROUTINE_CONTROL = 0x31
  REQUEST_DOWNLOAD = 0x34
  REQUEST_UPLOAD = 0x35
  TRANSFER_DATA = 0x36
  REQUEST_TRANSFER_EXIT = 0x37

class SESSION_TYPE(IntEnum):
  DEFAULT = 1
  PROGRAMMING = 2
  EXTENDED_DIAGNOSTIC = 3
  SAFETY_SYSTEM_DIAGNOSTIC = 4

class RESET_TYPE(IntEnum):
  HARD = 1
  KEY_OFF_ON = 2
  SOFT = 3
  ENABLE_RAPID_POWER_SHUTDOWN = 4
  DISABLE_RAPID_POWER_SHUTDOWN = 5

class ACCESS_TYPE(IntEnum):
  REQUEST_SEED = 1
  SEND_KEY = 2

class CONTROL_TYPE(IntEnum):
  ENABLE_RX_ENABLE_TX = 0
  ENABLE_RX_DISABLE_TX = 1
  DISABLE_RX_ENABLE_TX = 2
  DISABLE_RX_DISABLE_TX = 3

class MESSAGE_TYPE(IntEnum):
  NORMAL = 1
  NETWORK_MANAGEMENT = 2
  NORMAL_AND_NETWORK_MANAGEMENT = 3

class TIMING_PARAMETER_TYPE(IntEnum):
  READ_EXTENDED_SET = 1
  SET_TO_DEFAULT_VALUES = 2
  READ_CURRENTLY_ACTIVE = 3
  SET_TO_GIVEN_VALUES = 4

class DTC_SETTING_TYPE(IntEnum):
  ON = 1
  OFF = 2

class RESPONSE_EVENT_TYPE(IntEnum):
  STOP_RESPONSE_ON_EVENT = 0
  ON_DTC_STATUS_CHANGE = 1
  ON_TIMER_INTERRUPT = 2
  ON_CHANGE_OF_DATA_IDENTIFIER = 3
  REPORT_ACTIVATED_EVENTS = 4
  START_RESPONSE_ON_EVENT = 5
  CLEAR_RESPONSE_ON_EVENT = 6
  ON_COMPARISON_OF_VALUES = 7

class LINK_CONTROL_TYPE(IntEnum):
  VERIFY_BAUDRATE_TRANSITION_WITH_FIXED_BAUDRATE = 1
  VERIFY_BAUDRATE_TRANSITION_WITH_SPECIFIC_BAUDRATE = 2
  TRANSITION_BAUDRATE = 3

class BAUD_RATE_TYPE(IntEnum):
  PC9600 = 1
  PC19200 = 2
  PC38400 = 3
  PC57600 = 4
  PC115200 = 5
  CAN125000 = 16
  CAN250000 = 17
  CAN500000 = 18
  CAN1000000 = 19

class DATA_IDENTIFIER_TYPE(IntEnum):
  BOOT_SOFTWARE_IDENTIFICATION = 0xF180
  APPLICATION_SOFTWARE_IDENTIFICATION = 0xF181
  APPLICATION_DATA_IDENTIFICATION = 0xF182
  BOOT_SOFTWARE_FINGERPRINT = 0xF183
  APPLICATION_SOFTWARE_FINGERPRINT = 0xF184
  APPLICATION_DATA_FINGERPRINT = 0xF185
  ACTIVE_DIAGNOSTIC_SESSION = 0xF186
  VEHICLE_MANUFACTURER_SPARE_PART_NUMBER = 0xF187
  VEHICLE_MANUFACTURER_ECU_SOFTWARE_NUMBER = 0xF188
  VEHICLE_MANUFACTURER_ECU_SOFTWARE_VERSION_NUMBER = 0xF189
  SYSTEM_SUPPLIER_IDENTIFIER = 0xF18A
  ECU_MANUFACTURING_DATE = 0xF18B
  ECU_SERIAL_NUMBER = 0xF18C
  SUPPORTED_FUNCTIONAL_UNITS = 0xF18D
  VEHICLE_MANUFACTURER_KIT_ASSEMBLY_PART_NUMBER = 0xF18E
  VIN = 0xF190
  VEHICLE_MANUFACTURER_ECU_HARDWARE_NUMBER = 0xF191
  SYSTEM_SUPPLIER_ECU_HARDWARE_NUMBER = 0xF192
  SYSTEM_SUPPLIER_ECU_HARDWARE_VERSION_NUMBER = 0xF193
  SYSTEM_SUPPLIER_ECU_SOFTWARE_NUMBER = 0xF194
  SYSTEM_SUPPLIER_ECU_SOFTWARE_VERSION_NUMBER = 0xF195
  EXHAUST_REGULATION_OR_TYPE_APPROVAL_NUMBER = 0xF196
  SYSTEM_NAME_OR_ENGINE_TYPE = 0xF197
  REPAIR_SHOP_CODE_OR_TESTER_SERIAL_NUMBER = 0xF198
  PROGRAMMING_DATE = 0xF199
  CALIBRATION_REPAIR_SHOP_CODE_OR_CALIBRATION_EQUIPMENT_SERIAL_NUMBER = 0xF19A
  CALIBRATION_DATE = 0xF19B
  CALIBRATION_EQUIPMENT_SOFTWARE_NUMBER = 0xF19C
  ECU_INSTALLATION_DATE = 0xF19D
  ODX_FILE = 0xF19E
  ENTITY = 0xF19F

class TRANSMISSION_MODE_TYPE(IntEnum):
  SEND_AT_SLOW_RATE = 1
  SEND_AT_MEDIUM_RATE = 2
  SEND_AT_FAST_RATE = 3
  STOP_SENDING = 4

class DYNAMIC_DEFINITION_TYPE(IntEnum):
  DEFINE_BY_IDENTIFIER = 1
  DEFINE_BY_MEMORY_ADDRESS = 2
  CLEAR_DYNAMICALLY_DEFINED_DATA_IDENTIFIER = 3

class ISOTP_FRAME_TYPE(IntEnum):
  SINGLE = 0
  FIRST = 1
  CONSECUTIVE = 2
  FLOW = 3

class DynamicSourceDefinition(NamedTuple):
  data_identifier: int
  position: int
  memory_size: int
  memory_address: int

class DTC_GROUP_TYPE(IntEnum):
  EMISSIONS = 0x000000
  ALL = 0xFFFFFF

class DTC_REPORT_TYPE(IntEnum):
  NUMBER_OF_DTC_BY_STATUS_MASK = 0x01
  DTC_BY_STATUS_MASK = 0x02
  DTC_SNAPSHOT_IDENTIFICATION = 0x03
  DTC_SNAPSHOT_RECORD_BY_DTC_NUMBER = 0x04
  DTC_SNAPSHOT_RECORD_BY_RECORD_NUMBER = 0x05
  DTC_EXTENDED_DATA_RECORD_BY_DTC_NUMBER = 0x06
  NUMBER_OF_DTC_BY_SEVERITY_MASK_RECORD = 0x07
  DTC_BY_SEVERITY_MASK_RECORD = 0x08
  SEVERITY_INFORMATION_OF_DTC = 0x09
  SUPPORTED_DTC = 0x0A
  FIRST_TEST_FAILED_DTC = 0x0B
  FIRST_CONFIRMED_DTC = 0x0C
  MOST_RECENT_TEST_FAILED_DTC = 0x0D
  MOST_RECENT_CONFIRMED_DTC = 0x0E
  MIRROR_MEMORY_DTC_BY_STATUS_MASK = 0x0F
  MIRROR_MEMORY_DTC_EXTENDED_DATA_RECORD_BY_DTC_NUMBER = 0x10
  NUMBER_OF_MIRROR_MEMORY_DTC_BY_STATUS_MASK = 0x11
  NUMBER_OF_EMISSIONS_RELATED_OBD_DTC_BY_STATUS_MASK = 0x12
  EMISSIONS_RELATED_OBD_DTC_BY_STATUS_MASK = 0x13
  DTC_FAULT_DETECTION_COUNTER = 0x14
  DTC_WITH_PERMANENT_STATUS = 0x15

class DTC_STATUS_MASK_TYPE(IntEnum):
  TEST_FAILED = 0x01
  TEST_FAILED_THIS_OPERATION_CYCLE = 0x02
  PENDING_DTC = 0x04
  CONFIRMED_DTC = 0x08
  TEST_NOT_COMPLETED_SINCE_LAST_CLEAR = 0x10
  TEST_FAILED_SINCE_LAST_CLEAR = 0x20
  TEST_NOT_COMPLETED_THIS_OPERATION_CYCLE = 0x40
  WARNING_INDICATOR_REQUESTED = 0x80
  ALL = 0xFF

class DTC_SEVERITY_MASK_TYPE(IntEnum):
  MAINTENANCE_ONLY = 0x20
  CHECK_AT_NEXT_HALT = 0x40
  CHECK_IMMEDIATELY = 0x80
  ALL = 0xE0

class CONTROL_PARAMETER_TYPE(IntEnum):
  RETURN_CONTROL_TO_ECU = 0
  RESET_TO_DEFAULT = 1
  FREEZE_CURRENT_STATE = 2
  SHORT_TERM_ADJUSTMENT = 3

class ROUTINE_CONTROL_TYPE(IntEnum):
  START = 1
  STOP = 2
  REQUEST_RESULTS = 3

class ROUTINE_IDENTIFIER_TYPE(IntEnum):
  ERASE_MEMORY = 0xFF00
  CHECK_PROGRAMMING_DEPENDENCIES = 0xFF01
  ERASE_MIRROR_MEMORY_DTCS = 0xFF02

class MessageTimeoutError(Exception):
  pass

class NegativeResponseError(Exception):
  def __init__(self, message, service_id, error_code):
    super().__init__()
    self.message = message
    self.service_id = service_id
    self.error_code = error_code

  def __str__(self):
    return self.message

class InvalidServiceIdError(Exception):
  pass

class InvalidSubFunctionError(Exception):
  pass

class InvalidSubAddressError(Exception):
  pass

_negative_response_codes = {
    0x00: 'positive response',
    0x10: 'general reject',
    0x11: 'service not supported',
    0x12: 'sub-function not supported',
    0x13: 'incorrect message length or invalid format',
    0x14: 'response too long',
    0x21: 'busy repeat request',
    0x22: 'conditions not correct',
    0x24: 'request sequence error',
    0x25: 'no response from subnet component',
    0x26: 'failure prevents execution of requested action',
    0x31: 'request out of range',
    0x33: 'security access denied',
    0x35: 'invalid key',
    0x36: 'exceed number of attempts',
    0x37: 'required time delay not expired',
    0x70: 'upload download not accepted',
    0x71: 'transfer data suspended',
    0x72: 'general programming failure',
    0x73: 'wrong block sequence counter',
    0x78: 'request correctly received - response pending',
    0x7e: 'sub-function not supported in active session',
    0x7f: 'service not supported in active session',
    0x81: 'rpm too high',
    0x82: 'rpm too low',
    0x83: 'engine is running',
    0x84: 'engine is not running',
    0x85: 'engine run time too low',
    0x86: 'temperature too high',
    0x87: 'temperature too low',
    0x88: 'vehicle speed too high',
    0x89: 'vehicle speed too low',
    0x8a: 'throttle/pedal too high',
    0x8b: 'throttle/pedal too low',
    0x8c: 'transmission not in neutral',
    0x8d: 'transmission not in gear',
    0x8f: 'brake switch(es) not closed',
    0x90: 'shifter lever not in park',
    0x91: 'torque converter clutch locked',
    0x92: 'voltage too high',
    0x93: 'voltage too low',
}

def get_dtc_num_as_str(dtc_num_bytes):
  # ISO 15031-6
  designator = {
    0b00: "P",
    0b01: "C",
    0b10: "B",
    0b11: "U",
  }
  d = designator[dtc_num_bytes[0] >> 6]
  n = bytes([dtc_num_bytes[0] & 0x3F]) + dtc_num_bytes[1:]
  return d + n.hex()

def get_dtc_status_names(status):
  result = list()
  for m in DTC_STATUS_MASK_TYPE:
    if m == DTC_STATUS_MASK_TYPE.ALL:
      continue
    if status & m.value:
      result.append(m.name)
  return result

class CanClient():
  def __init__(self, can_send: Callable[[int, bytes, int], None], can_recv: Callable[[], List[Tuple[int, int, bytes, int]]],
               tx_addr: int, rx_addr: int, bus: int, sub_addr: Optional[int] = None, debug: bool = False):
    self.tx = can_send
    self.rx = can_recv
    self.tx_addr = tx_addr
    self.rx_addr = rx_addr
    self.rx_buff: Deque[bytes] = deque()
    self.sub_addr = sub_addr
    self.bus = bus
    self.debug = debug

  def _recv_filter(self, bus: int, addr: int) -> bool:
    # handle functional addresses (switch to first addr to respond)
    if self.tx_addr == 0x7DF:
      is_response = addr >= 0x7E8 and addr <= 0x7EF
      if is_response:
        if self.debug:
          print(f"switch to physical addr {hex(addr)}")
        self.tx_addr = addr - 8
        self.rx_addr = addr
      return is_response
    if self.tx_addr == 0x18DB33F1:
      is_response = addr >= 0x18DAF100 and addr <= 0x18DAF1FF
      if is_response:
        if self.debug:
          print(f"switch to physical addr {hex(addr)}")
        self.tx_addr = 0x18DA00F1 + (addr << 8 & 0xFF00)
        self.rx_addr = addr
    return bus == self.bus and addr == self.rx_addr

  def _recv_buffer(self, drain: bool = False) -> None:
    while True:
      msgs = self.rx()
      if drain:
        if self.debug:
          print("CAN-RX: drain - {}".format(len(msgs)))
        self.rx_buff.clear()
      else:
        for rx_addr, _, rx_data, rx_bus in msgs or []:
          if self._recv_filter(rx_bus, rx_addr) and len(rx_data) > 0:
            rx_data = bytes(rx_data)  # convert bytearray to bytes

            if self.debug:
              print(f"CAN-RX: {hex(rx_addr)} - 0x{bytes.hex(rx_data)}")

            # Cut off sub addr in first byte
            if self.sub_addr is not None:
              if rx_data[0] != self.sub_addr:
                raise InvalidSubAddressError(f"isotp - rx: invalid sub-address: {rx_data[0]}, expected: {self.sub_addr}")
              rx_data = rx_data[1:]

            self.rx_buff.append(rx_data)
      # break when non-full buffer is processed
      if len(msgs) < 254:
        return

  def recv(self, drain: bool = False) -> Generator[bytes, None, None]:
    # buffer rx messages in case two response messages are received at once
    # (e.g. response pending and success/failure response)
    self._recv_buffer(drain)
    try:
      while True:
        yield self.rx_buff.popleft()
    except IndexError:
      pass  # empty

  def send(self, msgs: List[bytes], delay: float = 0) -> None:
    for i, msg in enumerate(msgs):
      if delay and i != 0:
        if self.debug:
          print(f"CAN-TX: delay - {delay}")
        time.sleep(delay)

      if self.sub_addr is not None:
        msg = bytes([self.sub_addr]) + msg

      if self.debug:
        print(f"CAN-TX: {hex(self.tx_addr)} - 0x{bytes.hex(msg)}")
      assert len(msg) <= 8

      self.tx(self.tx_addr, msg, self.bus)
      # prevent rx buffer from overflowing on large tx
      if i % 10 == 9:
        self._recv_buffer()

class IsoTpMessage():
  def __init__(self, can_client: CanClient, timeout: float = 1, single_frame_mode: bool = False, separation_time: float = 0,
               debug: bool = False, max_len: int = 8):
    self._can_client = can_client
    self.timeout = timeout
    self.single_frame_mode = single_frame_mode
    self.debug = debug
    self.max_len = max_len

    # <= 127, separation time in milliseconds
    # 0xF1 to 0xF9 UF, 100 to 900 microseconds
    if 1e-4 <= separation_time <= 9e-4:
      offset = int(round(separation_time, 4) * 1e4) - 1
      separation_time = 0xF1 + offset
    elif 0 <= separation_time <= 0.127:
      separation_time = round(separation_time * 1000)
    else:
      raise Exception("Separation time not in range")

    self.flow_control_msg = bytes([
      0x30,  # flow control
      0x01 if self.single_frame_mode else 0x00,  # block size
      separation_time,
    ]).ljust(self.max_len, b"\x00")

  def send(self, dat: bytes, setup_only: bool = False) -> None:
    # throw away any stale data
    self._can_client.recv(drain=True)

    self.tx_dat = dat
    self.tx_len = len(dat)
    self.tx_idx = 0
    self.tx_done = False

    self.rx_dat = b""
    self.rx_len = 0
    self.rx_idx = 0
    self.rx_done = False

    if self.debug and not setup_only:
      print(f"ISO-TP: REQUEST - {hex(self._can_client.tx_addr)} 0x{bytes.hex(self.tx_dat)}")
    self._tx_first_frame(setup_only=setup_only)

  def _tx_first_frame(self, setup_only: bool = False) -> None:
    if self.tx_len < self.max_len:
      # single frame (send all bytes)
      if self.debug and not setup_only:
        print(f"ISO-TP: TX - single frame - {hex(self._can_client.tx_addr)}")
      msg = (bytes([self.tx_len]) + self.tx_dat).ljust(self.max_len, b"\x00")
      self.tx_done = True
    else:
      # first frame (send first 6 bytes)
      if self.debug and not setup_only:
        print(f"ISO-TP: TX - first frame - {hex(self._can_client.tx_addr)}")
      msg = (struct.pack("!H", 0x1000 | self.tx_len) + self.tx_dat[:self.max_len - 2]).ljust(self.max_len - 2, b"\x00")
    if not setup_only:
      self._can_client.send([msg])

  def recv(self, timeout=None) -> Tuple[Optional[bytes], bool]:
    if timeout is None:
      timeout = self.timeout

    start_time = time.monotonic()
    rx_in_progress = False
    try:
      while True:
        for msg in self._can_client.recv():
          frame_type = self._isotp_rx_next(msg)
          start_time = time.monotonic()
          rx_in_progress = frame_type == ISOTP_FRAME_TYPE.CONSECUTIVE
          if self.tx_done and self.rx_done:
            return self.rx_dat, False
        # no timeout indicates non-blocking
        if timeout == 0:
          return None, rx_in_progress
        if time.monotonic() - start_time > timeout:
          raise MessageTimeoutError("timeout waiting for response")
    finally:
      if self.debug and self.rx_dat:
        print(f"ISO-TP: RESPONSE - {hex(self._can_client.rx_addr)} 0x{bytes.hex(self.rx_dat)}")

  def _isotp_rx_next(self, rx_data: bytes) -> ISOTP_FRAME_TYPE:
    # TODO: Handle CAN frame data optimization, which is allowed with some frame types
    # # ISO 15765-2 specifies an eight byte CAN frame for ISO-TP communication
    # assert len(rx_data) == self.max_len, f"isotp - rx: invalid CAN frame length: {len(rx_data)}"

    if rx_data[0] >> 4 == ISOTP_FRAME_TYPE.SINGLE:
      self.rx_len = rx_data[0] & 0x0F
      assert self.rx_len < self.max_len, f"isotp - rx: invalid single frame length: {self.rx_len}"
      self.rx_dat = rx_data[1:1 + self.rx_len]
      self.rx_idx = 0
      self.rx_done = True
      if self.debug:
        print(f"ISO-TP: RX - single frame - {hex(self._can_client.rx_addr)} idx={self.rx_idx} done={self.rx_done}")
      return ISOTP_FRAME_TYPE.SINGLE

    elif rx_data[0] >> 4 == ISOTP_FRAME_TYPE.FIRST:
      self.rx_len = ((rx_data[0] & 0x0F) << 8) + rx_data[1]
      assert self.max_len <= self.rx_len, f"isotp - rx: invalid first frame length: {self.rx_len}"
      self.rx_dat = rx_data[2:]
      self.rx_idx = 0
      self.rx_done = False
      if self.debug:
        print(f"ISO-TP: RX - first frame - {hex(self._can_client.rx_addr)} idx={self.rx_idx} done={self.rx_done}")
      if self.debug:
        print(f"ISO-TP: TX - flow control continue - {hex(self._can_client.tx_addr)}")
      # send flow control message
      self._can_client.send([self.flow_control_msg])
      return ISOTP_FRAME_TYPE.FIRST

    elif rx_data[0] >> 4 == ISOTP_FRAME_TYPE.CONSECUTIVE:
      assert not self.rx_done, "isotp - rx: consecutive frame with no active frame"
      self.rx_idx += 1
      assert self.rx_idx & 0xF == rx_data[0] & 0xF, "isotp - rx: invalid consecutive frame index"
      rx_size = self.rx_len - len(self.rx_dat)
      self.rx_dat += rx_data[1:1 + rx_size]
      if self.rx_len == len(self.rx_dat):
        self.rx_done = True
      elif self.single_frame_mode:
        # notify ECU to send next frame
        self._can_client.send([self.flow_control_msg])
      if self.debug:
        print(f"ISO-TP: RX - consecutive frame - {hex(self._can_client.rx_addr)} idx={self.rx_idx} done={self.rx_done}")
      return ISOTP_FRAME_TYPE.CONSECUTIVE

    elif rx_data[0] >> 4 == ISOTP_FRAME_TYPE.FLOW:
      assert not self.tx_done, "isotp - rx: flow control with no active frame"
      assert rx_data[0] != 0x32, "isotp - rx: flow-control overflow/abort"
      assert rx_data[0] == 0x30 or rx_data[0] == 0x31, "isotp - rx: flow-control transfer state indicator invalid"
      if rx_data[0] == 0x30:
        if self.debug:
          print(f"ISO-TP: RX - flow control continue - {hex(self._can_client.tx_addr)}")
        delay_ts = rx_data[2] & 0x7F
        # scale is 1 milliseconds if first bit == 0, 100 micro seconds if first bit == 1
        delay_div = 1000. if rx_data[2] & 0x80 == 0 else 10000.
        delay_sec = delay_ts / delay_div

        # first frame = 6 bytes, each consecutive frame = 7 bytes
        num_bytes = self.max_len - 1
        start = self.max_len - 2 + self.tx_idx * num_bytes
        count = rx_data[1]
        end = start + count * num_bytes if count > 0 else self.tx_len
        tx_msgs = []
        for i in range(start, end, num_bytes):
          self.tx_idx += 1
          # consecutive tx messages
          msg = (bytes([0x20 | (self.tx_idx & 0xF)]) + self.tx_dat[i:i + num_bytes]).ljust(self.max_len, b"\x00")
          tx_msgs.append(msg)
        # send consecutive tx messages
        self._can_client.send(tx_msgs, delay=delay_sec)
        if end >= self.tx_len:
          self.tx_done = True
        if self.debug:
          print(f"ISO-TP: TX - consecutive frame - {hex(self._can_client.tx_addr)} idx={self.tx_idx} done={self.tx_done}")
      elif rx_data[0] == 0x31:
        # wait (do nothing until next flow control message)
        if self.debug:
          print(f"ISO-TP: TX - flow control wait - {hex(self._can_client.tx_addr)}")
      return ISOTP_FRAME_TYPE.FLOW

    # 4-15 - reserved
    else:
      raise Exception(f"isotp - rx: invalid frame type: {rx_data[0] >> 4}")


FUNCTIONAL_ADDRS = [0x7DF, 0x18DB33F1]


def get_rx_addr_for_tx_addr(tx_addr, rx_offset=0x8):
  if tx_addr in FUNCTIONAL_ADDRS:
    return None

  if tx_addr < 0xFFF8:
    # pseudo-standard 11 bit response addr (add 8) works for most manufacturers
    # allow override; some manufacturers use other offsets for non-OBD2 access
    return tx_addr + rx_offset

  if tx_addr > 0x10000000 and tx_addr < 0xFFFFFFFF:
    # standard 29 bit response addr (flip last two bytes)
    return (tx_addr & 0xFFFF0000) + (tx_addr << 8 & 0xFF00) + (tx_addr >> 8 & 0xFF)

  raise ValueError("invalid tx_addr: {}".format(tx_addr))


class UdsClient():
  def __init__(self, panda, tx_addr: int, rx_addr: Optional[int] = None, bus: int = 0, sub_addr: Optional[int] = None, timeout: float = 1,
               debug: bool = False, tx_timeout: float = 1, response_pending_timeout: float = 10):
    self.bus = bus
    self.tx_addr = tx_addr
    self.rx_addr = rx_addr if rx_addr is not None else get_rx_addr_for_tx_addr(tx_addr)
    self.sub_addr = sub_addr
    self.timeout = timeout
    self.debug = debug
    can_send_with_timeout = partial(panda.can_send, timeout=int(tx_timeout*1000))
    self._can_client = CanClient(can_send_with_timeout, panda.can_recv, self.tx_addr, self.rx_addr, self.bus, self.sub_addr, debug=self.debug)
    self.response_pending_timeout = response_pending_timeout

  # generic uds request
  def _uds_request(self, service_type: SERVICE_TYPE, subfunction: Optional[int] = None, data: Optional[bytes] = None) -> bytes:
    req = bytes([service_type])
    if subfunction is not None:
      req += bytes([subfunction])
    if data is not None:
      req += data

    # send request, wait for response
    max_len = 8 if self.sub_addr is None else 7
    isotp_msg = IsoTpMessage(self._can_client, timeout=self.timeout, debug=self.debug, max_len=max_len)
    isotp_msg.send(req)
    response_pending = False
    while True:
      timeout = self.response_pending_timeout if response_pending else self.timeout
      resp, _ = isotp_msg.recv(timeout)

      if resp is None:
        continue

      response_pending = False
      resp_sid = resp[0] if len(resp) > 0 else None

      # negative response
      if resp_sid == 0x7F:
        service_id = resp[1] if len(resp) > 1 else -1
        try:
          service_desc = SERVICE_TYPE(service_id).name
        except BaseException:
          service_desc = 'NON_STANDARD_SERVICE'
        error_code = resp[2] if len(resp) > 2 else -1
        try:
          error_desc = _negative_response_codes[error_code]
        except BaseException:
          error_desc = resp[3:].hex()
        # wait for another message if response pending
        if error_code == 0x78:
          response_pending = True
          if self.debug:
            print("UDS-RX: response pending")
          continue
        raise NegativeResponseError('{} - {}'.format(service_desc, error_desc), service_id, error_code)

      # positive response
      if service_type + 0x40 != resp_sid:
        resp_sid_hex = hex(resp_sid) if resp_sid is not None else None
        raise InvalidServiceIdError('invalid response service id: {}'.format(resp_sid_hex))

      if subfunction is not None:
        resp_sfn = resp[1] if len(resp) > 1 else None
        if subfunction != resp_sfn:
          resp_sfn_hex = hex(resp_sfn) if resp_sfn is not None else None
          raise InvalidSubFunctionError(f'invalid response subfunction: {resp_sfn_hex}')

      # return data (exclude service id and sub-function id)
      return resp[(1 if subfunction is None else 2):]

  # services
  def diagnostic_session_control(self, session_type: SESSION_TYPE):
    self._uds_request(SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL, subfunction=session_type)

  def ecu_reset(self, reset_type: RESET_TYPE):
    resp = self._uds_request(SERVICE_TYPE.ECU_RESET, subfunction=reset_type)
    power_down_time = None
    if reset_type == RESET_TYPE.ENABLE_RAPID_POWER_SHUTDOWN:
      power_down_time = resp[0]
      return power_down_time

  def security_access(self, access_type: ACCESS_TYPE, security_key: bytes = b'', data_record: bytes = b''):
    request_seed = access_type % 2 != 0
    if request_seed and len(security_key) != 0:
      raise ValueError('security_key not allowed')
    if not request_seed and len(security_key) == 0:
      raise ValueError('security_key is missing')
    if not request_seed and len(data_record) != 0:
      raise ValueError('data_record not allowed')
    data = security_key + data_record
    resp = self._uds_request(SERVICE_TYPE.SECURITY_ACCESS, subfunction=access_type, data=data)
    if request_seed:
      security_seed = resp
      return security_seed

  def communication_control(self, control_type: CONTROL_TYPE, message_type: MESSAGE_TYPE):
    data = bytes([message_type])
    self._uds_request(SERVICE_TYPE.COMMUNICATION_CONTROL, subfunction=control_type, data=data)

  def tester_present(self, ):
    self._uds_request(SERVICE_TYPE.TESTER_PRESENT, subfunction=0x00)

  def access_timing_parameter(self, timing_parameter_type: TIMING_PARAMETER_TYPE, parameter_values: Optional[bytes] = None):
    write_custom_values = timing_parameter_type == TIMING_PARAMETER_TYPE.SET_TO_GIVEN_VALUES
    read_values = (timing_parameter_type == TIMING_PARAMETER_TYPE.READ_CURRENTLY_ACTIVE or
                   timing_parameter_type == TIMING_PARAMETER_TYPE.READ_EXTENDED_SET)
    if not write_custom_values and parameter_values is not None:
      raise ValueError('parameter_values not allowed')
    if write_custom_values and parameter_values is None:
      raise ValueError('parameter_values is missing')
    resp = self._uds_request(SERVICE_TYPE.ACCESS_TIMING_PARAMETER, subfunction=timing_parameter_type, data=parameter_values)
    if read_values:
      # TODO: parse response into values?
      parameter_values = resp
      return parameter_values

  def secured_data_transmission(self, data: bytes):
    # TODO: split data into multiple input parameters?
    resp = self._uds_request(SERVICE_TYPE.SECURED_DATA_TRANSMISSION, subfunction=None, data=data)
    # TODO: parse response into multiple output values?
    return resp

  def control_dtc_setting(self, dtc_setting_type: DTC_SETTING_TYPE):
    self._uds_request(SERVICE_TYPE.CONTROL_DTC_SETTING, subfunction=dtc_setting_type)

  def response_on_event(self, response_event_type: RESPONSE_EVENT_TYPE, store_event: bool, window_time: int,
                        event_type_record: int, service_response_record: int):
    if store_event:
      response_event_type |= 0x20  # type: ignore
    # TODO: split record parameters into arrays
    data = bytes([window_time, event_type_record, service_response_record])
    resp = self._uds_request(SERVICE_TYPE.RESPONSE_ON_EVENT, subfunction=response_event_type, data=data)

    if response_event_type == RESPONSE_EVENT_TYPE.REPORT_ACTIVATED_EVENTS:
      return {
        "num_of_activated_events": resp[0],
        "data": resp[1:],  # TODO: parse the reset of response
      }

    return {
      "num_of_identified_events": resp[0],
      "event_window_time": resp[1],
      "data": resp[2:],  # TODO: parse the reset of response
    }

  def link_control(self, link_control_type: LINK_CONTROL_TYPE, baud_rate_type: Optional[BAUD_RATE_TYPE] = None):
    data: Optional[bytes]

    if link_control_type == LINK_CONTROL_TYPE.VERIFY_BAUDRATE_TRANSITION_WITH_FIXED_BAUDRATE:
      # baud_rate_type = BAUD_RATE_TYPE
      data = bytes([cast(int, baud_rate_type)])
    elif link_control_type == LINK_CONTROL_TYPE.VERIFY_BAUDRATE_TRANSITION_WITH_SPECIFIC_BAUDRATE:
      # baud_rate_type = custom value (3 bytes big-endian)
      data = struct.pack('!I', baud_rate_type)[1:]
    else:
      data = None
    self._uds_request(SERVICE_TYPE.LINK_CONTROL, subfunction=link_control_type, data=data)

  def read_data_by_identifier(self, data_identifier_type: DATA_IDENTIFIER_TYPE):
    # TODO: support list of identifiers
    data = struct.pack('!H', data_identifier_type)
    resp = self._uds_request(SERVICE_TYPE.READ_DATA_BY_IDENTIFIER, subfunction=None, data=data)
    resp_id = struct.unpack('!H', resp[0:2])[0] if len(resp) >= 2 else None
    if resp_id != data_identifier_type:
      raise ValueError('invalid response data identifier: {} expected: {}'.format(hex(resp_id), hex(data_identifier_type)))
    return resp[2:]

  def read_memory_by_address(self, memory_address: int, memory_size: int, memory_address_bytes: int = 4, memory_size_bytes: int = 1):
    if memory_address_bytes < 1 or memory_address_bytes > 4:
      raise ValueError('invalid memory_address_bytes: {}'.format(memory_address_bytes))
    if memory_size_bytes < 1 or memory_size_bytes > 4:
      raise ValueError('invalid memory_size_bytes: {}'.format(memory_size_bytes))
    data = bytes([memory_size_bytes << 4 | memory_address_bytes])

    if memory_address >= 1 << (memory_address_bytes * 8):
      raise ValueError('invalid memory_address: {}'.format(memory_address))
    data += struct.pack('!I', memory_address)[4 - memory_address_bytes:]
    if memory_size >= 1 << (memory_size_bytes * 8):
      raise ValueError('invalid memory_size: {}'.format(memory_size))
    data += struct.pack('!I', memory_size)[4 - memory_size_bytes:]

    resp = self._uds_request(SERVICE_TYPE.READ_MEMORY_BY_ADDRESS, subfunction=None, data=data)
    return resp

  def read_scaling_data_by_identifier(self, data_identifier_type: DATA_IDENTIFIER_TYPE):
    data = struct.pack('!H', data_identifier_type)
    resp = self._uds_request(SERVICE_TYPE.READ_SCALING_DATA_BY_IDENTIFIER, subfunction=None, data=data)
    resp_id = struct.unpack('!H', resp[0:2])[0] if len(resp) >= 2 else None
    if resp_id != data_identifier_type:
      raise ValueError('invalid response data identifier: {}'.format(hex(resp_id)))
    return resp[2:]  # TODO: parse the response

  def read_data_by_periodic_identifier(self, transmission_mode_type: TRANSMISSION_MODE_TYPE, periodic_data_identifier: int):
    # TODO: support list of identifiers
    data = bytes([transmission_mode_type, periodic_data_identifier])
    self._uds_request(SERVICE_TYPE.READ_DATA_BY_PERIODIC_IDENTIFIER, subfunction=None, data=data)

  def dynamically_define_data_identifier(self, dynamic_definition_type: DYNAMIC_DEFINITION_TYPE, dynamic_data_identifier: int,
                                         source_definitions: List[DynamicSourceDefinition], memory_address_bytes: int = 4, memory_size_bytes: int = 1):
    if memory_address_bytes < 1 or memory_address_bytes > 4:
      raise ValueError('invalid memory_address_bytes: {}'.format(memory_address_bytes))
    if memory_size_bytes < 1 or memory_size_bytes > 4:
      raise ValueError('invalid memory_size_bytes: {}'.format(memory_size_bytes))

    data = struct.pack('!H', dynamic_data_identifier)
    if dynamic_definition_type == DYNAMIC_DEFINITION_TYPE.DEFINE_BY_IDENTIFIER:
      for s in source_definitions:
        data += struct.pack('!H', s.data_identifier) + bytes([s.position, s.memory_size])
    elif dynamic_definition_type == DYNAMIC_DEFINITION_TYPE.DEFINE_BY_MEMORY_ADDRESS:
      data += bytes([memory_size_bytes << 4 | memory_address_bytes])
      for s in source_definitions:
        if s.memory_address >= 1 << (memory_address_bytes * 8):
          raise ValueError('invalid memory_address: {}'.format(s.memory_address))
        data += struct.pack('!I', s.memory_address)[4 - memory_address_bytes:]
        if s.memory_size >= 1 << (memory_size_bytes * 8):
          raise ValueError('invalid memory_size: {}'.format(s.memory_size))
        data += struct.pack('!I', s.memory_size)[4 - memory_size_bytes:]
    elif dynamic_definition_type == DYNAMIC_DEFINITION_TYPE.CLEAR_DYNAMICALLY_DEFINED_DATA_IDENTIFIER:
      pass
    else:
      raise ValueError('invalid dynamic identifier type: {}'.format(hex(dynamic_definition_type)))
    self._uds_request(SERVICE_TYPE.DYNAMICALLY_DEFINE_DATA_IDENTIFIER, subfunction=dynamic_definition_type, data=data)

  def write_data_by_identifier(self, data_identifier_type: DATA_IDENTIFIER_TYPE, data_record: bytes):
    data = struct.pack('!H', data_identifier_type) + data_record
    resp = self._uds_request(SERVICE_TYPE.WRITE_DATA_BY_IDENTIFIER, subfunction=None, data=data)
    resp_id = struct.unpack('!H', resp[0:2])[0] if len(resp) >= 2 else None
    if resp_id != data_identifier_type:
      raise ValueError('invalid response data identifier: {}'.format(hex(resp_id)))

  def write_memory_by_address(self, memory_address: int, memory_size: int, data_record: bytes, memory_address_bytes: int = 4, memory_size_bytes: int = 1):
    if memory_address_bytes < 1 or memory_address_bytes > 4:
      raise ValueError('invalid memory_address_bytes: {}'.format(memory_address_bytes))
    if memory_size_bytes < 1 or memory_size_bytes > 4:
      raise ValueError('invalid memory_size_bytes: {}'.format(memory_size_bytes))
    data = bytes([memory_size_bytes << 4 | memory_address_bytes])

    if memory_address >= 1 << (memory_address_bytes * 8):
      raise ValueError('invalid memory_address: {}'.format(memory_address))
    data += struct.pack('!I', memory_address)[4 - memory_address_bytes:]
    if memory_size >= 1 << (memory_size_bytes * 8):
      raise ValueError('invalid memory_size: {}'.format(memory_size))
    data += struct.pack('!I', memory_size)[4 - memory_size_bytes:]

    data += data_record
    self._uds_request(SERVICE_TYPE.WRITE_MEMORY_BY_ADDRESS, subfunction=0x00, data=data)

  def clear_diagnostic_information(self, dtc_group_type: DTC_GROUP_TYPE):
    data = struct.pack('!I', dtc_group_type)[1:]  # 3 bytes
    self._uds_request(SERVICE_TYPE.CLEAR_DIAGNOSTIC_INFORMATION, subfunction=None, data=data)

  def read_dtc_information(self, dtc_report_type: DTC_REPORT_TYPE, dtc_status_mask_type: DTC_STATUS_MASK_TYPE = DTC_STATUS_MASK_TYPE.ALL,
                           dtc_severity_mask_type: DTC_SEVERITY_MASK_TYPE = DTC_SEVERITY_MASK_TYPE.ALL, dtc_mask_record: int = 0xFFFFFF,
                           dtc_snapshot_record_num: int = 0xFF, dtc_extended_record_num: int = 0xFF):
    data = b''
    # dtc_status_mask_type
    if dtc_report_type == DTC_REPORT_TYPE.NUMBER_OF_DTC_BY_STATUS_MASK or \
       dtc_report_type == DTC_REPORT_TYPE.DTC_BY_STATUS_MASK or \
       dtc_report_type == DTC_REPORT_TYPE.MIRROR_MEMORY_DTC_BY_STATUS_MASK or \
       dtc_report_type == DTC_REPORT_TYPE.NUMBER_OF_MIRROR_MEMORY_DTC_BY_STATUS_MASK or \
       dtc_report_type == DTC_REPORT_TYPE.NUMBER_OF_EMISSIONS_RELATED_OBD_DTC_BY_STATUS_MASK or \
       dtc_report_type == DTC_REPORT_TYPE.EMISSIONS_RELATED_OBD_DTC_BY_STATUS_MASK:
       data += bytes([dtc_status_mask_type])
    # dtc_mask_record
    if dtc_report_type == DTC_REPORT_TYPE.DTC_SNAPSHOT_IDENTIFICATION or \
       dtc_report_type == DTC_REPORT_TYPE.DTC_SNAPSHOT_RECORD_BY_DTC_NUMBER or \
       dtc_report_type == DTC_REPORT_TYPE.DTC_EXTENDED_DATA_RECORD_BY_DTC_NUMBER or \
       dtc_report_type == DTC_REPORT_TYPE.MIRROR_MEMORY_DTC_EXTENDED_DATA_RECORD_BY_DTC_NUMBER or \
       dtc_report_type == DTC_REPORT_TYPE.SEVERITY_INFORMATION_OF_DTC:
       data += struct.pack('!I', dtc_mask_record)[1:]  # 3 bytes
    # dtc_snapshot_record_num
    if dtc_report_type == DTC_REPORT_TYPE.DTC_SNAPSHOT_IDENTIFICATION or \
       dtc_report_type == DTC_REPORT_TYPE.DTC_SNAPSHOT_RECORD_BY_DTC_NUMBER or \
       dtc_report_type == DTC_REPORT_TYPE.DTC_SNAPSHOT_RECORD_BY_RECORD_NUMBER:
       data += bytes([dtc_snapshot_record_num])
    # dtc_extended_record_num
    if dtc_report_type == DTC_REPORT_TYPE.DTC_EXTENDED_DATA_RECORD_BY_DTC_NUMBER or \
       dtc_report_type == DTC_REPORT_TYPE.MIRROR_MEMORY_DTC_EXTENDED_DATA_RECORD_BY_DTC_NUMBER:
       data += bytes([dtc_extended_record_num])
    # dtc_severity_mask_type
    if dtc_report_type == DTC_REPORT_TYPE.NUMBER_OF_DTC_BY_SEVERITY_MASK_RECORD or \
       dtc_report_type == DTC_REPORT_TYPE.DTC_BY_SEVERITY_MASK_RECORD:
       data += bytes([dtc_severity_mask_type, dtc_status_mask_type])

    resp = self._uds_request(SERVICE_TYPE.READ_DTC_INFORMATION, subfunction=dtc_report_type, data=data)

    # TODO: parse response
    return resp

  def input_output_control_by_identifier(self, data_identifier_type: DATA_IDENTIFIER_TYPE, control_parameter_type: CONTROL_PARAMETER_TYPE,
                                         control_option_record: bytes = b'', control_enable_mask_record: bytes = b''):
    data = struct.pack('!H', data_identifier_type) + bytes([control_parameter_type]) + control_option_record + control_enable_mask_record
    resp = self._uds_request(SERVICE_TYPE.INPUT_OUTPUT_CONTROL_BY_IDENTIFIER, subfunction=None, data=data)
    resp_id = struct.unpack('!H', resp[0:2])[0] if len(resp) >= 2 else None
    if resp_id != data_identifier_type:
      raise ValueError('invalid response data identifier: {}'.format(hex(resp_id)))
    return resp[2:]

  def routine_control(self, routine_control_type: ROUTINE_CONTROL_TYPE, routine_identifier_type: ROUTINE_IDENTIFIER_TYPE, routine_option_record: bytes = b''):
    data = struct.pack('!H', routine_identifier_type) + routine_option_record
    resp = self._uds_request(SERVICE_TYPE.ROUTINE_CONTROL, subfunction=routine_control_type, data=data)
    resp_id = struct.unpack('!H', resp[0:2])[0] if len(resp) >= 2 else None
    if resp_id != routine_identifier_type:
      raise ValueError('invalid response routine identifier: {}'.format(hex(resp_id)))
    return resp[2:]

  def request_download(self, memory_address: int, memory_size: int, memory_address_bytes: int = 4, memory_size_bytes: int = 4, data_format: int = 0x00):
    data = bytes([data_format])

    if memory_address_bytes < 1 or memory_address_bytes > 4:
      raise ValueError('invalid memory_address_bytes: {}'.format(memory_address_bytes))
    if memory_size_bytes < 1 or memory_size_bytes > 4:
      raise ValueError('invalid memory_size_bytes: {}'.format(memory_size_bytes))
    data += bytes([memory_size_bytes << 4 | memory_address_bytes])

    if memory_address >= 1 << (memory_address_bytes * 8):
      raise ValueError('invalid memory_address: {}'.format(memory_address))
    data += struct.pack('!I', memory_address)[4 - memory_address_bytes:]
    if memory_size >= 1 << (memory_size_bytes * 8):
      raise ValueError('invalid memory_size: {}'.format(memory_size))
    data += struct.pack('!I', memory_size)[4 - memory_size_bytes:]

    resp = self._uds_request(SERVICE_TYPE.REQUEST_DOWNLOAD, subfunction=None, data=data)
    max_num_bytes_len = resp[0] >> 4 if len(resp) > 0 else 0
    if max_num_bytes_len >= 1 and max_num_bytes_len <= 4:
      max_num_bytes = struct.unpack('!I', (b"\x00" * (4 - max_num_bytes_len)) + resp[1:max_num_bytes_len + 1])[0]
    else:
      raise ValueError('invalid max_num_bytes_len: {}'.format(max_num_bytes_len))

    return max_num_bytes  # max number of bytes per transfer data request

  def request_upload(self, memory_address: int, memory_size: int, memory_address_bytes: int = 4, memory_size_bytes: int = 4, data_format: int = 0x00):
    data = bytes([data_format])

    if memory_address_bytes < 1 or memory_address_bytes > 4:
      raise ValueError('invalid memory_address_bytes: {}'.format(memory_address_bytes))
    if memory_size_bytes < 1 or memory_size_bytes > 4:
      raise ValueError('invalid memory_size_bytes: {}'.format(memory_size_bytes))
    data += bytes([memory_size_bytes << 4 | memory_address_bytes])

    if memory_address >= 1 << (memory_address_bytes * 8):
      raise ValueError('invalid memory_address: {}'.format(memory_address))
    data += struct.pack('!I', memory_address)[4 - memory_address_bytes:]
    if memory_size >= 1 << (memory_size_bytes * 8):
      raise ValueError('invalid memory_size: {}'.format(memory_size))
    data += struct.pack('!I', memory_size)[4 - memory_size_bytes:]

    resp = self._uds_request(SERVICE_TYPE.REQUEST_UPLOAD, subfunction=None, data=data)
    max_num_bytes_len = resp[0] >> 4 if len(resp) > 0 else 0
    if max_num_bytes_len >= 1 and max_num_bytes_len <= 4:
      max_num_bytes = struct.unpack('!I', (b"\x00" * (4 - max_num_bytes_len)) + resp[1:max_num_bytes_len + 1])[0]
    else:
      raise ValueError('invalid max_num_bytes_len: {}'.format(max_num_bytes_len))

    return max_num_bytes  # max number of bytes per transfer data request

  def transfer_data(self, block_sequence_count: int, data: bytes = b''):
    data = bytes([block_sequence_count]) + data
    resp = self._uds_request(SERVICE_TYPE.TRANSFER_DATA, subfunction=None, data=data)
    resp_id = resp[0] if len(resp) > 0 else None
    if resp_id != block_sequence_count:
      raise ValueError('invalid block_sequence_count: {}'.format(resp_id))
    return resp[1:]

  def request_transfer_exit(self):
    self._uds_request(SERVICE_TYPE.REQUEST_TRANSFER_EXIT, subfunction=None)
