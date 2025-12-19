import sys
import time
import struct
from enum import IntEnum, Enum
from dataclasses import dataclass


@dataclass
class ExchangeStationIdsReturn:
  id_length: int
  data_type: int
  available: int
  protected: int


@dataclass
class GetDaqListSizeReturn:
  list_size: int
  first_pid: int


@dataclass
class GetSessionStatusReturn:
  status: int
  info: int | None


@dataclass
class DiagnosticServiceReturn:
  length: int
  type: int


@dataclass
class ActionServiceReturn:
  length: int
  type: int


class COMMAND_CODE(IntEnum):
  CONNECT = 0x01
  SET_MTA = 0x02
  DNLOAD = 0x03
  UPLOAD = 0x04
  TEST = 0x05
  START_STOP = 0x06
  DISCONNECT = 0x07
  START_STOP_ALL = 0x08
  GET_ACTIVE_CAL_PAGE = 0x09
  SET_S_STATUS = 0x0C
  GET_S_STATUS = 0x0D
  BUILD_CHKSUM = 0x0E
  SHORT_UP = 0x0F
  CLEAR_MEMORY = 0x10
  SELECT_CAL_PAGE = 0x11
  GET_SEED = 0x12
  UNLOCK = 0x13
  GET_DAQ_SIZE = 0x14
  SET_DAQ_PTR = 0x15
  WRITE_DAQ = 0x16
  EXCHANGE_ID = 0x17
  PROGRAM = 0x18
  MOVE = 0x19
  GET_CCP_VERSION = 0x1B
  DIAG_SERVICE = 0x20
  ACTION_SERVICE = 0x21
  PROGRAM_6 = 0x22
  DNLOAD_6 = 0x23


COMMAND_RETURN_CODES = {
  0x00: "acknowledge / no error",
  0x01: "DAQ processor overload",
  0x10: "command processor busy",
  0x11: "DAQ processor busy",
  0x12: "internal timeout",
  0x18: "key request",
  0x19: "session status request",
  0x20: "cold start request",
  0x21: "cal. data init. request",
  0x22: "DAQ list init. request",
  0x23: "code update request",
  0x30: "unknown command",
  0x31: "command syntax",
  0x32: "parameter(s) out of range",
  0x33: "access denied",
  0x34: "overload",
  0x35: "access locked",
  0x36: "resource/function not available",
}


class BYTE_ORDER(Enum):
  LITTLE_ENDIAN = '<'
  BIG_ENDIAN = '>'


class CommandTimeoutError(Exception):
  pass


class CommandCounterError(Exception):
  pass


class CommandResponseError(Exception):
  def __init__(self, message, return_code):
    super().__init__()
    self.message = message
    self.return_code = return_code

  def __str__(self):
    return self.message


class CcpClient:
  def __init__(self, panda, tx_addr: int, rx_addr: int, bus: int=0, byte_order: BYTE_ORDER=BYTE_ORDER.BIG_ENDIAN, debug=False):
    self.tx_addr = tx_addr
    self.rx_addr = rx_addr
    self.can_bus = bus
    self.byte_order = byte_order
    self.debug = debug
    self._panda = panda
    self._command_counter = -1

  def _send_cro(self, cmd: int, dat: bytes = b"") -> None:
    self._command_counter = (self._command_counter + 1) & 0xFF
    tx_data = (bytes([cmd, self._command_counter]) + dat).ljust(8, b"\x00")
    if self.debug:
      print(f"CAN-TX: {hex(self.tx_addr)} - 0x{bytes.hex(tx_data)}")
    assert len(tx_data) == 8, "data is not 8 bytes"
    self._panda.can_clear(self.can_bus)
    self._panda.can_clear(0xFFFF)
    self._panda.can_send(self.tx_addr, tx_data, self.can_bus)

  def _recv_dto(self, timeout: float) -> bytes:
    start_time = time.time()
    while time.time() - start_time < timeout:
      msgs = self._panda.can_recv() or []
      if len(msgs) >= 256:
        print("CAN RX buffer overflow!!!", file=sys.stderr)
      for rx_addr, rx_data_bytearray, rx_bus in msgs:
        if rx_bus == self.can_bus and rx_addr == self.rx_addr:
          rx_data = bytes(rx_data_bytearray)
          if self.debug:
            print(f"CAN-RX: {hex(rx_addr)} - 0x{bytes.hex(rx_data)}")
          assert len(rx_data) == 8, f"message length not 8: {len(rx_data)}"

          pid = rx_data[0]
          if pid == 0xFF or pid == 0xFE:
            err = rx_data[1]
            err_desc = COMMAND_RETURN_CODES.get(err, "unknown error")
            ctr = rx_data[2]
            dat = rx_data[3:]

            if pid == 0xFF and self._command_counter != ctr:
              raise CommandCounterError(f"counter invalid: {ctr} != {self._command_counter}")

            if err >= 0x10 and err <= 0x12:
              if self.debug:
                print(f"CCP-WAIT: {hex(err)} - {err_desc}")
              start_time = time.time()
              continue

            if err >= 0x30:
              raise CommandResponseError(f"{hex(err)} - {err_desc}", err)
          else:
            dat = rx_data[1:]

          return dat
      time.sleep(0.001)

    raise CommandTimeoutError("timeout waiting for response")

  # commands
  def connect(self, station_addr: int) -> None:
    if station_addr > 65535:
      raise ValueError("station address must be less than 65536")
    # NOTE: station address is always little endian
    self._send_cro(COMMAND_CODE.CONNECT, struct.pack("<H", station_addr))
    self._recv_dto(0.025)

  def exchange_station_ids(self, device_id_info: bytes = b"") -> ExchangeStationIdsReturn:
    self._send_cro(COMMAND_CODE.EXCHANGE_ID, device_id_info)
    resp = self._recv_dto(0.025)
    return ExchangeStationIdsReturn(id_length=resp[0], data_type=resp[1], available=resp[2], protected=resp[3])

  def get_seed(self, resource_mask: int) -> bytes:
    if resource_mask > 255:
      raise ValueError("resource mask must be less than 256")
    self._send_cro(COMMAND_CODE.GET_SEED, bytes([resource_mask]))
    resp = self._recv_dto(0.025)
    # protected = resp[0] == 0
    seed = resp[1:]
    return seed

  def unlock(self, key: bytes) -> int:
    if len(key) > 6:
      raise ValueError("max key size is 6 bytes")
    self._send_cro(COMMAND_CODE.UNLOCK, key)
    resp = self._recv_dto(0.025)
    status = resp[0]
    return status

  def set_memory_transfer_address(self, mta_num: int, addr_ext: int, addr: int) -> None:
    if mta_num > 255:
      raise ValueError("MTA number must be less than 256")
    if addr_ext > 255:
      raise ValueError("address extension must be less than 256")
    self._send_cro(COMMAND_CODE.SET_MTA, bytes([mta_num, addr_ext]) + struct.pack(f"{self.byte_order.value}I", addr))
    self._recv_dto(0.025)

  def download(self, data: bytes) -> int:
    if len(data) > 5:
      raise ValueError("max data size is 5 bytes")
    self._send_cro(COMMAND_CODE.DNLOAD, bytes([len(data)]) + data)
    resp = self._recv_dto(0.025)
    # mta_addr_ext = resp[0]
    mta_addr = struct.unpack(f"{self.byte_order.value}I", resp[1:5])[0]
    return mta_addr  # type: ignore

  def download_6_bytes(self, data: bytes) -> int:
    if len(data) != 6:
      raise ValueError("data size must be 6 bytes")
    self._send_cro(COMMAND_CODE.DNLOAD_6, data)
    resp = self._recv_dto(0.025)
    # mta_addr_ext = resp[0]
    mta_addr = struct.unpack(f"{self.byte_order.value}I", resp[1:5])[0]
    return mta_addr  # type: ignore

  def upload(self, size: int) -> bytes:
    if size > 5:
      raise ValueError("size must be less than 6")
    self._send_cro(COMMAND_CODE.UPLOAD, bytes([size]))
    return self._recv_dto(0.025)[:size]

  def short_upload(self, size: int, addr_ext: int, addr: int) -> bytes:
    if size > 5:
      raise ValueError("size must be less than 6")
    if addr_ext > 255:
      raise ValueError("address extension must be less than 256")
    self._send_cro(COMMAND_CODE.SHORT_UP, bytes([size, addr_ext]) + struct.pack(f"{self.byte_order.value}I", addr))
    return self._recv_dto(0.025)[:size]

  def select_calibration_page(self) -> None:
    self._send_cro(COMMAND_CODE.SELECT_CAL_PAGE)
    self._recv_dto(0.025)

  def get_daq_list_size(self, list_num: int, can_id: int = 0) -> GetDaqListSizeReturn:
    if list_num > 255:
      raise ValueError("list number must be less than 256")
    self._send_cro(COMMAND_CODE.GET_DAQ_SIZE, bytes([list_num, 0]) + struct.pack(f"{self.byte_order.value}I", can_id))
    resp = self._recv_dto(0.025)
    return GetDaqListSizeReturn(list_size=resp[0], first_pid=resp[1])

  def set_daq_list_pointer(self, list_num: int, odt_num: int, element_num: int) -> None:
    if list_num > 255:
      raise ValueError("list number must be less than 256")
    if odt_num > 255:
      raise ValueError("ODT number must be less than 256")
    if element_num > 255:
      raise ValueError("element number must be less than 256")
    self._send_cro(COMMAND_CODE.SET_DAQ_PTR, bytes([list_num, odt_num, element_num]))
    self._recv_dto(0.025)

  def write_daq_list_entry(self, size: int, addr_ext: int, addr: int) -> None:
    if size > 255:
      raise ValueError("size must be less than 256")
    if addr_ext > 255:
      raise ValueError("address extension must be less than 256")
    self._send_cro(COMMAND_CODE.WRITE_DAQ, bytes([size, addr_ext]) + struct.pack(f"{self.byte_order.value}I", addr))
    self._recv_dto(0.025)

  def start_stop_transmission(self, mode: int, list_num: int, odt_num: int, channel_num: int, rate_prescaler: int = 0) -> None:
    if mode > 255:
      raise ValueError("mode must be less than 256")
    if list_num > 255:
      raise ValueError("list number must be less than 256")
    if odt_num > 255:
      raise ValueError("ODT number must be less than 256")
    if channel_num > 255:
      raise ValueError("channel number must be less than 256")
    if rate_prescaler > 65535:
      raise ValueError("rate prescaler must be less than 65536")
    self._send_cro(COMMAND_CODE.START_STOP, bytes([mode, list_num, odt_num, channel_num]) + struct.pack(f"{self.byte_order.value}H", rate_prescaler))
    self._recv_dto(0.025)

  def disconnect(self, station_addr: int, temporary: bool = False) -> None:
    if station_addr > 65535:
      raise ValueError("station address must be less than 65536")
    # NOTE: station address is always little endian
    self._send_cro(COMMAND_CODE.DISCONNECT, bytes([int(not temporary), 0x00]) + struct.pack("<H", station_addr))
    self._recv_dto(0.025)

  def set_session_status(self, status: int) -> None:
    if status > 255:
      raise ValueError("status must be less than 256")
    self._send_cro(COMMAND_CODE.SET_S_STATUS, bytes([status]))
    self._recv_dto(0.025)

  def get_session_status(self) -> GetSessionStatusReturn:
    self._send_cro(COMMAND_CODE.GET_S_STATUS)
    resp = self._recv_dto(0.025)
    info = resp[2] if resp[1] else None
    return GetSessionStatusReturn(status=resp[0], info=info)

  def build_checksum(self, size: int) -> bytes:
    self._send_cro(COMMAND_CODE.BUILD_CHKSUM, struct.pack(f"{self.byte_order.value}I", size))
    resp = self._recv_dto(30.0)
    chksum_size = resp[0]
    assert chksum_size <= 4, "checksum more than 4 bytes"
    chksum = resp[1:1+chksum_size]
    return chksum

  def clear_memory(self, size: int) -> None:
    self._send_cro(COMMAND_CODE.CLEAR_MEMORY, struct.pack(f"{self.byte_order.value}I", size))
    self._recv_dto(30.0)

  def program(self, size: int, data: bytes) -> int:
    if size > 5:
      raise ValueError("size must be less than 6")
    if len(data) > 5:
      raise ValueError("max data size is 5 bytes")
    self._send_cro(COMMAND_CODE.PROGRAM, bytes([size]) + data)
    resp = self._recv_dto(0.1)
    # mta_addr_ext = resp[0]
    mta_addr = struct.unpack(f"{self.byte_order.value}I", resp[1:5])[0]
    return mta_addr  # type: ignore

  def program_6_bytes(self, data: bytes) -> int:
    if len(data) != 6:
      raise ValueError("data size must be 6 bytes")
    self._send_cro(COMMAND_CODE.PROGRAM_6, data)
    resp = self._recv_dto(0.1)
    # mta_addr_ext = resp[0]
    mta_addr = struct.unpack(f"{self.byte_order.value}I", resp[1:5])[0]
    return mta_addr  # type: ignore

  def move_memory_block(self, size: int) -> None:
    self._send_cro(COMMAND_CODE.MOVE, struct.pack(f"{self.byte_order.value}I", size))
    self._recv_dto(0.025)

  def diagnostic_service(self, service_num: int, data: bytes = b"") -> DiagnosticServiceReturn:
    if service_num > 65535:
      raise ValueError("service number must be less than 65536")
    if len(data) > 4:
      raise ValueError("max data size is 4 bytes")
    self._send_cro(COMMAND_CODE.DIAG_SERVICE, struct.pack(f"{self.byte_order.value}H", service_num) + data)
    resp = self._recv_dto(0.025)
    return DiagnosticServiceReturn(length=resp[0], type=resp[1])

  def action_service(self, service_num: int, data: bytes = b"") -> ActionServiceReturn:
    if service_num > 65535:
      raise ValueError("service number must be less than 65536")
    if len(data) > 4:
      raise ValueError("max data size is 4 bytes")
    self._send_cro(COMMAND_CODE.ACTION_SERVICE, struct.pack(f"{self.byte_order.value}H", service_num) + data)
    resp = self._recv_dto(0.025)
    return ActionServiceReturn(length=resp[0], type=resp[1])

  def test_availability(self, station_addr: int) -> None:
    if station_addr > 65535:
      raise ValueError("station address must be less than 65536")
    # NOTE: station address is always little endian
    self._send_cro(COMMAND_CODE.TEST, struct.pack("<H", station_addr))
    self._recv_dto(0.025)

  def start_stop_synchronised_transmission(self, mode: int) -> None:
    if mode > 255:
      raise ValueError("mode must be less than 256")
    self._send_cro(COMMAND_CODE.START_STOP_ALL, bytes([mode]))
    self._recv_dto(0.025)

  def get_active_calibration_page(self):
    self._send_cro(COMMAND_CODE.GET_ACTIVE_CAL_PAGE)
    resp = self._recv_dto(0.025)
    # cal_addr_ext = resp[0]
    cal_addr = struct.unpack(f"{self.byte_order.value}I", resp[1:5])[0]
    return cal_addr

  def get_version(self, desired_version: float = 2.1) -> float:
    major, minor = map(int, str(desired_version).split("."))
    self._send_cro(COMMAND_CODE.GET_CCP_VERSION, bytes([major, minor]))
    resp = self._recv_dto(0.025)
    return float(f"{resp[0]}.{resp[1]}")
