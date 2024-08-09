import sys
import time
import struct
from enum import IntEnum

class COMMAND_CODE(IntEnum):
  CONNECT = 0xFF
  DISCONNECT = 0xFE
  GET_STATUS = 0xFD
  SYNCH = 0xFC
  GET_COMM_MODE_INFO = 0xFB
  GET_ID = 0xFA
  SET_REQUEST = 0xF9
  GET_SEED = 0xF8
  UNLOCK = 0xF7
  SET_MTA = 0xF6
  UPLOAD = 0xF5
  SHORT_UPLOAD = 0xF4
  BUILD_CHECKSUM = 0xF3
  TRANSPORT_LAYER_CMD = 0xF2
  USER_CMD = 0xF1
  DOWNLOAD = 0xF0
  DOWNLOAD_NEXT = 0xEF
  DOWNLOAD_MAX = 0xEE
  SHORT_DOWNLOAD = 0xED
  MODIFY_BITS = 0xEC
  SET_CAL_PAGE = 0xEB
  GET_CAL_PAGE = 0xEA
  GET_PAG_PROCESSOR_INFO = 0xE9
  GET_SEGMENT_INFO = 0xE8
  GET_PAGE_INFO = 0xE7
  SET_SEGMENT_MODE = 0xE6
  GET_SEGMENT_MODE = 0xE5
  COPY_CAL_PAGE = 0xE4
  CLEAR_DAQ_LIST = 0xE3
  SET_DAQ_PTR = 0xE2
  WRITE_DAQ = 0xE1
  SET_DAQ_LIST_MODE = 0xE0
  GET_DAQ_LIST_MODE = 0xDF
  START_STOP_DAQ_LIST = 0xDE
  START_STOP_SYNCH = 0xDD
  GET_DAQ_CLOCK = 0xDC
  READ_DAQ = 0xDB
  GET_DAQ_PROCESSOR_INFO = 0xDA
  GET_DAQ_RESOLUTION_INFO = 0xD9
  GET_DAQ_LIST_INFO = 0xD8
  GET_DAQ_EVENT_INFO = 0xD7
  FREE_DAQ = 0xD6
  ALLOC_DAQ = 0xD5
  ALLOC_ODT = 0xD4
  ALLOC_ODT_ENTRY = 0xD3
  PROGRAM_START = 0xD2
  PROGRAM_CLEAR = 0xD1
  PROGRAM = 0xD0
  PROGRAM_RESET = 0xCF
  GET_PGM_PROCESSOR_INFO = 0xCE
  GET_SECTOR_INFO = 0xCD
  PROGRAM_PREPARE = 0xCC
  PROGRAM_FORMAT = 0xCB
  PROGRAM_NEXT = 0xCA
  PROGRAM_MAX = 0xC9
  PROGRAM_VERIFY = 0xC8

ERROR_CODES = {
  0x00: "Command processor synchronization",
  0x10: "Command was not executed",
  0x11: "Command rejected because DAQ is running",
  0x12: "Command rejected because PGM is running",
  0x20: "Unknown command or not implemented optional command",
  0x21: "Command syntax invalid",
  0x22: "Command syntax valid but command parameter(s) out of range",
  0x23: "The memory location is write protected",
  0x24: "The memory location is not accessible",
  0x25: "Access denied, Seed & Key is required",
  0x26: "Selected page not available",
  0x27: "Selected page mode not available",
  0x28: "Selected segment not valid",
  0x29: "Sequence error",
  0x2A: "DAQ configuration not valid",
  0x30: "Memory overflow error",
  0x31: "Generic error",
  0x32: "The slave internal program verify routine detects an error",
}

class CONNECT_MODE(IntEnum):
  NORMAL = 0x00,
  USER_DEFINED = 0x01,

class GET_ID_REQUEST_TYPE(IntEnum):
  ASCII = 0x00,
  ASAM_MC2_FILE = 0x01,
  ASAM_MC2_PATH = 0x02,
  ASAM_MC2_URL = 0x03,
  ASAM_MC2_UPLOAD = 0x04,
  # 128-255 user defined

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

class XcpClient():
  def __init__(self, panda, tx_addr: int, rx_addr: int, bus: int=0, timeout: float=0.1, debug=False, pad=True):
    self.tx_addr = tx_addr
    self.rx_addr = rx_addr
    self.can_bus = bus
    self.timeout = timeout
    self.debug = debug
    self._panda = panda
    self._byte_order = ">"
    self._max_cto = 8
    self._max_dto = 8
    self.pad = pad

  def _send_cto(self, cmd: int, dat: bytes = b"") -> None:
    tx_data = (bytes([cmd]) + dat)

    # Some ECUs don't respond if the packets are not padded to 8 bytes
    if self.pad:
      tx_data = tx_data.ljust(8, b"\x00")

    if self.debug:
      print("CAN-CLEAR: TX")
    self._panda.can_clear(self.can_bus)
    if self.debug:
      print("CAN-CLEAR: RX")
    self._panda.can_clear(0xFFFF)
    if self.debug:
      print(f"CAN-TX: {hex(self.tx_addr)} - 0x{bytes.hex(tx_data)}")
    self._panda.can_send(self.tx_addr, tx_data, self.can_bus)

  def _recv_dto(self, timeout: float) -> bytes:
    start_time = time.time()
    while time.time() - start_time < timeout:
      msgs = self._panda.can_recv() or []
      if len(msgs) >= 256:
        print("CAN RX buffer overflow!!!", file=sys.stderr)
      for rx_addr, rx_data, rx_bus in msgs:
        if rx_bus == self.can_bus and rx_addr == self.rx_addr:
          rx_data = bytes(rx_data)  # convert bytearray to bytes
          if self.debug:
            print(f"CAN-RX: {hex(rx_addr)} - 0x{bytes.hex(rx_data)}")

          pid = rx_data[0]
          if pid == 0xFE:
            err = rx_data[1]
            err_desc = ERROR_CODES.get(err, "unknown error")
            dat = rx_data[2:]
            raise CommandResponseError(f"{hex(err)} - {err_desc} {dat}", err)

          return bytes(rx_data[1:])
      time.sleep(0.001)

    raise CommandTimeoutError("timeout waiting for response")

  # commands
  def connect(self, connect_mode: CONNECT_MODE=CONNECT_MODE.NORMAL) -> dict:
    self._send_cto(COMMAND_CODE.CONNECT, bytes([connect_mode]))
    resp = self._recv_dto(self.timeout)
    assert len(resp) == 7, f"incorrect data length: {len(resp)}"
    self._byte_order = ">" if resp[1] & 0x01 else "<"
    self._slave_block_mode = resp[1] & 0x40 != 0
    self._max_cto = resp[2]
    self._max_dto = struct.unpack(f"{self._byte_order}H", resp[3:5])[0]
    return {
      "cal_support": resp[0] & 0x01 != 0,
      "daq_support": resp[0] & 0x04 != 0,
      "stim_support": resp[0] & 0x08 != 0,
      "pgm_support": resp[0] & 0x10 != 0,
      "byte_order": self._byte_order,
      "address_granularity": 2**((resp[1] & 0x06) >> 1),
      "slave_block_mode": self._slave_block_mode,
      "optional": resp[1] & 0x80 != 0,
      "max_cto": self._max_cto,
      "max_dto": self._max_dto,
      "protocol_version": resp[5],
      "transport_version": resp[6],
    }

  def disconnect(self) -> None:
    self._send_cto(COMMAND_CODE.DISCONNECT)
    resp = self._recv_dto(self.timeout)
    assert len(resp) == 0, f"incorrect data length: {len(resp)}"

  def get_id(self, req_id_type: GET_ID_REQUEST_TYPE = GET_ID_REQUEST_TYPE.ASCII) -> dict:
    if req_id_type > 255:
      raise ValueError("request id type must be less than 255")
    self._send_cto(COMMAND_CODE.GET_ID, bytes([req_id_type]))
    resp = self._recv_dto(self.timeout)
    return {
      # mode = 0 means MTA was set
      # mode = 1 means data is at end (only CAN-FD has space for this)
      "mode": resp[0],
      "length": struct.unpack(f"{self._byte_order}I", resp[3:7])[0],
      "identifier": resp[7:] if self._max_cto > 8 else None
    }

  def get_seed(self, mode: int = 0) -> bytes:
    if mode > 255:
      raise ValueError("mode must be less than 255")
    self._send_cto(COMMAND_CODE.GET_SEED, bytes([0, mode]))

    # TODO: add support for longer seeds spread over multiple blocks
    ret = self._recv_dto(self.timeout)
    length = ret[0]
    return ret[1:length+1]

  def unlock(self, key: bytes) -> bytes:
    # TODO: add support for longer keys spread over multiple blocks
    self._send_cto(COMMAND_CODE.UNLOCK, bytes([len(key)]) + key)
    return self._recv_dto(self.timeout)

  def set_mta(self, addr: int, addr_ext: int = 0) -> bytes:
    if addr_ext > 255:
      raise ValueError("address extension must be less than 256")
    # TODO: this looks broken (missing addr extension)
    self._send_cto(COMMAND_CODE.SET_MTA, bytes([0x00, 0x00, addr_ext]) + struct.pack(f"{self._byte_order}I", addr))
    return self._recv_dto(self.timeout)

  def upload(self, size: int) -> bytes:
    if size > 255:
      raise ValueError("size must be less than 256")
    if not self._slave_block_mode and size > self._max_dto - 1:
      raise ValueError("block mode not supported")

    self._send_cto(COMMAND_CODE.UPLOAD, bytes([size]))
    resp = b""
    while len(resp) < size:
      resp += self._recv_dto(self.timeout)[:size - len(resp) + 1]
    return resp[:size] # trim off bytes with undefined values

  def short_upload(self, size: int, addr_ext: int, addr: int) -> bytes:
    if size > 6:
      raise ValueError("size must be less than 7")
    if addr_ext > 255:
      raise ValueError("address extension must be less than 256")
    self._send_cto(COMMAND_CODE.SHORT_UPLOAD, bytes([size, 0x00, addr_ext]) + struct.pack(f"{self._byte_order}I", addr))
    return self._recv_dto(self.timeout)[:size] # trim off bytes with undefined values

  def download(self, data: bytes) -> bytes:
    size = len(data)
    if size > 255:
      raise ValueError("size must be less than 256")
    if not self._slave_block_mode and size > self._max_dto - 2:
      raise ValueError("block mode not supported")

    self._send_cto(COMMAND_CODE.DOWNLOAD, bytes([size]) + data)
    return self._recv_dto(self.timeout)[:size]
