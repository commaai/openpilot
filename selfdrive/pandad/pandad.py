#!/usr/bin/env python3
import os
import ctypes
import threading


from openpilot.common.swaglog import cloudlog
import msgq
from cereal import log

# Load library
def get_libpath():
  path = os.path.dirname(os.path.abspath(__file__))
  return os.path.join(path, "libpandad.so")

try:
  _lib: ctypes.CDLL | None = ctypes.CDLL(get_libpath())
except OSError:
  _lib = None


# Constants
PANDA_BUS_OFFSET = 4
PANDA_CAN_CNT = 3

class CanFrame_C(ctypes.Structure):
  _fields_ = [
    ("address", ctypes.c_long),
    ("dat", ctypes.POINTER(ctypes.c_uint8)),
    ("dat_len", ctypes.c_size_t),
    ("src", ctypes.c_long),
  ]

class CanFrame_Flat(ctypes.Structure):
  _fields_ = [
    ("address", ctypes.c_long),
    ("dat", ctypes.c_uint8 * 64),
    ("dat_len", ctypes.c_size_t),
    ("src", ctypes.c_long),
  ]


def can_list_to_can_capnp(can_msgs, msgtype='can', valid=True):
  if not isinstance(can_msgs, list):
    raise TypeError("can_msgs must be a list")
  sendcan = (msgtype == 'sendcan')
  count = len(can_msgs)
  c_frames = (CanFrame_C * count)()
  for i, (addr, dat, src) in enumerate(can_msgs):
    c_frames[i].address = addr
    c_frames[i].dat = ctypes.cast(ctypes.c_char_p(dat), ctypes.POINTER(ctypes.c_uint8))
    c_frames[i].dat_len = len(dat)
    c_frames[i].src = src

  out_len = ctypes.c_size_t()

  ret_ptr = _lib.can_list_to_capnp(c_frames, count, sendcan, valid, ctypes.byref(out_len))
  if not ret_ptr:
    return b""

  try:
    return ctypes.string_at(ret_ptr, out_len.value)
  finally:
    _lib.panda_free_str(ret_ptr)

def can_capnp_to_list(strings, msgtype='can'):
  sendcan = (msgtype == 'sendcan')
  if not isinstance(strings, list):
    raise TypeError("strings must be a list")

  ret = []
  for s in strings:
    ptr = _lib.can_capnp_to_list_create(s, len(s), sendcan)
    if not ptr:
      continue

    try:
      count = _lib.can_capnp_handler_size(ptr)
      for i in range(count):
        nanos = _lib.can_capnp_handler_get_nanos(ptr, i)
        frame_cnt = _lib.can_capnp_handler_get_frame_count(ptr, i)
        frames = []
        c_frame = CanFrame_Flat()
        for j in range(frame_cnt):
            _lib.can_capnp_handler_get_frame(ptr, i, j, ctypes.byref(c_frame))
            dat = bytes(c_frame.dat[:c_frame.dat_len])
            frames.append((c_frame.address, dat, c_frame.src))
        ret.append((nanos, frames))
    finally:
      _lib.can_capnp_handler_free(ptr)

  return ret

# Set CAPI types for helpers
if _lib:
  _lib.can_list_to_capnp.argtypes = [ctypes.POINTER(CanFrame_C), ctypes.c_size_t, ctypes.c_bool, ctypes.c_bool, ctypes.POINTER(ctypes.c_size_t)]
  _lib.can_list_to_capnp.restype = ctypes.c_void_p

  _lib.can_capnp_to_list_create.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_bool]
  _lib.can_capnp_to_list_create.restype = ctypes.c_void_p

  _lib.can_capnp_handler_size.argtypes = [ctypes.c_void_p]
  _lib.can_capnp_handler_size.restype = ctypes.c_size_t

  _lib.can_capnp_handler_get_nanos.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
  _lib.can_capnp_handler_get_nanos.restype = ctypes.c_uint64

  _lib.can_capnp_handler_get_frame_count.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
  _lib.can_capnp_handler_get_frame_count.restype = ctypes.c_size_t

  _lib.can_capnp_handler_get_frame.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(CanFrame_Flat)]

  _lib.can_capnp_handler_free.argtypes = [ctypes.c_void_p]


class PandaHealth_C(ctypes.Structure):
  _fields_ = [
    ("voltage_pkt", ctypes.c_uint16),
    ("current_pkt", ctypes.c_uint16),
    ("uptime_pkt", ctypes.c_uint32),
    ("rx_buffer_overflow_pkt", ctypes.c_uint32),
    ("tx_buffer_overflow_pkt", ctypes.c_uint32),
    ("faults_pkt", ctypes.c_uint32),
    ("ignition_line_pkt", ctypes.c_uint8),
    ("ignition_can_pkt", ctypes.c_uint8),
    ("controls_allowed_pkt", ctypes.c_uint8),
    ("safety_mode_pkt", ctypes.c_uint8),
    ("safety_param_pkt", ctypes.c_uint16),
    ("fault_status_pkt", ctypes.c_uint8),
    ("power_save_enabled_pkt", ctypes.c_uint8),
    ("heartbeat_lost_pkt", ctypes.c_uint8),
    ("alternative_experience_pkt", ctypes.c_uint16),
    ("car_harness_status_pkt", ctypes.c_uint8),
    ("safety_tx_blocked_pkt", ctypes.c_uint8),
    ("safety_rx_invalid_pkt", ctypes.c_uint8),
    ("safety_rx_checks_invalid_pkt", ctypes.c_uint8),
    ("interrupt_load_pkt", ctypes.c_uint32),
    ("fan_power", ctypes.c_uint16),
    ("spi_error_count_pkt", ctypes.c_uint32),
    ("sbu1_voltage_mV", ctypes.c_uint16),
    ("sbu2_voltage_mV", ctypes.c_uint16),
  ]

# Set return types
if _lib:
  _lib.panda_create.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
  _lib.panda_create.restype = ctypes.c_void_p
  _lib.panda_delete.argtypes = [ctypes.c_void_p]

  _lib.panda_connected.argtypes = [ctypes.c_void_p]
  _lib.panda_connected.restype = ctypes.c_bool

  _lib.panda_comms_healthy.argtypes = [ctypes.c_void_p]
  _lib.panda_comms_healthy.restype = ctypes.c_bool

  _lib.panda_get_serial.argtypes = [ctypes.c_void_p]
  _lib.panda_get_serial.restype = ctypes.c_void_p # char* needed to free

  _lib.panda_free_str.argtypes = [ctypes.c_void_p]

  _lib.panda_set_safety_model.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint16]

  _lib.panda_get_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(PandaHealth_C)]
  _lib.panda_get_state.restype = ctypes.c_bool

  _lib.panda_can_send.argtypes = [ctypes.c_void_p, ctypes.POINTER(CanFrame_C), ctypes.c_size_t]

  _lib.panda_can_receive.argtypes = [ctypes.c_void_p, ctypes.POINTER(CanFrame_Flat), ctypes.c_size_t]
  _lib.panda_can_receive.restype = ctypes.c_int

  _lib.panda_send_heartbeat.argtypes = [ctypes.c_void_p, ctypes.c_bool]
  _lib.panda_set_power_saving.argtypes = [ctypes.c_void_p, ctypes.c_bool]

  _lib.panda_get_type.argtypes = [ctypes.c_void_p]
  _lib.panda_get_type.restype = ctypes.c_uint16


def pandad_main(pandas):
  if not pandas:
    return

  # Start send thread
  stop_event = threading.Event()
  send_t = threading.Thread(target=can_send_thread, args=(pandas, stop_event))
  send_t.start()

  pm = msgq.PubMaster(["can", "pandaStates", "peripheralState"])
  sm = msgq.SubMaster(["deviceState", "controlsState"])

  # Rate keepers
  rk = msgq.Ratekeeper(100, print_delay_threshold=None)

  try:
    while True:
      # Check connections
      if not comms_checks(pandas):
        break

      sm.update(0)

      # CAN Recv
      all_frames = []
      for p in pandas:
        frames = p.can_receive()
        if frames:
          all_frames.extend(frames)

      # Publish raw CAN
      if all_frames:
        msg = msgq.MessageBuilder()
        evt = msg.initEvent()
        evt.valid = True
        can_data = evt.initCan(len(all_frames))
        for i, (addr, _, dat, src) in enumerate(all_frames):
          can_data[i].address = addr
          can_data[i].dat = dat
          can_data[i].src = src
        pm.send("can", msg)

      # 10 Hz - PandaState
      if rk.frame % 10 == 0:
        msg = msgq.MessageBuilder()
        evt = msg.initEvent()
        evt.valid = True
        ps = evt.initPandaStates(len(pandas))

        for i, p in enumerate(pandas):
          s = p.get_state()
          if s:
            ps[i].voltage = s.voltage_pkt
            ps[i].current = s.current_pkt
            ps[i].uptime = s.uptime_pkt
            ps[i].rxBufferOverflow = s.rx_buffer_overflow_pkt
            ps[i].txBufferOverflow = s.tx_buffer_overflow_pkt
            ps[i].faults = s.faults_pkt
            ps[i].ignitionLine = bool(s.ignition_line_pkt)
            ps[i].ignitionCan = bool(s.ignition_can_pkt)
            ps[i].controlsAllowed = bool(s.controls_allowed_pkt)
            ps[i].safetyModel = int(s.safety_mode_pkt)
            ps[i].safetyParam = s.safety_param_pkt
            ps[i].faultStatus = int(s.fault_status_pkt)
            ps[i].powerSaveEnabled = bool(s.power_save_enabled_pkt)
            ps[i].heartbeatLost = bool(s.heartbeat_lost_pkt)
            ps[i].alternativeExperience = s.alternative_experience_pkt
            ps[i].carHarnessStatus = int(s.car_harness_status_pkt)
            ps[i].safetyTxBlocked = bool(s.safety_tx_blocked_pkt)
            ps[i].safetyRxInvalid = bool(s.safety_rx_invalid_pkt)
            ps[i].safetyRxChecksInvalid = bool(s.safety_rx_checks_invalid_pkt)
            ps[i].interruptLoad = s.interrupt_load_pkt
            ps[i].fanPower = s.fan_power
            ps[i].spiErrorCount = s.spi_error_count_pkt
            ps[i].sbu1Voltage = s.sbu1_voltage_mV
            ps[i].sbu2Voltage = s.sbu2_voltage_mV
            ps[i].pandaType = p.get_type()

        pm.send("pandaStates", msg)

        # Heartbeat
        engaged = sm['controlsState'].enabled
        for p in pandas:
          p.send_heartbeat(engaged)

      # 2 Hz - PeripheralState
      if rk.frame % 50 == 0:
        msg = msgq.MessageBuilder()
        evt = msg.initEvent()
        evt.valid = True
        evt.initPeripheralState()
        pm.send("peripheralState", msg)

      rk.keep_time()

  finally:
    stop_event.set()
    send_t.join()


class Panda:
  def __init__(self, serial=None, index=0):
    if serial is not None:
      serial = serial.encode('utf-8')
    self.ptr = _lib.panda_create(serial, index * PANDA_BUS_OFFSET)
    if not self.ptr:
      raise Exception("Failed to create Panda")
    self.serial = serial.decode('utf-8') if serial else self.get_serial()

  def __del__(self):
    if self.ptr:
      _lib.panda_delete(self.ptr)

  def connected(self):
    return _lib.panda_connected(self.ptr)

  def comms_healthy(self):
    return _lib.panda_comms_healthy(self.ptr)

  def get_type(self):
    return _lib.panda_get_type(self.ptr)

  def get_serial(self):
    ptr = _lib.panda_get_serial(self.ptr)
    if not ptr:
      return ""
    try:
      return ctypes.cast(ptr, ctypes.c_char_p).value.decode('utf-8')
    finally:
      _lib.panda_free_str(ptr)

  def set_safety_model(self, mode, param=0):
    _lib.panda_set_safety_model(self.ptr, int(mode), param)

  def get_state(self):
    h = PandaHealth_C()
    if _lib.panda_get_state(self.ptr, ctypes.byref(h)):
      return h
    return None

  def send_heartbeat(self, engaged):
    _lib.panda_send_heartbeat(self.ptr, engaged)

  def set_power_saving(self, enable):
    _lib.panda_set_power_saving(self.ptr, enable)

  def can_send(self, frames):
    # frames = list of (addr, dat, src)
    count = len(frames)
    c_frames = (CanFrame_C * count)()
    for i, (addr, dat, src) in enumerate(frames):
      c_frames[i].address = addr
      c_frames[i].dat = ctypes.cast(ctypes.c_char_p(dat), ctypes.POINTER(ctypes.c_uint8))
      c_frames[i].dat_len = len(dat)
      c_frames[i].src = src
    _lib.panda_can_send(self.ptr, c_frames, count)

  def can_receive(self):
    max_len = 512
    c_frames = (CanFrame_Flat * max_len)()
    cnt = _lib.panda_can_receive(self.ptr, c_frames, max_len)
    if cnt < 0:
      return None

    ret = []
    for i in range(cnt):
      f = c_frames[i]
      dat = bytes(f.dat[:f.dat_len])
      ret.append((f.address, 0, dat, f.src))
    return ret

  @staticmethod
  def list():
    from panda import Panda as PyPanda
    return PyPanda.list()


def comms_checks(pandas):
  if any(not p.connected() for p in pandas):
    return False
  return True

def can_send_thread(pandas, stop_event):
  sm = msgq.SubSocket(msgq.Context(), "sendcan")
  sm.setTimeout(100)

  while not stop_event.is_set():
    if not comms_checks(pandas):
      break

    msg = sm.receive(True)
    if msg is None:
      continue

    evt = log.Event.from_bytes(msg)
    if evt.which() == 'sendcan':
      for p in pandas:
        frames = []
        for c in evt.sendcan:
            frames.append((c.address, c.dat, c.src))
        p.can_send(frames)

def main():
  from panda import Panda as PyPanda
  serials = PyPanda.list()

  if not serials:
    cloudlog.warning("no pandas found, exiting")
    return

  cloudlog.warning(f"connecting to pandas: {serials}")
  pandas = []
  for i, s in enumerate(serials):
    try:
      p = Panda(s, i)
      pandas.append(p)
    except Exception:
      cloudlog.exception(f"failed to connect to {s}")

  if pandas:
    pandad_main(pandas)

if __name__ == "__main__":
  main()
