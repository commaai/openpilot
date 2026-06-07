#!/usr/bin/env python3
from __future__ import annotations

import os
import signal
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any

import usb1

import cereal.messaging as messaging
from cereal import car, log
from panda import DLC_TO_LEN, LEN_TO_DLC, FW_PATH, McuType, Panda, PandaDFU, PandaProtocolMismatch, PandaSpiException
from panda.python import PandaSpiHandle
from panda.python import spi as panda_spi
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper, config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.pandad.pandad_api_impl import can_list_to_can_capnp
from openpilot.system.hardware import HARDWARE, PC


PANDA_CAN_CNT = 3
PANDA_BUS_OFFSET = 4
CAN_RECV_SIZE = 0x4000
CAN_SEND_SOFT_LIMIT = 0x100

MAX_IR_PANDA_VAL = 50
CUTOFF_IL = 400
SATURATE_IL = 1000

CAN_SEND_TIMEOUT_MS = 5
CAN_RECV_TIMEOUT_MS = 5
CONTROL_TIMEOUT_MS = 100
PANDA_LOG_READ_INTERVAL = 10
PANDA_SPI_ACK_TIMEOUT_MS = 20
PANDA_SPI_MAX_ACK_TIMEOUT_MS = 500
PANDA_SPI_CLEANUP_TIMEOUT_MS = 1

FAULT_NAME_BY_VALUE = {v: k for k, v in log.PandaState.FaultType.schema.enumerants.items()}

HEALTH_FIELDS = (
  "uptime", "voltage", "current", "safety_tx_blocked", "safety_rx_invalid",
  "tx_buffer_overflow", "rx_buffer_overflow", "faults", "ignition_line",
  "ignition_can", "controls_allowed", "car_harness_status", "safety_mode",
  "safety_param", "fault_status", "power_save_enabled", "heartbeat_lost",
  "alternative_experience", "interrupt_load", "fan_power",
  "safety_rx_checks_invalid", "spi_error_count", "sbu1_voltage_mV",
  "sbu2_voltage_mV", "som_reset_triggered", "sound_output_level",
)

CAN_HEALTH_FIELDS = (
  "bus_off", "bus_off_cnt", "error_warning", "error_passive", "last_error",
  "last_stored_error", "last_data_error", "last_data_stored_error",
  "receive_error_cnt", "transmit_error_cnt", "total_error_cnt",
  "total_tx_lost_cnt", "total_rx_lost_cnt", "total_tx_cnt", "total_rx_cnt",
  "total_fwd_cnt", "total_tx_checksum_error_cnt", "can_speed",
  "can_data_speed", "canfd_enabled", "brs_enabled", "canfd_non_iso",
  "irq0_call_rate", "irq1_call_rate", "irq2_call_rate",
  "can_core_reset_count",
)

exit_event = threading.Event()
shutdown_event = threading.Event()


def _patch_panda_spi() -> None:
  panda_spi.MIN_ACK_TIMEOUT_MS = PANDA_SPI_ACK_TIMEOUT_MS
  panda_spi.SPI_ACK_TIMEOUT_MS = PANDA_SPI_MAX_ACK_TIMEOUT_MS

  original_init = PandaSpiHandle.__init__
  original_close = PandaSpiHandle.close

  def patched_init(self) -> None:
    original_init(self)
    self.connected = True
    self.comms_healthy = True

  def patched_close(self) -> None:
    self.connected = False
    original_close(self)

  def patched_wait_for_ack(self, spi, ack_val: int, timeout: int, tx: int, length: int = 1) -> bytes:
    if timeout == 0:
      timeout = PANDA_SPI_MAX_ACK_TIMEOUT_MS
    timeout_s = timeout * 1e-3

    start = time.monotonic()
    while (time.monotonic() - start) < timeout_s:
      dat = spi.xfer2([tx] * length)
      if dat[0] == ack_val:
        return bytes(dat)
      if dat[0] == panda_spi.NACK:
        raise panda_spi.PandaSpiNackResponse

    raise panda_spi.PandaSpiMissingAck

  def patched_transfer_spidev(self, spi, endpoint: int, data, timeout: int, max_rx_len: int = 1000,
                              expect_disconnect: bool = False) -> bytes:
    max_rx_len = max(panda_spi.USBPACKET_MAX_SIZE, max_rx_len)

    packet = self.HEADER.pack(panda_spi.SYNC, endpoint, len(data), max_rx_len)
    packet += bytes([self._calc_checksum(packet)])
    spi.xfer2(packet)
    self._wait_for_ack(spi, panda_spi.HACK, timeout, 0x11)

    packet = bytes([*data, self._calc_checksum(data)])
    spi.xfer2(packet)

    if expect_disconnect:
      return b""

    dat = self._wait_for_ack(spi, panda_spi.DACK, timeout, 0x13, length=3)
    response_len = panda_spi.struct.unpack("<H", dat[1:3])[0]
    if response_len > max_rx_len:
      raise panda_spi.PandaSpiException(f"response length greater than max ({max_rx_len} {response_len})")

    dat += bytes(spi.readbytes(response_len + 1))
    dat = dat[:3 + response_len + 1]
    if self._calc_checksum(dat) != 0:
      raise panda_spi.PandaSpiBadChecksum

    return dat[3:-1]

  def patched_transfer(self, endpoint: int, data, timeout: int, max_rx_len: int = 1000,
                       expect_disconnect: bool = False) -> bytes:
    nack_count = 0
    timeout_count = 0
    exc = panda_spi.PandaSpiException()
    while self.connected:
      with self.dev.acquire() as spi:
        try:
          ret = self._transfer_spidev(spi, endpoint, data, timeout, max_rx_len, expect_disconnect)
          self.comms_healthy = True
          return ret
        except panda_spi.PandaSpiException as e:
          exc = e
          if self.no_retry:
            break

          if isinstance(e, panda_spi.PandaSpiMissingAck):
            timeout_count += 1
            if timeout != 0 and timeout_count > panda_spi.MAX_XFER_RETRY_COUNT:
              break

          if isinstance(e, panda_spi.PandaSpiNackResponse):
            nack_count += 1
            if nack_count > 3:
              time.sleep(min(max(nack_count * 10, 200), 2000) * 1e-6)

          cleanup_nack_count = 0
          attempts = 5
          while cleanup_nack_count <= 3 and attempts > 0:
            attempts -= 1
            try:
              self._wait_for_ack(spi, panda_spi.NACK, PANDA_SPI_CLEANUP_TIMEOUT_MS, 0x14,
                                  length=panda_spi.XFER_SIZE // 2)
              cleanup_nack_count += 1
            except panda_spi.PandaSpiException:
              cleanup_nack_count = 0

    self.comms_healthy = False
    raise exc

  PandaSpiHandle.__init__ = patched_init
  PandaSpiHandle.close = patched_close
  PandaSpiHandle._wait_for_ack = patched_wait_for_ack
  PandaSpiHandle._transfer_spidev = patched_transfer_spidev
  PandaSpiHandle._transfer = patched_transfer


_patch_panda_spi()


def _enum_value(value: Any) -> int:
  return int(getattr(value, "raw", value))


@dataclass
class HwmonState:
  voltage: int = 0
  current: int = 0
  initialized: bool = False


class PandadPanda:
  def __init__(self, serial: str | None):
    self._context, self._handle, self.serial, self.bootstub = Panda.spi_connect(serial)
    if self._handle is None or self.serial is None:
      raise PandaSpiException("failed to connect to panda over SPI")
    if self.bootstub:
      raise PandaProtocolMismatch("panda is in bootstub. run firmware recovery before pandad")

    self.comms_healthy = True
    self.can_rx_overflow_buffer = b""
    self.hw_type = self.get_hw_type()
    self._check_packet_versions()

    self.can_reset_communications()
    cloudlog.warning(f"connected to {self.serial} over SPI")

  @classmethod
  def list_pandas(cls) -> list[str]:
    return Panda.spi_list()

  @property
  def connected(self) -> bool:
    return self._handle is not None and getattr(self._handle, "connected", True)

  def close(self) -> None:
    if self._handle is not None:
      self._handle.close()

  def _mark_unhealthy(self) -> None:
    self.comms_healthy = False
    if self._handle is not None:
      self._handle.comms_healthy = False

  def _control_write(self, request: int, value: int = 0, index: int = 0, timeout: int = CONTROL_TIMEOUT_MS) -> None:
    try:
      self._handle.controlWrite(Panda.REQUEST_OUT, request, value, index, b"", timeout=timeout)
    except PandaSpiException:
      self._mark_unhealthy()
      raise

  def _control_read(self, request: int, value: int, index: int, length: int, timeout: int = CONTROL_TIMEOUT_MS) -> bytes:
    try:
      return self._handle.controlRead(Panda.REQUEST_IN, request, value, index, length, timeout=timeout)
    except PandaSpiException:
      self._mark_unhealthy()
      raise

  def _bulk_write(self, endpoint: int, data: bytes, timeout: int = CAN_SEND_TIMEOUT_MS) -> int:
    try:
      return self._handle.bulkWrite(endpoint, data, timeout=timeout)
    except PandaSpiException:
      self._mark_unhealthy()
      raise

  def _bulk_read(self, endpoint: int, length: int, timeout: int = CAN_RECV_TIMEOUT_MS) -> bytes:
    try:
      return self._handle.bulkRead(endpoint, length, timeout=timeout)
    except PandaSpiException:
      self._mark_unhealthy()
      raise

  def _check_packet_versions(self) -> None:
    dat = self._control_read(0xdd, 0, 0, 8)
    if len(dat) != 8:
      raise PandaProtocolMismatch("failed to read panda packet versions")
    health_version, can_version = struct.unpack("<II", dat)
    if health_version != Panda.HEALTH_PACKET_VERSION:
      raise PandaProtocolMismatch(
        f"health packet version mismatch: panda's firmware v{health_version}, library v{Panda.HEALTH_PACKET_VERSION}. reflash panda."
      )
    if can_version != Panda.CAN_PACKET_VERSION:
      raise PandaProtocolMismatch(
        f"CAN packet version mismatch: panda's firmware v{can_version}, library v{Panda.CAN_PACKET_VERSION}. reflash panda."
      )

  def get_hw_type(self) -> int:
    return self._control_read(0xc1, 0, 0, 1)[0]

  def is_internal(self) -> bool:
    return self.hw_type in (int(log.PandaState.PandaType.tres), int(log.PandaState.PandaType.cuatro))

  def health(self) -> dict[str, Any] | None:
    try:
      dat = self._control_read(0xd2, 0, 0, Panda.HEALTH_STRUCT.size)
      return dict(zip(HEALTH_FIELDS, Panda.HEALTH_STRUCT.unpack(dat), strict=True))
    except (PandaSpiException, struct.error):
      return None

  def can_health(self, can_number: int) -> dict[str, Any] | None:
    try:
      dat = self._control_read(0xc2, int(can_number), 0, Panda.CAN_HEALTH_STRUCT.size)
      return dict(zip(CAN_HEALTH_FIELDS, Panda.CAN_HEALTH_STRUCT.unpack(dat), strict=True))
    except (PandaSpiException, struct.error):
      return None

  def set_safety_mode(self, mode: int, param: int = 0) -> None:
    self._control_write(0xdc, _enum_value(mode), int(param))

  def set_alternative_experience(self, alternative_experience: int) -> None:
    self._control_write(0xdf, int(alternative_experience), 0)

  def serial_read(self, port_number: int = 0, maxlen: int = 1024, timeout: int = CONTROL_TIMEOUT_MS) -> bytes:
    ret = b""
    while len(ret) < maxlen:
      try:
        dat = self._control_read(0xe0, port_number, 0, 0x40, timeout=timeout)
      except PandaSpiException:
        break
      if len(dat) == 0:
        break
      ret += dat
    return ret

  def set_fan_power(self, percentage: int) -> None:
    self._control_write(0xb1, int(percentage), 0)

  def get_fan_rpm(self) -> int:
    try:
      dat = self._control_read(0xb2, 0, 0, 2)
      return struct.unpack("<H", dat)[0]
    except (PandaSpiException, struct.error):
      return 0

  def set_ir_power(self, percentage: int) -> None:
    self._control_write(0xb0, int(percentage), 0)

  def set_can_loopback(self, enable: bool) -> None:
    self._control_write(0xe5, int(enable), 0)

  def set_power_save(self, power_save_enabled: bool) -> None:
    self._control_write(0xe7, int(power_save_enabled), 0)

  def send_heartbeat(self, engaged: bool) -> None:
    self._control_write(0xf3, int(engaged), 0)

  def set_canfd_auto(self, bus: int, enabled: bool) -> None:
    self._control_write(0xe8, int(bus), int(enabled))

  def can_reset_communications(self) -> None:
    self._control_write(0xc0, 0, 0)

  def can_send_many(self, can_msgs) -> None:
    tx = bytearray()
    for address, dat, bus in can_msgs:
      if bus >= PANDA_BUS_OFFSET:
        continue
      dat = bytes(dat)
      assert len(dat) <= 64
      data_len_code = LEN_TO_DLC[len(dat)]
      assert len(dat) == DLC_TO_LEN[data_len_code]

      extended = 1 if address >= 0x800 else 0
      word_4b = (int(address) << 3) | (extended << 2)
      header = bytearray(6)
      header[0] = (data_len_code << 4) | (int(bus) << 1)
      header[1] = word_4b & 0xff
      header[2] = (word_4b >> 8) & 0xff
      header[3] = (word_4b >> 16) & 0xff
      header[4] = (word_4b >> 24) & 0xff
      header[5] = 0
      header[5] = _calculate_checksum(header + dat)

      tx.extend(header)
      tx.extend(dat)
      if len(tx) >= CAN_SEND_SOFT_LIMIT:
        self._bulk_write(3, bytes(tx))
        tx.clear()

    if len(tx) > 0:
      self._bulk_write(3, bytes(tx))

  def can_receive(self) -> tuple[bool, list[tuple[int, bytes, int]]]:
    try:
      dat = self._bulk_read(1, CAN_RECV_SIZE)
      if "PANDAD_MAXOUT" in os.environ and len(dat) < CAN_RECV_SIZE:
        self._bulk_read(0xab, CAN_RECV_SIZE - len(dat))
    except PandaSpiException:
      return False, []

    if not self.comms_healthy:
      return False, []

    if len(dat) == 0:
      return True, []

    try:
      can_msgs, self.can_rx_overflow_buffer = _unpack_can_buffer(self.can_rx_overflow_buffer + dat)
      return True, can_msgs
    except ValueError:
      cloudlog.error("Panda CAN checksum failed")
      self.can_rx_overflow_buffer = b""
      try:
        self.can_reset_communications()
      except PandaSpiException:
        pass
      return False, []


class PandaFirmwareSpi(Panda):
  def connect(self, claim=True, wait=False):
    self.close()

    self._handle = None
    while self._handle is None:
      self._context, self._handle, serial, self.bootstub = self.spi_connect(self._connect_serial)
      if not wait:
        break
      if self._handle is None:
        time.sleep(0.1)

    if self._handle is None:
      raise Exception("failed to connect to panda over SPI")

    self._serial = serial
    self._connect_serial = serial
    self._handle_open = True
    self.health_version, self.can_version = self.get_packets_versions()

    if self._disable_checks:
      self.set_heartbeat_disabled()
      self.set_power_save(0)

    self.can_reset_communications()
    for bus in range(PANDA_CAN_CNT):
      self.set_canfd_auto(bus, False)

    for bus in range(PANDA_CAN_CNT):
      self.set_can_speed_kbps(bus, self._can_speed_kbps)


class PandaSafety:
  def __init__(self, panda: PandadPanda):
    self.panda = panda
    self.params = Params()
    self.initialized = False
    self.safety_configured = False
    self.prev_obd_multiplexing = False
    self.log_once = False

  def configure_safety_mode(self, is_onroad: bool) -> None:
    if is_onroad and not self.safety_configured:
      self._update_multiplexing_mode()
      params_string = self._fetch_car_params()
      if params_string:
        cloudlog.warning(f"got {len(params_string)} bytes CarParams")
        self._set_safety_mode(params_string)
        self.safety_configured = True
    elif not is_onroad:
      self.initialized = False
      self.safety_configured = False
      self.log_once = False

  def _update_multiplexing_mode(self) -> None:
    if not self.initialized:
      self.prev_obd_multiplexing = False
      self.panda.set_safety_mode(car.CarParams.SafetyModel.elm327, 1)
      self.initialized = True

    obd_multiplexing_requested = self.params.get_bool("ObdMultiplexingEnabled")
    if obd_multiplexing_requested != self.prev_obd_multiplexing:
      safety_param = 0 if obd_multiplexing_requested else 1
      self.panda.set_safety_mode(car.CarParams.SafetyModel.elm327, safety_param)
      self.prev_obd_multiplexing = obd_multiplexing_requested
      self.params.put_bool("ObdMultiplexingChanged", True)

  def _fetch_car_params(self) -> bytes | None:
    if not self.params.get_bool("FirmwareQueryDone"):
      return None

    if not self.log_once:
      cloudlog.warning("Finished FW query, Waiting for params to set safety model")
      self.log_once = True

    if not self.params.get_bool("ControlsReady"):
      return None
    return self.params.get("CarParams")

  def _set_safety_mode(self, params_string: bytes) -> None:
    with car.CarParams.from_bytes(params_string) as car_params:
      safety_config = car_params.safetyConfigs[0]
      safety_model = _enum_value(safety_config.safetyModel)
      safety_param = safety_config.safetyParam
      alternative_experience = car_params.alternativeExperience

    cloudlog.warning(
      f"setting safety model: {safety_model}, param: {safety_param}, alternative experience: {alternative_experience}"
    )
    self.panda.set_alternative_experience(alternative_experience)
    self.panda.set_safety_mode(safety_model, safety_param)


class PeripheralController:
  def __init__(self):
    self.params = Params()
    self.sm = messaging.SubMaster(["deviceState", "driverCameraState"])
    self.last_driver_camera_t = 0
    self.prev_fan_speed = 999
    self.ir_pwr = 0
    self.prev_ir_pwr = 999
    self.prev_frame_id = 0xffffffff
    self.driver_view = False
    self.not_car = False
    self.not_car_checked = False
    self.integ_lines_filter = FirstOrderFilter(0, 30.0, 0.05)
    self.integ_lines_filter_driver_view = FirstOrderFilter(0, 5.0, 0.05)

  def update(self, panda: PandadPanda, no_fan_control: bool, is_onroad: bool) -> None:
    self.sm.update(0)
    if self.sm.updated["deviceState"] and not no_fan_control:
      fan_speed = self.sm["deviceState"].fanSpeedPercentDesired
      if fan_speed != self.prev_fan_speed or self.sm.frame % 100 == 0:
        panda.set_fan_power(fan_speed)
        self.prev_fan_speed = fan_speed

    if self.sm.updated["driverCameraState"]:
      event = self.sm["driverCameraState"]
      cur_integ_lines = event.integLines

      if event.frameId < self.prev_frame_id:
        self.integ_lines_filter.reset(0)
        self.integ_lines_filter_driver_view.reset(0)
        self.driver_view = self.params.get_bool("IsDriverViewEnabled")
      self.prev_frame_id = event.frameId

      if self.driver_view:
        cur_integ_lines = self.integ_lines_filter_driver_view.update(cur_integ_lines)
      else:
        cur_integ_lines = self.integ_lines_filter.update(cur_integ_lines)
      self.last_driver_camera_t = self.sm.logMonoTime["driverCameraState"]

      if cur_integ_lines <= CUTOFF_IL:
        self.ir_pwr = 0
      elif cur_integ_lines > SATURATE_IL:
        self.ir_pwr = 100
      else:
        self.ir_pwr = int(100 * (cur_integ_lines - CUTOFF_IL) / (SATURATE_IL - CUTOFF_IL))

    if time.monotonic_ns() - self.last_driver_camera_t > 1_000_000_000:
      self.ir_pwr = 0

    if not self.not_car_checked and is_onroad:
      cp_bytes = self.params.get("CarParams")
      if cp_bytes:
        with car.CarParams.from_bytes(cp_bytes) as car_params:
          self.not_car = car_params.notCar
        self.not_car_checked = True
    if self.not_car:
      self.ir_pwr = 0

    if self.ir_pwr != self.prev_ir_pwr or self.sm.frame % 100 == 0:
      ir_panda = max(0, min(MAX_IR_PANDA_VAL, self.ir_pwr * MAX_IR_PANDA_VAL // 100))
      panda.set_ir_power(ir_panda)
      HARDWARE.set_ir_power(self.ir_pwr)
      self.prev_ir_pwr = self.ir_pwr


def _calculate_checksum(data: bytes | bytearray) -> int:
  checksum = 0
  for b in data:
    checksum ^= b
  return checksum


def _unpack_can_buffer(dat: bytes) -> tuple[list[tuple[int, bytes, int]], bytes]:
  ret = []
  pos = 0

  while len(dat) - pos >= 6:
    data_len = DLC_TO_LEN[dat[pos] >> 4]
    packet_len = 6 + data_len
    if len(dat) - pos < packet_len:
      break

    packet = dat[pos:pos + packet_len]
    if _calculate_checksum(packet) != 0:
      raise ValueError("CAN packet checksum incorrect")

    bus = (packet[0] >> 1) & 0x7
    address = (packet[4] << 24 | packet[3] << 16 | packet[2] << 8 | packet[1]) >> 3
    if (packet[1] >> 1) & 0x1:
      bus += 128
    if packet[1] & 0x1:
      bus += 192

    ret.append((address, bytes(packet[6:packet_len]), bus))
    pos += packet_len

  return ret, dat[pos:]


def get_expected_signature() -> bytes:
  fn = os.path.join(FW_PATH, McuType.H7.config.app_fn)
  return Panda.get_signature_from_firmware(fn)


def flash_panda(panda_serial: str) -> None:
  panda = PandaFirmwareSpi(panda_serial, cli=False)
  fw_signature = get_expected_signature()
  internal_panda = panda.is_internal()

  panda_version = "bootstub" if panda.bootstub else panda.get_version()
  panda_signature = b"" if panda.bootstub else panda.get_signature()
  cloudlog.warning(
    f"Panda {panda_serial} connected, version: {panda_version}, signature {panda_signature.hex()[:16]}, expected {fw_signature.hex()[:16]}"
  )

  if panda.bootstub or panda_signature != fw_signature:
    cloudlog.info("Panda firmware out of date, update required")
    panda.flash()
    cloudlog.info("Done flashing")

  if panda.bootstub:
    bootstub_version = panda.get_version()
    cloudlog.info(f"Flashed firmware not booting, flashing development bootloader. {bootstub_version=}, {internal_panda=}")
    if internal_panda:
      HARDWARE.recover_internal_panda()
    panda.recover(reset=(not internal_panda))
    cloudlog.info("Done flashing bootstub")

  if panda.bootstub:
    cloudlog.info("Panda still not booting, exiting")
    raise AssertionError

  panda_signature = panda.get_signature()
  if panda_signature != fw_signature:
    cloudlog.info("Version mismatch after flashing, exiting")
    raise AssertionError

  panda.close()


def connect(serial: str | None) -> PandadPanda | None:
  try:
    panda = PandadPanda(serial)
  except Exception:
    return None

  if "BOARDD_LOOPBACK" in os.environ:
    panda.set_can_loopback(True)

  for bus in range(PANDA_CAN_CNT):
    panda.set_canfd_auto(bus, True)

  return panda


def can_send_thread(panda: PandadPanda, fake_send: bool) -> None:
  sock = messaging.sub_sock("sendcan", conflate=False, timeout=100)

  while not exit_event.is_set() and panda.connected:
    dat = sock.receive()
    if dat is None:
      continue

    with log.Event.from_bytes(dat, traversal_limit_in_words=2**64 - 1) as event:
      if time.monotonic_ns() - event.logMonoTime < 1_000_000_000 and not fake_send:
        try:
          panda.can_send_many((can.address, can.dat, can.src) for can in event.sendcan)
          time.sleep(0)
        except PandaSpiException:
          cloudlog.exception("pandad.can_send_failed")
          exit_event.set()
      else:
        cloudlog.error(f"sendcan too old to send: {time.monotonic_ns()}, {event.logMonoTime}")


def can_recv_thread(panda: PandadPanda) -> None:
  pm = messaging.PubMaster(["can"])
  rk = Ratekeeper(100, print_delay_threshold=None)

  while not exit_event.is_set() and panda.connected:
    comms_healthy, can_msgs = panda.can_receive()
    can_msgs = [msg for msg in can_msgs if msg[2] < 192]
    pm.send("can", can_list_to_can_capnp(can_msgs, valid=comms_healthy))
    rk.keep_time()


def hwmon_thread(hwmon_state: HwmonState, lock: threading.Lock) -> None:
  while not exit_event.is_set():
    read_time = time.monotonic()
    voltage = int(HARDWARE.get_voltage())
    current = int(HARDWARE.get_current())
    read_time = (time.monotonic() - read_time) * 1000
    if read_time > 50:
      cloudlog.warning(f"reading hwmon took {read_time}ms")

    with lock:
      hwmon_state.voltage = voltage
      hwmon_state.current = current
      hwmon_state.initialized = True

    exit_event.wait(0.5)


def fill_panda_state(ps, hw_type: int, health: dict[str, Any]) -> None:
  ps.voltage = health["voltage"]
  ps.current = health["current"]
  ps.uptime = health["uptime"]
  ps.safetyTxBlocked = health["safety_tx_blocked"]
  ps.safetyRxInvalid = health["safety_rx_invalid"]
  ps.ignitionLine = bool(health["ignition_line"])
  ps.ignitionCan = bool(health["ignition_can"])
  ps.controlsAllowed = bool(health["controls_allowed"])
  ps.txBufferOverflow = health["tx_buffer_overflow"]
  ps.rxBufferOverflow = health["rx_buffer_overflow"]
  ps.pandaType = hw_type
  ps.safetyModel = health["safety_mode"]
  ps.safetyParam = health["safety_param"]
  ps.faultStatus = health["fault_status"]
  ps.powerSaveEnabled = bool(health["power_save_enabled"])
  ps.heartbeatLost = bool(health["heartbeat_lost"])
  ps.alternativeExperience = health["alternative_experience"]
  ps.harnessStatus = health["car_harness_status"]
  ps.interruptLoad = health["interrupt_load"]
  ps.fanPower = health["fan_power"]
  ps.safetyRxChecksInvalid = bool(health["safety_rx_checks_invalid"])
  ps.spiErrorCount = health["spi_error_count"]
  ps.sbu1Voltage = health["sbu1_voltage_mV"] / 1000.0
  ps.sbu2Voltage = health["sbu2_voltage_mV"] / 1000.0
  ps.soundOutputLevel = health["sound_output_level"]


def fill_panda_can_state(cs, can_health: dict[str, Any]) -> None:
  cs.busOff = bool(can_health["bus_off"])
  cs.busOffCnt = can_health["bus_off_cnt"]
  cs.errorWarning = bool(can_health["error_warning"])
  cs.errorPassive = bool(can_health["error_passive"])
  cs.lastError = can_health["last_error"]
  cs.lastStoredError = can_health["last_stored_error"]
  cs.lastDataError = can_health["last_data_error"]
  cs.lastDataStoredError = can_health["last_data_stored_error"]
  cs.receiveErrorCnt = can_health["receive_error_cnt"]
  cs.transmitErrorCnt = can_health["transmit_error_cnt"]
  cs.totalErrorCnt = can_health["total_error_cnt"]
  cs.totalTxLostCnt = can_health["total_tx_lost_cnt"]
  cs.totalRxLostCnt = can_health["total_rx_lost_cnt"]
  cs.totalTxCnt = can_health["total_tx_cnt"]
  cs.totalRxCnt = can_health["total_rx_cnt"]
  cs.totalFwdCnt = can_health["total_fwd_cnt"]
  cs.canSpeed = can_health["can_speed"]
  cs.canDataSpeed = can_health["can_data_speed"]
  cs.canfdEnabled = bool(can_health["canfd_enabled"])
  cs.brsEnabled = bool(can_health["brs_enabled"])
  cs.canfdNonIso = bool(can_health["canfd_non_iso"])
  cs.irq0CallRate = can_health["irq0_call_rate"]
  cs.irq1CallRate = can_health["irq1_call_rate"]
  cs.irq2CallRate = can_health["irq2_call_rate"]
  cs.canCoreResetCnt = can_health["can_core_reset_count"]


def send_panda_states(pm: messaging.PubMaster, panda: PandadPanda, is_onroad: bool, spoofing_started: bool) -> bool | None:
  health = panda.health()
  if health is None:
    return None

  can_health = []
  for bus in range(PANDA_CAN_CNT):
    state = panda.can_health(bus)
    if state is None:
      return None
    can_health.append(state)

  if spoofing_started:
    health["ignition_line"] = 1

  ignition_local = bool(health["ignition_line"] or health["ignition_can"])

  if health["safety_mode"] == int(car.CarParams.SafetyModel.silent):
    panda.set_safety_mode(car.CarParams.SafetyModel.noOutput)

  power_save_desired = not ignition_local
  if bool(health["power_save_enabled"]) != power_save_desired:
    panda.set_power_save(power_save_desired)

  should_close_relay = not ignition_local or not is_onroad
  if should_close_relay and health["safety_mode"] != int(car.CarParams.SafetyModel.noOutput):
    panda.set_safety_mode(car.CarParams.SafetyModel.noOutput)

  msg = messaging.new_message("pandaStates", 1, valid=panda.comms_healthy)
  ps = msg.pandaStates[0]
  fill_panda_state(ps, panda.hw_type, health)
  for name, state in zip(("canState0", "canState1", "canState2"), can_health, strict=True):
    fill_panda_can_state(ps.init(name), state)

  fault_values = [fault for fault in range(int(log.PandaState.FaultType.heartbeatLoopWatchdog) + 1)
                  if health["faults"] & (1 << fault) and fault in FAULT_NAME_BY_VALUE]
  faults = ps.init("faults", len(fault_values))
  for i, fault in enumerate(fault_values):
    faults[i] = FAULT_NAME_BY_VALUE[fault]

  pm.send("pandaStates", msg)
  return ignition_local


def send_peripheral_state(panda: PandadPanda, pm: messaging.PubMaster, hwmon_state: HwmonState, lock: threading.Lock) -> None:
  with lock:
    if not hwmon_state.initialized:
      return
    voltage = hwmon_state.voltage
    current = hwmon_state.current

  msg = messaging.new_message("peripheralState", valid=panda.comms_healthy)
  ps = msg.peripheralState
  ps.pandaType = panda.hw_type
  ps.voltage = voltage
  ps.current = current

  if ps.voltage == 0 and ps.current == 0:
    health = panda.health()
    if health is not None:
      ps.voltage = health["voltage"]
      ps.current = health["current"]

  ps.fanSpeedRpm = panda.get_fan_rpm()
  pm.send("peripheralState", msg)


def process_panda_state(panda: PandadPanda, pm: messaging.PubMaster, engaged: bool,
                        is_onroad: bool, spoofing_started: bool) -> None:
  ignition = send_panda_states(pm, panda, is_onroad, spoofing_started)
  if ignition is None:
    cloudlog.error("Failed to get ignition state")
    return

  if not ignition and not panda.comms_healthy:
    cloudlog.error("Reconnecting, communication to panda not healthy")
    exit_event.set()

  panda.send_heartbeat(engaged)


def pandad_run(panda: PandadPanda) -> None:
  if not PC:
    config_realtime_process(3, 54)

  no_fan_control = "NO_FAN_CONTROL" in os.environ
  spoofing_started = "STARTED" in os.environ
  fake_send = "FAKESEND" in os.environ

  hwmon_state = HwmonState()
  hwmon_lock = threading.Lock()
  recv_thread = threading.Thread(target=can_recv_thread, args=(panda,), name="pandad_can_recv")
  send_thread = threading.Thread(target=can_send_thread, args=(panda, fake_send), name="pandad_can_send")
  hardware_thread = threading.Thread(target=hwmon_thread, args=(hwmon_state, hwmon_lock), name="pandad_hwmon")
  recv_thread.start()
  send_thread.start()
  hardware_thread.start()

  rk = Ratekeeper(100)
  sm = messaging.SubMaster(["selfdriveState", "deviceState"])
  pm = messaging.PubMaster(["pandaStates", "peripheralState"])
  panda_safety = PandaSafety(panda)
  peripheral_controller = PeripheralController()
  engaged = False
  is_onroad = False

  try:
    while not exit_event.is_set() and panda.connected:
      if rk.frame % 5 == 0:
        peripheral_controller.update(panda, no_fan_control, is_onroad)

      if rk.frame % 10 == 0:
        sm.update(0)
        engaged = sm.all_checks(["selfdriveState"]) and sm["selfdriveState"].enabled
        if sm.updated["deviceState"]:
          is_onroad = sm["deviceState"].started
        process_panda_state(panda, pm, engaged, is_onroad, spoofing_started)
        panda_safety.configure_safety_mode(is_onroad)

      if rk.frame % 50 == 0:
        send_peripheral_state(panda, pm, hwmon_state, hwmon_lock)

      if not is_onroad and rk.frame % PANDA_LOG_READ_INTERVAL == 0:
        panda_log = panda.serial_read(timeout=CAN_RECV_TIMEOUT_MS)
        if panda_log:
          text = panda_log.decode("utf-8", "replace")
          if "Register 0x" in text:
            cloudlog.error(text)
          else:
            cloudlog.debug(text)

      rk.keep_time()
  finally:
    if is_onroad and not engaged and panda.connected:
      try:
        panda.set_safety_mode(car.CarParams.SafetyModel.noOutput)
      except PandaSpiException:
        pass

    exit_event.set()
    recv_thread.join()
    send_thread.join()
    hardware_thread.join()


def pandad_main_thread(serial: str | None = None) -> None:
  if serial is None:
    serials = PandadPanda.list_pandas()
    if not serials:
      cloudlog.warning("no pandas found, exiting")
      return
    serial = serials[0]

  cloudlog.warning(f"connecting to panda: {serial}")
  panda = None
  while not exit_event.is_set():
    panda = connect(serial)
    if panda is not None:
      break
    time.sleep(0.1)

  if not exit_event.is_set() and panda is not None:
    cloudlog.warning("connected to panda")
    try:
      pandad_run(panda)
    finally:
      panda.close()


def check_heartbeat_lost() -> None:
  try:
    for serial in PandadPanda.list_pandas():
      panda = PandadPanda(serial)
      try:
        health = panda.health()
        if health is not None and panda.is_internal() and health["heartbeat_lost"]:
          Params().put_bool("PandaHeartbeatLost", True, block=True)
          cloudlog.event("heartbeat lost", deviceState=health)
      finally:
        panda.close()
  except Exception:
    cloudlog.exception("pandad.uncaught_exception")


def main() -> None:
  def signal_handler(signum, frame) -> None:
    cloudlog.info(f"Caught signal {signum}, exiting")
    shutdown_event.set()
    exit_event.set()

  signal.signal(signal.SIGINT, signal_handler)
  signal.signal(signal.SIGTERM, signal_handler)

  check_heartbeat_lost()

  count = 0
  while not shutdown_event.is_set():
    try:
      cloudlog.event("pandad.flash_and_connect", count=count)
      if (count % 2) == 0:
        HARDWARE.reset_internal_panda()
      else:
        HARDWARE.recover_internal_panda()
      count += 1

      for serial in PandaDFU.list():
        cloudlog.info(f"Panda in DFU mode found, flashing recovery {serial}")
        PandaDFU(serial).recover()
        time.sleep(1)

      panda_serials = PandadPanda.list_pandas()
      if len(panda_serials) == 0:
        time.sleep(0.1)
        continue
      assert len(panda_serials) == 1

      cloudlog.info(f"{len(panda_serials)} panda found, connecting - {panda_serials}")
      flash_panda(panda_serials[0])

      os.environ["MANAGER_DAEMON"] = "pandad"
      exit_event.clear()
      pandad_main_thread(panda_serials[0])
    except (usb1.USBErrorNoDevice, usb1.USBErrorPipe):
      cloudlog.exception("Panda USB exception while setting up")
    except PandaProtocolMismatch:
      cloudlog.exception("pandad.protocol_mismatch")
    except Exception:
      cloudlog.exception("pandad.uncaught_exception")


if __name__ == "__main__":
  main()
