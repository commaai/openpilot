#!/usr/bin/env python3
"""
Python implementation of the pandad daemon.
This replaces the C++ pandad binary with a pure Python implementation.
"""
import os
import sys
import time
import threading
from typing import Optional
from dataclasses import dataclass

import cereal.messaging as messaging
from cereal import car, log
from panda import Panda

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.common.realtime import Ratekeeper, config_realtime_process
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.system.hardware import HARDWARE, PC

# Constants from C++ pandad
PANDA_CAN_CNT = 3  # Number of CAN buses per panda (matches panda firmware)
PANDA_BUS_OFFSET = 4
MAX_IR_PANDA_VAL = 50
CUTOFF_IL = 400
SATURATE_IL = 1000


def map_val(x: float, x_min: float, x_max: float, y_min: float, y_max: float) -> float:
  """Map a value from one range to another."""
  return y_min + (y_max - y_min) * (x - x_min) / (x_max - x_min)


@dataclass
class CanFrame:
  """CAN frame data structure."""
  address: int
  data: bytes
  src: int


class PandaWrapper:
  """Wrapper around Panda to add bus offset and daemon-specific functionality."""

  def __init__(self, panda: Panda, bus_offset: int):
    self._panda = panda
    self.bus_offset = bus_offset
    self._serial = panda.get_usb_serial()
    self._hw_type = self._get_hw_type()

  def _get_hw_type(self) -> int:
    """Get the hardware type enum value."""
    hw_type_bytes = bytes(self._panda.get_type())
    # Map hardware type bytes to cereal enum values
    hw_type_map = {
      Panda.HW_TYPE_RED_PANDA: log.PandaState.PandaType.redPanda,
      Panda.HW_TYPE_TRES: log.PandaState.PandaType.tres,
      Panda.HW_TYPE_CUATRO: log.PandaState.PandaType.cuatro,
    }
    return hw_type_map.get(hw_type_bytes, log.PandaState.PandaType.unknown)

  @property
  def hw_type(self) -> int:
    return self._hw_type

  @property
  def hw_serial(self) -> str:
    return self._serial

  def connected(self) -> bool:
    return self._panda.connected

  def comms_healthy(self) -> bool:
    # For now, assume comms are healthy if connected
    # The Python Panda library doesn't have a direct comms_healthy status
    return self._panda.connected

  def health(self) -> dict:
    return self._panda.health()

  def can_health(self, bus: int) -> dict:
    return self._panda.can_health(bus)

  def can_recv(self) -> list[tuple[int, bytes, int]]:
    """Receive CAN messages and adjust bus numbers with offset."""
    msgs = self._panda.can_recv()
    result = []
    for addr, data, bus in msgs:
      # Apply bus offset for multi-panda setups
      adjusted_bus = bus
      if bus < 128:  # Not a returned/rejected message
        adjusted_bus = bus + self.bus_offset
      elif bus < 192:  # Returned message
        adjusted_bus = (bus - 128) + self.bus_offset + 128
      else:  # Rejected message
        adjusted_bus = (bus - 192) + self.bus_offset + 192
      result.append((addr, data, adjusted_bus))
    return result

  def can_send(self, msgs: list) -> None:
    """Send CAN messages, adjusting bus numbers to remove offset."""
    adjusted_msgs = []
    for addr, data, bus in msgs:
      # Remove bus offset to get the actual bus number for this panda
      actual_bus = bus - self.bus_offset
      if 0 <= actual_bus < PANDA_CAN_CNT:
        adjusted_msgs.append((addr, data, actual_bus))
    if adjusted_msgs:
      self._panda.can_send_many(adjusted_msgs)

  def set_safety_model(self, mode: int, param: int = 0) -> None:
    self._panda.set_safety_mode(mode, param)

  def set_alternative_experience(self, alt_exp: int) -> None:
    self._panda.set_alternative_experience(alt_exp)

  def set_power_saving(self, enabled: bool) -> None:
    self._panda.set_power_save(int(enabled))

  def set_fan_speed(self, speed: int) -> None:
    self._panda.set_fan_power(speed)

  def get_fan_speed(self) -> int:
    return self._panda.get_fan_rpm()

  def set_ir_pwr(self, pwr: int) -> None:
    self._panda.set_ir_power(pwr)

  def set_can_fd_auto(self, bus: int, enabled: bool) -> None:
    self._panda.set_canfd_auto(bus, enabled)

  def set_loopback(self, enabled: bool) -> None:
    self._panda.set_can_loopback(enabled)

  def send_heartbeat(self, engaged: bool) -> None:
    self._panda.send_heartbeat(engaged)

  def serial_read(self) -> str:
    data = self._panda.serial_read(Panda.SERIAL_DEBUG)
    return data.decode('utf-8', errors='ignore') if data else ""

  def is_internal(self) -> bool:
    return self._panda.is_internal()

  def up_to_date(self) -> bool:
    return self._panda.up_to_date()

  def close(self) -> None:
    self._panda.close()


class PandaSafety:
  """Manages safety configuration for pandas."""

  def __init__(self, pandas: list[PandaWrapper]):
    self.pandas = pandas
    self.params = Params()
    self.initialized = False
    self.log_once = False
    self.safety_configured = False
    self.prev_obd_multiplexing = False

  def configure_safety_mode(self, is_onroad: bool) -> None:
    """Configure safety mode based on onroad state."""
    if is_onroad and not self.safety_configured:
      self._update_multiplexing_mode()

      car_params = self._fetch_car_params()
      if car_params:
        cloudlog.warning(f"got {len(car_params)} bytes CarParams")
        self._set_safety_mode(car_params)
        self.safety_configured = True
    elif not is_onroad:
      self.initialized = False
      self.safety_configured = False
      self.log_once = False

  def _update_multiplexing_mode(self) -> None:
    """Update OBD multiplexing mode."""
    # Initialize to ELM327 without OBD multiplexing for initial fingerprinting
    if not self.initialized:
      self.prev_obd_multiplexing = False
      for panda in self.pandas:
        panda.set_safety_model(int(car.CarParams.SafetyModel.elm327), 1)
      self.initialized = True

    # Switch between multiplexing modes based on the OBD multiplexing request
    obd_multiplexing_requested = self.params.get_bool("ObdMultiplexingEnabled")
    if obd_multiplexing_requested != self.prev_obd_multiplexing:
      for i, panda in enumerate(self.pandas):
        safety_param = 1 if (i > 0 or not obd_multiplexing_requested) else 0
        panda.set_safety_model(int(car.CarParams.SafetyModel.elm327), safety_param)
      self.prev_obd_multiplexing = obd_multiplexing_requested
      self.params.put_bool("ObdMultiplexingChanged", True)

  def _fetch_car_params(self) -> Optional[bytes]:
    """Fetch car parameters from params."""
    if not self.params.get_bool("FirmwareQueryDone"):
      return None

    if not self.log_once:
      cloudlog.warning("Finished FW query, Waiting for params to set safety model")
      self.log_once = True

    if not self.params.get_bool("ControlsReady"):
      return None

    return self.params.get("CarParams")

  def _set_safety_mode(self, params_bytes: bytes) -> None:
    """Set safety mode from CarParams."""
    car_params = messaging.log_from_bytes(params_bytes, car.CarParams)
    safety_configs = car_params.safetyConfigs
    alternative_experience = car_params.alternativeExperience

    for i, panda in enumerate(self.pandas):
      # Default to SILENT safety model if not specified
      safety_model = int(car.CarParams.SafetyModel.silent)
      safety_param = 0
      if i < len(safety_configs):
        safety_model = safety_configs[i].safetyModel.raw
        safety_param = safety_configs[i].safetyParam

      cloudlog.warning(f"Panda {i}: setting safety model: {safety_model}, param: {safety_param}, alternative experience: {alternative_experience}")
      panda.set_alternative_experience(int(alternative_experience))
      panda.set_safety_model(safety_model, safety_param)


class PeripheralState:
  """Manages peripheral state (fan, IR power)."""

  def __init__(self):
    self.params = Params()
    self.last_driver_camera_t = 0
    self.prev_fan_speed = 999
    self.ir_pwr = 0
    self.prev_ir_pwr = 999
    self.prev_frame_id = 0xFFFFFFFF
    self.driver_view = False

    # Filters for integration lines (from C++)
    # FirstOrderFilter(x0, rc, dt)
    self.integ_lines_filter = FirstOrderFilter(0, 30.0, 0.05)
    self.integ_lines_filter_driver_view = FirstOrderFilter(0, 5.0, 0.05)

  def process(self, panda: PandaWrapper, sm: messaging.SubMaster, frame: int, no_fan_control: bool) -> None:
    """Process peripheral state updates."""
    if sm.updated['deviceState'] and not no_fan_control:
      # Fan speed
      fan_speed = sm['deviceState'].fanSpeedPercentDesired
      if fan_speed != self.prev_fan_speed or frame % 100 == 0:
        panda.set_fan_speed(fan_speed)
        self.prev_fan_speed = fan_speed

    if sm.updated['driverCameraState']:
      event = sm['driverCameraState']
      cur_integ_lines = event.integLines

      # Reset the filter when camerad restarts
      if event.frameId < self.prev_frame_id:
        self.integ_lines_filter = FirstOrderFilter(0, 30.0, 0.05)
        self.integ_lines_filter_driver_view = FirstOrderFilter(0, 5.0, 0.05)
        self.driver_view = self.params.get_bool("IsDriverViewEnabled")
      self.prev_frame_id = event.frameId

      if self.driver_view:
        cur_integ_lines = self.integ_lines_filter_driver_view.update(cur_integ_lines)
      else:
        cur_integ_lines = self.integ_lines_filter.update(cur_integ_lines)
      self.last_driver_camera_t = event.timestampSof

      if cur_integ_lines <= CUTOFF_IL:
        self.ir_pwr = 0
      elif cur_integ_lines > SATURATE_IL:
        self.ir_pwr = 100
      else:
        self.ir_pwr = int(100 * (cur_integ_lines - CUTOFF_IL) / (SATURATE_IL - CUTOFF_IL))

    # Disable IR on input timeout (1 second)
    if time.monotonic_ns() - self.last_driver_camera_t > 1e9:
      self.ir_pwr = 0

    if self.ir_pwr != self.prev_ir_pwr or frame % 100 == 0:
      ir_panda = int(map_val(self.ir_pwr, 0, 100, 0, MAX_IR_PANDA_VAL))
      panda.set_ir_pwr(ir_panda)
      HARDWARE.set_ir_power(self.ir_pwr)
      self.prev_ir_pwr = self.ir_pwr


class Pandad:
  """Main pandad daemon class."""

  def __init__(self, serials: list[str]):
    self.serials = serials
    self.pandas: list[PandaWrapper] = []
    self.do_exit = False
    self.params = Params()

    # Environment flags
    self.no_fan_control = os.environ.get("NO_FAN_CONTROL") is not None
    self.spoofing_started = os.environ.get("STARTED") is not None
    self.fake_send = os.environ.get("FAKESEND") is not None
    self.loopback = os.environ.get("BOARDD_LOOPBACK") is not None

  def connect(self) -> bool:
    """Connect to all pandas."""
    self.pandas = []

    for i, serial in enumerate(self.serials):
      try:
        panda = Panda(serial, claim=True, disable_checks=False, cli=False)

        # Common panda config
        if self.loopback:
          panda.set_can_loopback(True)

        for bus in range(PANDA_CAN_CNT):
          panda.set_canfd_auto(bus, True)

        if not panda.up_to_date() and not os.environ.get("BOARDD_SKIP_FW_CHECK"):
          raise RuntimeError("Panda firmware out of date. Run pandad.py to update.")

        wrapper = PandaWrapper(panda, i * PANDA_BUS_OFFSET)
        self.pandas.append(wrapper)
        cloudlog.info(f"Connected to panda {serial}")
      except Exception as e:
        cloudlog.exception(f"Failed to connect to panda {serial}: {e}")
        return False

    return len(self.pandas) == len(self.serials)

  def check_all_connected(self) -> bool:
    """Check if all pandas are still connected."""
    for panda in self.pandas:
      if not panda.connected():
        self.do_exit = True
        return False
    return True

  def can_send_thread(self) -> None:
    """Thread for sending CAN messages."""
    cloudlog.info("can_send_thread started")

    sm = messaging.SubMaster(['sendcan'])

    while not self.do_exit and self.check_all_connected():
      sm.update(100)

      if not sm.updated['sendcan']:
        continue

      # Don't send if older than 1 second
      event = sm['sendcan']
      age_ns = time.monotonic_ns() - sm.logMonoTime['sendcan']

      if age_ns < 1e9 and not self.fake_send:
        # Convert sendcan messages to format expected by panda
        msgs = [(msg.address, msg.dat, msg.src) for msg in event]

        for panda in self.pandas:
          panda.can_send(msgs)
      else:
        cloudlog.error(f"sendcan too old to send: age={age_ns / 1e6:.1f}ms")

    cloudlog.info("can_send_thread exiting")

  def can_recv(self, pm: messaging.PubMaster) -> None:
    """Receive CAN messages from all pandas and publish."""
    raw_can_data = []
    comms_healthy = True

    for panda in self.pandas:
      try:
        msgs = panda.can_recv()
        raw_can_data.extend(msgs)
        comms_healthy &= panda.comms_healthy()
      except Exception:
        cloudlog.exception("CAN receive error")
        comms_healthy = False

    # Build and send the message
    msg = messaging.new_message('can', len(raw_can_data), valid=comms_healthy)
    for i, (addr, data, bus) in enumerate(raw_can_data):
      msg.can[i].address = addr
      msg.can[i].dat = data
      msg.can[i].src = bus

    pm.send('can', msg)

  def send_panda_states(self, pm: messaging.PubMaster, is_onroad: bool) -> Optional[bool]:
    """Send panda states and return ignition state."""
    ignition_local = False

    msg = messaging.new_message('pandaStates', len(self.pandas))

    # Check for red panda comma three setup
    red_panda_comma_three = (
      len(self.pandas) == 2 and
      self.pandas[0].hw_type == log.PandaState.PandaType.dos and
      self.pandas[1].hw_type == log.PandaState.PandaType.redPanda
    )

    panda_states = []
    panda_can_states = []

    for panda in self.pandas:
      try:
        health = panda.health()
        can_healths = [panda.can_health(i) for i in range(PANDA_CAN_CNT)]
      except Exception:
        cloudlog.exception("Failed to get panda state")
        return None

      panda_states.append(health)
      panda_can_states.append(can_healths)

      if self.spoofing_started:
        health['ignition_line'] = 1

      # On comma three with red panda, ignore DOS ignition (false positives)
      if red_panda_comma_three and panda.hw_type == log.PandaState.PandaType.dos:
        health['ignition_line'] = 0

      ignition_local |= (health['ignition_line'] != 0) or (health['ignition_can'] != 0)

    for i, panda in enumerate(self.pandas):
      health = panda_states[i]
      can_healths = panda_can_states[i]

      # Make sure CAN buses are live
      if health['safety_mode'] == car.CarParams.SafetyModel.silent:
        panda.set_safety_model(int(car.CarParams.SafetyModel.noOutput))

      # Power save mode
      power_save_desired = not ignition_local
      if health['power_save_enabled'] != power_save_desired:
        panda.set_power_saving(power_save_desired)

      # Set safety mode to NO_OUTPUT when car is off or not onroad
      should_close_relay = not ignition_local or not is_onroad
      if should_close_relay and health['safety_mode'] != car.CarParams.SafetyModel.noOutput:
        panda.set_safety_model(int(car.CarParams.SafetyModel.noOutput))

      if not panda.comms_healthy():
        msg.valid = False

      # Fill panda state
      ps = msg.pandaStates[i]
      ps.voltage = health['voltage']
      ps.current = health['current']
      ps.uptime = health['uptime']
      ps.safetyTxBlocked = health['safety_tx_blocked']
      ps.safetyRxInvalid = health['safety_rx_invalid']
      ps.ignitionLine = bool(health['ignition_line'])
      ps.ignitionCan = bool(health['ignition_can'])
      ps.controlsAllowed = bool(health['controls_allowed'])
      ps.txBufferOverflow = health['tx_buffer_overflow']
      ps.rxBufferOverflow = health['rx_buffer_overflow']
      ps.pandaType = panda.hw_type
      ps.safetyModel = health['safety_mode']
      ps.safetyParam = health['safety_param']
      ps.faultStatus = health['fault_status']
      ps.powerSaveEnabled = bool(health['power_save_enabled'])
      ps.heartbeatLost = bool(health['heartbeat_lost'])
      ps.alternativeExperience = health['alternative_experience']
      ps.harnessStatus = health['car_harness_status']
      ps.interruptLoad = health['interrupt_load']
      ps.fanPower = health['fan_power']
      ps.safetyRxChecksInvalid = bool(health['safety_rx_checks_invalid'])
      ps.spiErrorCount = health.get('spi_error_count', 0)
      ps.sbu1Voltage = health.get('sbu1_voltage_mV', 0) / 1000.0
      ps.sbu2Voltage = health.get('sbu2_voltage_mV', 0) / 1000.0

      # Fill CAN states
      # Map error code strings back to enum values
      lec_error_map = {
        "No error": 0, "Stuff error": 1, "Form error": 2, "AckError": 3,
        "Bit1Error": 4, "Bit0Error": 5, "CRCError": 6, "NoChange": 7,
      }
      for j, can_health in enumerate(can_healths):
        cs = getattr(ps, f'canState{j}')
        cs.busOff = bool(can_health['bus_off'])
        cs.busOffCnt = can_health['bus_off_cnt']
        cs.errorWarning = bool(can_health['error_warning'])
        cs.errorPassive = bool(can_health['error_passive'])
        cs.lastError = lec_error_map.get(can_health['last_error'], 0)
        cs.lastStoredError = lec_error_map.get(can_health['last_stored_error'], 0)
        cs.lastDataError = lec_error_map.get(can_health['last_data_error'], 0)
        cs.lastDataStoredError = lec_error_map.get(can_health['last_data_stored_error'], 0)
        cs.receiveErrorCnt = can_health['receive_error_cnt']
        cs.transmitErrorCnt = can_health['transmit_error_cnt']
        cs.totalErrorCnt = can_health['total_error_cnt']
        cs.totalTxLostCnt = can_health['total_tx_lost_cnt']
        cs.totalRxLostCnt = can_health['total_rx_lost_cnt']
        cs.totalTxCnt = can_health['total_tx_cnt']
        cs.totalRxCnt = can_health['total_rx_cnt']
        cs.totalFwdCnt = can_health['total_fwd_cnt']
        cs.canSpeed = can_health['can_speed']
        cs.canDataSpeed = can_health['can_data_speed']
        cs.canfdEnabled = bool(can_health['canfd_enabled'])
        cs.brsEnabled = bool(can_health['brs_enabled'])
        cs.canfdNonIso = bool(can_health['canfd_non_iso'])
        cs.irq0CallRate = can_health['irq0_call_rate']
        cs.irq1CallRate = can_health['irq1_call_rate']
        cs.irq2CallRate = can_health['irq2_call_rate']
        cs.canCoreResetCnt = can_health['can_core_reset_count']

      # Fill faults
      faults_pkt = health['faults']
      fault_list = []
      for f in range(32):
        if faults_pkt & (1 << f):
          fault_list.append(f)
      ps.init('faults', len(fault_list))
      for j, fault in enumerate(fault_list):
        ps.faults[j] = fault

    pm.send('pandaStates', msg)
    return ignition_local

  def process_panda_state(self, pm: messaging.PubMaster, sm: messaging.SubMaster, engaged: bool, is_onroad: bool) -> None:
    """Process panda state at 10 Hz."""
    connected_serials = [p.hw_serial for p in self.pandas]

    ignition_opt = self.send_panda_states(pm, is_onroad)
    if ignition_opt is None:
      cloudlog.error("Failed to get ignition_opt")
      return

    # Check if we should reconnect
    if not ignition_opt:
      comms_healthy = all(p.comms_healthy() for p in self.pandas)

      if not comms_healthy:
        cloudlog.error("Reconnecting, communication to pandas not healthy")
        self.do_exit = True
      else:
        # Check for new pandas
        for s in Panda.list(usb_only=True):
          if s not in connected_serials:
            cloudlog.warning(f"Reconnecting to new panda: {s}")
            self.do_exit = True
            break

    # Send heartbeat
    for panda in self.pandas:
      panda.send_heartbeat(engaged)

  def send_peripheral_state(self, panda: PandaWrapper, pm: messaging.PubMaster) -> None:
    """Send peripheral state message."""
    msg = messaging.new_message('peripheralState', valid=panda.comms_healthy())
    ps = msg.peripheralState

    ps.pandaType = panda.hw_type

    # Read voltage and current
    start_time = time.monotonic()
    voltage = HARDWARE.get_voltage()
    current = HARDWARE.get_current()
    read_time = (time.monotonic() - start_time) * 1000

    if read_time > 50:
      cloudlog.warning(f"reading hwmon took {read_time:.1f}ms")

    ps.voltage = voltage
    ps.current = current

    # Fall back to panda's voltage and current measurement
    if ps.voltage == 0 and ps.current == 0:
      try:
        health = panda.health()
        ps.voltage = health['voltage']
        ps.current = health['current']
      except Exception:
        pass

    ps.fanSpeedRpm = panda.get_fan_speed()

    pm.send('peripheralState', msg)

  def run(self) -> None:
    """Main daemon loop."""
    cloudlog.info("pandad_run starting")

    # Start CAN send thread
    send_thread = threading.Thread(target=self.can_send_thread, daemon=True)
    send_thread.start()

    # Initialize messaging
    sm = messaging.SubMaster(['selfdriveState', 'deviceState', 'driverCameraState'])
    pm = messaging.PubMaster(['can', 'pandaStates', 'peripheralState'])

    # Initialize components
    panda_safety = PandaSafety(self.pandas)
    peripheral_state = PeripheralState()
    peripheral_panda = self.pandas[0]

    engaged = False
    is_onroad = False

    # Main loop at 100 Hz
    rk = Ratekeeper(100, print_delay_threshold=None)

    while not self.do_exit and self.check_all_connected():
      # Receive CAN messages (100 Hz)
      self.can_recv(pm)

      # Process peripheral state at 20 Hz
      if rk.frame % 5 == 0:
        sm.update(0)
        peripheral_state.process(peripheral_panda, sm, rk.frame, self.no_fan_control)

      # Process panda state at 10 Hz
      if rk.frame % 10 == 0:
        sm.update(0)
        if sm.all_checks(['selfdriveState']):
          engaged = sm['selfdriveState'].enabled
        else:
          engaged = False
        is_onroad = self.params.get_bool("IsOnroad")
        self.process_panda_state(pm, sm, engaged, is_onroad)
        panda_safety.configure_safety_mode(is_onroad)

      # Send peripheral state at 2 Hz
      if rk.frame % 50 == 0:
        self.send_peripheral_state(peripheral_panda, pm)

      # Forward logs from pandas
      for panda in self.pandas:
        log_data = panda.serial_read()
        if log_data:
          if "Register 0x" in log_data:
            cloudlog.error(log_data)
          else:
            cloudlog.debug(log_data)

      rk.keep_time()

    # Close relay on exit to prevent a fault
    if is_onroad and not engaged:
      for panda in self.pandas:
        if panda.connected():
          panda.set_safety_model(int(car.CarParams.SafetyModel.noOutput))

    # Wait for send thread to finish
    send_thread.join(timeout=1.0)

    cloudlog.info("pandad_run exiting")

  def cleanup(self) -> None:
    """Clean up resources."""
    for panda in self.pandas:
      try:
        panda.close()
      except Exception:
        pass


def pandad_main_thread(serials: list[str]) -> None:
  """Main entry point for pandad."""
  if not serials:
    serials = Panda.list()
    if not serials:
      cloudlog.warning("no pandas found, exiting")
      return

  serials_str = ", ".join(serials)
  cloudlog.warning(f"connecting to pandas: {serials_str}")

  daemon = Pandad(serials)

  if not daemon.connect():
    cloudlog.error("Failed to connect to all pandas")
    daemon.cleanup()
    return

  cloudlog.warning("connected to all pandas")

  try:
    daemon.run()
  finally:
    daemon.cleanup()


def main() -> None:
  """Main entry point."""
  cloudlog.warning("starting pandad (Python)")

  if not PC:
    config_realtime_process([3], 54)

  serials = sys.argv[1:] if len(sys.argv) > 1 else []
  pandad_main_thread(serials)


if __name__ == "__main__":
  main()
