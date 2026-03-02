#!/usr/bin/env python3
"""Python pandad daemon - replaces the C++ pandad binary.

Single-threaded design: the main 100Hz loop handles CAN recv, CAN send,
health/state publishing, safety mode, IR power, and fan speed.

The only background thread is for reading hwmon (voltage/current) since
that can block indefinitely and is best-effort.
"""
import os
import threading
import time

from cereal import log, messaging
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper, config_realtime_process
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware import HARDWARE, PC
from opendbc.car.structs import CarParams
from panda import Panda

PANDA_CAN_CNT = 3

MAX_IR_PANDA_VAL = 50
CUTOFF_IL = 400
SATURATE_IL = 1000

FAULT_RELAY_MALFUNCTION = 0
FAULT_HEARTBEAT_LOOP_WATCHDOG = 26

# Reverse map from Python panda library's LEC string to capnp enum name
LEC_STRING_TO_CAPNP = {
  "No error": "noError",
  "Stuff error": "stuffError",
  "Form error": "formError",
  "AckError": "ackError",
  "Bit1Error": "bit1Error",
  "Bit0Error": "bit0Error",
  "CRCError": "crcError",
  "NoChange": "noChange",
}

do_exit = False


def check_connected(panda):
  global do_exit
  if not panda.connected:
    do_exit = True
    return False
  return True


def connect(serial):
  """Connect to panda and configure for daemon use."""
  try:
    panda = Panda(serial, claim=True, disable_checks=False, cli=False)
  except Exception:
    return None

  # common panda config
  if os.getenv("BOARDD_LOOPBACK"):
    panda.set_can_loopback(True)

  for i in range(PANDA_CAN_CNT):
    panda.set_canfd_auto(i, True)

  if not panda.up_to_date() and not os.getenv("BOARDD_SKIP_FW_CHECK"):
    raise RuntimeError("Panda firmware out of date. Run pandad.py to update.")

  return panda


HW_TYPE_MAP = {
  Panda.HW_TYPE_UNKNOWN: log.PandaState.PandaType.unknown,
  Panda.HW_TYPE_RED_PANDA: log.PandaState.PandaType.redPanda,
  Panda.HW_TYPE_TRES: log.PandaState.PandaType.tres,
  Panda.HW_TYPE_CUATRO: log.PandaState.PandaType.cuatro,
}


def get_hw_type_capnp(panda):
  """Convert panda hardware type bytes to capnp PandaType enum."""
  hw_type = panda.get_type()
  return HW_TYPE_MAP.get(hw_type, log.PandaState.PandaType.unknown)


def can_send(panda, sendcan_sock, fake_send):
  """Drain all pending sendcan messages and forward to panda."""
  for evt in messaging.drain_sock(sendcan_sock):
    # Don't send if older than 1 second
    cur_time = int(time.monotonic() * 1e9)
    if (cur_time - evt.logMonoTime) < 1e9 and not fake_send:
      can_msgs = [(msg.address, msg.dat, msg.src) for msg in evt.sendcan]
      panda.can_send_many(can_msgs)
    else:
      cloudlog.error(f"sendcan too old to send: {cur_time}, {evt.logMonoTime}")


def can_recv(panda, pm):
  """Receive CAN messages from panda and publish to 'can' topic."""
  try:
    raw_can_data = panda.can_recv()
    comms_healthy = True
  except Exception:
    raw_can_data = []
    comms_healthy = False

  msg = messaging.new_message('can', len(raw_can_data))
  msg.valid = comms_healthy
  for i, (address, dat, src) in enumerate(raw_can_data):
    msg.can[i].address = address
    msg.can[i].dat = dat
    msg.can[i].src = src
  pm.send('can', msg)
  return comms_healthy


def fill_panda_state(ps, hw_type, health):
  """Fill a PandaState capnp builder from health dict."""
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
  ps.pandaType = hw_type
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
  ps.spiErrorCount = health['spi_error_count']
  ps.sbu1Voltage = health['sbu1_voltage_mV'] / 1000.0
  ps.sbu2Voltage = health['sbu2_voltage_mV'] / 1000.0


def fill_panda_can_state(cs, can_health):
  """Fill a PandaCanState capnp builder from can_health dict."""
  cs.busOff = bool(can_health['bus_off'])
  cs.busOffCnt = can_health['bus_off_cnt']
  cs.errorWarning = bool(can_health['error_warning'])
  cs.errorPassive = bool(can_health['error_passive'])
  cs.lastError = LEC_STRING_TO_CAPNP.get(can_health['last_error'], 'noError')
  cs.lastStoredError = LEC_STRING_TO_CAPNP.get(can_health['last_stored_error'], 'noError')
  cs.lastDataError = LEC_STRING_TO_CAPNP.get(can_health['last_data_error'], 'noError')
  cs.lastDataStoredError = LEC_STRING_TO_CAPNP.get(can_health['last_data_stored_error'], 'noError')
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


def send_panda_states(pm, panda, hw_type, is_onroad, spoofing_started):
  """Build and publish pandaStates message. Returns ignition state or None on error."""
  msg = messaging.new_message('pandaStates', 1)

  try:
    health = panda.health()
  except Exception:
    return None

  can_healths = []
  for i in range(PANDA_CAN_CNT):
    try:
      can_healths.append(panda.can_health(i))
    except Exception:
      return None

  if spoofing_started:
    health['ignition_line'] = 1

  ignition_local = health['ignition_line'] != 0 or health['ignition_can'] != 0

  # Make sure CAN buses are live: safety_setter_thread does not work if Panda CAN are silent
  if health['safety_mode'] == int(CarParams.SafetyModel.silent):
    panda.set_safety_mode(CarParams.SafetyModel.noOutput)

  power_save_desired = not ignition_local
  if bool(health['power_save_enabled']) != power_save_desired:
    panda.set_power_save(int(power_save_desired))

  # set safety mode to NO_OUTPUT when car is off or we're not onroad
  should_close_relay = not ignition_local or not is_onroad
  if should_close_relay and health['safety_mode'] != int(CarParams.SafetyModel.noOutput):
    panda.set_safety_mode(CarParams.SafetyModel.noOutput)

  msg.valid = True

  ps = msg.pandaStates[0]
  fill_panda_state(ps, hw_type, health)

  can_state_builders = [ps.canState0, ps.canState1, ps.canState2]
  for j in range(PANDA_CAN_CNT):
    fill_panda_can_state(can_state_builders[j], can_healths[j])

  # Convert faults bitset to capnp list
  faults_pkt = health['faults']
  fault_list = []
  for f in range(FAULT_RELAY_MALFUNCTION, FAULT_HEARTBEAT_LOOP_WATCHDOG + 1):
    if faults_pkt & (1 << f):
      fault_list.append(f)

  if fault_list:
    faults = ps.init('faults', len(fault_list))
    for j, f in enumerate(fault_list):
      faults[j] = f

  pm.send('pandaStates', msg)
  return ignition_local


class HwmonReader:
  """Background thread for reading hwmon voltage/current (can block indefinitely)."""

  def __init__(self):
    self.voltage = 0
    self.current = 0
    self._lock = threading.Lock()
    self._thread = None

  def start(self):
    if PC:
      return
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()

  def _run(self):
    while not do_exit:
      try:
        read_time = time.monotonic() * 1000
        voltage = HARDWARE.get_voltage()
        current = HARDWARE.get_current()
        read_time = time.monotonic() * 1000 - read_time
        if read_time > 50:
          cloudlog.warning(f"reading hwmon took {read_time:.1f}ms")
        with self._lock:
          self.voltage = voltage
          self.current = current
      except Exception:
        cloudlog.exception("hwmon read failed")
      time.sleep(0.5)

  def get(self):
    with self._lock:
      return self.voltage, self.current


def send_peripheral_state(panda, pm, hw_type, hwmon_reader):
  """Build and publish peripheralState message at 2Hz."""
  msg = messaging.new_message('peripheralState')
  msg.valid = True

  ps = msg.peripheralState
  ps.pandaType = hw_type

  voltage, current = hwmon_reader.get()
  ps.voltage = voltage
  ps.current = current

  # fall back to panda's voltage and current measurement
  if ps.voltage == 0 and ps.current == 0:
    try:
      health = panda.health()
      ps.voltage = health['voltage']
      ps.current = health['current']
    except Exception:
      pass

  try:
    ps.fanSpeedRpm = panda.get_fan_rpm()
  except Exception:
    pass

  pm.send('peripheralState', msg)


class PeripheralStateProcessor:
  """Manages fan speed and IR power control at 20Hz."""

  def __init__(self):
    self.sm = messaging.SubMaster(['deviceState', 'driverCameraState'])
    self.params = Params()
    self.last_driver_camera_t = 0
    self.prev_fan_speed = 999
    self.ir_pwr = 0
    self.prev_ir_pwr = 999
    self.prev_frame_id = 0xFFFFFFFF
    self.driver_view = False
    self.integ_lines_filter = FirstOrderFilter(0, 30.0, 0.05)
    self.integ_lines_filter_driver_view = FirstOrderFilter(0, 5.0, 0.05)
    self.frame = 0

  def update(self, panda, no_fan_control):
    self.sm.update(0)
    self.frame += 1

    if self.sm.updated['deviceState'] and not no_fan_control:
      fan_speed = self.sm['deviceState'].fanSpeedPercentDesired
      if fan_speed != self.prev_fan_speed or self.frame % 100 == 0:
        panda.set_fan_power(fan_speed)
        self.prev_fan_speed = fan_speed

    if self.sm.updated['driverCameraState']:
      event = self.sm['driverCameraState']
      cur_integ_lines = event.integLines

      # reset the filter when camerad restarts
      if event.frameId < self.prev_frame_id:
        self.integ_lines_filter = FirstOrderFilter(0, 30.0, 0.05)
        self.integ_lines_filter_driver_view = FirstOrderFilter(0, 5.0, 0.05)
        self.driver_view = self.params.get_bool("IsDriverViewEnabled")
      self.prev_frame_id = event.frameId

      if self.driver_view:
        cur_integ_lines = self.integ_lines_filter_driver_view.update(cur_integ_lines)
      else:
        cur_integ_lines = self.integ_lines_filter.update(cur_integ_lines)
      self.last_driver_camera_t = self.sm.logMonoTime['driverCameraState']

      if cur_integ_lines <= CUTOFF_IL:
        self.ir_pwr = 0
      elif cur_integ_lines > SATURATE_IL:
        self.ir_pwr = 100
      else:
        self.ir_pwr = 100 * (cur_integ_lines - CUTOFF_IL) // (SATURATE_IL - CUTOFF_IL)

    # Disable IR on input timeout
    cur_time = int(time.monotonic() * 1e9)
    if cur_time - self.last_driver_camera_t > 1e9:
      self.ir_pwr = 0

    if self.ir_pwr != self.prev_ir_pwr or self.frame % 100 == 0:
      # map ir_pwr 0-100 to 0-MAX_IR_PANDA_VAL
      ir_panda = round(self.ir_pwr * MAX_IR_PANDA_VAL / 100)
      panda.set_ir_power(ir_panda)
      HARDWARE.set_ir_power(self.ir_pwr)
      self.prev_ir_pwr = self.ir_pwr


class PandaSafety:
  """Manages safety mode configuration for panda."""

  def __init__(self, panda):
    self.panda = panda
    self.params = Params()
    self.initialized = False
    self.log_once = False
    self.safety_configured = False
    self.prev_obd_multiplexing = False

  def configure_safety_mode(self, is_onroad):
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

  def _update_multiplexing_mode(self):
    # Initialize to ELM327 without OBD multiplexing for initial fingerprinting
    if not self.initialized:
      self.prev_obd_multiplexing = False
      self.panda.set_safety_mode(CarParams.SafetyModel.elm327, 1)
      self.initialized = True

    # Switch between multiplexing modes based on the OBD multiplexing request
    obd_multiplexing_requested = self.params.get_bool("ObdMultiplexingEnabled")
    if obd_multiplexing_requested != self.prev_obd_multiplexing:
      safety_param = 0 if obd_multiplexing_requested else 1
      self.panda.set_safety_mode(CarParams.SafetyModel.elm327, safety_param)
      self.prev_obd_multiplexing = obd_multiplexing_requested
      self.params.put_bool("ObdMultiplexingChanged", True)

  def _fetch_car_params(self):
    if not self.params.get_bool("FirmwareQueryDone"):
      return None

    if not self.log_once:
      cloudlog.warning("Finished FW query, Waiting for params to set safety model")
      self.log_once = True

    if not self.params.get_bool("ControlsReady"):
      return None
    return self.params.get("CarParams")

  def _set_safety_mode(self, params_string):
    with CarParams.from_bytes(params_string) as car_params:
      safety_configs = car_params.safetyConfigs
      alternative_experience = car_params.alternativeExperience

      # .raw converts capnp _DynamicEnum to int for struct.pack in SPI controlWrite
      safety_model = safety_configs[0].safetyModel.raw
      safety_param = safety_configs[0].safetyParam

      cloudlog.warning(f"setting safety model: {safety_model}, param: {safety_param}, alternative experience: {alternative_experience}")
      self.panda.set_alternative_experience(alternative_experience)
      self.panda.set_safety_mode(safety_model, safety_param)


def pandad_run(panda):
  """Main daemon loop running at 100Hz. Single-threaded except for hwmon."""
  global do_exit

  no_fan_control = os.getenv("NO_FAN_CONTROL") is not None
  spoofing_started = os.getenv("STARTED") is not None
  fake_send = os.getenv("FAKESEND") is not None

  params = Params()
  rk = Ratekeeper(100)
  sendcan_sock = messaging.sub_sock('sendcan', conflate=False, timeout=0)
  sm = messaging.SubMaster(['selfdriveState'])
  pm = messaging.PubMaster(['can', 'pandaStates', 'peripheralState'])
  panda_safety = PandaSafety(panda)
  peripheral_processor = PeripheralStateProcessor()
  hwmon_reader = HwmonReader()
  hwmon_reader.start()
  hw_type = get_hw_type_capnp(panda)
  engaged = False
  is_onroad = False
  comms_healthy = True

  # Main loop: single-threaded CAN recv/send and state processing
  while not do_exit and check_connected(panda):
    comms_healthy = can_recv(panda, pm)
    can_send(panda, sendcan_sock, fake_send)

    # Process peripheral state at 20 Hz
    if rk.frame % 5 == 0:
      peripheral_processor.update(panda, no_fan_control)

    # Process panda state at 10 Hz
    if rk.frame % 10 == 0:
      sm.update(0)
      engaged = sm.all_checks(['selfdriveState']) and sm['selfdriveState'].enabled
      is_onroad = params.get_bool("IsOnroad")

      # send panda states and check connection
      ignition_opt = send_panda_states(pm, panda, hw_type, is_onroad, spoofing_started)
      if ignition_opt is None:
        cloudlog.error("Failed to get ignition_opt")
      else:
        # check if we should have pandad reconnect
        if not ignition_opt and not comms_healthy:
          cloudlog.error("Reconnecting, communication to panda not healthy")
          do_exit = True

      panda.send_heartbeat(engaged)
      panda_safety.configure_safety_mode(is_onroad)

    # Send out peripheralState at 2Hz
    if rk.frame % 50 == 0:
      send_peripheral_state(panda, pm, hw_type, hwmon_reader)

    # Forward logs from panda to cloudlog if available
    try:
      log_data = panda.serial_read(Panda.SERIAL_DEBUG)
      if log_data:
        log_str = log_data.decode('utf-8', errors='replace')
        if "Register 0x" in log_str:
          cloudlog.error(log_str)
        else:
          cloudlog.debug(log_str)
    except Exception:
      pass

    rk.keep_time()

  # Close relay on exit to prevent a fault
  if is_onroad and not engaged:
    if panda.connected:
      panda.set_safety_mode(CarParams.SafetyModel.noOutput)


def pandad_main_thread(serial=""):
  """Main entry point for the pandad daemon."""
  global do_exit
  do_exit = False

  cloudlog.warning("starting pandad")

  if not PC:
    config_realtime_process([3], 54)

  if not serial:
    serials = Panda.list()
    if not serials:
      cloudlog.warning("no pandas found, exiting")
      return
    serial = serials[0]

  cloudlog.warning(f"connecting to panda: {serial}")

  panda = None
  while not do_exit:
    panda = connect(serial)
    if panda:
      break
    time.sleep(0.1)

  if not do_exit and panda is not None:
    cloudlog.warning("connected to panda")
    pandad_run(panda)

  if panda is not None:
    panda.close()
