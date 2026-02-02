import os
import threading
import time

from cereal import car, log
from cereal.messaging import SubMaster, PubMaster
import cereal.messaging as messaging
from panda import Panda
from panda.python import PANDA_CAN_CNT
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.system.hardware import HARDWARE

FAKE_SEND = os.getenv("FAKESEND") == "1"
SPOOFING_STARTED = os.getenv("STARTED") == "1"
NO_FAN_CONTROL = os.getenv("NO_FAN_CONTROL") == "1"

LEC_ERROR_CODE = {
  "No error": 0, "Stuff error": 1, "Form error": 2, "AckError": 3,
  "Bit1Error": 4, "Bit0Error": 5, "CRCError": 6, "NoChange": 7,
}

MAX_IR_PANDA_VAL = 50
CUTOFF_IL = 400
SATURATE_IL = 1000


class HardwareReader:
  """Separate thread needed here since the read can stall."""
  def __init__(self):
    self.voltage = 0
    self.current = 0
    self.running = True
    self.thread = threading.Thread(target=self._read_loop, daemon=True)
    self.thread.start()

  def _read_loop(self):
    while self.running:
      start = time.monotonic()
      try:
        self.voltage = HARDWARE.get_voltage()
        self.current = HARDWARE.get_current()
        elapsed = (time.monotonic() - start) * 1000
        if elapsed > 50:
          cloudlog.warning(f"hwmon read took {elapsed:.2f}ms")
      except Exception as e:
        cloudlog.error(f"Hardware read error: {e}")
      time.sleep(0.5)  # 500ms update rate

  def get_values(self):
    return self.voltage, self.current

  def stop(self):
    self.running = False
    self.thread.join()


class PandaStateManager:
  def __init__(self, pandas, hw_types):
    self.pandas = pandas
    self.hw_types = hw_types

    self.is_comma_three_red = (
      len(pandas) == 2 and
      self.hw_types[0] == log.PandaState.PandaType.dos and
      self.hw_types[1] == log.PandaState.PandaType.redPanda
    )

  def process(self, engaged, pm) -> bool:
    msg = messaging.new_message('pandaStates', len(self.pandas))
    msg.valid = True
    panda_states = msg.pandaStates
    ignition = False

    for i, panda in enumerate(self.pandas):
      health = panda.health() or {}

      if SPOOFING_STARTED:
        health['ignition_line'] = 1

      # on comma three setups with a red panda, the dos can
      # get false positive ignitions due to the harness box
      # without a harness connector, so ignore it
      if self.is_comma_three_red and i == 0:
        health['ignition_line'] = 0

      ignition |= bool(health['ignition_line']) or bool(health['ignition_can'])
      ps = panda_states[i]
      self._fill_state(ps, self.hw_types[i], health)

      # Fill can state
      for j in range(PANDA_CAN_CNT):
        can_health = panda.can_health(j)
        can_state = ps.init(f'canState{j}')
        self._fill_can_state(can_state, can_health)

      # Set faults
      fault_bits = int(health.get('faults', 0))
      faults_list = [f for f in range(log.PandaState.FaultType.relayMalfunction,
                                      log.PandaState.FaultType.heartbeatLoopWatchdog + 1)
                    if fault_bits & (1 << f)]
      faults = ps.init('faults', len(faults_list))
      for idx, fault in enumerate(faults_list):
        faults[idx] = fault

    for panda, ps in zip(self.pandas, panda_states, strict=False):
      if ps.safetyModel == car.CarParams.SafetyModel.silent or (
          not ignition and ps.safetyModel != car.CarParams.SafetyModel.noOutput):
        panda.set_safety_mode(car.CarParams.SafetyModel.noOutput)

      if ps.powerSaveEnabled != (not ignition):
        panda.set_power_save(not ignition)

      panda.send_heartbeat(engaged)

    pm.send("pandaStates", msg)
    return ignition

  def _fill_state(self, ps, hw_type, health):
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

  def _fill_can_state(self, cs, can_health):
    cs.busOff = bool(can_health['bus_off'])
    cs.busOffCnt = can_health['bus_off_cnt']
    cs.errorWarning = bool(can_health['error_warning'])
    cs.errorPassive = bool(can_health['error_passive'])
    cs.lastError = LEC_ERROR_CODE[can_health['last_error']]
    cs.lastStoredError = LEC_ERROR_CODE[can_health['last_stored_error']]
    cs.lastDataError = LEC_ERROR_CODE[can_health['last_data_error']]
    cs.lastDataStoredError = LEC_ERROR_CODE[can_health['last_data_stored_error']]
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


class PeripheralManager:
  def __init__(self, panda, hw_type):
    self.panda = panda
    self.last_camera_t = 0
    self.prev_fan = 999
    self.prev_ir_pwr = 999
    self.ir_pwr = 0
    self.filter = FirstOrderFilter(0, 30.0, 0.05)
    self.hw_type = hw_type
    self.hw_reader = HardwareReader()

  def process(self, sm):
    if sm.updated["deviceState"] and not NO_FAN_CONTROL:
      fan = sm["deviceState"].fanSpeedPercentDesired
      if fan != self.prev_fan or sm.frame % 100 == 0:
        self.panda.set_fan_power(fan)
        self.prev_fan = fan

    if sm.updated["driverCameraState"]:
      state = sm["driverCameraState"]
      lines = self.filter.update(state.integLines)
      self.last_camera_t = sm.logMonoTime['driverCameraState']
      if lines <= CUTOFF_IL:
        self.ir_pwr = 0
      elif lines > SATURATE_IL:
        self.ir_pwr = 100
      else:
        self.ir_pwr = 100 * (lines - CUTOFF_IL) / (SATURATE_IL - CUTOFF_IL)

    if time.monotonic_ns() - self.last_camera_t > 1e9:
      self.ir_pwr = 0

    if self.ir_pwr != self.prev_ir_pwr or sm.frame % 100 == 0:
      ir_panda = int(self.ir_pwr * MAX_IR_PANDA_VAL / 100)
      self.panda.set_ir_power(ir_panda)

      HARDWARE.set_ir_power(self.ir_pwr)
      self.prev_ir_pwr = self.ir_pwr

  def send_state(self, pm):
    msg = messaging.new_message('peripheralState')
    msg.valid = True
    ps = msg.peripheralState
    ps.pandaType = self.hw_type
    ps.voltage, ps.current = self.hw_reader.get_values()

    if not (ps.voltage or ps.current):
      h = self.panda.health() or {}
      ps.voltage = h.get("voltage", 0)
      ps.current = h.get("current", 0)

    ps.fanSpeedRpm = self.panda.get_fan_rpm()

    pm.send("peripheralState", msg)

  def cleanup(self):
    self.hw_reader.stop()


class PandaSafetyManager:
  def __init__(self, pandas: list[Panda]):
    self.pandas = pandas
    self.params = Params()
    self.safety_configured = False
    self.initialized = False
    self.prev_obd_multiplexing = False
    self.log_once = False

  def configure_safety_mode(self):
    is_onroad = self.params.get_bool("IsOnroad")

    if is_onroad and not self.safety_configured:
      self.update_multiplexing_mode()
      car_params = self.fetch_car_params()
      if car_params:
        cloudlog.warning(f"got {len(car_params)} bytes CarParams")
        self.set_safety_mode(car_params)
        self.safety_configured = True
    elif not is_onroad:
      self.initialized = False
      self.safety_configured = False
      self.log_once = False

  def update_multiplexing_mode(self):
    # Initialize to ELM327 without OBD multiplexing for initial fingerprinting
    if not self.initialized:
      self.prev_obd_multiplexing = False
      for panda in self.pandas:
        panda.set_safety_mode(car.CarParams.SafetyModel.elm327, 1)
      self.initialized = True

    # Switch between multiplexing modes based on OBD multiplexing request
    obd_multiplexing_requested = self.params.get_bool("ObdMultiplexingEnabled")
    if obd_multiplexing_requested != self.prev_obd_multiplexing:
      for i, panda in enumerate(self.pandas):
        safety_param = 1 if i > 0 or not obd_multiplexing_requested else 0
        panda.set_safety_mode(car.CarParams.SafetyModel.elm327, safety_param)
      self.prev_obd_multiplexing = obd_multiplexing_requested
      self.params.put_bool("ObdMultiplexingChanged", True)

  def fetch_car_params(self) -> bytes:
    if not self.params.get_bool("FirmwareQueryDone"):
      return b""

    if not self.log_once:
      cloudlog.warning("Finished FW query, waiting for params to set safety model")
      self.log_once = True

    if not self.params.get_bool("ControlsReady"):
      return b""
    return self.params.get("CarParams") or b""

  def set_safety_mode(self, params_bytes: bytes):
    # Parse CarParams from bytes
    with car.CarParams.from_bytes(params_bytes) as car_params:
      safety_configs = car_params.safetyConfigs
      alternative_experience = car_params.alternativeExperience

    for i, panda in enumerate(self.pandas):
      # Default to SILENT if no config for this panda
      safety_model = car.CarParams.SafetyModel.silent

      safety_param = 0
      if i < len(safety_configs):
        safety_model = car.CarParams.SafetyModel.schema.enumerants[safety_configs[i].safetyModel]
        safety_param = safety_configs[i].safetyParam

      cloudlog.warning(f"Panda {i}: setting safety model: {safety_model}, param: {safety_param}, alternative experience: {alternative_experience}")
      panda._handle.controlWrite(Panda.REQUEST_OUT, 0xdf, alternative_experience, 0, b'')
      panda.set_safety_mode(safety_model, safety_param)


class PandaRunner:
  def __init__(self, serials, pandas):
    self.pandas = pandas
    self.usb_pandas = {p.get_usb_serial() for p in pandas if not p.spi}
    self.hw_types = [int.from_bytes(p.get_type(), 'big') for p in pandas]

    for panda in self.pandas:
      if os.getenv("BOARDD_LOOPBACK"):
        panda.set_can_loopback(True)
      for i in range(3):
        panda.set_canfd_auto(i, True)

    self.sm = SubMaster(["selfdriveState", "deviceState", "driverCameraState"])
    self.pm = PubMaster(["can", "pandaStates", "peripheralState"])
    self.sendcan_sock = messaging.sub_sock('sendcan', timeout=10)
    self.sendcan_buffer = None

    self.state_mgr = PandaStateManager(pandas, self.hw_types)
    self.periph_mgr = PeripheralManager(pandas[0], self.hw_types[0])
    self.safety_mgr = PandaSafetyManager(pandas)

  def _can_send(self):
    # TODO: this needs to have a strict timeout of <10ms and handle NACKs well (buffer the data)
    while (msg := messaging.recv_one_or_none(self.sendcan_sock)):
      # drop msg if too old
      if (time.monotonic_ns() - msg.logMonoTime) / 1e9 > 1.0:
        cloudlog.warning("skipping CAN send, too old")
        continue

      # Group CAN messages by panda based on bus offset
      panda_msgs = [[] for _ in self.pandas]
      for c in msg.sendcan:
        panda_idx = c.src // 4  # Each panda handles 4 buses
        if panda_idx < len(self.pandas):
          # Adjust bus number for the panda (remove offset)
          adjusted_bus = c.src % 4
          panda_msgs[panda_idx].append((c.address, c.dat, adjusted_bus))

      # Send messages to each panda
      for panda_idx, can_msgs in enumerate(panda_msgs):
        if can_msgs:
          self.pandas[panda_idx].can_send_many(can_msgs)

  def _can_recv(self):
    cans = []
    for panda_idx, p in enumerate(self.pandas):
      bus_offset = panda_idx * 4  # Each panda gets 4 buses
      for address, dat, src in p.can_recv():
        if src >= 192:  # Rejected message
          base_bus = src - 192
          adjusted_src = base_bus + bus_offset + 192
        elif src >= 128:  # Returned message
          base_bus = src - 128
          adjusted_src = base_bus + bus_offset + 128
        else:  # Normal message
          adjusted_src = src + bus_offset
        cans.append((address, dat, adjusted_src))

    msg = messaging.new_message('can', len(cans) if cans else 0)
    msg.valid = True
    if cans:
      for i, can_info in enumerate(cans):
        can = msg.can[i]
        can.address, can.dat, can.src = can_info
    self.pm.send("can", msg)

  def run(self, evt):
    rk = Ratekeeper(100, print_delay_threshold=None)
    engaged = False

    try:
      while not evt.is_set():
        # receive CAN messages
        self._can_recv()

        # send CAN messages
        self._can_send()

        # Process peripheral state at 20 Hz
        if rk.frame % 5 == 0:
          self.sm.update(0)
          engaged = self.sm.all_checks() and self.sm["selfdriveState"].enabled
          self.periph_mgr.process(self.sm)

        # Process panda state at 10 Hz
        if rk.frame % 10 == 0:
          ignition = self.state_mgr.process(engaged, self.pm)
          if not ignition and rk.frame % 100 == 0:
            if set(Panda.list(usb_only=True)) != self.usb_pandas:
              cloudlog.warning("Reconnecting to new panda")
              evt.set()
              break

          self.safety_mgr.configure_safety_mode()

        # Send out peripheralState at 2Hz
        if rk.frame % 50 == 0:
          self.periph_mgr.send_state(self.pm)

        rk.keep_time()
    except Exception as e:
      cloudlog.error(f"Exception in main loop: {e}")
    finally:
      evt.set()
      self.periph_mgr.cleanup()

      # Close relay on exit to prevent a fault
      is_onroad = Params().get_bool("IsOnroad")
      if is_onroad and not engaged:
        for p in self.pandas:
          p.set_safety_mode(car.CarParams.SafetyModel.noOutput)
