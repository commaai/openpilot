import os
import time
from collections import deque

from opendbc.car.structs import car
from openpilot.cereal import log
from openpilot.cereal.messaging import SubMaster, PubMaster
import openpilot.cereal.messaging as messaging
from panda import Panda
from panda.python import CANPACKET_HEAD_SIZE, PANDA_CAN_CNT
from panda.python.spi import PandaSpiMissingAck, PandaSpiNackResponse
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.hardware import HARDWARE

SafetyModel = car.CarParams.SafetyModel

FAKE_SEND = os.getenv("FAKESEND") == "1"
SPOOFING_STARTED = os.getenv("STARTED") == "1"
NO_FAN_CONTROL = os.getenv("NO_FAN_CONTROL") == "1"

LEC_ERROR_CODE = {
  "No error": 0, "Stuff error": 1, "Form error": 2, "AckError": 3,
  "Bit1Error": 4, "Bit0Error": 5, "CRCError": 6, "NoChange": 7,
}

# capnp enum lists only accept enumerant names, not their integer values
FAULT_NAMES = {v: k for k, v in log.PandaState.FaultType.schema.enumerants.items()}

MAX_IR_PANDA_VAL = 50
CUTOFF_IL = 400
SATURATE_IL = 1000

# a NACK means the panda's CAN TX buffers are full: one attempt per chunk, we try again next frame
CAN_SEND_TIMEOUT_MS = 1
# one chunk is one SPI transaction: it's either accepted whole or NACKed whole, so it can be retried
CAN_SEND_CHUNK_BYTES = 256
# frames older than this are useless to the car
CAN_SEND_MAX_AGE_NS = int(1e9)
# ~2 s of a saturated bus; beyond that we're not keeping up and the oldest go
CAN_SEND_MAX_PENDING = 4000

# anything but a dead transport only costs the frame it happened in. but if the 10 Hz state work
# (health, heartbeat, pandaStates) hasn't gone through for this long, a reconnect is the way out.
STATE_TIMEOUT_S = 1.0


# every safety mode write goes through here, so the 10 Hz state check can compare what the panda reports against what we asked for
def set_safety_mode(panda: Panda, mode, param: int = 0) -> None:
  panda.set_safety_mode(mode, param)
  panda.expected_safety = (int(mode), int(param))


class PandaStateManager:
  def __init__(self, panda: Panda, hw_type: int):
    self.panda = panda
    self.hw_type = hw_type
    self.last_warn: dict[str, float] = {}

  def _warn(self, tag: str, msg: str, interval: float = 5.0) -> None:
    now = time.monotonic()
    if now - self.last_warn.get(tag, -interval) >= interval:
      self.last_warn[tag] = now
      cloudlog.error(f"{tag}: {msg}")

  def process(self, engaged: bool, is_onroad: bool, pm) -> bool:
    health = self.panda.health()
    if SPOOFING_STARTED:
      health['ignition_line'] = True
    ignition = health['ignition_line'] or health['ignition_can']

    msg = messaging.new_message('pandaStates', 1, valid=True)
    ps = msg.pandaStates[0]
    self._fill_state(ps, health)
    for j in range(PANDA_CAN_CNT):
      self._fill_can_state(ps.init(f'canState{j}'), self.panda.can_health(j))
    ps.faults = [FAULT_NAMES[f] for f in FAULT_NAMES if health['faults'] & (1 << f)]

    # Is the panda in the safety state we think it is? The firmware can change it on its own
    # (heartbeat loss -> SILENT), and so can anything else talking to the panda.
    if health['heartbeat_disabled']:
      self._warn("pandad.heartbeat_disabled", "panda heartbeat checks are disabled, re-arming with the next heartbeat")
    expected = getattr(self.panda, "expected_safety", None)
    actual = (health['safety_mode'], health['safety_param'])
    if expected is not None and actual != expected:
      self._warn("pandad.safety_mode_mismatch", f"panda reports safety {actual}, pandad set {expected}")

    # Make sure CAN buses are live: the safety setter does not work if the panda's CAN is silent
    # and there is only one other CAN node. Also set NO_OUTPUT when the car is off or we're not onroad.
    should_close_relay = not ignition or not is_onroad
    if health['safety_mode'] == SafetyModel.silent or (should_close_relay and health['safety_mode'] != SafetyModel.noOutput):
      set_safety_mode(self.panda, SafetyModel.noOutput)

    if health['power_save_enabled'] != (not ignition):
      self.panda.set_power_save(not ignition)

    self.panda.send_heartbeat(engaged)

    pm.send("pandaStates", msg)
    return ignition

  def _fill_state(self, ps, health):
    ps.voltage = health['voltage']
    ps.current = health['current']
    ps.uptime = health['uptime']
    ps.safetyTxBlocked = health['safety_tx_blocked']
    ps.safetyRxInvalid = health['safety_rx_invalid']
    ps.ignitionLine = health['ignition_line']
    ps.ignitionCan = health['ignition_can']
    ps.controlsAllowed = health['controls_allowed']
    ps.txBufferOverflow = health['tx_buffer_overflow']
    ps.rxBufferOverflow = health['rx_buffer_overflow']
    ps.pandaType = self.hw_type
    ps.safetyModel = health['safety_mode']
    ps.safetyParam = health['safety_param']
    ps.faultStatus = health['fault_status']
    ps.powerSaveEnabled = health['power_save_enabled']
    ps.heartbeatLost = health['heartbeat_lost']
    ps.alternativeExperience = health['alternative_experience']
    ps.harnessStatus = health['car_harness_status']
    ps.interruptLoad = health['interrupt_load']
    ps.fanPower = health['fan_power']
    ps.safetyRxChecksInvalid = health['safety_rx_checks_invalid']
    ps.spiErrorCount = health['spi_error_count']
    ps.sbu1Voltage = health['sbu1_voltage_mV'] / 1000.0
    ps.sbu2Voltage = health['sbu2_voltage_mV'] / 1000.0
    ps.nmiReset = health['nmi_reset']
    ps.hardfaultReset = health['hardfault_reset']
    ps.heartbeatDisabled = health['heartbeat_disabled']
    ps.soundOutputLevel = health['sound_output_level']

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
  def __init__(self, panda: Panda, hw_type: int):
    self.panda = panda
    self.hw_type = hw_type
    self.params = Params()
    self.last_camera_t = 0
    self.prev_fan = 999
    self.prev_ir_pwr = 999
    self.ir_pwr = 0
    self.filter = FirstOrderFilter(0, 30.0, 0.05)
    self.filter_driver_view = FirstOrderFilter(0, 5.0, 0.05)
    self.prev_frame_id = None
    self.driver_view = False
    self.not_car = None

  def process(self, sm):
    if sm.updated["deviceState"] and not NO_FAN_CONTROL:
      fan = sm["deviceState"].fanSpeedPercentDesired
      if fan != self.prev_fan or sm.frame % 100 == 0:
        self.panda.set_fan_power(fan)
        self.prev_fan = fan

    if sm.updated["cabinCameraState"]:
      state = sm["cabinCameraState"]
      # reset the filters when camerad restarts
      if self.prev_frame_id is None or state.frameId < self.prev_frame_id:
        self.filter.x = 0
        self.filter_driver_view.x = 0
        self.driver_view = self.params.get_bool("IsDriverViewEnabled")
      self.prev_frame_id = state.frameId
      self.last_camera_t = sm.logMonoTime['cabinCameraState']

      lines = int((self.filter_driver_view if self.driver_view else self.filter).update(state.integLines))
      # integer like the C++, so ir_pwr != prev_ir_pwr doesn't fire every frame
      self.ir_pwr = min(max(100 * (lines - CUTOFF_IL) // (SATURATE_IL - CUTOFF_IL), 0), 100)

    # Disable IR on input timeout or when requested offroad
    is_onroad = sm["deviceState"].started
    if time.monotonic_ns() - self.last_camera_t > 1e9 or (not is_onroad and self.params.get_bool("DisableDriverCameraIR")):
      self.ir_pwr = 0

    # turn off IR leds if body
    if self.not_car is None and is_onroad:
      cp_bytes = self.params.get("CarParams")
      if cp_bytes:
        with car.CarParams.from_bytes(cp_bytes) as CP:
          self.not_car = CP.notCar
    if self.not_car:
      self.ir_pwr = 0

    if self.ir_pwr != self.prev_ir_pwr or sm.frame % 100 == 0:
      self.prev_ir_pwr = self.ir_pwr
      self.panda.set_ir_power(self.ir_pwr * MAX_IR_PANDA_VAL // 100)
      HARDWARE.set_ir_power(self.ir_pwr)

  def send_state(self, pm):
    msg = messaging.new_message('peripheralState', valid=True)
    ps = msg.peripheralState
    ps.pandaType = self.hw_type
    health = self.panda.health()
    ps.voltage, ps.current = health['voltage'], health['current']
    ps.fanSpeedRpm = self.panda.get_fan_rpm()
    pm.send("peripheralState", msg)


class PandaSafetyManager:
  def __init__(self, panda: Panda):
    self.panda = panda
    self.params = Params()
    self.safety_configured = False
    self.initialized = False
    self.prev_obd_multiplexing = False
    self.log_once = False

  def configure_safety_mode(self, is_onroad: bool):
    if is_onroad and not self.safety_configured:
      self.update_multiplexing_mode()
      car_params = self.fetch_car_params()
      if car_params:
        cloudlog.warning(f"got {len(car_params)} bytes CarParams")
        self.set_car_safety_mode(car_params)
        self.safety_configured = True
    elif not is_onroad:
      self.initialized = False
      self.safety_configured = False
      self.log_once = False

  def update_multiplexing_mode(self):
    # Initialize to ELM327 without OBD multiplexing for initial fingerprinting
    if not self.initialized:
      self.prev_obd_multiplexing = False
      set_safety_mode(self.panda, SafetyModel.elm327, 1)
      self.initialized = True

    # Switch between multiplexing modes based on OBD multiplexing request
    obd_multiplexing_requested = self.params.get_bool("ObdMultiplexingEnabled")
    if obd_multiplexing_requested != self.prev_obd_multiplexing:
      set_safety_mode(self.panda, SafetyModel.elm327, 0 if obd_multiplexing_requested else 1)
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

  def set_car_safety_mode(self, params_bytes: bytes):
    with car.CarParams.from_bytes(params_bytes) as CP:
      alternative_experience = CP.alternativeExperience
      # Default to SILENT if there's no safety config
      safety_model, safety_param = SafetyModel.silent, 0
      if CP.safetyConfigs:
        safety_model = SafetyModel.schema.enumerants[CP.safetyConfigs[0].safetyModel]
        safety_param = CP.safetyConfigs[0].safetyParam

    cloudlog.warning(f"setting safety model: {safety_model}, param: {safety_param}, alternative experience: {alternative_experience}")
    self.panda.set_alternative_experience(alternative_experience)
    set_safety_mode(self.panda, safety_model, safety_param)


class PandaRunner:
  def __init__(self, panda: Panda):
    assert panda.spi, "only the internal SPI panda is supported"
    self.panda = panda
    hw_type = int.from_bytes(panda.get_type(), 'big')

    if os.getenv("BOARDD_LOOPBACK"):
      panda.set_can_loopback(True)
    for i in range(PANDA_CAN_CNT):
      panda.set_canfd_auto(i, True)

    self.sm = SubMaster(["selfdriveState", "deviceState", "cabinCameraState"])
    self.pm = PubMaster(["can", "pandaStates", "peripheralState"])
    self.sendcan_poller = messaging.Poller()
    self.sendcan_sock = messaging.sub_sock('sendcan', poller=self.sendcan_poller)
    self.pending: deque = deque()

    self.state_mgr = PandaStateManager(panda, hw_type)
    self.periph_mgr = PeripheralManager(panda, hw_type)
    self.safety_mgr = PandaSafetyManager(panda)

  def _can_send(self):
    """Queue what was published on sendcan and push as much as the panda will take, never waiting on it:
    when its CAN TX buffers are full it NACKs right away, the chunk stays queued, and we try again next poll."""
    while (msg := messaging.recv_one_or_none(self.sendcan_sock)):
      if time.monotonic_ns() - msg.logMonoTime > CAN_SEND_MAX_AGE_NS:
        cloudlog.warning("skipping CAN send, too old")
        continue
      for c in msg.sendcan:
        if c.src < PANDA_CAN_CNT:
          self.pending.append((msg.logMonoTime, c.address, c.dat, c.src))

    q = self.pending
    if FAKE_SEND:
      q.clear()
      return

    if len(q) > CAN_SEND_MAX_PENDING:
      cloudlog.warning(f"dropping {len(q) - CAN_SEND_MAX_PENDING} queued CAN frames, panda isn't keeping up")
      while len(q) > CAN_SEND_MAX_PENDING:
        q.popleft()

    now_ns = time.monotonic_ns()
    while q:
      if now_ns - q[0][0] > CAN_SEND_MAX_AGE_NS:
        q.popleft()
        cloudlog.warning("dropping queued CAN frame, too old")
        continue

      # one chunk = one transfer
      chunk, size = [], 0
      for item in q:
        size += CANPACKET_HEAD_SIZE + len(item[2])
        if chunk and size > CAN_SEND_CHUNK_BYTES:
          break
        chunk.append(item)

      try:
        self.panda.can_send_many([(address, dat, bus) for _, address, dat, bus in chunk], timeout=CAN_SEND_TIMEOUT_MS)
      except PandaSpiNackResponse:
        break  # the panda's buffers are full. the chunk was rejected whole, keep it for the next try
      except PandaSpiMissingAck:
        raise
      except Exception:
        cloudlog.exception("pandad.can_send_exception")
        for _ in chunk:
          q.popleft()
        break
      for _ in chunk:
        q.popleft()

  def _can_recv(self):
    cans, valid = [], True
    try:
      cans = self.panda.can_recv()
    except AssertionError:
      # unpack_can_buffer asserts on a CAN packet checksum mismatch. Like the C++: log, drop the
      # buffer, reset the panda's CAN comms, and publish this frame as invalid.
      cloudlog.exception("pandad.can_checksum_failed")
      self.panda.can_rx_overflow_buffer = b''
      self.panda.can_reset_communications()
      valid = False

    msg = messaging.new_message('can', len(cans), valid=valid)
    for i, can_info in enumerate(cans):
      can = msg.can[i]
      can.address, can.dat, can.src = can_info
    self.pm.send("can", msg)

  def run(self, evt):
    rk = Ratekeeper(100, print_delay_threshold=None)
    engaged = False
    is_onroad = False
    last_state_t = time.monotonic()

    try:
      while not evt.is_set():
        try:
          self._can_recv()
          self._can_send()

          # Process peripheral state at 20 Hz
          if rk.frame % 5 == 0:
            self.sm.update(0)
            # no frequency check: selfdriveState is 100 Hz but we only poll it at 20 Hz on a conflated socket
            engaged = self.sm.all_alive(["selfdriveState"]) and self.sm.all_valid(["selfdriveState"]) and self.sm["selfdriveState"].enabled
            is_onroad = self.sm["deviceState"].started
            try:
              self.periph_mgr.process(self.sm)
            except Exception:
              # fan/IR trouble must never cost us the heartbeat below
              cloudlog.exception("pandad.peripheral_exception")

          # Process panda state at 10 Hz
          if rk.frame % 10 == 0:
            self.state_mgr.process(engaged, is_onroad, self.pm)
            self.safety_mgr.configure_safety_mode(is_onroad)
            last_state_t = time.monotonic()

          # Forward logs from panda to cloudlog at 10 Hz, offset from the state work
          if rk.frame % 10 == 5:
            dbg = self.panda.debug_read().decode(errors="replace")
            if dbg:
              # register divergent faults are errors
              (cloudlog.error if "Register 0x" in dbg else cloudlog.debug)(dbg)

          # Send out peripheralState at 2 Hz, offset from the state work
          if rk.frame % 50 == 25:
            self.periph_mgr.send_state(self.pm)
        except PandaSpiMissingAck:
          # the panda is gone or the transport is broken: leave so pandad.py reconnects
          cloudlog.exception("pandad.panda_disconnected")
          break
        except Exception:
          # e.g. a bad health packet or a capnp error: skip this frame, don't reset the panda over it
          cloudlog.exception("pandad.runner_exception")
          if time.monotonic() - last_state_t > STATE_TIMEOUT_S:
            cloudlog.error(f"pandad: no panda state for {STATE_TIMEOUT_S} s, reconnecting")
            break

        # Idle until the next frame, but send CAN the moment it's published (the C++ did this with a send thread)
        rk.monitor_time()
        deadline = time.monotonic() + rk.remaining
        while not evt.is_set() and (remaining := deadline - time.monotonic()) > 0:
          if remaining < 1e-3:
            time.sleep(remaining)
            break
          if not self.sendcan_poller.poll(int(remaining * 1000)):
            break  # poll slept the whole time
          try:
            self._can_send()
          except PandaSpiMissingAck:
            cloudlog.exception("pandad.panda_disconnected")
            evt.set()
          except Exception:
            cloudlog.exception("pandad.runner_exception")
            break
    finally:
      # Close relay on exit to prevent a fault
      if is_onroad and not engaged:
        try:
          set_safety_mode(self.panda, SafetyModel.noOutput)
        except Exception:
          cloudlog.exception("pandad.relay_close_failed")
