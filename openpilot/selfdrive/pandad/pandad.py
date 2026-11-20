#!/usr/bin/env python3
import logging
import os
import signal
import time

from setproctitle import setproctitle

from opendbc.car.structs import CarParams
import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.hardware import HARDWARE
from openpilot.common.params import Params
from openpilot.common.realtime import Priority, Ratekeeper, config_realtime_process
from openpilot.common.swaglog import ForwardingHandler, cloudlog
from openpilot.selfdrive.pandad.panda_safety import PandaSafety
from openpilot.selfdrive.pandad.pandad_api_impl import can_capnp_to_list, can_list_to_can_capnp
from panda import Panda, PandaDFU
from panda.python import LEN_TO_DLC, PANDA_CAN_CNT

SafetyModel = CarParams.SafetyModel

PANDA_BUS_OFFSET = 4  # sendcan frames from this bus up are for another panda
CAN_SEND_MAX_AGE_NS = 1_000_000_000
SENDCAN_DRAIN_MAX = 100  # messages per pass, 1 s of card's output: a flood can't hold up the tick, a lagging loop still catches up

MAX_IR_PANDA_VAL = 50
CUTOFF_IL = 400
SATURATE_IL = 1000

# capnp lists of enums only take enumerant names. bits above the last fault type are dropped
FAULT_NAMES = [name for name, _ in sorted(log.PandaState.FaultType.schema.enumerants.items(), key=lambda kv: kv[1])]
# Panda.can_health() reports the last error codes as strings
LEC_ERROR_CODE = {name: code for code, name in Panda.LEC_ERROR_CODE.items()}

do_exit: bool = False


# *** CAN ***

class CanSender:
  """sendcan -> panda, never waiting on the panda: frames it has no room for are tried again on the next pass, for up to 1 s"""

  def __init__(self, panda: Panda, sock: messaging.SubSocket, fake_send: bool):
    self.panda, self.sock, self.fake_send = panda, sock, fake_send
    self.poller = messaging.Poller()
    self.poller.registerSocket(sock)
    self.pending: list[tuple[int, list]] = []  # (logMonoTime, frames) of each message, oldest first
    self.dropping = False

  def wait(self, deadline: float) -> None:
    """Until the deadline, sends sendcan as soon as it arrives"""
    self.send()
    while (timeout_ms := int((deadline - time.monotonic()) * 1000)) > 0:
      if self.poller.poll(timeout_ms):
        self.send()
    # the rest, under a ms: poll takes whole ms
    time.sleep(max(0., deadline - time.monotonic()))

  def send(self) -> None:
    for _ in range(SENDCAN_DRAIN_MAX):
      if (dat := self.sock.receive(non_blocking=True)) is None:
        break
      self.queue(dat)

    # Don't send if older than 1 second or timestamped in the future
    now = time.monotonic_ns()
    fresh = [m for m in self.pending if 0 <= now - m[0] < CAN_SEND_MAX_AGE_NS]
    if len(fresh) < len(self.pending) and not self.dropping:
      # once until everything has gone out: a dead bus ages out a message every pass
      cloudlog.error("sendcan: dropping messages, too old to send")
      self.dropping = True
    self.pending = fresh
    if not self.pending:
      return

    # one try, up to the first NACK (CAN TX buffers full)
    sent = self.panda.can_send_nonblocking([f for _, frames in self.pending for f in frames])
    # drop what went out. a message can go out in parts
    while sent > 0:
      t, frames = self.pending.pop(0)
      if sent < len(frames):
        self.pending.insert(0, (t, frames[sent:]))
      sent -= len(frames)
    if not self.pending:
      self.dropping = False

  def queue(self, dat: bytes) -> None:
    try:
      (t, frames), = can_capnp_to_list([dat], 'sendcan')
      frames = [f for f in frames if f[2] < PANDA_BUS_OFFSET]
      if any(len(f[1]) not in LEN_TO_DLC for f in frames):
        raise ValueError("invalid CAN frame length")
    except Exception:
      # dropped on its own
      cloudlog.exception("sendcan: invalid message")
      return
    if frames and not self.fake_send:
      self.pending.append((t, frames))


def can_recv(panda: Panda, pm: messaging.PubMaster) -> None:
  valid = True
  try:
    frames = panda.can_recv()
  except AssertionError:
    # bad checksum. the panda library dropped its buffer and reset the panda's CAN comms
    cloudlog.error("Panda CAN checksum failed")
    frames, valid = [], False
  # published every frame, even when empty
  pm.send('can', can_list_to_can_capnp(frames, 'can', valid))


# *** panda state ***

def update_safety(panda: Panda, health: dict, is_onroad: bool) -> None:
  """Keeps the panda's safety mode and power save in line with ignition, all from one health read"""
  ignition = health['ignition_line'] or health['ignition_can']

  # Make sure CAN buses are live: openpilot can't talk to the car while the panda is SILENT
  if health['safety_mode'] == SafetyModel.silent:
    panda.set_safety_mode(SafetyModel.noOutput)

  power_save_desired = not ignition
  if health['power_save_enabled'] != power_save_desired:
    panda.set_power_save(power_save_desired)

  # set safety mode to NO_OUTPUT when car is off or we're not onroad. ELM327 is an alternative if we want to leverage athenad/connect
  should_close_relay = not ignition or not is_onroad
  if should_close_relay and health['safety_mode'] != SafetyModel.noOutput:
    panda.set_safety_mode(SafetyModel.noOutput)


def panda_states_msg(hw_type: int, health: dict, can_healths: list[dict]):
  msg = messaging.new_message('pandaStates', 1, valid=True)
  ps = msg.pandaStates[0]
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
  ps.pandaType = hw_type
  ps.safetyModel = health['safety_mode']
  ps.safetyParam = health['safety_param']
  ps.faultStatus = health['fault_status']
  ps.powerSaveEnabled = health['power_save_enabled']
  ps.heartbeatLost = health['heartbeat_lost']
  # uint16 on the panda, Int16 in the schema
  alternative_experience = health['alternative_experience']
  ps.alternativeExperience = alternative_experience - 0x10000 if alternative_experience >= 0x8000 else alternative_experience
  ps.harnessStatus = health['car_harness_status']
  ps.interruptLoad = health['interrupt_load']
  ps.fanPower = health['fan_power']
  ps.safetyRxChecksInvalid = health['safety_rx_checks_invalid']
  ps.nmiReset = health['nmi_reset']
  ps.hardfaultReset = health['hardfault_reset']
  ps.spiErrorCount = health['spi_error_count']
  ps.sbu1Voltage = health['sbu1_voltage_mV'] / 1000.0
  ps.sbu2Voltage = health['sbu2_voltage_mV'] / 1000.0
  ps.soundOutputLevel = health['sound_output_level']
  ps.faults = [name for bit, name in enumerate(FAULT_NAMES) if health['faults'] & (1 << bit)]
  for i, can_health in enumerate(can_healths):
    fill_panda_can_state(ps.init(f'canState{i}'), can_health)
  return msg


def fill_panda_can_state(cs, can_health: dict) -> None:
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


def send_peripheral_state(pm: messaging.PubMaster, panda: Panda, hw_type: int, health: dict) -> None:
  msg = messaging.new_message('peripheralState', valid=True)
  msg.peripheralState.pandaType = hw_type
  msg.peripheralState.voltage = health['voltage']
  msg.peripheralState.current = health['current']
  msg.peripheralState.fanSpeedRpm = panda.get_fan_rpm()
  pm.send('peripheralState', msg)


class PeripheralControl:
  """Fan, and the IR LEDs for the driver camera"""

  def __init__(self, panda: Panda, no_fan_control: bool):
    self.panda = panda
    self.no_fan_control = no_fan_control
    self.params = Params()
    self.prev_fan_speed = -1
    self.ir_pwr = 0
    self.prev_ir_pwr = -1
    self.prev_frame_id = 2**32  # above any frameId, so the first frame starts the filter
    self.integ_lines_filter = FirstOrderFilter(0, 30.0, 0.05)

  def update(self, sm: messaging.SubMaster, is_onroad: bool, not_car: bool) -> None:
    if sm.updated['deviceState'] and not self.no_fan_control:
      fan_speed = sm['deviceState'].fanSpeedPercentDesired
      if fan_speed != self.prev_fan_speed or sm.frame % 100 == 0:
        self.panda.set_fan_power(fan_speed)
        self.prev_fan_speed = fan_speed

    if sm.updated['cabinCameraState']:
      cs = sm['cabinCameraState']
      # start over when camerad restarts. the driver camera preview ramps faster
      if cs.frameId < self.prev_frame_id:
        self.integ_lines_filter.x = 0
        self.integ_lines_filter.update_alpha(5.0 if self.params.get_bool("IsDriverViewEnabled") else 30.0)
      self.prev_frame_id = cs.frameId
      integ_lines = int(self.integ_lines_filter.update(cs.integLines))
      self.ir_pwr = min(max(100 * (integ_lines - CUTOFF_IL) // (SATURATE_IL - CUTOFF_IL), 0), 100)

    # off on input timeout, when requested offroad, or for a body
    camera_timeout = time.monotonic() - sm.recv_time['cabinCameraState'] > 1.0
    if camera_timeout or (not is_onroad and self.params.get_bool("DisableDriverCameraIR")) or not_car:
      self.ir_pwr = 0

    if self.ir_pwr != self.prev_ir_pwr or sm.frame % 100 == 0:
      self.panda.set_ir_power(self.ir_pwr * MAX_IR_PANDA_VAL // 100)
      try:
        HARDWARE.set_ir_power(self.ir_pwr)
      except OSError:
        cloudlog.exception("pandad.set_ir_power_failed")
      self.prev_ir_pwr = self.ir_pwr


def forward_logs(panda: Panda) -> None:
  dbg = panda.debug_read().decode('utf-8', errors='replace')
  # Log register divergent faults as errors
  if "Register 0x" in dbg:
    cloudlog.error(dbg)
  elif dbg:
    cloudlog.debug(dbg)


# *** main loop ***

class Pandad:
  def __init__(self, panda: Panda, pm: messaging.PubMaster, sm: messaging.SubMaster, sendcan: messaging.SubSocket):
    self.panda, self.pm, self.sm = panda, pm, sm
    self.spoofing_started = "STARTED" in os.environ
    self.sender = CanSender(panda, sendcan, "FAKESEND" in os.environ)
    self.panda_safety = PandaSafety(panda)
    self.peripherals = PeripheralControl(panda, "NO_FAN_CONTROL" in os.environ)
    self.engaged = self.is_onroad = False

    # common panda config
    if "BOARDD_LOOPBACK" in os.environ:
      panda.set_can_loopback(True)
    for i in range(PANDA_CAN_CNT):
      panda.set_canfd_auto(i, True)
    self.hw_type = panda.get_type()[0]

  def step(self, frame: int) -> None:
    """One 10 ms tick: receive CAN first, then the lower priority panda and peripheral state"""
    can_recv(self.panda, self.pm)

    # 20 Hz: inputs (engaged, onroad), fan and IR
    if frame % 5 == 0:
      sm = self.sm
      sm.update(0)
      self.engaged = sm.all_alive(['selfdriveState']) and sm.all_valid(['selfdriveState']) and sm['selfdriveState'].enabled
      self.is_onroad = sm['deviceState'].started
      self.peripherals.update(sm, self.is_onroad, self.panda_safety.not_car)

    # Process panda state at 10 Hz
    if frame % 10 == 0:
      health = self.panda.health()
      can_healths = [self.panda.can_health(i) for i in range(PANDA_CAN_CNT)]
      if self.spoofing_started:
        health['ignition_line'] = True
      update_safety(self.panda, health, self.is_onroad)
      self.pm.send('pandaStates', panda_states_msg(self.hw_type, health, can_healths))
      # only after a publish: no heartbeat when the panda can't be read
      self.panda.send_heartbeat(self.engaged)
      self.panda_safety.configure_safety_mode(self.is_onroad)

      # Send out peripheralState at 2 Hz
      if frame % 50 == 0:
        send_peripheral_state(self.pm, self.panda, self.hw_type, health)

    # every tick: the panda's 1 KB debug ring overwrites its oldest bytes when full
    forward_logs(self.panda)

  def run(self) -> None:
    """Until a stop signal. Any exception, e.g. the panda going silent, ends it and main() resets the panda"""
    rk = Ratekeeper(100, print_delay_threshold=None)
    while not do_exit:
      self.step(rk.frame)
      rk.monitor_time()
      self.sender.wait(time.monotonic() + rk.remaining)

    # Close relay on exit to prevent a fault
    if self.is_onroad and not self.engaged:
      self.panda.set_safety_mode(SafetyModel.noOutput)


# *** firmware ***

def panda_ready() -> bool:
  """Is the panda up and running the firmware we ship?"""
  try:
    # a healthy panda answers on SPI within ~200ms of reset
    if not Panda.wait_for_panda(None, timeout=1):
      cloudlog.warning("no panda found")
      return False
    with Panda(cli=False) as p:
      if p.bootstub:
        cloudlog.warning(f"panda stuck in bootstub, version {p.get_version()}")
        return False
      if not p.up_to_date():
        cloudlog.warning(f"panda firmware out of date, version {p.get_version()}")
        return False
      return True
  except Exception:
    # e.g. SPI protocol mismatch with old firmware
    cloudlog.exception("pandad.panda_check_failed")
    return False


def flash_panda() -> None:
  """Reflash bootstub + app from the ST bootloader. Works from any state the panda can be in."""
  cloudlog.warning("flashing panda")
  HARDWARE.recover_internal_panda()
  assert Panda.wait_for_dfu(None, timeout=5), "panda didn't enter DFU"
  with PandaDFU(None) as dfu:
    dfu.recover()

  assert Panda.wait_for_panda(None, timeout=5), "panda didn't boot into bootstub"
  with Panda(cli=False) as p:
    p.flash()
  cloudlog.warning("done flashing panda")


def main() -> None:
  # signal pandad to close the relay and exit
  def signal_handler(signum, frame):
    global do_exit
    cloudlog.info(f"Caught signal {signum}, exiting")
    do_exit = True
  for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, signal_handler)

  # for pidof pandad
  setproctitle("pandad")
  # the panda library's SPI errors go to swaglog
  logging.getLogger('panda').handlers = [ForwardingHandler(cloudlog)]

  # check health for lost heartbeat
  try:
    for s in Panda.list():
      with Panda(s) as p:
        health = p.health()
        if p.is_internal() and health["heartbeat_lost"]:
          Params().put_bool("PandaHeartbeatLost", True, block=True)
          cloudlog.event("heartbeat lost", deviceState=health)
  except Exception:
    cloudlog.exception("pandad.uncaught_exception")

  pm = messaging.PubMaster(['can', 'pandaStates', 'peripheralState'])
  sm = messaging.SubMaster(['selfdriveState', 'deviceState', 'cabinCameraState'])
  sendcan = messaging.sub_sock('sendcan')
  config_realtime_process(3, Priority.PANDAD)

  count = 0
  while not do_exit:
    try:
      cloudlog.event("pandad.flash_and_connect", count=count)
      count += 1

      HARDWARE.reset_internal_panda()
      if not panda_ready():
        flash_panda()
        assert panda_ready(), "panda not ready after flashing"

      # the internal (SPI) panda. pandad sends the heartbeat, so the checks stay enabled
      with Panda(Panda.spi_list()[0], cli=False, disable_checks=False) as panda:
        Pandad(panda, pm, sm, sendcan).run()
    except Exception:
      cloudlog.exception("pandad.uncaught_exception")


if __name__ == "__main__":
  main()
