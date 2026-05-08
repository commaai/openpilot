#!/usr/bin/env python3
import argparse
import signal
import time
from dataclasses import dataclass
from typing import Any

PANDA_LED_REQUEST = 0xf7
STARTUP_TIMEOUT = 300
RUNTIME_TIMEOUT = 3
STARTUP_GRACE = 30.
RUNTIME_HZ = 4.
ACCELERATION_DUE_TO_GRAVITY = 9.81
DEFAULT_MAX_LAT_ACCEL = 3.0
STARTED_AT = time.monotonic()

AudibleAlert: Any = None
Ratekeeper: Any = None
log: Any = None
messaging: Any = None


def cloudlog_exception(message: str) -> None:
  try:
    from openpilot.common.swaglog import cloudlog
    cloudlog.exception(message)
  except Exception as e:
    print(f"{message}: {e}")


@dataclass(frozen=True)
class LedState:
  name: str
  red: int
  green: int
  blue: int


STARTING = LedState("starting", 0, 36, 180)
READY = LedState("ready", 0, 36, 180)
ENGAGED = LedState("engaged", 0, 180, 35)
YELLOW = LedState("yellow", 180, 130, 0)
WARNING = LedState("warning", 180, 130, 0)
CRITICAL = LedState("critical", 180, 0, 0)
OFF = LedState("off", 0, 0, 0)


def pack_rgb565(red: int, green: int, blue: int) -> int:
  return ((red & 0xF8) << 8) | ((green & 0xFC) << 3) | (blue >> 3)


def clamp(value: float, low: float, high: float) -> float:
  return max(low, min(high, value))


def interp(value: float, x0: float, x1: float, y0: float, y1: float) -> float:
  if x1 == x0:
    return y1
  return y0 + (clamp((value - x0) / (x1 - x0), 0., 1.) * (y1 - y0))


def blend(name: str, start: LedState, end: LedState, amount: float) -> LedState:
  amount = clamp(amount, 0., 1.)
  return LedState(
    name,
    int(start.red + (end.red - start.red) * amount),
    int(start.green + (end.green - start.green) * amount),
    int(start.blue + (end.blue - start.blue) * amount),
  )


class PandaLed:
  def __init__(self) -> None:
    self.handle = None
    self.last_state: LedState | None = None
    self.last_sent = 0.

  def connect(self):
    if self.handle is None:
      from panda.python.spi import PandaSpiHandle
      self.handle = PandaSpiHandle()
    return self.handle

  def close(self) -> None:
    if self.handle is not None:
      self.handle.close()
      self.handle = None

  def set(self, state: LedState, timeout: int = RUNTIME_TIMEOUT, force: bool = False) -> None:
    now = time.monotonic()
    if not force and state == self.last_state and (now - self.last_sent) < 1.0:
      return

    try:
      self.connect().controlWrite(0, PANDA_LED_REQUEST, pack_rgb565(state.red, state.green, state.blue), timeout, b'', timeout=500)
      self.last_state = state
      self.last_sent = now
    except Exception:
      cloudlog_exception("failed to set panda LED")
      self.close()

  def clear(self) -> None:
    try:
      self.connect().controlWrite(0, PANDA_LED_REQUEST, 0, 0, b'', timeout=500)
      self.last_state = None
    except Exception:
      cloudlog_exception("failed to clear panda LED")
    finally:
      self.close()


def manager_failed(sm) -> bool:
  if time.monotonic() - STARTED_AT < STARTUP_GRACE:
    return False
  if not sm.seen['managerState']:
    return False

  for process in sm['managerState'].processes:
    if process.name == "ledd":
      continue
    if process.shouldBeRunning and not process.running:
      return True
  return False


def panda_disconnected(sm) -> bool:
  if not sm.seen['pandaStates'] or not sm.alive['pandaStates']:
    return True

  panda_states = sm['pandaStates']
  if len(panda_states) == 0:
    return True

  for panda_state in panda_states:
    if panda_state.pandaType == log.PandaState.PandaType.unknown:
      return True
  return False


def panda_failed(sm) -> bool:
  if time.monotonic() - STARTED_AT < STARTUP_GRACE:
    return False
  if panda_disconnected(sm):
    return False

  for panda_state in sm['pandaStates']:
    if panda_state.faultStatus == log.PandaState.FaultStatus.faultPerm:
      return True
    if len(panda_state.faults) > 0 or panda_state.heartbeatLost:
      return True
  return False


def driver_monitoring_alert_state(sm) -> LedState | None:
  if not sm.seen['driverMonitoringState'] or not sm.alive['driverMonitoringState']:
    return None

  dm_state = sm['driverMonitoringState']
  if dm_state.lockout or dm_state.alwaysOnLockout or dm_state.alertLevel == log.DriverMonitoringState.AlertLevel.three:
    return CRITICAL
  if dm_state.alertLevel in (log.DriverMonitoringState.AlertLevel.one, log.DriverMonitoringState.AlertLevel.two):
    return WARNING
  return None


def alert_state(sm) -> LedState | None:
  dm_alert = driver_monitoring_alert_state(sm)
  if dm_alert is not None:
    return dm_alert

  if not sm.seen['selfdriveState'] or not sm.alive['selfdriveState']:
    return None

  selfdrive_state = sm['selfdriveState']
  alert_sound = selfdrive_state.alertSound.raw
  has_alert = (
    selfdrive_state.alertStatus != log.SelfdriveState.AlertStatus.normal or
    selfdrive_state.alertSize != log.SelfdriveState.AlertSize.none or
    alert_sound != AudibleAlert.none or
    bool(selfdrive_state.alertText1) or
    bool(selfdrive_state.alertText2)
  )
  if not has_alert:
    return None

  if (
    selfdrive_state.state == log.SelfdriveState.OpenpilotState.softDisabling or
    selfdrive_state.alertStatus == log.SelfdriveState.AlertStatus.critical or
    selfdrive_state.alertSize == log.SelfdriveState.AlertSize.full or
    alert_sound == AudibleAlert.warningImmediate
  ):
    return CRITICAL
  return WARNING


def steering_utilization(sm) -> float:
  if not sm.seen['carControl'] or not sm.alive['carControl'] or not sm['carControl'].latActive:
    return 0.
  if not sm.seen['controlsState'] or not sm.alive['controlsState']:
    return 0.

  controls_state = sm['controlsState']
  lac = getattr(controls_state.lateralControlState, controls_state.lateralControlState.which())
  util = 0.

  if controls_state.lateralControlState.which() == 'angleState':
    if sm.seen['carState'] and sm.alive['carState'] and sm.seen['liveParameters'] and sm.alive['liveParameters']:
      car_state = sm['carState']
      live_parameters = sm['liveParameters']
      actual_lateral_accel = controls_state.curvature * car_state.vEgo ** 2
      desired_lateral_accel = controls_state.desiredCurvature * car_state.vEgo ** 2
      accel_diff = desired_lateral_accel - actual_lateral_accel
      roll_compensation = live_parameters.roll * ACCELERATION_DUE_TO_GRAVITY * interp(car_state.vEgo, 5., 15., 0., 1.)
      lateral_acceleration = actual_lateral_accel - roll_compensation
      max_lateral_acceleration = DEFAULT_MAX_LAT_ACCEL
      if sm.seen['carParams'] and sm['carParams'].maxLateralAccel > 0.:
        max_lateral_acceleration = sm['carParams'].maxLateralAccel
      util = abs(clamp((lateral_acceleration + accel_diff) / max_lateral_acceleration, -1., 1.))
  elif sm.seen['carOutput'] and sm.alive['carOutput']:
    util = abs(clamp(sm['carOutput'].actuatorsOutput.torque, -1., 1.))

  if getattr(lac, "saturated", False):
    util = max(util, 0.95)
  return util


def engaged_state(sm) -> LedState:
  util = steering_utilization(sm)
  if util < 0.65:
    return ENGAGED
  if util < 0.85:
    return blend("engaged_yellow", ENGAGED, YELLOW, (util - 0.65) / 0.20)
  return blend("engaged_red", YELLOW, CRITICAL, (util - 0.85) / 0.15)


def led_state(sm) -> tuple[LedState | None, bool]:
  if panda_disconnected(sm):
    return None, False

  if manager_failed(sm) or panda_failed(sm):
    return CRITICAL, True

  if not sm.seen['deviceState']:
    return STARTING, True

  alert = alert_state(sm)
  if alert is not None:
    return alert, True

  if sm.seen['selfdriveState'] and sm.alive['selfdriveState']:
    selfdrive_state = sm['selfdriveState']
    if selfdrive_state.enabled:
      return engaged_state(sm), False
    if sm['deviceState'].started and not selfdrive_state.engageable:
      return WARNING, True

  return READY, False


def set_startup_once(timeout: int = STARTUP_TIMEOUT) -> None:
  led = PandaLed()
  led.set(STARTING, timeout=timeout, force=True)
  led.close()


def startup_blink() -> None:
  led = PandaLed()
  done = False

  def sigterm_handler(signum, frame) -> None:
    nonlocal done
    done = True

  signal.signal(signal.SIGINT, sigterm_handler)
  signal.signal(signal.SIGTERM, sigterm_handler)

  end_time = time.monotonic() + STARTUP_TIMEOUT
  on = False
  while not done and time.monotonic() < end_time:
    on = not on
    led.set(STARTING if on else OFF, force=True)
    time.sleep(0.35)
  led.clear()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--startup-once", action="store_true")
  parser.add_argument("--startup-blink", action="store_true")
  parser.add_argument("--clear", action="store_true")
  args = parser.parse_args()

  led = PandaLed()
  if args.clear:
    led.clear()
    return
  if args.startup_once:
    set_startup_once()
    return
  if args.startup_blink:
    startup_blink()
    return

  global AudibleAlert, Ratekeeper, log, messaging
  from cereal import car, log as cereal_log, messaging as cereal_messaging
  from openpilot.common.realtime import Ratekeeper as OpenpilotRatekeeper
  AudibleAlert = car.CarControl.HUDControl.AudibleAlert
  Ratekeeper = OpenpilotRatekeeper
  log = cereal_log
  messaging = cereal_messaging

  done = False

  def sigterm_handler(signum, frame) -> None:
    nonlocal done
    done = True

  signal.signal(signal.SIGINT, sigterm_handler)
  signal.signal(signal.SIGTERM, sigterm_handler)

  sm = messaging.SubMaster([
    'carControl',
    'carOutput',
    'carParams',
    'carState',
    'controlsState',
    'deviceState',
    'driverMonitoringState',
    'liveParameters',
    'managerState',
    'pandaStates',
    'selfdriveState',
  ], ignore_avg_freq=['managerState'])
  rk = Ratekeeper(RUNTIME_HZ)

  while not done:
    sm.update(0)
    state, should_blink = led_state(sm)
    if state is None:
      if led.last_state is not None:
        led.clear()
    elif should_blink and int(time.monotonic() * 2) % 2 == 0:
      led.set(OFF, force=True)
    else:
      led.set(state, force=should_blink)
    rk.keep_time()

  led.clear()


if __name__ == "__main__":
  main()
