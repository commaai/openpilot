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
STANDBY = LedState("standby", 8, 12, 32)
READY = LedState("ready", 0, 150, 120)
ENGAGED = LedState("engaged", 0, 180, 35)
WARNING = LedState("warning", 180, 80, 0)
CRITICAL = LedState("critical", 180, 0, 0)


def pack_rgb565(red: int, green: int, blue: int) -> int:
  return ((red & 0xF8) << 8) | ((green & 0xFC) << 3) | (blue >> 3)


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


def panda_failed(sm) -> bool:
  if time.monotonic() - STARTED_AT < STARTUP_GRACE:
    return False
  if not sm.seen['pandaStates'] or not sm.alive['pandaStates']:
    return sm.seen['managerState']

  panda_states = sm['pandaStates']
  if len(panda_states) == 0:
    return True

  for panda_state in panda_states:
    if panda_state.pandaType == log.PandaState.PandaType.unknown:
      return True
    if panda_state.faultStatus == log.PandaState.FaultStatus.faultPerm:
      return True
    if len(panda_state.faults) > 0 or panda_state.heartbeatLost:
      return True
  return False


def alert_state(sm) -> LedState | None:
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
    selfdrive_state.alertStatus == log.SelfdriveState.AlertStatus.critical or
    selfdrive_state.alertSize == log.SelfdriveState.AlertSize.full or
    alert_sound == AudibleAlert.warningImmediate
  ):
    return CRITICAL
  return WARNING


def led_state(sm) -> LedState:
  if manager_failed(sm) or panda_failed(sm):
    return CRITICAL

  if not sm.seen['deviceState']:
    return STARTING

  alert = alert_state(sm)
  if alert is not None:
    return alert

  if sm.seen['selfdriveState'] and sm.alive['selfdriveState']:
    selfdrive_state = sm['selfdriveState']
    if selfdrive_state.enabled:
      return ENGAGED
    if sm['deviceState'].started and not selfdrive_state.engageable:
      return WARNING

  if sm['deviceState'].started:
    return READY
  return STANDBY


def set_startup_once() -> None:
  led = PandaLed()
  led.set(STARTING, timeout=STARTUP_TIMEOUT, force=True)
  led.close()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--startup-once", action="store_true")
  parser.add_argument("--clear", action="store_true")
  args = parser.parse_args()

  led = PandaLed()
  if args.clear:
    led.clear()
    return
  if args.startup_once:
    set_startup_once()
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

  sm = messaging.SubMaster(['deviceState', 'managerState', 'pandaStates', 'selfdriveState'], ignore_avg_freq=['managerState'])
  rk = Ratekeeper(1.0)
  led.set(STARTING, force=True)

  while not done:
    sm.update(0)
    led.set(led_state(sm))
    rk.keep_time()

  led.clear()


if __name__ == "__main__":
  main()
