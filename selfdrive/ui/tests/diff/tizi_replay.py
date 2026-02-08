#!/usr/bin/env python3
import os
import time
import coverage
import pyray as rl
from dataclasses import dataclass
from collections.abc import Callable
from openpilot.selfdrive.ui.tests.diff.diff import DIFF_OUT_DIR

os.environ["BIG"] = "1"
os.environ["RECORD"] = "1"
if "RECORD_OUTPUT" not in os.environ:
  os.environ["RECORD_OUTPUT"] = "tizi_ui_replay.mp4"

os.environ["RECORD_OUTPUT"] = os.path.join(DIFF_OUT_DIR, os.environ["RECORD_OUTPUT"])

from cereal import car, log, messaging
from cereal.messaging import PubMaster
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
from openpilot.system.updated.updated import parse_release_notes
from openpilot.system.version import terms_version, training_version
from openpilot.system.ui.lib.application import gui_app, MousePos, MouseEvent
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.layouts.main import MainLayout, MainState

FPS = 60
HEADLESS = os.getenv("WINDOWED", "0") == "1"
HOLD = int(FPS * 0.75)

AlertSize = log.SelfdriveState.AlertSize
AlertStatus = log.SelfdriveState.AlertStatus

BRANCH_NAME = "this-is-a-really-super-mega-ultra-max-extreme-ultimate-long-branch-name"

# Persistent per-frame sender function, set by setup callbacks to keep sending cereal messages
_frame_fn: Callable | None = None


@dataclass
class DummyEvent:
  click_pos: tuple[int, int] | None = None
  setup: Callable | None = None


# --- Setup helper functions ---

def put_update_params(params: Params):
  params.put("UpdaterCurrentReleaseNotes", parse_release_notes(BASEDIR))
  params.put("UpdaterNewReleaseNotes", parse_release_notes(BASEDIR))
  params.put("UpdaterTargetBranch", BRANCH_NAME)


def setup_offroad_alerts():
  put_update_params(Params())
  set_offroad_alert("Offroad_TemperatureTooHigh", True, extra_text='99C')
  set_offroad_alert("Offroad_ExcessiveActuation", True, extra_text='longitudinal')
  set_offroad_alert("Offroad_IsTakingSnapshot", True)


def setup_update_available():
  params = Params()
  params.put_bool("UpdateAvailable", True)
  params.put("UpdaterNewDescription", f"0.10.1 / {BRANCH_NAME} / 7864838 / Oct 03")
  put_update_params(params)


def setup_developer_params():
  CP = car.CarParams()
  CP.alphaLongitudinalAvailable = True
  Params().put("CarParamsPersistent", CP.to_bytes())


def dismiss_modal():
  gui_app.set_modal_overlay(None)


def send_onroad(pm):
  ds = messaging.new_message('deviceState')
  ds.deviceState.started = True
  ds.deviceState.networkType = log.DeviceState.NetworkType.wifi

  ps = messaging.new_message('pandaStates', 1)
  ps.pandaStates[0].pandaType = log.PandaState.PandaType.dos
  ps.pandaStates[0].ignitionLine = True

  pm.send('deviceState', ds)
  pm.send('pandaStates', ps)


def make_onroad_setup(pm):
  def _send():
    send_onroad(pm)

  def setup():
    global _frame_fn
    _frame_fn = _send
    send_onroad(pm)
  return setup


def make_alert_setup(pm, size, text1, text2, status):
  def _send():
    send_onroad(pm)
    alert = messaging.new_message('selfdriveState')
    ss = alert.selfdriveState
    ss.alertSize = size
    ss.alertText1 = text1
    ss.alertText2 = text2
    ss.alertStatus = status
    pm.send('selfdriveState', alert)

  def setup():
    global _frame_fn
    _frame_fn = _send
    _send()
  return setup


def build_script(pm, main_layout):
  t = 0
  script = []

  def add(dt, event):
    nonlocal t
    t += dt
    script.append((t, event))

  def make_home_refresh_setup(fn):
    """Set up state and force an immediate refresh on the home layout."""
    def setup():
      fn()
      main_layout._layouts[MainState.HOME].last_refresh = 0
    return setup

  # === Homescreen (clean) ===
  add(0, DummyEvent())
  add(FPS, DummyEvent())  # dwell 1s

  # === Offroad Alerts (auto-transitions via HomeLayout refresh) ===
  add(0, DummyEvent(setup=make_home_refresh_setup(setup_offroad_alerts)))
  add(HOLD, DummyEvent())

  # === Update Available (auto-transitions via HomeLayout refresh) ===
  add(0, DummyEvent(setup=make_home_refresh_setup(setup_update_available)))
  add(HOLD, DummyEvent())

  # === Settings - Device (click sidebar settings button) ===
  # Sidebar SETTINGS_BTN = rl.Rectangle(50, 35, 200, 117), center ~(150, 93)
  add(0, DummyEvent(click_pos=(150, 93)))
  add(HOLD, DummyEvent())

  # === Settings - Network ===
  # Nav buttons start at y=300, height=110, x centered ~278
  add(0, DummyEvent(click_pos=(278, 450)))
  add(HOLD, DummyEvent())

  # === Settings - Toggles ===
  add(0, DummyEvent(click_pos=(278, 600)))
  add(HOLD, DummyEvent())

  # === Settings - Software ===
  add(0, DummyEvent(setup=lambda: put_update_params(Params())))
  add(int(FPS * 0.2), DummyEvent(click_pos=(278, 720)))
  add(HOLD, DummyEvent())

  # === Settings - Firehose ===
  add(0, DummyEvent(click_pos=(278, 845)))
  add(HOLD, DummyEvent())

  # === Settings - Developer (set CarParamsPersistent first) ===
  add(0, DummyEvent(setup=setup_developer_params))
  add(int(FPS * 0.2), DummyEvent(click_pos=(278, 950)))
  add(HOLD, DummyEvent())

  # === Keyboard modal (SSH keys button in developer panel) ===
  add(0, DummyEvent(click_pos=(1930, 470)))
  add(HOLD, DummyEvent(setup=dismiss_modal))
  add(int(FPS * 0.3), DummyEvent())

  # === Close settings (close button center ~(250, 160)) ===
  add(0, DummyEvent(click_pos=(250, 160)))
  add(HOLD, DummyEvent())

  # === Onroad ===
  add(0, DummyEvent(setup=make_onroad_setup(pm)))
  add(int(FPS * 1.5), DummyEvent())  # wait for transition

  # === Onroad with sidebar (click onroad to toggle) ===
  add(0, DummyEvent(click_pos=(1000, 500)))
  add(HOLD, DummyEvent())

  # === Onroad alerts ===
  # Small alert
  add(0, DummyEvent(setup=make_alert_setup(pm, AlertSize.small, "Small Alert", "This is a small alert", AlertStatus.normal)))
  add(HOLD, DummyEvent())

  # Medium alert
  add(0, DummyEvent(setup=make_alert_setup(pm, AlertSize.mid, "Medium Alert", "This is a medium alert", AlertStatus.userPrompt)))
  add(HOLD, DummyEvent())

  # Full alert
  add(0, DummyEvent(setup=make_alert_setup(pm, AlertSize.full, "DISENGAGE IMMEDIATELY", "Driver Distracted", AlertStatus.critical)))
  add(HOLD, DummyEvent())

  # Full alert multiline
  add(0, DummyEvent(setup=make_alert_setup(pm, AlertSize.full, "Reverse\nGear", "", AlertStatus.normal)))
  add(HOLD, DummyEvent())

  # Full alert long text
  add(0, DummyEvent(setup=make_alert_setup(pm, AlertSize.full, "TAKE CONTROL IMMEDIATELY",
                     "Calibration Invalid: Remount Device & Recalibrate", AlertStatus.userPrompt)))
  add(HOLD, DummyEvent())

  # End
  add(0, DummyEvent())

  return script


def setup_state():
  params = Params()
  params.put("HasAcceptedTerms", terms_version)
  params.put("CompletedTrainingVersion", training_version)
  params.put("DongleId", "test123456789")
  params.put("UpdaterCurrentDescription", "0.10.1 / test-branch / abc1234 / Nov 30")

  pm = PubMaster(["deviceState", "pandaStates", "driverStateV2", "selfdriveState"])

  # Seed initial offroad device state
  ds = messaging.new_message('deviceState')
  ds.deviceState.networkType = log.DeviceState.NetworkType.wifi
  pm.send('deviceState', ds)

  return pm


def inject_click(x, y):
  events = [
    MouseEvent(pos=MousePos(x, y), slot=0, left_pressed=True, left_released=False, left_down=False, t=time.monotonic()),
    MouseEvent(pos=MousePos(x, y), slot=0, left_pressed=False, left_released=True, left_down=False, t=time.monotonic()),
  ]
  with gui_app._mouse._lock:
    gui_app._mouse._events.extend(events)


def handle_event(event: DummyEvent):
  if event.setup:
    event.setup()
  if event.click_pos:
    inject_click(*event.click_pos)


def run_replay():
  global _frame_fn
  pm = setup_state()
  os.makedirs(DIFF_OUT_DIR, exist_ok=True)

  if not HEADLESS:
    rl.set_config_flags(rl.FLAG_WINDOW_HIDDEN)
  gui_app.init_window("ui diff test", fps=FPS)
  main_layout = MainLayout()
  main_layout.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))

  SCRIPT = build_script(pm, main_layout)

  frame = 0
  script_index = 0

  for should_render in gui_app.render():
    while script_index < len(SCRIPT) and SCRIPT[script_index][0] == frame:
      _, event = SCRIPT[script_index]
      handle_event(event)
      script_index += 1

    # Keep sending cereal messages for persistent states (onroad, alerts)
    if _frame_fn:
      _frame_fn()

    ui_state.update()

    if should_render:
      main_layout.render()

    frame += 1

    if script_index >= len(SCRIPT):
      break

  gui_app.close()

  print(f"Total frames: {frame}")
  print(f"Video saved to: {os.environ['RECORD_OUTPUT']}")


def main():
  cov = coverage.coverage(source=['openpilot.selfdrive.ui.layouts'])
  with cov.collect():
    run_replay()
  cov.stop()
  cov.save()
  cov.report()
  cov.html_report(directory=os.path.join(DIFF_OUT_DIR, 'htmlcov'))
  print("HTML report: htmlcov/index.html")


if __name__ == "__main__":
  main()
