#!/usr/bin/env python3
import os
import time
import coverage
import pyray as rl
from dataclasses import dataclass
from collections.abc import Callable
from openpilot.selfdrive.ui.tests.diff.diff import DIFF_OUT_DIR

os.environ["RECORD"] = "1"
if "RECORD_OUTPUT" not in os.environ:
  os.environ["RECORD_OUTPUT"] = "mici_ui_replay.mp4"

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
from openpilot.selfdrive.ui.mici.layouts.main import MiciMainLayout

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
  click: bool = False
  click_pos: tuple[int, int] | None = None
  swipe_left: bool = False
  swipe_right: bool = False
  swipe_down: bool = False
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

  def make_alert_refresh_setup(alert_fn):
    """Set up alerts and force an immediate refresh on the alerts layout."""
    def setup():
      alert_fn()
      main_layout._alerts_layout._last_refresh = 0
    return setup

  # === Homescreen (clean) ===
  add(0, DummyEvent())
  add(FPS, DummyEvent())  # dwell 1s

  # === Offroad Alerts ===
  add(0, DummyEvent(setup=make_alert_refresh_setup(setup_offroad_alerts)))
  add(int(FPS * 0.5), DummyEvent(swipe_left=True))
  add(HOLD, DummyEvent())  # dwell on alerts

  # Back to homescreen
  add(0, DummyEvent(swipe_right=True))
  add(HOLD, DummyEvent())

  # === Update Available ===
  add(0, DummyEvent(setup=make_alert_refresh_setup(setup_update_available)))
  add(int(FPS * 0.5), DummyEvent(swipe_left=True))
  add(HOLD, DummyEvent())  # dwell on update available

  # Back to homescreen
  add(0, DummyEvent(swipe_right=True))
  add(HOLD, DummyEvent())

  # === Settings ===
  # Click anywhere on homescreen to open settings
  add(0, DummyEvent(click=True))
  add(HOLD, DummyEvent())

  # Settings scroller button centers (vertical scroller, BigButton 402x180, spacing=20, pad_start=20):
  # toggles: y=110, network: y=310, device: y=510, pair: y=710, firehose: y=910, developer: y=1110
  # All centered at x=1080

  # Toggles panel
  add(0, DummyEvent(click_pos=(1080, 110)))
  add(HOLD, DummyEvent(swipe_down=True))
  add(int(FPS * 0.4), DummyEvent())

  # Network panel
  add(0, DummyEvent(click_pos=(1080, 310)))
  add(HOLD, DummyEvent(swipe_down=True))
  add(int(FPS * 0.4), DummyEvent())

  # Device panel
  add(0, DummyEvent(click_pos=(1080, 510)))
  add(HOLD, DummyEvent(swipe_down=True))
  add(int(FPS * 0.4), DummyEvent())

  # Pair button (opens pairing modal)
  add(0, DummyEvent(click_pos=(1080, 710)))
  add(HOLD, DummyEvent(setup=dismiss_modal))
  add(int(FPS * 0.3), DummyEvent())

  # Firehose panel
  add(0, DummyEvent(click_pos=(1080, 910)))
  add(HOLD, DummyEvent(swipe_down=True))
  add(int(FPS * 0.4), DummyEvent())

  # Developer panel (set CarParamsPersistent first to show alpha long toggle)
  add(0, DummyEvent(setup=setup_developer_params))
  add(int(FPS * 0.2), DummyEvent(click_pos=(1080, 1050)))
  add(HOLD, DummyEvent())

  # SSH keys / keyboard in developer panel (BigButton at 3rd position, y_center=510)
  add(0, DummyEvent(click_pos=(1080, 510)))
  add(HOLD, DummyEvent(setup=dismiss_modal))
  add(int(FPS * 0.3), DummyEvent())

  # Navigate back: developer panel -> settings scroller -> homescreen
  add(0, DummyEvent(swipe_down=True))
  add(int(FPS * 0.3), DummyEvent(swipe_down=True))
  add(int(FPS * 0.3), DummyEvent(swipe_down=True))
  add(int(FPS * 0.5), DummyEvent())

  # === Onroad ===
  add(0, DummyEvent(setup=make_onroad_setup(pm)))
  add(int(FPS * 3), DummyEvent())  # wait for ONROAD_DELAY (2.5s) + buffer

  # Onroad: small alert
  add(0, DummyEvent(setup=make_alert_setup(pm, AlertSize.small, "Small Alert", "This is a small alert", AlertStatus.normal)))
  add(HOLD, DummyEvent())

  # Onroad: medium alert
  add(0, DummyEvent(setup=make_alert_setup(pm, AlertSize.mid, "Medium Alert", "This is a medium alert", AlertStatus.userPrompt)))
  add(HOLD, DummyEvent())

  # Onroad: full alert
  add(0, DummyEvent(setup=make_alert_setup(pm, AlertSize.full, "DISENGAGE IMMEDIATELY", "Driver Distracted", AlertStatus.critical)))
  add(HOLD, DummyEvent())

  # Onroad: full alert multiline
  add(0, DummyEvent(setup=make_alert_setup(pm, AlertSize.full, "Reverse\nGear", "", AlertStatus.normal)))
  add(HOLD, DummyEvent())

  # Onroad: full alert long text
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


def inject_click(coords):
  events = []
  x, y = coords[0]
  events.append(MouseEvent(pos=MousePos(x, y), slot=0, left_pressed=True, left_released=False, left_down=False, t=time.monotonic()))
  for x, y in coords[1:]:
    events.append(MouseEvent(pos=MousePos(x, y), slot=0, left_pressed=False, left_released=False, left_down=True, t=time.monotonic()))
  x, y = coords[-1]
  events.append(MouseEvent(pos=MousePos(x, y), slot=0, left_pressed=False, left_released=True, left_down=False, t=time.monotonic()))

  with gui_app._mouse._lock:
    gui_app._mouse._events.extend(events)


def handle_event(event: DummyEvent):
  if event.setup:
    event.setup()
  if event.click:
    inject_click([(gui_app.width // 2, gui_app.height // 2)])
  if event.click_pos:
    inject_click([event.click_pos])
  if event.swipe_left:
    inject_click([(gui_app.width * 3 // 4, gui_app.height // 2),
                  (gui_app.width // 4, gui_app.height // 2),
                  (0, gui_app.height // 2)])
  if event.swipe_right:
    inject_click([(gui_app.width // 4, gui_app.height // 2),
                  (gui_app.width * 3 // 4, gui_app.height // 2),
                  (gui_app.width, gui_app.height // 2)])
  if event.swipe_down:
    inject_click([(gui_app.width // 2, gui_app.height // 4),
                  (gui_app.width // 2, gui_app.height * 3 // 4),
                  (gui_app.width // 2, gui_app.height)])


def run_replay():
  global _frame_fn
  pm = setup_state()
  os.makedirs(DIFF_OUT_DIR, exist_ok=True)

  if not HEADLESS:
    rl.set_config_flags(rl.FLAG_WINDOW_HIDDEN)
  gui_app.init_window("ui diff test", fps=FPS)
  main_layout = MiciMainLayout()
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
  cov = coverage.coverage(source=['openpilot.selfdrive.ui.mici'])
  with cov.collect():
    run_replay()
  cov.stop()
  cov.save()
  cov.report()
  cov.html_report(directory=os.path.join(DIFF_OUT_DIR, 'htmlcov'))
  print("HTML report: htmlcov/index.html")


if __name__ == "__main__":
  main()
