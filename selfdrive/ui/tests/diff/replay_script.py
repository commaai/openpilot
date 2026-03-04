from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Callable
from dataclasses import dataclass

from cereal import car, log, messaging
from cereal.messaging import PubMaster
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
from openpilot.selfdrive.ui.tests.diff.replay import FPS, LayoutVariant
from openpilot.system.updated.updated import parse_release_notes

WAIT = int(FPS * 0.5)  # Default frames to wait after events

AlertSize = log.SelfdriveState.AlertSize
AlertStatus = log.SelfdriveState.AlertStatus

BRANCH_NAME = "this-is-a-really-super-mega-ultra-max-extreme-ultimate-long-branch-name"


@dataclass
class ScriptEvent:
  if TYPE_CHECKING:
    # Only import for type checking to avoid excluding the application code from coverage
    from openpilot.system.ui.lib.application import MouseEvent

  setup: Callable | None = None  # Setup function to run prior to adding mouse events
  mouse_events: list[MouseEvent] | None = None  # Mouse events to send to the application on this event's frame
  send_fn: Callable | None = None  # When set, the main loop uses this as the new persistent sender


ScriptEntry = tuple[int, ScriptEvent]  # (frame, event)


class Script:
  def __init__(self, fps: int) -> None:
    self.fps = fps
    self.frame = 0
    self.entries: list[ScriptEntry] = []

  def get_frame_time(self) -> float:
    return self.frame / self.fps

  def add(self, event: ScriptEvent, before: int = 0, after: int = 0) -> None:
    """Add event to the script, optionally with the given number of frames to wait before or after the event."""
    self.frame += before
    self.entries.append((self.frame, event))
    self.frame += after

  def end(self) -> None:
    """Add a final empty event to mark the end of the script."""
    self.add(ScriptEvent())  # Without this, it will just end on the last event without waiting for any specified delay after it

  def wait(self, frames: int) -> None:
    """Add a delay for the given number of frames followed by an empty event."""
    self.add(ScriptEvent(), before=frames)

  def setup(self, fn: Callable, wait_after: int = WAIT) -> None:
    """Add a setup function to be called immediately followed by a delay of the given number of frames."""
    self.add(ScriptEvent(setup=fn), after=wait_after)

  def set_send(self, fn: Callable, wait_after: int = WAIT) -> None:
    """Set a new persistent send function to be called every frame."""
    self.add(ScriptEvent(send_fn=fn), after=wait_after)

  # TODO: Also add more complex gestures, like swipe or drag
  def click(self, x: int, y: int, wait_after: int = WAIT, wait_between: int = 2) -> None:
    """Add a click event to the script for the given position and specify frames to wait between mouse events or after the click."""
    # NOTE: By default we wait a couple frames between mouse events so pressed states will be rendered
    from openpilot.system.ui.lib.application import MouseEvent, MousePos

    # TODO: Add support for long press (left_down=True)
    mouse_down = MouseEvent(pos=MousePos(x, y), slot=0, left_pressed=True, left_released=False, left_down=False, t=self.get_frame_time())
    self.add(ScriptEvent(mouse_events=[mouse_down]), after=wait_between)
    mouse_up = MouseEvent(pos=MousePos(x, y), slot=0, left_pressed=False, left_released=True, left_down=False, t=self.get_frame_time())
    self.add(ScriptEvent(mouse_events=[mouse_up]), after=wait_after)


# --- Setup functions ---

def put_update_params(params: Params | None = None) -> None:
  if params is None:
    params = Params()
  params.put("UpdaterCurrentReleaseNotes", parse_release_notes(BASEDIR))
  params.put("UpdaterNewReleaseNotes", parse_release_notes(BASEDIR))
  params.put("UpdaterTargetBranch", BRANCH_NAME)


def setup_offroad_alerts() -> None:
  put_update_params(Params())
  set_offroad_alert("Offroad_TemperatureTooHigh", True, extra_text='99C')
  set_offroad_alert("Offroad_ExcessiveActuation", True, extra_text='longitudinal')
  set_offroad_alert("Offroad_IsTakingSnapshot", True)


def setup_update_available() -> None:
  params = Params()
  params.put_bool("UpdateAvailable", True)
  params.put("UpdaterNewDescription", f"0.10.2 / {BRANCH_NAME} / 0a1b2c3 / Jan 01")
  put_update_params(params)


def setup_developer_params() -> None:
  CP = car.CarParams()
  CP.alphaLongitudinalAvailable = True
  Params().put("CarParamsPersistent", CP.to_bytes())


# --- Send functions ---

def send_onroad(pm: PubMaster) -> None:
  ds = messaging.new_message('deviceState')
  ds.deviceState.started = True
  ds.deviceState.networkType = log.DeviceState.NetworkType.wifi

  ps = messaging.new_message('pandaStates', 1)
  ps.pandaStates[0].pandaType = log.PandaState.PandaType.dos
  ps.pandaStates[0].ignitionLine = True

  pm.send('deviceState', ds)
  pm.send('pandaStates', ps)


def make_network_state_setup(pm: PubMaster, network_type) -> Callable:
  def _send() -> None:
    ds = messaging.new_message('deviceState')
    ds.deviceState.networkType = network_type
    pm.send('deviceState', ds)
  return _send


def make_alert_setup(pm: PubMaster, size, text1, text2, status) -> Callable:
  def _send() -> None:
    send_onroad(pm)
    alert = messaging.new_message('selfdriveState')
    ss = alert.selfdriveState
    ss.alertSize = size
    ss.alertText1 = text1
    ss.alertText2 = text2
    ss.alertStatus = status
    pm.send('selfdriveState', alert)
  return _send


# --- Script builders ---

def build_mici_script(pm: PubMaster, main_layout, script: Script) -> None:
  """Build the replay script for the mici layout."""
  from openpilot.system.ui.lib.application import gui_app

  center = (gui_app.width // 2, gui_app.height // 2)

  # TODO: Explore more
  script.wait(FPS)
  script.click(*center, FPS)  # Open settings
  script.click(*center, FPS)  # Open toggles
  script.end()


def build_tizi_script(pm: PubMaster, main_layout, script: Script) -> None:
  """Build the replay script for the tizi layout."""

  def make_home_refresh_setup(fn: Callable) -> Callable:
    """Return setup function that calls the given function to modify state and forces an immediate refresh on the home layout."""
    from openpilot.selfdrive.ui.layouts.main import MainState

    def setup():
      fn()
      main_layout._layouts[MainState.HOME].last_refresh = 0

    return setup

  # TODO: Better way of organizing the events

  # === Homescreen ===
  script.set_send(make_network_state_setup(pm, log.DeviceState.NetworkType.wifi))

  # === Offroad Alerts (auto-transitions via HomeLayout refresh) ===
  script.setup(make_home_refresh_setup(setup_offroad_alerts))

  # === Update Available (auto-transitions via HomeLayout refresh) ===
  script.setup(make_home_refresh_setup(setup_update_available))

  # === Settings - Device (click sidebar settings button) ===
  script.click(150, 90)
  script.click(1985, 790)  # reset calibration confirmation
  script.click(650, 750)  # cancel

  # === Settings - Network ===
  script.click(278, 450)
  script.click(1880, 100)  # advanced network settings
  script.click(630, 80)  # back

  # === Settings - Toggles ===
  script.click(278, 600)
  script.click(1200, 280)  # experimental mode description

  # === Settings - Software ===
  script.setup(put_update_params, wait_after=0)
  script.click(278, 720)

  # === Settings - Firehose ===
  script.click(278, 845)

  # === Settings - Developer (set CarParamsPersistent first) ===
  script.setup(setup_developer_params, wait_after=0)
  script.click(278, 950)
  script.click(2000, 960)  # toggle alpha long
  script.click(1500, 875)  # confirm

  # === Keyboard modal (SSH keys button in developer panel) ===
  script.click(1930, 470)  # click SSH keys
  script.click(1930, 115)  # click cancel on keyboard

  # === Close settings ===
  script.click(250, 160)

  # === Onroad ===
  script.set_send(lambda: send_onroad(pm))
  script.click(1000, 500)  # click onroad to toggle sidebar

  # === Onroad alerts ===
  # Small alert (normal)
  script.set_send(make_alert_setup(pm, AlertSize.small, "Small Alert", "This is a small alert", AlertStatus.normal))
  # Medium alert (userPrompt)
  script.set_send(make_alert_setup(pm, AlertSize.mid, "Medium Alert", "This is a medium alert", AlertStatus.userPrompt))
  # Full alert (critical)
  script.set_send(make_alert_setup(pm, AlertSize.full, "DISENGAGE IMMEDIATELY", "Driver Distracted", AlertStatus.critical))
  # Full alert multiline
  script.set_send(make_alert_setup(pm, AlertSize.full, "Reverse\nGear", "", AlertStatus.normal))
  # Full alert long text
  script.set_send(make_alert_setup(pm, AlertSize.full, "TAKE CONTROL IMMEDIATELY", "Calibration Invalid: Remount Device & Recalibrate", AlertStatus.userPrompt))

  # End
  script.end()


def build_script(pm: PubMaster, main_layout, variant: LayoutVariant) -> list[ScriptEntry]:
  """Build the replay script for the appropriate layout variant and return list of script entries."""
  print(f"Building {variant} replay script...")

  script = Script(FPS)
  builder = build_tizi_script if variant == 'tizi' else build_mici_script
  builder(pm, main_layout, script)

  print(f"Built replay script with {len(script.entries)} events and {script.frame} frames ({script.get_frame_time():.2f} seconds)")

  return script.entries
