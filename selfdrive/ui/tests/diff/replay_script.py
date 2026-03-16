from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Callable
from dataclasses import dataclass

import math

from cereal import car, log, messaging
from cereal.messaging import PubMaster
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
from openpilot.selfdrive.ui.lib.prime_state import PrimeType
from openpilot.selfdrive.ui.tests.diff.replay import FPS, LayoutVariant
from openpilot.system.updated.updated import parse_release_notes

# Default frames to wait after events
WAIT_LONG = FPS
WAIT_SHORT = FPS // 2
FAST_CLICK = FPS // 6

# Direction vectors for drag gestures
DIR_LEFT = (-1, 0)
DIR_RIGHT = (1, 0)
DIR_UP = (0, -1)
DIR_DOWN = (0, 1)

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

  def setup(self, fn: Callable, wait_after: int = WAIT_SHORT) -> None:
    """Add a setup function to be called immediately followed by a delay of the given number of frames."""
    self.add(ScriptEvent(setup=fn), after=wait_after)

  def set_send(self, fn: Callable, wait_after: int = WAIT_SHORT) -> None:
    """Set a new persistent send function to be called every frame."""
    self.add(ScriptEvent(send_fn=fn), after=wait_after)

  def click(self, x: int, y: int, wait_after: int = WAIT_SHORT, wait_between: int = 2) -> None:
    """Add a click event to the script for the given position and specify frames to wait between mouse events or after the click."""
    # NOTE: By default we wait a couple frames between mouse events so pressed states will be rendered
    from openpilot.system.ui.lib.application import MouseEvent, MousePos

    mouse_down = MouseEvent(pos=MousePos(x, y), slot=0, left_pressed=True, left_released=False, left_down=False, t=self.get_frame_time())
    self.add(ScriptEvent(mouse_events=[mouse_down]), after=wait_between)
    mouse_up = MouseEvent(pos=MousePos(x, y), slot=0, left_pressed=False, left_released=True, left_down=False, t=self.get_frame_time())
    self.add(ScriptEvent(mouse_events=[mouse_up]), after=wait_after)

  def drag(self, start_x: int, start_y: int, direction: tuple[int, int], distance: int, duration_frames: int, wait_after: int = WAIT_LONG) -> None:
    """Add a drag gesture to the script from start position in the specified direction by the given distance over the given number of frames."""
    from openpilot.system.ui.lib.application import MouseEvent, MousePos

    # Calculate delta and end position based on direction and distance
    delta_x, delta_y = direction[0] * distance, direction[1] * distance
    end_x, end_y = start_x + delta_x, start_y + delta_y

    # Mouse down at start
    mouse_down = MouseEvent(pos=MousePos(start_x, start_y), slot=0, left_pressed=True, left_released=False, left_down=True, t=self.get_frame_time())
    self.add(ScriptEvent(mouse_events=[mouse_down]), after=1)

    # Interpolate positions over duration_frames
    for i in range(1, duration_frames):
      t = i / duration_frames
      x, y = int(start_x + delta_x * t), int(start_y + delta_y * t)
      mouse_move = MouseEvent(pos=MousePos(x, y), slot=0, left_pressed=False, left_released=False, left_down=True, t=self.get_frame_time())
      self.add(ScriptEvent(mouse_events=[mouse_move]), after=1)

    # Mouse up at end
    mouse_up = MouseEvent(pos=MousePos(end_x, end_y), slot=0, left_pressed=False, left_released=True, left_down=False, t=self.get_frame_time())
    self.add(ScriptEvent(mouse_events=[mouse_up]), after=wait_after)


# --- Setup functions ---


def set_prime_state(prime_type: PrimeType) -> None:
  from openpilot.selfdrive.ui.ui_state import ui_state
  ui_state.prime_state.set_type(prime_type)


def setup_offroad_alerts() -> None:
  set_offroad_alert("Offroad_TemperatureTooHigh", True, extra_text='99C')
  set_offroad_alert("Offroad_ExcessiveActuation", True, extra_text='longitudinal')
  set_offroad_alert("Offroad_IsTakingSnapshot", True)


def setup_update_available(available: bool = True) -> None:
  params = Params()
  params.put_bool("UpdateAvailable", available)
  params.put("UpdaterAvailableBranches", ",".join(["test-branch", "test-branch-2", BRANCH_NAME]))
  if available:
    params.put("UpdaterNewDescription", f"0.10.2 / {BRANCH_NAME} / 0a1b2c3 / Jan 01")
    params.put("UpdaterNewReleaseNotes", parse_release_notes(BASEDIR))
    params.put("UpdaterTargetBranch", BRANCH_NAME)
  else:
    params.remove("UpdaterNewDescription")
    params.remove("UpdaterNewReleaseNotes")
    params.remove("UpdaterTargetBranch")


def setup_calibration_params() -> None:
  params = Params()
  # live calibration
  calib = messaging.new_message('liveCalibration')
  calib.liveCalibration.calStatus = log.LiveCalibrationData.Status.calibrated
  calib.liveCalibration.rpyCalib = [0.0, math.radians(2.5), math.radians(-1.2)]
  params.put("CalibrationParams", calib.to_bytes())
  # live delay
  delay = messaging.new_message('liveDelay')
  delay.liveDelay.calPerc = 75
  params.put("LiveDelay", delay.to_bytes())
  # live torque parameters
  torque = messaging.new_message('liveTorqueParameters')
  torque.liveTorqueParameters.useParams = True
  torque.liveTorqueParameters.calPerc = 60
  params.put("LiveTorqueParameters", torque.to_bytes())


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
    alert = messaging.new_message('selfdriveState')
    ss = alert.selfdriveState
    ss.alertSize = size
    ss.alertText1 = text1
    ss.alertText2 = text2
    ss.alertStatus = status
    pm.send('selfdriveState', alert)
  return _send


def test_onroad_alerts(script: Script, pm: PubMaster) -> None:
  """Go through various alert types and sizes and add them to the script to test alert rendering.
    Each alert is sent as a separate event with a delay in between."""
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


# --- Script builders ---

def build_mici_script(pm: PubMaster, main_layout, script: Script) -> None:
  """Build the replay script for the mici layout."""
  from openpilot.system.ui.lib.application import gui_app

  width, height = gui_app.width, gui_app.height
  center = (width // 2, height // 2)
  right = (width * 4 // 5, height // 2)
  left = (width // 5, height // 2)
  top = (width // 2, height // 10)
  bottom = (width // 2, height * 9 // 10)

  DURATION = 5
  SWIPE_WAIT = FPS * 3 // 4

  def click(times: int = 1, wait_after: int = WAIT_SHORT) -> None:
    """Click at the center of the screen the given number of times with optional delay after."""
    for _ in range(times):
      script.click(*center, wait_after=wait_after)

  def press(x: int, y: int, duration_frames: int = DURATION, wait_after: int = WAIT_SHORT) -> None:
    """Perform a drag with no movement to simulate a left_down mouse event at the given position for the specified duration and delay after."""
    script.drag(x, y, (0, 0), 0, duration_frames, wait_after=wait_after)

  def swipe_left(distance: int = right[0] - left[0], duration_frames: int = DURATION, wait_after: int = SWIPE_WAIT) -> None:
    """Drag from right edge to left (scroll right / slide confirmation)."""
    script.drag(*right, DIR_LEFT, distance, duration_frames, wait_after)

  def swipe_right(distance: int = right[0] - left[0], duration_frames: int = DURATION, wait_after: int = SWIPE_WAIT) -> None:
    """Drag from left edge to right (scroll left)."""
    script.drag(*left, DIR_RIGHT, distance, duration_frames, wait_after)

  def swipe_down(distance: int = bottom[1] - top[1], duration_frames: int = DURATION, wait_after: int = SWIPE_WAIT) -> None:
    """Drag from top edge to bottom (scroll up / go back)."""
    script.drag(*top, DIR_DOWN, distance, duration_frames, wait_after)

  def swipe_up(distance: int = bottom[1] - top[1], duration_frames: int = DURATION, wait_after: int = SWIPE_WAIT) -> None:
    """Drag from bottom edge to top (scroll down)."""
    script.drag(*bottom, DIR_UP, distance, duration_frames, wait_after)

  ActionFn = Callable[[], None] | None
  Cases = list[ActionFn]

  def run_actions(*actions: ActionFn, after_each: ActionFn = None) -> None:
    """Helper function to run a sequence of actions in order for interaction tests, calling after_each callback after each action if provided."""
    for action in actions:
      if action is not None:
        action()
      if after_each is not None:
        after_each()

  def explore_setting(*actions: ActionFn) -> None:
    """Helper function to open a settings item, run the given actions, and go back."""
    run_actions(click, *actions, swipe_down)  # open, interact, go back

  def scroll_through_cases(cases: Cases) -> None:
    """Helper function to explore a panel by calling the interaction callbacks for each item/page before swiping to the next one."""
    run_actions(*cases, after_each=lambda: swipe_left(210, 10))  # swipe to roughly the center of the next toggle after each case

  def interact_keyboard() -> None:
    """Interact with the keyboard in various ways to test different actions and states.
    Assumes it's a password keyboard with 8 characters required. Closes by pressing confirm at the end."""
    KEY = (250, 160)  # key in the middle of the keyboard ('G')
    SHIFT = (50, 210)
    NUMBERS = (480, 210)
    SPACE = (500, 160)
    BACKSPACE = (490, 30)
    CONFIRM = (50, 30)
    # Begin interactions
    press(*CONFIRM, wait_after=FAST_CLICK)  # confirm while disabled should do nothing
    swipe_left(duration_frames=FPS // 2)  # swipe to type
    swipe_up(duration_frames=FPS // 2)  # swipe out of keyboard (nothing typed)
    # press various keys to test different states:
    for key in [
      SHIFT, KEY, KEY, SHIFT, SHIFT, KEY, KEY,  # test casing (upper, lower, caps lock)
      SPACE, SPACE, BACKSPACE, BACKSPACE,  # test multiple space and backspace
      NUMBERS, KEY, center, SHIFT, KEY  # test numbers and symbols
    ]:
      press(*key, wait_after=FAST_CLICK)
    # press confirm to close
    script.wait(WAIT_SHORT)  # wait for confirm to enable
    press(*CONFIRM)

  toggle_cases: Cases = [
    lambda: click(times=3, wait_after=FAST_CLICK),  # first toggle is personality, which has 3 states
    None, None, None, None, None, None,  # skip other toggles to save time
    lambda: click(times=2, wait_after=FAST_CLICK),  # test final toggle (enable openpilot)
  ]

  network_cases: Cases = [
    explore_setting,  # select wifi (just open and close)
    None, None,
    lambda: run_actions(click, interact_keyboard),  # tether password keyboard
  ]

  device_cases: Cases = [
    None,
    click,  # update
    explore_setting,  # pairing (just open and close)
    lambda: explore_setting(
      # training guide
      lambda: swipe_left(width * 2), click,  # first page, click next
      lambda: swipe_left(width * 2), swipe_down  # second page, go back (TODO: make driver cam preview work)
    ),
    None,  # TODO: preview driver camera; enabling this causes MultiplePublishersError later in onroad alert tests
    lambda: explore_setting(swipe_left),  # terms & conditions (swipe to view QR code)
    lambda: explore_setting(lambda: swipe_up(height * 3), lambda: swipe_down(height * 3)),  # regulatory info
    lambda: run_actions(click, lambda: swipe_left(width)),  # reset calibration confirm (goes back automatically)
    lambda: explore_setting(lambda: swipe_left(width)),  # uninstall
    lambda: run_actions(
      lambda: explore_setting(lambda: swipe_left(width)),  # reboot
      lambda: script.click(430, 120), lambda: swipe_left(width), swipe_down,  # shutdown
    ),
  ]

  developer_cases: Cases = [
    lambda: click(times=2, wait_after=FAST_CLICK),  # toggle ssh mode
    explore_setting,  # SSH keys keyboard (just open and close)
    None,  # joystick mode
    lambda: click(wait_after=FAST_CLICK),  # longitudinal maneuver mode (disabled; should do nothing)
    lambda: click(times=2, wait_after=FAST_CLICK),  # toggle UI debug mode
  ]

  settings_cases: Cases = [
    lambda: scroll_through_cases(toggle_cases),
    lambda: scroll_through_cases(network_cases),
    lambda: scroll_through_cases(device_cases),
    lambda: script.wait(WAIT_SHORT),  # pairing
    lambda: run_actions(lambda: swipe_up(height * 3), lambda: swipe_down(height * 3)),  # firehose (scroll down and back up)
    lambda: scroll_through_cases(developer_cases),
  ]

  # === Homescreen === #
  script.wait(WAIT_SHORT)
  swipe_left(width, wait_after=WAIT_SHORT)  # onroad screen
  swipe_right(width, wait_after=WAIT_SHORT)  # back to home

  # === Offroad Alerts ===
  def setup_offroad_alerts_and_refresh() -> None:
    """Setup function to trigger offroad alerts and force a refresh on the alerts layout."""
    setup_offroad_alerts()
    main_layout._alerts_layout.refresh()

  swipe_right(width, wait_after=WAIT_SHORT)  # open alerts
  script.setup(setup_offroad_alerts_and_refresh)  # show alerts
  swipe_up(height)  # scroll alerts
  swipe_left(width, wait_after=WAIT_SHORT)  # close alerts

  # === Settings === #
  click()  # open settings
  scroll_through_cases([lambda case=case: explore_setting(case) for case in settings_cases])  # explore settings
  swipe_down()  # back to home

  # === Onroad ===
  script.set_send(lambda: send_onroad(pm))
  swipe_left(width, wait_after=WAIT_SHORT)  # onroad screen
  test_onroad_alerts(script, pm)
  swipe_right(width)  # back to home

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

  def add_prime_state_setup(prime_type: PrimeType) -> None:
    script.set_send(lambda: set_prime_state(prime_type))

  def do_onboarding() -> None:
    """Click through the training guide and close."""
    from openpilot.selfdrive.ui.layouts.onboarding import STEP_RECTS
    step = 0
    for step_rect in STEP_RECTS:
      if step < len(STEP_RECTS) - 1:
        script.click(int(step_rect.x), int(step_rect.y), wait_after=FAST_CLICK)
      else:
        script.click(950, 900)  # On the last step, click Finish instead of restart
      step += 1

  def type_keyboard() -> None:
    """Types 8 characters using the big keyboard to test different layouts and interactions."""
    KEY = (150, 430)  # e.g. 'Q' key
    SHIFT = (150, 750)  # also symbols key in number mode
    NUMBERS = (150, 950)
    SPACE = (1060, 950)
    BACKSPACE = (2000, 780)
    for key in [
      SHIFT, KEY, KEY, SHIFT, SHIFT, KEY, KEY,  # test casing (upper, lower, caps lock)
      SPACE, SPACE, BACKSPACE, BACKSPACE,  # test multiple space and backspace
      NUMBERS, KEY, KEY, SHIFT, KEY, KEY  # test numbers and symbols
    ]:
      script.click(*key, wait_after=FAST_CLICK)

  # TODO: Better way of organizing the events

  # === Homescreen ===
  script.set_send(make_network_state_setup(pm, log.DeviceState.NetworkType.wifi))
  # Go through different prime state layouts
  add_prime_state_setup(PrimeType.LITE)
  add_prime_state_setup(PrimeType.NONE)
  add_prime_state_setup(PrimeType.UNPAIRED)

  # === Update Available (auto-transitions via HomeLayout refresh) ===
  script.setup(make_home_refresh_setup(setup_update_available))

  # === Offroad Alerts (auto-transitions via HomeLayout refresh, overrides update) ===
  script.setup(make_home_refresh_setup(setup_offroad_alerts))
  script.click(620, 950)  # close alerts

  # === Settings (click sidebar settings button) ===
  script.click(150, 90)

  # === Settings - Device ===
  # pair device
  script.click(2000, 450)  # pair device
  script.click(110, 110)  # close pairing dialog
  add_prime_state_setup(PrimeType.NONE)  # changed from unpaired to hide pair device button
  # calibration
  script.setup(setup_calibration_params, wait_after=0)
  script.click(1000, 620)  # expand calibration description
  script.click(2000, 620)  # reset calibration confirmation
  script.click(1500, 750)  # confirm reset
  script.click(1000, 620)  # collapse calibration description
  # training guide
  script.click(2000, 800)  # open training guide
  do_onboarding()
  # regulatory info
  script.click(2000, 970)  # regulatory button
  script.click(2000, 970)  # OK

  # === Settings - Network ===
  script.click(278, 450)
  # TODO: mock networks
  script.click(1880, 100)  # advanced network settings

  # Keyboard (tethering password)
  script.click(2000, 420, wait_after=FAST_CLICK)  # open tether password keyboard
  script.click(2000, 950, wait_after=FAST_CLICK)  # click confirm (disabled, should not close)
  script.click(2000, 115)  # cancel (close without typing)
  script.click(2000, 420, wait_after=FAST_CLICK)  # open keyboard again
  type_keyboard()  # test various keyboard layouts and interactions
  script.click(2050, 250, wait_after=FAST_CLICK)  # toggle show/hide password
  script.click(2000, 950)  # confirm (close keyboard)

  script.click(630, 80)  # back from advanced network

  # === Settings - Toggles ===
  script.click(278, 600)
  script.click(1200, 280)  # expand experimental mode description

  # === Settings - Software ===
  script.setup(lambda: setup_update_available(False), wait_after=0)  # start with no update available
  script.click(278, 720)  # software
  for _ in range(2):
    script.click(720, 120)  # toggle current release notes
  script.setup(setup_update_available)  # set update available
  for _ in range(2):
    script.click(720, 450)  # toggle new release notes
  script.click(2000, 630)  # open select branch dialog
  script.click(1000, 300)  # select 1st option
  script.click(1600, 900)  # confirm selection
  script.click(2000, 800)  # uninstall
  script.click(650, 750)  # cancel uninstall

  # === Settings - Firehose ===
  script.click(278, 845)

  # === Settings - Developer (set CarParamsPersistent first) ===
  script.setup(setup_developer_params, wait_after=0)
  script.click(278, 950)
  script.click(1930, 470)  # SSH keys (keyboard)
  script.click(1930, 115)  # click cancel on keyboard
  script.click(2000, 960)  # toggle alpha long
  script.click(1500, 875)  # confirm

  # === Close settings ===
  script.click(250, 160)

  # === Onroad ===
  script.set_send(lambda: send_onroad(pm))
  script.click(1000, 500)  # click onroad to toggle sidebar
  test_onroad_alerts(script, pm)

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
