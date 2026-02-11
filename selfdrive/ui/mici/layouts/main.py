import pyray as rl
from enum import IntEnum
import cereal.messaging as messaging
from openpilot.selfdrive.ui.mici.layouts.home import MiciHomeLayout
from openpilot.selfdrive.ui.mici.layouts.settings.settings import SettingsLayout
from openpilot.selfdrive.ui.mici.layouts.offroad_alerts import MiciOffroadAlerts
from openpilot.selfdrive.ui.mici.onroad.augmented_road_view import AugmentedRoadView
from openpilot.selfdrive.ui.ui_state import device, ui_state
from openpilot.selfdrive.ui.mici.layouts.onboarding import OnboardingWindow
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.scroller import Scroller
from openpilot.system.ui.lib.application import gui_app


ONROAD_DELAY = 2.5  # seconds


class MainState(IntEnum):
  MAIN = 0
  SETTINGS = 1


class MiciMainLayout(Widget):
  def __init__(self):
    super().__init__()

    self._pm = messaging.PubMaster(['bookmarkButton'])

    self._current_mode: MainState | None = None
    self._prev_onroad = False
    self._prev_standstill = False
    self._onroad_time_delay: float | None = None
    self._setup = False

    # Initialize widgets
    self._home_layout = MiciHomeLayout()
    self._alerts_layout = MiciOffroadAlerts()
    self._settings_layout = SettingsLayout()
    self._onroad_layout = AugmentedRoadView(bookmark_callback=self._on_bookmark_clicked)

    # Initialize widget rects
    for widget in (self._home_layout, self._settings_layout, self._alerts_layout, self._onroad_layout):
      # TODO: set parent rect and use it if never passed rect from render (like in Scroller)
      widget.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))

    self._scroller = Scroller([
      self._alerts_layout,
      self._home_layout,
      self._onroad_layout,
    ], spacing=0, pad_start=0, pad_end=0, scroll_indicator=False)
    self._scroller.set_reset_scroll_at_show(False)

    # Disable scrolling when onroad is interacting with bookmark
    self._scroller.set_scrolling_enabled(lambda: not self._onroad_layout.is_swiping_left())

    self._layouts = {
      MainState.MAIN: self._scroller,
      MainState.SETTINGS: self._settings_layout,
    }

    # Set callbacks
    self._setup_callbacks()

    # Start onboarding if terms or training not completed
    self._onboarding_window = OnboardingWindow()
    if not self._onboarding_window.completed:
      gui_app.set_modal_overlay(self._onboarding_window)

  def _setup_callbacks(self):
    self._home_layout.set_callbacks(on_settings=self._on_settings_clicked)
    self._settings_layout.set_callbacks(on_close=self._on_settings_closed)
    self._onroad_layout.set_click_callback(lambda: self._scroll_to(self._home_layout))
    device.add_interactive_timeout_callback(self._set_mode_for_started)

  def _scroll_to(self, layout: Widget):
    layout_x = int(layout.rect.x)
    self._scroller.scroll_to(layout_x, smooth=True)

  def _render(self, _):
    # Initial show event
    if self._current_mode is None:
      self._set_mode(MainState.MAIN)

    if not self._setup:
      if self._alerts_layout.active_alerts() > 0:
        self._scroller.scroll_to(self._alerts_layout.rect.x)
      else:
        self._scroller.scroll_to(self._rect.width)
      self._setup = True

    # Render
    if self._current_mode == MainState.MAIN:
      self._scroller.render(self._rect)

    elif self._current_mode == MainState.SETTINGS:
      self._settings_layout.render(self._rect)

    self._handle_transitions()

  def _set_mode(self, mode: MainState):
    if mode != self._current_mode:
      if self._current_mode is not None:
        self._layouts[self._current_mode].hide_event()
      self._layouts[mode].show_event()
      self._current_mode = mode

  def _handle_transitions(self):
    if ui_state.started != self._prev_onroad:
      self._prev_onroad = ui_state.started

      if ui_state.started:
        self._onroad_time_delay = rl.get_time()
      else:
        self._set_mode_for_started(True)

    # delay so we show home for a bit after starting
    if self._onroad_time_delay is not None and rl.get_time() - self._onroad_time_delay >= ONROAD_DELAY:
      self._set_mode_for_started(True)
      self._onroad_time_delay = None

    CS = ui_state.sm["carState"]
    if not CS.standstill and self._prev_standstill:
      self._set_mode(MainState.MAIN)
      self._scroll_to(self._onroad_layout)
    self._prev_standstill = CS.standstill

  def _set_mode_for_started(self, onroad_transition: bool = False):
    if ui_state.started:
      CS = ui_state.sm["carState"]
      # Only go onroad if car starts or is not at a standstill
      if not CS.standstill or onroad_transition:
        self._set_mode(MainState.MAIN)
        self._scroll_to(self._onroad_layout)
    else:
      # Stay in settings if car turns off while in settings
      if not onroad_transition or self._current_mode != MainState.SETTINGS:
        self._set_mode(MainState.MAIN)
        self._scroll_to(self._home_layout)

  def _on_settings_clicked(self):
    self._set_mode(MainState.SETTINGS)

  def _on_settings_closed(self):
    self._set_mode(MainState.MAIN)

  def _on_bookmark_clicked(self):
    user_bookmark = messaging.new_message('bookmarkButton')
    user_bookmark.valid = True
    self._pm.send('bookmarkButton', user_bookmark)
