import math
import pyray as rl
import cereal.messaging as messaging
from openpilot.common.filter_simple import FirstOrderFilter
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


class MiciMainLayout(Scroller):
  def __init__(self):
    super().__init__(snap_items=True, spacing=0, pad=0, scroll_indicator=False, edge_shadows=False)

    self._pm = messaging.PubMaster(['bookmarkButton'])

    self._prev_onroad = False
    self._prev_standstill = False
    self._onroad_time_delay: float | None = None
    self._setup = False

    # Transition effect state
    self._transition_alpha = FirstOrderFilter(0.0, 0.3, 1 / 60.0)
    self._transition_active = False
    self._transition_start = 0.0

    # Initialize widgets
    self._home_layout = MiciHomeLayout()
    self._alerts_layout = MiciOffroadAlerts()
    self._settings_layout = SettingsLayout()
    self._onroad_layout = AugmentedRoadView(bookmark_callback=self._on_bookmark_clicked)

    # Initialize widget rects
    for widget in (self._home_layout, self._settings_layout, self._alerts_layout, self._onroad_layout):
      # TODO: set parent rect and use it if never passed rect from render (like in Scroller)
      widget.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))

    self._scroller.add_widgets([
      self._alerts_layout,
      self._home_layout,
      self._onroad_layout,
    ])
    self._scroller.set_reset_scroll_at_show(False)

    # Disable scrolling when onroad is interacting with bookmark
    self._scroller.set_scrolling_enabled(lambda: not self._onroad_layout.is_swiping_left())

    # Set callbacks
    self._setup_callbacks()

    gui_app.add_nav_stack_tick(self._handle_transitions)
    gui_app.push_widget(self)

    # Start onboarding if terms or training not completed, make sure to push after self
    self._onboarding_window = OnboardingWindow(lambda: gui_app.pop_widgets_to(self))
    if not self._onboarding_window.completed:
      gui_app.push_widget(self._onboarding_window)

  def _setup_callbacks(self):
    self._home_layout.set_callbacks(on_settings=lambda: gui_app.push_widget(self._settings_layout))
    self._onroad_layout.set_click_callback(self._onroad_layout.cycle_lead_style)
    device.add_interactive_timeout_callback(self._on_interactive_timeout)

  def _scroll_to(self, layout: Widget):
    layout_x = int(layout.rect.x)
    self._scroller.scroll_to(layout_x, smooth=True)

  def _render(self, _):
    if not self._setup:
      if self._alerts_layout.active_alerts() > 0:
        self._scroller.scroll_to(self._alerts_layout.rect.x)
      else:
        self._scroller.scroll_to(self._rect.width)
      self._setup = True

    # Render
    super()._render(self._rect)

    # Transition overlay effect
    if self._transition_active:
      elapsed = rl.get_time() - self._transition_start
      if elapsed < ONROAD_DELAY:
        # Subtle pulsing overlay that builds anticipation
        pulse = 0.5 + 0.5 * math.sin(elapsed * 3.0)
        alpha = int(min(40, elapsed / ONROAD_DELAY * 40) * pulse)
        rl.draw_rectangle(int(self._rect.x), int(self._rect.y), int(self._rect.width), int(self._rect.height),
                          rl.Color(0, 20, 40, alpha))
      else:
        self._transition_active = False

  def _handle_transitions(self):
    # Don't pop if onboarding
    if gui_app.widget_in_stack(self._onboarding_window):
      return

    if ui_state.started != self._prev_onroad:
      self._prev_onroad = ui_state.started

      # onroad: after delay, pop nav stack and scroll to onroad
      # offroad: immediately scroll to home, but don't pop nav stack (can stay in settings)
      if ui_state.started:
        self._onroad_time_delay = rl.get_time()
        self._transition_active = True
        self._transition_start = rl.get_time()
      else:
        self._scroll_to(self._home_layout)

    # FIXME: these two pops can interrupt user interacting in the settings
    if self._onroad_time_delay is not None and rl.get_time() - self._onroad_time_delay >= ONROAD_DELAY:
      gui_app.pop_widgets_to(self, lambda: self._scroll_to(self._onroad_layout))
      self._onroad_time_delay = None

    # When car leaves standstill, pop nav stack and scroll to onroad
    CS = ui_state.sm["carState"]
    if not CS.standstill and self._prev_standstill:
      gui_app.pop_widgets_to(self, lambda: self._scroll_to(self._onroad_layout))
    self._prev_standstill = CS.standstill

  def _on_interactive_timeout(self):
    # Don't pop if onboarding
    if gui_app.widget_in_stack(self._onboarding_window):
      return

    if ui_state.started:
      # Don't pop if at standstill
      if not ui_state.sm["carState"].standstill:
        gui_app.pop_widgets_to(self, lambda: self._scroll_to(self._onroad_layout))
    else:
      # Screen turns off on timeout offroad, so pop immediately without animation
      gui_app.pop_widgets_to(self, instant=True)
      self._scroll_to(self._home_layout)

  def _on_bookmark_clicked(self):
    user_bookmark = messaging.new_message('bookmarkButton')
    user_bookmark.valid = True
    self._pm.send('bookmarkButton', user_bookmark)
