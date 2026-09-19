import pyray as rl
import openpilot.cereal.messaging as messaging
from openpilot.selfdrive.ui.mici.layouts.home import MiciHomeLayout
from openpilot.selfdrive.ui.mici.layouts.settings.settings import SettingsLayout
from openpilot.selfdrive.ui.mici.layouts.offroad_alerts import MiciOffroadAlerts
from openpilot.selfdrive.ui.ui_state import device, ui_state
from openpilot.common.version import terms_version, training_version
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.scroller import Scroller
from openpilot.system.ui.lib.application import gui_app


ONROAD_DELAY = 2.5  # seconds


class DeferredRoadView(Widget):
  """Keep layout/input geometry available while deferring camera/GL setup."""
  def __init__(self, factory):
    super().__init__()
    self._factory = factory
    self._view = None
    self._view_click = None

  def prepare(self):
    if self._view is None:
      self._view = self._factory()
      self._view.set_rect(self.rect)
      self._view.set_click_callback(self._view_click)
      self._view.show_event()

  def set_click_callback(self, callback):
    self._view_click = callback
    if self._view is not None:
      self._view.set_click_callback(callback)

  def is_swiping_left(self):
    return self._view is not None and self._view.is_swiping_left()

  def _render(self, rect):
    self.prepare()
    self._view.set_enabled(self.enabled)
    self._view.set_parent_rect(self._parent_rect)
    self._view.set_touch_valid_callback(self._touch_valid_callback)
    return self._view.render(rect)

  def show_event(self):
    if self._view is not None:
      self._view.show_event()

  def hide_event(self):
    if self._view is not None:
      self._view.hide_event()


class MiciMainLayout(Scroller):
  def __init__(self):
    super().__init__(snap_items=True, spacing=0, pad=0, scroll_indicator=False, edge_shadows=False)

    self._pm = messaging.PubMaster(['bookmarkButton', 'userBookmark'])

    self._prev_onroad = False
    self._prev_standstill = False
    self._onroad_time_delay: float | None = None
    self._setup = False

    # Initialize widgets
    self._home_layout = MiciHomeLayout()
    self._alerts_layout = MiciOffroadAlerts()
    self._settings_layout = SettingsLayout()
    def car_view():
      from openpilot.selfdrive.ui.mici.onroad.augmented_road_view import AugmentedRoadView
      return AugmentedRoadView(bookmark_callback=self._on_bookmark_clicked)

    def body_view():
      from openpilot.selfdrive.ui.body.layouts.onroad import BodyLayout
      return BodyLayout()

    self._car_onroad_layout = DeferredRoadView(car_view)
    self._body_onroad_layout = DeferredRoadView(body_view)

    # Initialize widget rects
    for widget in (self._home_layout, self._alerts_layout, self._settings_layout,
                   self._car_onroad_layout, self._body_onroad_layout):
      # TODO: set parent rect and use it if never passed rect from render (like in Scroller)
      widget.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))

    self._scroller.add_widgets([
      self._alerts_layout,
      self._home_layout,
      self._car_onroad_layout,
      self._body_onroad_layout,
    ])
    self._scroller.set_reset_scroll_at_show(False)

    # Disable scrolling when onroad is interacting with bookmark
    self._scroller.set_scrolling_enabled(lambda: not self._car_onroad_layout.is_swiping_left())

    # Set callbacks
    self._setup_callbacks()

    gui_app.add_nav_stack_tick(self._handle_transitions)
    gui_app.push_widget(self)

    # Start onboarding if terms or training not completed, make sure to push after self
    self._onboarding_window = None
    if (ui_state.params.get("HasAcceptedTerms") != terms_version or
        ui_state.params.get("CompletedTrainingVersion") != training_version):
      from openpilot.selfdrive.ui.mici.layouts.onboarding import OnboardingWindow
      self._onboarding_window = OnboardingWindow(lambda: gui_app.pop_widgets_to(self))
      gui_app.push_widget(self._onboarding_window)

    # initialize correct onroad layout
    self._on_body_changed()

  @property
  def _onroad_layout(self) -> Widget:
    # For scroll_to
    return self._body_onroad_layout if ui_state.is_body else self._car_onroad_layout

  def _setup_callbacks(self):
    self._alerts_layout.set_enabled(lambda: self.enabled)
    self._alerts_layout.set_pairing_callback(self._settings_layout.show_pairing)
    self._home_layout.set_callbacks(
      on_settings=lambda: gui_app.push_widget(self._settings_layout),
      on_alerts=lambda: self._scroll_to(self._alerts_layout),
      alert_count_callback=self._alerts_layout.active_alerts,
      alert_icon_callback=self._alerts_layout.highest_severity_icon,
    )
    for layout in (self._car_onroad_layout, self._body_onroad_layout):
      layout.set_click_callback(lambda: self._scroll_to(self._home_layout))

    device.add_interactive_timeout_callback(self._on_interactive_timeout)
    ui_state.add_on_body_changed_callbacks(self._on_body_changed)

  def _scroll_to(self, layout: Widget):
    layout_x = int(layout.rect.x)
    self._scroller.scroll_to(layout_x, smooth=True)

  def _update_state(self):
    super()._update_state()
    # TODO: Hack to run alert updates while not in view. Add a nav stack tick?
    self._alerts_layout._update_state()

  def _render(self, _):
    if not self._setup:
      if self._alerts_layout.active_alerts() > 0:
        self._scroller.scroll_to(self._alerts_layout.rect.x)
      else:
        self._scroller.scroll_to(self._rect.width)
      self._setup = True

    # Render
    super()._render(self._rect)

  def _handle_transitions(self):
    # Prepare the driving views just after the home screen's first frame.
    if gui_app.frame > 0:
      self._car_onroad_layout.prepare()
      self._body_onroad_layout.prepare()
    # Don't pop if onboarding
    if gui_app.widget_in_stack(self._onboarding_window):
      return

    if ui_state.started != self._prev_onroad:
      self._prev_onroad = ui_state.started

      # onroad: after delay, pop nav stack and scroll to onroad
      # offroad: immediately scroll to home, but don't pop nav stack (can stay in settings)
      if ui_state.started:
        self._onroad_time_delay = rl.get_time()
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
    for service in ('bookmarkButton', 'userBookmark'):
      msg = messaging.new_message(service, valid=True)
      self._pm.send(service, msg)

  def _on_body_changed(self):
    self._car_onroad_layout.set_visible(not ui_state.is_body)
    self._body_onroad_layout.set_visible(bool(ui_state.is_body))
