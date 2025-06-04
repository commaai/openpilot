import pyray as rl
from enum import IntEnum
from openpilot.selfdrive.ui.layouts.sidebar import Sidebar, SIDEBAR_WIDTH
from openpilot.selfdrive.ui.layouts.home import HomeLayout
from openpilot.selfdrive.ui.layouts.settings.settings import SettingsLayout
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.onroad.augmented_road_view import AugmentedRoadView


class MainState(IntEnum):
  HOME = 0
  SETTINGS = 1
  ONROAD = 2


class MainLayout:
  def __init__(self):
    self._sidebar = Sidebar()
    self._sidebar_visible = True
    self._current_mode = MainState.HOME
    self._prev_onroad = False
    self._window_rect = None
    self._current_callback: callable | None = None

    # Initialize layouts
    self._layouts = {MainState.HOME: HomeLayout(), MainState.SETTINGS: SettingsLayout(), MainState.ONROAD: AugmentedRoadView()}

    self._sidebar_rect = rl.Rectangle(0, 0, 0, 0)
    self._content_rect = rl.Rectangle(0, 0, 0, 0)

    # Set callbacks
    self._setup_callbacks()

  def render(self, rect):
    self._current_callback = None

    self._update_layout_rects(rect)
    self._handle_onroad_transition()
    self._render_main_content()
    self._handle_input()

    if self._current_callback:
      self._current_callback()

  def _setup_callbacks(self):
    self._sidebar.set_callbacks(
      on_settings=lambda: setattr(self, '_current_callback', self._on_settings_clicked),
      on_flag=lambda: setattr(self, '_current_callback', self._on_flag_clicked),
    )
    self._layouts[MainState.SETTINGS].set_callbacks(
      on_close=lambda: setattr(self, '_current_callback', self._set_mode_for_state)
    )

  def _update_layout_rects(self, rect):
    self._window_rect = rect
    self._sidebar_rect = rl.Rectangle(rect.x, rect.y, SIDEBAR_WIDTH, rect.height)

    x_offset = SIDEBAR_WIDTH if self._sidebar_visible else 0
    self._content_rect = rl.Rectangle(rect.y + x_offset, rect.y, rect.width - x_offset, rect.height)

  def _handle_onroad_transition(self):
    if ui_state.started != self._prev_onroad:
      self._prev_onroad = ui_state.started

      self._set_mode_for_state()

  def _set_mode_for_state(self):
    if ui_state.started:
      self._current_mode = MainState.ONROAD
      self._sidebar_visible = False
    else:
      self._current_mode = MainState.HOME
      self._sidebar_visible = True

  def _on_settings_clicked(self):
    self._current_mode = MainState.SETTINGS
    self._sidebar_visible = False

  def _on_flag_clicked(self):
    pass

  def _render_main_content(self):
    # Render sidebar
    if self._sidebar_visible:
      self._sidebar.render(self._sidebar_rect)

    content_rect = self._content_rect if self._sidebar_visible else self._window_rect
    self._layouts[self._current_mode].render(content_rect)

  def _handle_input(self):
    if self._current_mode != MainState.ONROAD or not rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT):
      return

    mouse_pos = rl.get_mouse_position()
    if rl.check_collision_point_rec(mouse_pos, self._content_rect):
      self._sidebar_visible = not self._sidebar_visible
