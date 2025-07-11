import pyray as rl
from enum import IntEnum
import cereal.messaging as messaging
from openpilot.selfdrive.ui.layouts.sidebar import Sidebar, SIDEBAR_WIDTH
from openpilot.selfdrive.ui.layouts.home import HomeLayout
from openpilot.selfdrive.ui.layouts.settings.settings import SettingsLayout, PanelType
from openpilot.selfdrive.ui.onroad.augmented_road_view import AugmentedRoadView
from openpilot.selfdrive.ui.ui_state import device, ui_state
from openpilot.system.ui.widgets import Widget


class MainState(IntEnum):
  HOME = 0
  SETTINGS = 1
  ONROAD = 2


class MainLayout(Widget):
  def __init__(self):
    super().__init__()

    self._pm = messaging.PubMaster(['userFlag'])

    self._sidebar = Sidebar()
    self._current_mode = MainState.HOME
    self._prev_onroad = False

    # Initialize layouts
    self._layouts = {MainState.HOME: HomeLayout(), MainState.SETTINGS: SettingsLayout(), MainState.ONROAD: AugmentedRoadView()}

    self._sidebar_rect = rl.Rectangle(0, 0, 0, 0)
    self._content_rect = rl.Rectangle(0, 0, 0, 0)

    # Set callbacks
    self._setup_callbacks()

  def _render(self, _):
    self._handle_onroad_transition()
    self._render_main_content()

  def _setup_callbacks(self):
    self._sidebar.set_callbacks(on_settings=self._on_settings_clicked,
                                on_flag=self._on_flag_clicked)
    self._layouts[MainState.HOME]._setup_widget.set_open_settings_callback(lambda: self.open_settings(PanelType.FIREHOSE))
    self._layouts[MainState.SETTINGS].set_callbacks(on_close=self._set_mode_for_state)
    self._layouts[MainState.ONROAD].set_callbacks(on_click=self._on_onroad_clicked)
    device.add_interactive_timeout_callback(self._set_mode_for_state)

  def _update_layout_rects(self):
    self._sidebar_rect = rl.Rectangle(self._rect.x, self._rect.y, SIDEBAR_WIDTH, self._rect.height)

    x_offset = SIDEBAR_WIDTH if self._sidebar.is_visible else 0
    self._content_rect = rl.Rectangle(self._rect.y + x_offset, self._rect.y, self._rect.width - x_offset, self._rect.height)

  def _handle_onroad_transition(self):
    if ui_state.started != self._prev_onroad:
      self._prev_onroad = ui_state.started

      self._set_mode_for_state()

  def _set_mode_for_state(self):
    if ui_state.started:
      # Don't hide sidebar from interactive timeout
      if self._current_mode != MainState.ONROAD:
        self._sidebar.set_visible(False)
      self._current_mode = MainState.ONROAD
    else:
      self._current_mode = MainState.HOME
      self._sidebar.set_visible(True)

  def open_settings(self, panel_type: PanelType):
    self._layouts[MainState.SETTINGS].set_current_panel(panel_type)
    self._current_mode = MainState.SETTINGS
    self._sidebar.set_visible(False)

  def _on_settings_clicked(self):
    self.open_settings(PanelType.DEVICE)

  def _on_flag_clicked(self):
    user_flag = messaging.new_message('userFlag')
    user_flag.valid = True
    self._pm.send('userFlag', user_flag)

  def _on_onroad_clicked(self):
    self._sidebar.set_visible(not self._sidebar.is_visible)

  def _render_main_content(self):
    # Render sidebar
    if self._sidebar.is_visible:
      self._sidebar.render(self._sidebar_rect)

    content_rect = self._content_rect if self._sidebar.is_visible else self._rect
    self._layouts[self._current_mode].render(content_rect)
