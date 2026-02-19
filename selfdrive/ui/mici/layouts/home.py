import time

from cereal import log
import pyray as rl
from collections.abc import Callable
from openpilot.system.ui.widgets.label import gui_label, MiciLabel, UnifiedLabel
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.lib.application import gui_app, FontWeight, DEFAULT_TEXT_COLOR, MousePos
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.text import wrap_text
from openpilot.system.version import training_version, RELEASE_BRANCHES

HEAD_BUTTON_FONT_SIZE = 40
HOME_PADDING = 8

NetworkType = log.DeviceState.NetworkType

NETWORK_TYPES = {
  NetworkType.none: "Offline",
  NetworkType.wifi: "WiFi",
  NetworkType.cell2G: "2G",
  NetworkType.cell3G: "3G",
  NetworkType.cell4G: "LTE",
  NetworkType.cell5G: "5G",
  NetworkType.ethernet: "Ethernet",
}


class DeviceStatus(Widget):
  def __init__(self):
    super().__init__()
    self.set_rect(rl.Rectangle(0, 0, 300, 175))
    self._update_state()
    self._version_text = self._get_version_text()

    self._do_welcome()

  def _do_welcome(self):
    ui_state.params.put("CompletedTrainingVersion", training_version)

  def refresh(self):
    self._update_state()
    self._version_text = self._get_version_text()

  def _get_version_text(self) -> str:
    brand = "openpilot"
    description = ui_state.params.get("UpdaterCurrentDescription")
    return f"{brand} {description}" if description else brand

  def _update_state(self):
    # TODO: refresh function that can be called periodically, not at 60 fps, so we can update version
    # update system status
    self._system_status = "SYSTEM READY ✓" if ui_state.panda_type != log.PandaState.PandaType.unknown else "BOOTING UP..."

    # update network status
    strength = ui_state.sm['deviceState'].networkStrength.raw
    strength_text = "● " * strength + "○ " * (4 - strength)  # ◌ also works
    network_type = NETWORK_TYPES[ui_state.sm['deviceState'].networkType.raw]
    self._network_status = f"{network_type} {strength_text}"

  def _render(self, _):
    # draw status
    status_rect = rl.Rectangle(self._rect.x, self._rect.y, self._rect.width, 40)
    gui_label(status_rect, self._system_status, font_size=HEAD_BUTTON_FONT_SIZE, color=DEFAULT_TEXT_COLOR,
              font_weight=FontWeight.BOLD, alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER)

    # draw network status
    network_rect = rl.Rectangle(self._rect.x, self._rect.y + 60, self._rect.width, 40)
    gui_label(network_rect, self._network_status, font_size=40, color=DEFAULT_TEXT_COLOR,
              font_weight=FontWeight.MEDIUM, alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER)

    # draw version
    version_font_size = 30
    version_rect = rl.Rectangle(self._rect.x, self._rect.y + 140, self._rect.width + 20, 40)
    wrapped_text = '\n'.join(wrap_text(self._version_text, version_font_size, version_rect.width))
    gui_label(version_rect, wrapped_text, font_size=version_font_size, color=DEFAULT_TEXT_COLOR,
              font_weight=FontWeight.MEDIUM, alignment=rl.GuiTextAlignment.TEXT_ALIGN_LEFT)


class MiciHomeLayout(Widget):
  def __init__(self):
    super().__init__()
    self._on_settings_click: Callable | None = None

    self._last_refresh = 0
    self._mouse_down_t: None | float = None
    self._did_long_press = False
    self._is_pressed_prev = False

    self._version_text = None
    self._experimental_mode = False

    self._settings_txt = gui_app.texture("icons_mici/settings.png", 48, 48)
    self._experimental_txt = gui_app.texture("icons_mici/experimental_mode.png", 48, 48)
    self._mic_txt = gui_app.texture("icons_mici/microphone.png", 32, 46)

    self._net_type = NETWORK_TYPES.get(NetworkType.none)
    self._net_strength = 0

    self._wifi_slash_txt = gui_app.texture("icons_mici/settings/network/wifi_strength_slash.png", 50, 44)
    self._wifi_none_txt = gui_app.texture("icons_mici/settings/network/wifi_strength_none.png", 50, 37)
    self._wifi_low_txt = gui_app.texture("icons_mici/settings/network/wifi_strength_low.png", 50, 37)
    self._wifi_medium_txt = gui_app.texture("icons_mici/settings/network/wifi_strength_medium.png", 50, 37)
    self._wifi_full_txt = gui_app.texture("icons_mici/settings/network/wifi_strength_full.png", 50, 37)

    self._cell_none_txt = gui_app.texture("icons_mici/settings/network/cell_strength_none.png", 54, 36)
    self._cell_low_txt = gui_app.texture("icons_mici/settings/network/cell_strength_low.png", 54, 36)
    self._cell_medium_txt = gui_app.texture("icons_mici/settings/network/cell_strength_medium.png", 54, 36)
    self._cell_high_txt = gui_app.texture("icons_mici/settings/network/cell_strength_high.png", 54, 36)
    self._cell_full_txt = gui_app.texture("icons_mici/settings/network/cell_strength_full.png", 54, 36)

    self._openpilot_label = MiciLabel("openpilot", font_size=96, color=rl.Color(255, 255, 255, int(255 * 0.9)), font_weight=FontWeight.DISPLAY)
    self._version_label = MiciLabel("", font_size=36, font_weight=FontWeight.ROMAN)
    self._large_version_label = MiciLabel("", font_size=64, color=rl.GRAY, font_weight=FontWeight.ROMAN)
    self._date_label = MiciLabel("", font_size=36, color=rl.GRAY, font_weight=FontWeight.ROMAN)
    self._branch_label = UnifiedLabel("", font_size=36, text_color=rl.GRAY, font_weight=FontWeight.ROMAN, scroll=True)
    self._version_commit_label = MiciLabel("", font_size=36, color=rl.GRAY, font_weight=FontWeight.ROMAN)

  def show_event(self):
    self._version_text = self._get_version_text()
    self._update_network_status(ui_state.sm['deviceState'])
    self._update_params()

  def _update_params(self):
    self._experimental_mode = ui_state.params.get_bool("ExperimentalMode")

  def _update_state(self):
    if self.is_pressed and not self._is_pressed_prev:
      self._mouse_down_t = time.monotonic()
    elif not self.is_pressed and self._is_pressed_prev:
      self._mouse_down_t = None
      self._did_long_press = False
    self._is_pressed_prev = self.is_pressed

    if self._mouse_down_t is not None:
      if time.monotonic() - self._mouse_down_t > 0.5:
        # long gating for experimental mode - only allow toggle if longitudinal control is available
        if ui_state.has_longitudinal_control:
          self._experimental_mode = not self._experimental_mode
          ui_state.params.put("ExperimentalMode", self._experimental_mode)
        self._mouse_down_t = None
        self._did_long_press = True

    if rl.get_time() - self._last_refresh > 5.0:
      device_state = ui_state.sm['deviceState']
      self._update_network_status(device_state)

      # Update version text
      self._version_text = self._get_version_text()
      self._last_refresh = rl.get_time()
      self._update_params()

  def _update_network_status(self, device_state):
    self._net_type = device_state.networkType
    strength = device_state.networkStrength
    self._net_strength = max(0, min(5, strength.raw + 1)) if strength.raw > 0 else 0

  def set_callbacks(self, on_settings: Callable | None = None):
    self._on_settings_click = on_settings

  def _handle_mouse_release(self, mouse_pos: MousePos):
    if not self._did_long_press:
      if self._on_settings_click:
        self._on_settings_click()
    self._did_long_press = False

  def _get_version_text(self) -> tuple[str, str, str, str] | None:
    description = ui_state.params.get("UpdaterCurrentDescription")

    if description is not None and len(description) > 0:
      # Expect "version / branch / commit / date"; be tolerant of other formats
      try:
        version, branch, commit, date = description.split(" / ")
        return version, branch, commit, date
      except Exception:
        return None

    return None

  def _render(self, _):
    # TODO: why is there extra space here to get it to be flush?
    text_pos = rl.Vector2(self.rect.x - 2 + HOME_PADDING, self.rect.y - 16)
    self._openpilot_label.set_position(text_pos.x, text_pos.y)
    self._openpilot_label.render()

    if self._version_text is not None:
      # release branch
      release_branch = self._version_text[1] in RELEASE_BRANCHES
      version_pos = rl.Rectangle(text_pos.x, text_pos.y + self._openpilot_label.font_size + 16, 100, 44)
      self._version_label.set_text(self._version_text[0])
      self._version_label.set_position(version_pos.x, version_pos.y)
      self._version_label.render()

      self._date_label.set_text(" " + self._version_text[3])
      self._date_label.set_position(version_pos.x + self._version_label.rect.width + 10, version_pos.y)
      self._date_label.render()

      self._branch_label.set_max_width(gui_app.width - self._version_label.rect.width - self._date_label.rect.width - 32)
      self._branch_label.set_text(" " + ("release" if release_branch else self._version_text[1]))
      self._branch_label.set_position(version_pos.x + self._version_label.rect.width + self._date_label.rect.width + 20, version_pos.y)
      self._branch_label.render()

      if not release_branch:
        # 2nd line
        self._version_commit_label.set_text(self._version_text[2])
        self._version_commit_label.set_position(version_pos.x, version_pos.y + self._date_label.font_size + 7)
        self._version_commit_label.render()

    self._render_bottom_status_bar()

  def _render_bottom_status_bar(self):
    # ***** Center-aligned bottom section icons *****

    # TODO: refactor repeated icon drawing into a small loop
    ITEM_SPACING = 18
    Y_CENTER = 24

    last_x = self.rect.x + HOME_PADDING

    # Draw settings icon in bottom left corner
    rl.draw_texture(self._settings_txt, int(last_x), int(self._rect.y + self.rect.height - self._settings_txt.height / 2 - Y_CENTER),
                    rl.Color(255, 255, 255, int(255 * 0.9)))
    last_x = last_x + self._settings_txt.width + ITEM_SPACING

    # draw network
    if self._net_type == NetworkType.wifi:
      # There is no 1
      draw_net_txt = {0: self._wifi_none_txt,
                      2: self._wifi_low_txt,
                      3: self._wifi_medium_txt,
                      4: self._wifi_full_txt,
                      5: self._wifi_full_txt}.get(self._net_strength, self._wifi_low_txt)
      rl.draw_texture(draw_net_txt, int(last_x),
                      int(self._rect.y + self.rect.height - draw_net_txt.height / 2 - Y_CENTER), rl.Color(255, 255, 255, int(255 * 0.9)))
      last_x += draw_net_txt.width + ITEM_SPACING

    elif self._net_type in (NetworkType.cell2G, NetworkType.cell3G, NetworkType.cell4G, NetworkType.cell5G):
      draw_net_txt = {0: self._cell_none_txt,
                      2: self._cell_low_txt,
                      3: self._cell_medium_txt,
                      4: self._cell_high_txt,
                      5: self._cell_full_txt}.get(self._net_strength, self._cell_none_txt)
      rl.draw_texture(draw_net_txt, int(last_x),
                      int(self._rect.y + self.rect.height - draw_net_txt.height / 2 - Y_CENTER), rl.Color(255, 255, 255, int(255 * 0.9)))
      last_x += draw_net_txt.width + ITEM_SPACING

    else:
      # No network
      # Offset by difference in height between slashless and slash icons to make center align match
      rl.draw_texture(self._wifi_slash_txt, int(last_x), int(self._rect.y + self.rect.height - self._wifi_slash_txt.height / 2 -
                                                             (self._wifi_slash_txt.height - self._wifi_none_txt.height) / 2 - Y_CENTER),
                      rl.Color(255, 255, 255, int(255 * 0.9)))
      last_x += self._wifi_slash_txt.width + ITEM_SPACING

    # draw experimental icon
    if self._experimental_mode:
      rl.draw_texture(self._experimental_txt, int(last_x),
                      int(self._rect.y + self.rect.height - self._experimental_txt.height / 2 - Y_CENTER), rl.Color(255, 255, 255, 255))
      last_x += self._experimental_txt.width + ITEM_SPACING

    # draw microphone icon when recording audio is enabled
    if ui_state.recording_audio:
      rl.draw_texture(self._mic_txt, int(last_x),
                      int(self._rect.y + self.rect.height - self._mic_txt.height / 2 - Y_CENTER), rl.Color(255, 255, 255, 255))
      last_x += self._mic_txt.width + ITEM_SPACING
