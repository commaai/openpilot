import datetime
import time

from cereal import log
import pyray as rl
from collections.abc import Callable
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.layouts import HBoxLayout
from openpilot.system.ui.widgets.icon_widget import IconWidget
from openpilot.system.ui.widgets.label import MiciLabel, UnifiedLabel
from openpilot.system.ui.lib.application import gui_app, FontWeight, MousePos
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.mici.effects import StaggeredReveal
from openpilot.system.version import RELEASE_BRANCHES

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


class NetworkIcon(Widget):
  def __init__(self):
    super().__init__()
    self.set_rect(rl.Rectangle(0, 0, 54, 44))  # max size of all icons
    self._net_type = NetworkType.none
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

  def _update_state(self):
    device_state = ui_state.sm['deviceState']
    self._net_type = device_state.networkType
    strength = device_state.networkStrength
    self._net_strength = max(0, min(5, strength.raw + 1)) if strength.raw > 0 else 0

  def _render(self, _):
    if self._net_type == NetworkType.wifi:
      # There is no 1
      draw_net_txt = {0: self._wifi_none_txt,
                      2: self._wifi_low_txt,
                      3: self._wifi_medium_txt,
                      4: self._wifi_full_txt,
                      5: self._wifi_full_txt}.get(self._net_strength, self._wifi_low_txt)
    elif self._net_type in (NetworkType.cell2G, NetworkType.cell3G, NetworkType.cell4G, NetworkType.cell5G):
      draw_net_txt = {0: self._cell_none_txt,
                      2: self._cell_low_txt,
                      3: self._cell_medium_txt,
                      4: self._cell_high_txt,
                      5: self._cell_full_txt}.get(self._net_strength, self._cell_none_txt)
    else:
      draw_net_txt = self._wifi_slash_txt

    draw_x = self._rect.x + (self._rect.width - draw_net_txt.width) / 2
    draw_y = self._rect.y + (self._rect.height - draw_net_txt.height) / 2

    if draw_net_txt == self._wifi_slash_txt:
      # Offset by difference in height between slashless and slash icons to make center align match
      draw_y -= (self._wifi_slash_txt.height - self._wifi_none_txt.height) / 2

    rl.draw_texture(draw_net_txt, int(draw_x), int(draw_y), rl.Color(255, 255, 255, int(255 * 0.9)))


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

    self._experimental_icon = IconWidget("icons_mici/experimental_mode.png", (48, 48))
    self._mic_icon = IconWidget("icons_mici/microphone.png", (32, 46))

    self._status_bar_layout = HBoxLayout([
      IconWidget("icons_mici/settings.png", (48, 48), opacity=0.9),
      NetworkIcon(),
      self._experimental_icon,
      self._mic_icon,
    ], spacing=18)

    self._openpilot_label = MiciLabel("openpilot", font_size=96, color=rl.Color(255, 255, 255, int(255 * 0.9)), font_weight=FontWeight.DISPLAY)
    self._version_label = MiciLabel("", font_size=36, font_weight=FontWeight.ROMAN)
    self._large_version_label = MiciLabel("", font_size=64, color=rl.GRAY, font_weight=FontWeight.ROMAN)
    self._date_label = MiciLabel("", font_size=36, color=rl.GRAY, font_weight=FontWeight.ROMAN)
    self._branch_label = UnifiedLabel("", font_size=36, text_color=rl.GRAY, font_weight=FontWeight.ROMAN, scroll=True)
    self._version_commit_label = MiciLabel("", font_size=36, color=rl.GRAY, font_weight=FontWeight.ROMAN)

    # Wake animation: staggered reveal for 4 elements (openpilot, version, date/branch, status bar)
    self._wake_anim = StaggeredReveal(4)

    # Magnetic snap: smooth filter for status bar x-offset toward touch
    self._magnetic_x_filter = FirstOrderFilter(0.0, 0.12, 1 / gui_app.target_fps)

  def show_event(self):
    self._version_text = self._get_version_text()
    self._update_params()
    self._wake_anim.trigger()

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

    # Magnetic snap: track touch and apply subtle pull to status bar
    if self.is_pressed:
      mouse_x = gui_app.last_mouse_event.pos.x if gui_app.last_mouse_event else self.rect.x + self.rect.width / 2
      center_x = self.rect.x + self.rect.width / 2
      pull = (mouse_x - center_x) * 0.03  # subtle pull toward touch
      pull = max(-8, min(8, pull))
      self._magnetic_x_filter.update(pull)
    else:
      self._magnetic_x_filter.update(0.0)

    if rl.get_time() - self._last_refresh > 5.0:
      # Update version text
      self._version_text = self._get_version_text()
      self._last_refresh = rl.get_time()
      self._update_params()

  def set_callbacks(self, on_settings: Callable | None = None):
    self._on_settings_click = on_settings

  def _handle_mouse_release(self, mouse_pos: MousePos):
    if not self._did_long_press:
      if self._on_settings_click:
        self._on_settings_click()
    self._did_long_press = False

  def _get_version_text(self) -> tuple[str, str, str, str] | None:
    version = ui_state.params.get("Version")
    branch = ui_state.params.get("GitBranch")
    commit = ui_state.params.get("GitCommit")

    if not all((version, branch, commit)):
      return None

    commit_date_raw = ui_state.params.get("GitCommitDate")
    try:
      # GitCommitDate format from get_commit_date(): '%ct %ci' e.g. "'1708012345 2024-02-15 ...'"
      unix_ts = int(commit_date_raw.strip("'").split()[0])
      date_str = datetime.datetime.fromtimestamp(unix_ts).strftime("%b %d")
    except (ValueError, IndexError, TypeError, AttributeError):
      date_str = ""

    return version, branch, commit[:7], date_str

  def _render(self, _):
    # ── Parallax: compute scroll offset relative to home's resting position ──
    scroll_offset = self.rect.x - gui_app.width  # home is at index 1

    # ── Wake animation progress per element ──
    prog_op = self._wake_anim.get_progress(0)      # openpilot text
    prog_ver = self._wake_anim.get_progress(1)      # version label
    prog_date = self._wake_anim.get_progress(2)     # date/branch row
    prog_footer = self._wake_anim.get_progress(3)   # status bar

    slide_op = self._wake_anim.get_slide_y(0)
    slide_ver = self._wake_anim.get_slide_y(1)
    slide_date = self._wake_anim.get_slide_y(2)
    slide_footer = self._wake_anim.get_slide_y(3)

    # ── Openpilot label ──
    # TODO: why is there extra space here to get it to be flush?
    parallax_op = scroll_offset * 0.15
    text_pos = rl.Vector2(self.rect.x - 2 + HOME_PADDING + parallax_op, self.rect.y - 16 + slide_op)
    self._openpilot_label._color = rl.Color(255, 255, 255, int(255 * 0.9 * prog_op))
    self._openpilot_label.set_position(text_pos.x, text_pos.y)
    self._openpilot_label.render()

    # ── Version / date / branch row ──
    if self._version_text is not None:
      release_branch = self._version_text[1] in RELEASE_BRANCHES

      parallax_ver = scroll_offset * 0.08
      ver_alpha = int(255 * 0.9 * prog_ver)
      date_alpha = int(255 * 0.9 * prog_date)

      version_pos = rl.Rectangle(text_pos.x + parallax_ver - parallax_op, text_pos.y + self._openpilot_label.font_size + 16 + slide_ver - slide_op, 100, 44)

      self._version_label.set_text(self._version_text[0])
      self._version_label._color = rl.Color(255, 255, 255, ver_alpha)
      self._version_label.set_position(version_pos.x, version_pos.y)
      self._version_label.render()

      self._date_label.set_text(" " + self._version_text[3])
      self._date_label._color = rl.Color(128, 128, 128, date_alpha)
      self._date_label.set_position(version_pos.x + self._version_label.rect.width + 10, version_pos.y + slide_date - slide_ver)
      self._date_label.render()

      self._branch_label.set_max_width(gui_app.width - self._version_label.rect.width - self._date_label.rect.width - 32)
      self._branch_label.set_text(" " + ("release" if release_branch else self._version_text[1]))
      self._branch_label._text_color = rl.Color(128, 128, 128, date_alpha)
      self._branch_label.set_position(version_pos.x + self._version_label.rect.width + self._date_label.rect.width + 20, version_pos.y + slide_date - slide_ver)
      self._branch_label.render()

      if not release_branch:
        # 2nd line
        self._version_commit_label.set_text(self._version_text[2])
        self._version_commit_label._color = rl.Color(128, 128, 128, date_alpha)
        self._version_commit_label.set_position(version_pos.x, version_pos.y + self._date_label.font_size + 7 + slide_date - slide_ver)
        self._version_commit_label.render()

    # ── Status bar (footer) ──
    self._experimental_icon.set_visible(self._experimental_mode)
    self._mic_icon.set_visible(ui_state.recording_audio)

    parallax_footer = scroll_offset * 0.05
    magnetic_x = self._magnetic_x_filter.x
    footer_alpha_factor = prog_footer  # used conceptually; icons rendered via layout

    footer_rect = rl.Rectangle(
      self.rect.x + HOME_PADDING + parallax_footer + magnetic_x,
      self.rect.y + self.rect.height - 48 + slide_footer,
      self.rect.width - HOME_PADDING,
      48,
    )
    self._status_bar_layout.render(footer_rect)
