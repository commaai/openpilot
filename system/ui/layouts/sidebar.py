import pyray as rl
import time
from cereal import log
from dataclasses import dataclass
from enum import IntEnum
from cereal import messaging
from openpilot.system.ui.lib.application import gui_app, FontWeight

SIDEBAR_WIDTH = 300
METRIC_HEIGHT = 126
METRIC_WIDTH = 240
METRIC_MARGIN = 30
BUTTON_SIZE = 192

SETTINGS_BTN_RECT = rl.Rectangle(50, 35, 200, 117)
HOME_BTN_RECT = rl.Rectangle(60, 860, 180, 180)

ThermalStatus = log.DeviceState.ThermalStatus
NetworkType = log.DeviceState.NetworkType

# Colors
SIDEBAR_BG = rl.Color(57, 57, 57, 255)
WHITE_DIM = rl.Color(255, 255, 255, 85)
GRAY = rl.Color(84, 84, 84, 255)
METRIC_BORDER = rl.Color(255, 255, 255, 85)


NETWORK_TYPES = {
  NetworkType.none: "None",
  NetworkType.wifi: "WiFi",
  NetworkType.cell2G: "2G",
  NetworkType.cell3G: "3G",
  NetworkType.cell4G: "LTE",
  NetworkType.cell5G: "5G",
  NetworkType.ethernet: "Ethernet",
}

class ItemStatus(IntEnum):
  GOOD = 0
  WARNING = 1
  DANGER = 2


STATUS_COLORS = {
  ItemStatus.GOOD: rl.Color(255, 255, 255, 255),
  ItemStatus.WARNING: rl.Color(218, 202, 37, 255),
  ItemStatus.DANGER: rl.Color(201, 34, 49, 255),
}


@dataclass
class MetricData:
  label: str
  value: str
  status: ItemStatus


class Sidebar:
  def __init__(self):
    self.onroad = False
    self.flag_pressed = False
    self.settings_pressed = False

    self.net_type = "WiFi"
    self.net_strength = 0

    self.temp_status = MetricData("TEMP", "GOOD", ItemStatus.GOOD)
    self.panda_status = MetricData("VEHICLE", "ONLINE", ItemStatus.GOOD)
    self.connect_status = MetricData("CONNECT", "OFFLINE", ItemStatus.WARNING)

    self.home_img = gui_app.texture("images/button_home.png", HOME_BTN_RECT.width, HOME_BTN_RECT.height)
    self.flag_img = gui_app.texture("images/button_flag.png", HOME_BTN_RECT.width, HOME_BTN_RECT.height)
    self.settings_img = gui_app.texture("images/button_settings.png", SETTINGS_BTN_RECT.width, SETTINGS_BTN_RECT.height)
    self.font_regular = gui_app.font(FontWeight.NORMAL)
    self.font_bold = gui_app.font(FontWeight.SEMI_BOLD)

    self._last_update = 0.0

  def draw(self, sm, rect: rl.Rectangle):
    self.sm = sm
    self.update_state(sm)

    # Background
    rl.draw_rectangle_rec(rect, SIDEBAR_BG)

    self._draw_buttons(rect)
    self._draw_network_indicator(rect)
    self._draw_metrics(rect)

  def update_state(self, sm: messaging.SubMaster):
    current_time = time.time()

    # Throttle updates to avoid excessive processing
    if current_time - self._last_update < 0.5:
      return
    self._last_update = current_time

    if not sm.valid['deviceState']:
      return

    device_state = sm['deviceState']

    self._update_network_status(device_state)
    self._update_temperature_status(device_state)
    self._update_connection_status(device_state)
    self._update_panda_status(sm)

  def _update_network_status(self, device_state):
    network_type = device_state.networkType
    self.net_type = NETWORK_TYPES.get(network_type, "Unknown")

    strength = device_state.networkStrength
    self.net_strength = max(0, min(5, strength.raw + 1)) if strength > 0 else 0

  def _update_temperature_status(self, device_state):
    thermal_status = device_state.thermalStatus

    if thermal_status == ThermalStatus.green:
      self.temp_status = MetricData("TEMP", "GOOD", ItemStatus.GOOD)
    elif thermal_status == ThermalStatus.yellow:
      self.temp_status = MetricData("TEMP", "OK", ItemStatus.WARNING)
    else:
      self.temp_status = MetricData("TEMP", "HIGH", ItemStatus.DANGER)

  def _update_connection_status(self, device_state):
    last_ping = device_state.lastAthenaPingTime
    current_time_ns = time.time_ns()

    if last_ping == 0:
      self.connect_status = MetricData("CONNECT", "OFFLINE", ItemStatus.WARNING)
    elif current_time_ns - last_ping < 80_000_000_000:  # 80 seconds in nanoseconds
      self.connect_status = MetricData("CONNECT", "ONLINE", ItemStatus.GOOD)
    else:
      self.connect_status = MetricData("CONNECT", "ERROR", ItemStatus.DANGER)

  def _update_panda_status(self, sm):
    if sm.valid['pandaStates'] and len(sm['pandaStates']) > 0:
      panda_state = sm['pandaStates'][0]
      if hasattr(panda_state, 'pandaType') and panda_state.pandaType != 0:  # UNKNOWN
        self.panda_status = MetricData("VEHICLE", "ONLINE", ItemStatus.GOOD)
      else:
        self.panda_status = MetricData("NO", "PANDA", ItemStatus.DANGER)
    else:
      self.panda_status = MetricData("NO", "PANDA", ItemStatus.DANGER)

  def handle_mouse_press(self, mouse_pos: rl.Vector2):
    x, y = mouse_pos.x, mouse_pos.y

    if rl.check_collision_point_rec(rl.Vector2(x, y), HOME_BTN_RECT):
      if self.onroad:
        self.flag_pressed = True
        return "flag"
      else:
        return "home"
    elif rl.check_collision_point_rec(rl.Vector2(x, y), SETTINGS_BTN_RECT):
      self.settings_pressed = True
      return "settings"

    return None

  def handle_mouse_release(self, mouse_pos: rl.Vector2):
    x, y = mouse_pos.x, mouse_pos.y
    action = None

    if self.flag_pressed:
      self.flag_pressed = False
      if rl.check_collision_point_rec(rl.Vector2(x, y), HOME_BTN_RECT):
        action = "send_flag"

    if self.settings_pressed:
      self.settings_pressed = False
      if rl.check_collision_point_rec(rl.Vector2(x, y), SETTINGS_BTN_RECT):
        action = "open_settings"

    return action

  def set_onroad(self, onroad: bool):
    self.onroad = onroad

  def _draw_buttons(self, rect: rl.Rectangle):
    # Settings button
    opacity = 0.65 if self.settings_pressed else 1.0

    tint = rl.Color(255, 255, 255, int(255 * opacity))
    rl.draw_texture(self.settings_img, int(SETTINGS_BTN_RECT.x), int(SETTINGS_BTN_RECT.y), tint)
    # Home/Flag button
    opacity = 0.65 if self.onroad and self.flag_pressed else 1.0
    button_img = self.flag_img if self.onroad else self.home_img

    tint = rl.Color(255, 255, 255, int(255 * opacity))
    rl.draw_texture(button_img, int(HOME_BTN_RECT.x), int(HOME_BTN_RECT.y), tint)

  def _draw_network_indicator(self, rect: rl.Rectangle):
    # Signal strength dots
    x_start = rect.x + 58
    y_pos = rect.y + 196
    dot_size = 27
    dot_spacing = 37

    for i in range(5):
      color = rl.WHITE if i < self.net_strength else GRAY
      rl.draw_circle(int(x_start + i * dot_spacing + dot_size // 2), int(y_pos + dot_size // 2), dot_size // 2, color)

    # Network type text
    text_y = rect.y + 247
    text_rect = rl.Rectangle(rect.x + 58, text_y, rect.width - 100, 50)
    self._draw_text_in_rect(
      self.net_type, self.font_regular, 35, text_rect, rl.WHITE, rl.GuiTextAlignment.TEXT_ALIGN_LEFT
    )

  def _draw_metrics(self, rect: rl.Rectangle):
    metrics = [(self.temp_status, 338), (self.panda_status, 496), (self.connect_status, 654)]

    for metric, y_offset in metrics:
      self._draw_metric(rect, metric, rect.y + y_offset)

  def _draw_metric(self, rect: rl.Rectangle, metric: MetricData, y: float):
    metric_rect = rl.Rectangle(rect.x + METRIC_MARGIN, y, METRIC_WIDTH, METRIC_HEIGHT)
    status_color = STATUS_COLORS.get(metric.status, rl.WHITE)

    # Draw colored left edge (clipped rounded rectangle)
    edge_rect = rl.Rectangle(metric_rect.x + 4, metric_rect.y + 4, 100, 118)
    rl.begin_scissor_mode(int(metric_rect.x + 4), int(metric_rect.y), 18, int(metric_rect.height))
    rl.draw_rectangle_rounded(edge_rect, 0.18, 10, status_color)
    rl.end_scissor_mode()

    # Draw border
    rl.draw_rectangle_rounded_lines_ex(metric_rect, 0.15, 10, 2, METRIC_BORDER)

    # Draw text
    text = f"{metric.label}\n{metric.value}"
    text_rect = rl.Rectangle(metric_rect.x + 22, metric_rect.y, metric_rect.width - 22, metric_rect.height)
    self._draw_text_in_rect(text, self.font_bold, 35, text_rect, rl.WHITE, rl.GuiTextAlignment.TEXT_ALIGN_CENTER)

  def _draw_text_in_rect(
    self, text: str, font: rl.Font, size: int, rect: rl.Rectangle, color: rl.Color, alignment: int
  ):
    text_size = rl.measure_text_ex(font, text, size, 0)
    if alignment == rl.GuiTextAlignment.TEXT_ALIGN_CENTER:
      x = rect.x + (rect.width - text_size.x) / 2
    elif alignment == rl.GuiTextAlignment.TEXT_ALIGN_LEFT:
      x = rect.x
    else:
      x = rect.x + rect.width - text_size.x
    y = rect.y + (rect.height - text_size.y) / 2
    rl.draw_text_ex(font, text, rl.Vector2(x, y), size, 0, color)
