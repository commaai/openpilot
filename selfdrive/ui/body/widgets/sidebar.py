import pyray as rl
from openpilot.selfdrive.ui.layouts.sidebar import Sidebar, Colors, MetricData
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FONT_SCALE
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached

BODY_SIDEBAR_HEIGHT = 200
METRIC_HEIGHT = 117
METRIC_WIDTH = 220
METRIC_MARGIN = 20
FONT_SIZE = 30
BATTERY_FONT_SIZE = 26

SETTINGS_BTN = rl.Rectangle(50, 35, 200, 117)

class Colors(Colors):
  BATTERY_GREEN = rl.Color(0, 200, 0, 255)
  BATTERY_LOW = rl.Color(201, 34, 49, 255)

class BodySidebar(Sidebar):
  """A top-dropping sidebar for the comma body, containing the same info as the regular sidebar."""

  def __init__(self):
    super().__init__()
    self.set_visible(False)

    self._battery_percent = 0.0
    self._battery_charging = False

  def _render(self, rect: rl.Rectangle):
    rl.draw_rectangle_rec(rect, rl.BLACK)

    self._draw_settings_button(rect)
    self._draw_network_indicator(rect)
    self._draw_metrics(rect)
    self._draw_battery_indicator(rect)

  def _update_battery_status(self):
    sm = ui_state.sm
    if sm.updated['carState']:
      car_state = sm['carState']
      self._battery_percent = max(0.0, min(1.0, car_state.fuelGauge))
      self._battery_charging = car_state.charging

  def _draw_settings_button(self, rect: rl.Rectangle):
    mouse_pos = rl.get_mouse_position()
    mouse_down = self.is_pressed and rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT)

    btn_x = int(rect.x + 30)
    btn_y = int(rect.y + 30)
    settings_rect = rl.Rectangle(btn_x, btn_y, SETTINGS_BTN.width, SETTINGS_BTN.height)
    settings_down = mouse_down and rl.check_collision_point_rec(mouse_pos, settings_rect)
    tint = Colors.BUTTON_PRESSED if settings_down else Colors.BUTTON_NORMAL
    rl.draw_texture(self._settings_img, btn_x, btn_y, tint)

  def _draw_battery_indicator(self, rect: rl.Rectangle):
    # Battery icon dimensions
    batt_w = 50
    batt_h = 28
    tip_w = 5
    tip_h = 12
    batt_x = int(rect.x + rect.width - batt_w - tip_w - 30)
    batt_y = int(rect.y + 30 + (METRIC_HEIGHT - batt_h) / 2)

    # Choose fill color based on level
    pct = self._battery_percent
    if pct <= 0.2:
      fill_color = Colors.BATTERY_LOW
    elif self._battery_charging:
      fill_color = Colors.BATTERY_GREEN
    else:
      fill_color = Colors.WHITE

    # Battery outline
    rl.draw_rectangle_rounded_lines_ex(rl.Rectangle(batt_x, batt_y, batt_w, batt_h), 0.2, 6, 2, Colors.WHITE)

    # Battery tip (positive terminal)
    tip_x = batt_x + batt_w
    tip_y = batt_y + (batt_h - tip_h) / 2
    rl.draw_rectangle_rounded(rl.Rectangle(tip_x, tip_y, tip_w, tip_h), 0.3, 4, Colors.WHITE)

    # Fill level
    fill_margin = 4
    fill_max_w = batt_w - 2 * fill_margin
    fill_w = max(0, int(fill_max_w * pct))
    if fill_w > 0:
      rl.draw_rectangle_rounded(
        rl.Rectangle(batt_x + fill_margin, batt_y + fill_margin, fill_w, batt_h - 2 * fill_margin),
        0.15, 4, fill_color
      )

    # Percentage text
    pct_text = f"{int(pct * 100)}%"
    if self._battery_charging:
      pct_text = pct_text
    pct_size = measure_text_cached(self._font_bold, pct_text, BATTERY_FONT_SIZE)
    pct_pos = rl.Vector2(batt_x + (batt_w - pct_size.x) / 2, batt_y + batt_h + 6)
    rl.draw_text_ex(self._font_bold, pct_text, pct_pos, BATTERY_FONT_SIZE, 0, Colors.WHITE)

  def _draw_network_indicator(self, rect: rl.Rectangle):
    # Draw network dots horizontally, positioned after the settings button
    x_start = rect.x + 260
    y_pos = rect.y + 40
    dot_size = 20
    dot_spacing = 28

    for i in range(5):
      color = Colors.WHITE if i < self._net_strength else Colors.GRAY
      x = int(x_start + i * dot_spacing + dot_size // 2)
      y = int(y_pos + dot_size // 2)
      rl.draw_circle(x, y, dot_size // 2, color)

    # Network type text below dots
    text_pos = rl.Vector2(x_start, y_pos + dot_size + 8)
    rl.draw_text_ex(self._font_regular, tr(self._net_type), text_pos, FONT_SIZE, 0, Colors.WHITE)

  def _draw_metrics(self, rect: rl.Rectangle):
    metrics = [self._temp_status, self._panda_status, self._connect_status]
    # Center the 3 metrics in the middle of the bar
    total_width = len(metrics) * METRIC_WIDTH + (len(metrics) - 1) * METRIC_MARGIN
    start_x = rect.x + (rect.width - total_width) / 2

    y = rect.y + 30

    for i, metric in enumerate(metrics):
      x = start_x + i * (METRIC_WIDTH + METRIC_MARGIN)
      self._draw_metric(metric, x, y)

  def _draw_metric(self, metric: MetricData, x: float, y: float):
    r = rl.Rectangle(x, y, METRIC_WIDTH, METRIC_HEIGHT)

    # Colored top edge (clipped rounded rect)
    rl.begin_scissor_mode(int(x), int(y + 4), int(METRIC_WIDTH), 18)
    rl.draw_rectangle_rounded(rl.Rectangle(x + 4, y + 4, METRIC_WIDTH - 8, 100), 0.3, 10, metric.color)
    rl.end_scissor_mode()

    rl.draw_rectangle_rounded_lines_ex(r, 0.3, 10, 2, Colors.METRIC_BORDER)

    # Center label and value below the top edge
    text_y = y + 22 + ((METRIC_HEIGHT - 22) / 2 - 2 * FONT_SIZE * FONT_SCALE)
    for label in (metric.label, metric.value):
      text = tr(label)
      size = measure_text_cached(self._font_bold, text, FONT_SIZE)
      text_y += size.y
      text_pos = rl.Vector2(
        x + (METRIC_WIDTH - size.x) / 2,
        text_y
      )
      rl.draw_text_ex(self._font_bold, text, text_pos, FONT_SIZE, 0, Colors.WHITE)
