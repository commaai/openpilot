import json
import pyray as rl
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from openpilot.common.params import Params
from openpilot.system.hardware import HARDWARE
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.widget import Widget


class AlertColors:
  HIGH_SEVERITY = rl.Color(226, 44, 44, 255)
  LOW_SEVERITY = rl.Color(41, 41, 41, 255)
  BACKGROUND = rl.Color(57, 57, 57, 255)
  BUTTON = rl.WHITE
  BUTTON_TEXT = rl.BLACK
  SNOOZE_BG = rl.Color(79, 79, 79, 255)
  TEXT = rl.WHITE


class AlertConstants:
  BUTTON_SIZE = (400, 125)
  SNOOZE_BUTTON_SIZE = (550, 125)
  REBOOT_BUTTON_SIZE = (600, 125)
  MARGIN = 50
  SPACING = 30
  FONT_SIZE = 48
  BORDER_RADIUS = 30
  ALERT_HEIGHT = 120
  ALERT_SPACING = 20


@dataclass
class AlertData:
  key: str
  text: str
  severity: int
  visible: bool = False


class AbstractAlert(Widget, ABC):
  def __init__(self, has_reboot_btn: bool = False):
    super().__init__()
    self.params = Params()
    self.has_reboot_btn = has_reboot_btn
    self.dismiss_callback: Callable | None = None

    self.dismiss_btn_rect = rl.Rectangle(0, 0, *AlertConstants.BUTTON_SIZE)
    self.snooze_btn_rect = rl.Rectangle(0, 0, *AlertConstants.SNOOZE_BUTTON_SIZE)
    self.reboot_btn_rect = rl.Rectangle(0, 0, *AlertConstants.REBOOT_BUTTON_SIZE)

    self.snooze_visible = False
    self.content_rect = rl.Rectangle(0, 0, 0, 0)
    self.scroll_panel_rect = rl.Rectangle(0, 0, 0, 0)
    self.scroll_panel = GuiScrollPanel()

  def set_dismiss_callback(self, callback: Callable):
    self.dismiss_callback = callback

  @abstractmethod
  def refresh(self) -> bool:
    pass

  @abstractmethod
  def get_content_height(self) -> float:
    pass

  def handle_input(self, mouse_pos: rl.Vector2, mouse_clicked: bool) -> bool:
    if not mouse_clicked or not self.scroll_panel.is_touch_valid():
      return False

    if rl.check_collision_point_rec(mouse_pos, self.dismiss_btn_rect):
      if self.dismiss_callback:
        self.dismiss_callback()
      return True

    if self.snooze_visible and rl.check_collision_point_rec(mouse_pos, self.snooze_btn_rect):
      self.params.put_bool("SnoozeUpdate", True)
      if self.dismiss_callback:
        self.dismiss_callback()
      return True

    if self.has_reboot_btn and rl.check_collision_point_rec(mouse_pos, self.reboot_btn_rect):
      HARDWARE.reboot()
      return True

    return False

  def _render(self, rect: rl.Rectangle):
    rl.draw_rectangle_rounded(rect, AlertConstants.BORDER_RADIUS / rect.width, 10, AlertColors.BACKGROUND)

    footer_height = AlertConstants.BUTTON_SIZE[1] + AlertConstants.SPACING
    content_height = rect.height - 2 * AlertConstants.MARGIN - footer_height

    self.content_rect = rl.Rectangle(
      rect.x + AlertConstants.MARGIN,
      rect.y + AlertConstants.MARGIN,
      rect.width - 2 * AlertConstants.MARGIN,
      content_height,
    )
    self.scroll_panel_rect = rl.Rectangle(
      self.content_rect.x, self.content_rect.y, self.content_rect.width, self.content_rect.height
    )

    self._render_scrollable_content()
    self._render_footer(rect)

  def _render_scrollable_content(self):
    content_total_height = self.get_content_height()
    content_bounds = rl.Rectangle(0, 0, self.scroll_panel_rect.width, content_total_height)
    scroll_offset = self.scroll_panel.handle_scroll(self.scroll_panel_rect, content_bounds)

    rl.begin_scissor_mode(
      int(self.scroll_panel_rect.x),
      int(self.scroll_panel_rect.y),
      int(self.scroll_panel_rect.width),
      int(self.scroll_panel_rect.height),
    )

    content_rect_with_scroll = rl.Rectangle(
      self.scroll_panel_rect.x,
      self.scroll_panel_rect.y + scroll_offset.y,
      self.scroll_panel_rect.width,
      content_total_height,
    )

    self._render_content(content_rect_with_scroll)
    rl.end_scissor_mode()

  @abstractmethod
  def _render_content(self, content_rect: rl.Rectangle):
    pass

  def _render_footer(self, rect: rl.Rectangle):
    footer_y = rect.y + rect.height - AlertConstants.MARGIN - AlertConstants.BUTTON_SIZE[1]
    font = gui_app.font(FontWeight.MEDIUM)

    self.dismiss_btn_rect.x = rect.x + AlertConstants.MARGIN
    self.dismiss_btn_rect.y = footer_y
    rl.draw_rectangle_rounded(self.dismiss_btn_rect, 0.3, 10, AlertColors.BUTTON)

    text = "Close"
    text_width = measure_text_cached(font, text, AlertConstants.FONT_SIZE).x
    text_x = self.dismiss_btn_rect.x + (AlertConstants.BUTTON_SIZE[0] - text_width) // 2
    text_y = self.dismiss_btn_rect.y + (AlertConstants.BUTTON_SIZE[1] - AlertConstants.FONT_SIZE) // 2
    rl.draw_text_ex(
      font, text, rl.Vector2(int(text_x), int(text_y)), AlertConstants.FONT_SIZE, 0, AlertColors.BUTTON_TEXT
    )

    if self.snooze_visible:
      self.snooze_btn_rect.x = rect.x + rect.width - AlertConstants.MARGIN - AlertConstants.SNOOZE_BUTTON_SIZE[0]
      self.snooze_btn_rect.y = footer_y
      rl.draw_rectangle_rounded(self.snooze_btn_rect, 0.3, 10, AlertColors.SNOOZE_BG)

      text = "Snooze Update"
      text_width = measure_text_cached(font, text, AlertConstants.FONT_SIZE).x
      text_x = self.snooze_btn_rect.x + (AlertConstants.SNOOZE_BUTTON_SIZE[0] - text_width) // 2
      text_y = self.snooze_btn_rect.y + (AlertConstants.SNOOZE_BUTTON_SIZE[1] - AlertConstants.FONT_SIZE) // 2
      rl.draw_text_ex(font, text, rl.Vector2(int(text_x), int(text_y)), AlertConstants.FONT_SIZE, 0, AlertColors.TEXT)

    elif self.has_reboot_btn:
      self.reboot_btn_rect.x = rect.x + rect.width - AlertConstants.MARGIN - AlertConstants.REBOOT_BUTTON_SIZE[0]
      self.reboot_btn_rect.y = footer_y
      rl.draw_rectangle_rounded(self.reboot_btn_rect, 0.3, 10, AlertColors.BUTTON)

      text = "Reboot and Update"
      text_width = measure_text_cached(font, text, AlertConstants.FONT_SIZE).x
      text_x = self.reboot_btn_rect.x + (AlertConstants.REBOOT_BUTTON_SIZE[0] - text_width) // 2
      text_y = self.reboot_btn_rect.y + (AlertConstants.REBOOT_BUTTON_SIZE[1] - AlertConstants.FONT_SIZE) // 2
      rl.draw_text_ex(
        font, text, rl.Vector2(int(text_x), int(text_y)), AlertConstants.FONT_SIZE, 0, AlertColors.BUTTON_TEXT
      )


class OffroadAlert(AbstractAlert):
  def __init__(self):
    super().__init__(has_reboot_btn=False)
    self.sorted_alerts: list[AlertData] = []

  def refresh(self):
    if not self.sorted_alerts:
      self._build_alerts()

    active_count = 0
    connectivity_needed = False

    for alert_data in self.sorted_alerts:
      text = ""
      bytes_data = self.params.get(alert_data.key)

      if bytes_data:
        try:
          alert_json = json.loads(bytes_data)
          text = alert_json.get("text", "").replace("{}", alert_json.get("extra", ""))
        except json.JSONDecodeError:
          text = ""

      alert_data.text = text
      alert_data.visible = bool(text)

      if alert_data.visible:
        active_count += 1

      if alert_data.key == "Offroad_ConnectivityNeeded" and alert_data.visible:
        connectivity_needed = True

    self.snooze_visible = connectivity_needed
    return active_count

  def get_content_height(self) -> float:
    if not self.sorted_alerts:
      return 0

    total_height = 20
    font = gui_app.font(FontWeight.NORMAL)

    for alert_data in self.sorted_alerts:
      if not alert_data.visible:
        continue

      text_width = int(self.content_rect.width - 90)
      wrapped_lines = wrap_text(font, alert_data.text, AlertConstants.FONT_SIZE, text_width)
      line_count = len(wrapped_lines)
      text_height = line_count * (AlertConstants.FONT_SIZE + 5)
      alert_item_height = max(text_height + 40, AlertConstants.ALERT_HEIGHT)
      total_height += alert_item_height + AlertConstants.ALERT_SPACING

    if total_height > 20:
      total_height = total_height - AlertConstants.ALERT_SPACING + 20

    return total_height

  def _build_alerts(self):
    self.sorted_alerts = []
    try:
      with open("../selfdrived/alerts_offroad.json", "rb") as f:
        alerts_config = json.load(f)
        for key, config in sorted(alerts_config.items(), key=lambda x: x[1].get("severity", 0), reverse=True):
          severity = config.get("severity", 0)
          alert_data = AlertData(key=key, text="", severity=severity)
          self.sorted_alerts.append(alert_data)
    except (FileNotFoundError, json.JSONDecodeError):
      pass

  def _render_content(self, content_rect: rl.Rectangle):
    y_offset = 20
    font = gui_app.font(FontWeight.NORMAL)

    for alert_data in self.sorted_alerts:
      if not alert_data.visible:
        continue

      bg_color = AlertColors.HIGH_SEVERITY if alert_data.severity > 0 else AlertColors.LOW_SEVERITY
      text_width = int(content_rect.width - 90)
      wrapped_lines = wrap_text(font, alert_data.text, AlertConstants.FONT_SIZE, text_width)
      line_count = len(wrapped_lines)
      text_height = line_count * (AlertConstants.FONT_SIZE + 5)
      alert_item_height = max(text_height + 40, AlertConstants.ALERT_HEIGHT)

      alert_rect = rl.Rectangle(
        content_rect.x + 10,
        content_rect.y + y_offset,
        content_rect.width - 30,
        alert_item_height,
      )

      rl.draw_rectangle_rounded(alert_rect, 0.2, 10, bg_color)

      text_x = alert_rect.x + 30
      text_y = alert_rect.y + 20

      for i, line in enumerate(wrapped_lines):
        rl.draw_text_ex(
          font,
          line,
          rl.Vector2(text_x, text_y + i * (AlertConstants.FONT_SIZE + 5)),
          AlertConstants.FONT_SIZE,
          0,
          AlertColors.TEXT,
        )

      y_offset += alert_item_height + AlertConstants.ALERT_SPACING


class UpdateAlert(AbstractAlert):
  def __init__(self):
    super().__init__(has_reboot_btn=True)
    self.release_notes = ""
    self._wrapped_release_notes = ""
    self._cached_content_height: float = 0.0

  def refresh(self) -> bool:
    update_available: bool = self.params.get_bool("UpdateAvailable")
    if update_available:
      self.release_notes = self.params.get("UpdaterNewReleaseNotes", encoding='utf-8')
      self._cached_content_height = 0

    return update_available

  def get_content_height(self) -> float:
    if not self.release_notes:
      return 100

    if self._cached_content_height == 0:
      self._wrapped_release_notes = self.release_notes
      size = measure_text_cached(gui_app.font(FontWeight.NORMAL), self._wrapped_release_notes, AlertConstants.FONT_SIZE)
      self._cached_content_height = max(size.y + 60, 100)

    return self._cached_content_height

  def _render_content(self, content_rect: rl.Rectangle):
    if self.release_notes:
      rl.draw_text_ex(
        gui_app.font(FontWeight.NORMAL),
        self._wrapped_release_notes,
        rl.Vector2(content_rect.x + 30, content_rect.y + 30),
        AlertConstants.FONT_SIZE,
        0.0,
        AlertColors.TEXT,
      )
    else:
      no_notes_text = "No release notes available."
      text_width = rl.measure_text(no_notes_text, AlertConstants.FONT_SIZE)
      text_x = content_rect.x + (content_rect.width - text_width) // 2
      text_y = content_rect.y + 50
      rl.draw_text(no_notes_text, int(text_x), int(text_y), AlertConstants.FONT_SIZE, AlertColors.TEXT)
