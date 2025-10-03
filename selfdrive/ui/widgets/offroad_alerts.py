import pyray as rl
from enum import IntEnum
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from openpilot.common.params import Params
from openpilot.system.hardware import HARDWARE
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.widgets import Widget
from openpilot.selfdrive.selfdrived.alertmanager import OFFROAD_ALERTS


class AlertColors:
  HIGH_SEVERITY = rl.Color(226, 44, 44, 255)
  LOW_SEVERITY = rl.Color(41, 41, 41, 255)
  BACKGROUND = rl.Color(57, 57, 57, 255)
  BUTTON = rl.WHITE
  BUTTON_PRESSED = rl.Color(200, 200, 200, 255)
  BUTTON_TEXT = rl.BLACK
  SNOOZE_BG = rl.Color(79, 79, 79, 255)
  SNOOZE_BG_PRESSED = rl.Color(100, 100, 100, 255)
  TEXT = rl.WHITE


class AlertConstants:
  MIN_BUTTON_WIDTH = 400
  BUTTON_HEIGHT = 125
  MARGIN = 50
  SPACING = 30
  FONT_SIZE = 48
  BORDER_RADIUS = 30 * 2  # matches Qt's 30px
  ALERT_HEIGHT = 120
  ALERT_SPACING = 10
  ALERT_INSET = 60


@dataclass
class AlertData:
  key: str
  text: str
  severity: int
  visible: bool = False


class ButtonStyle(IntEnum):
  LIGHT = 0
  DARK = 1


class ActionButton(Widget):
  def __init__(self, text: str, style: ButtonStyle = ButtonStyle.LIGHT,
               min_width: int = AlertConstants.MIN_BUTTON_WIDTH):
    super().__init__()
    self._style = style
    self._min_width = min_width
    self._font = gui_app.font(FontWeight.MEDIUM)
    self.set_text(text)

  def set_text(self, text: str):
    self._text = text
    self._text_width = measure_text_cached(gui_app.font(FontWeight.MEDIUM), self._text, AlertConstants.FONT_SIZE).x
    self._rect.width = max(self._text_width + 60 * 2, self._min_width)
    self._rect.height = AlertConstants.BUTTON_HEIGHT

  def _render(self, _):
    roundness = AlertConstants.BORDER_RADIUS / self._rect.height
    bg_color = AlertColors.BUTTON if self._style == ButtonStyle.LIGHT else AlertColors.SNOOZE_BG
    if self.is_pressed:
      bg_color = AlertColors.BUTTON_PRESSED if self._style == ButtonStyle.LIGHT else AlertColors.SNOOZE_BG_PRESSED

    rl.draw_rectangle_rounded(self._rect, roundness, 10, bg_color)

    # center text
    color = rl.WHITE if self._style == ButtonStyle.DARK else rl.BLACK
    text_x = int(self._rect.x + (self._rect.width - self._text_width) // 2)
    text_y = int(self._rect.y + (self._rect.height - AlertConstants.FONT_SIZE) // 2)
    rl.draw_text_ex(self._font, self._text, rl.Vector2(text_x, text_y), AlertConstants.FONT_SIZE, 0, color)


class AbstractAlert(Widget, ABC):
  def __init__(self, has_reboot_btn: bool = False):
    super().__init__()
    self.params = Params()
    self.has_reboot_btn = has_reboot_btn
    self.dismiss_callback: Callable | None = None

    def snooze_callback():
      self.params.put_bool("SnoozeUpdate", True)
      if self.dismiss_callback:
        self.dismiss_callback()

    def excessive_actuation_callback():
      self.params.remove("Offroad_ExcessiveActuation")
      if self.dismiss_callback:
        self.dismiss_callback()

    self.dismiss_btn = ActionButton("Close")

    self.snooze_btn = ActionButton("Snooze Update", style=ButtonStyle.DARK)
    self.snooze_btn.set_click_callback(snooze_callback)

    self.excessive_actuation_btn = ActionButton("Acknowledge Excessive Actuation", style=ButtonStyle.DARK, min_width=800)
    self.excessive_actuation_btn.set_click_callback(excessive_actuation_callback)

    self.reboot_btn = ActionButton("Reboot and Update", min_width=600)
    self.reboot_btn.set_click_callback(lambda: HARDWARE.reboot())

    # TODO: just use a Scroller?
    self.content_rect = rl.Rectangle(0, 0, 0, 0)
    self.scroll_panel_rect = rl.Rectangle(0, 0, 0, 0)
    self.scroll_panel = GuiScrollPanel()

  def set_dismiss_callback(self, callback: Callable):
    self.dismiss_callback = callback
    self.dismiss_btn.set_click_callback(self.dismiss_callback)

  @abstractmethod
  def refresh(self) -> bool:
    pass

  @abstractmethod
  def get_content_height(self) -> float:
    pass

  def _render(self, rect: rl.Rectangle):
    rl.draw_rectangle_rounded(rect, AlertConstants.BORDER_RADIUS / rect.height, 10, AlertColors.BACKGROUND)

    footer_height = AlertConstants.BUTTON_HEIGHT + AlertConstants.SPACING
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
    scroll_offset = self.scroll_panel.update(self.scroll_panel_rect, content_bounds)

    rl.begin_scissor_mode(
      int(self.scroll_panel_rect.x),
      int(self.scroll_panel_rect.y),
      int(self.scroll_panel_rect.width),
      int(self.scroll_panel_rect.height),
    )

    content_rect_with_scroll = rl.Rectangle(
      self.scroll_panel_rect.x,
      self.scroll_panel_rect.y + scroll_offset,
      self.scroll_panel_rect.width,
      content_total_height,
    )

    self._render_content(content_rect_with_scroll)
    rl.end_scissor_mode()

  @abstractmethod
  def _render_content(self, content_rect: rl.Rectangle):
    pass

  def _render_footer(self, rect: rl.Rectangle):
    footer_y = rect.y + rect.height - AlertConstants.MARGIN - AlertConstants.BUTTON_HEIGHT

    dismiss_x = rect.x + AlertConstants.MARGIN
    self.dismiss_btn.set_position(dismiss_x, footer_y)
    self.dismiss_btn.render()

    if self.has_reboot_btn:
      reboot_x = rect.x + rect.width - AlertConstants.MARGIN - self.reboot_btn.rect.width
      self.reboot_btn.set_position(reboot_x, footer_y)
      self.reboot_btn.render()

    elif self.excessive_actuation_btn.is_visible:
      actuation_x = rect.x + rect.width - AlertConstants.MARGIN - self.excessive_actuation_btn.rect.width
      self.excessive_actuation_btn.set_position(actuation_x, footer_y)
      self.excessive_actuation_btn.render()

    elif self.snooze_btn.is_visible:
      snooze_x = rect.x + rect.width - AlertConstants.MARGIN - self.snooze_btn.rect.width
      self.snooze_btn.set_position(snooze_x, footer_y)
      self.snooze_btn.render()


class OffroadAlert(AbstractAlert):
  def __init__(self):
    super().__init__(has_reboot_btn=False)
    self.sorted_alerts: list[AlertData] = []

  def refresh(self):
    if not self.sorted_alerts:
      self._build_alerts()

    active_count = 0
    connectivity_needed = False
    excessive_actuation = False

    for alert_data in self.sorted_alerts:
      text = ""
      alert_json = self.params.get(alert_data.key)

      if alert_json:
        text = alert_json.get("text", "").replace("%1", alert_json.get("extra", ""))

      alert_data.text = text
      alert_data.visible = bool(text)

      if alert_data.visible:
        active_count += 1

      if alert_data.key == "Offroad_ConnectivityNeeded" and alert_data.visible:
        connectivity_needed = True

      if alert_data.key == "Offroad_ExcessiveActuation" and alert_data.visible:
        excessive_actuation = True

    self.excessive_actuation_btn.set_visible(excessive_actuation)
    self.snooze_btn.set_visible(connectivity_needed and not excessive_actuation)
    return active_count

  def get_content_height(self) -> float:
    if not self.sorted_alerts:
      return 0

    total_height = 20
    font = gui_app.font(FontWeight.NORMAL)

    for alert_data in self.sorted_alerts:
      if not alert_data.visible:
        continue

      text_width = int(self.content_rect.width - (AlertConstants.ALERT_INSET * 2))
      wrapped_lines = wrap_text(font, alert_data.text, AlertConstants.FONT_SIZE, text_width)
      line_count = len(wrapped_lines)
      text_height = line_count * (AlertConstants.FONT_SIZE + 5)
      alert_item_height = max(text_height + (AlertConstants.ALERT_INSET * 2), AlertConstants.ALERT_HEIGHT)
      total_height += alert_item_height + AlertConstants.ALERT_SPACING

    if total_height > 20:
      total_height = total_height - AlertConstants.ALERT_SPACING + 20

    return total_height

  def _build_alerts(self):
    self.sorted_alerts = []
    for key, config in sorted(OFFROAD_ALERTS.items(), key=lambda x: x[1].get("severity", 0), reverse=True):
      severity = config.get("severity", 0)
      alert_data = AlertData(key=key, text="", severity=severity)
      self.sorted_alerts.append(alert_data)

  def _render_content(self, content_rect: rl.Rectangle):
    y_offset = AlertConstants.ALERT_SPACING
    font = gui_app.font(FontWeight.NORMAL)

    for alert_data in self.sorted_alerts:
      if not alert_data.visible:
        continue

      bg_color = AlertColors.HIGH_SEVERITY if alert_data.severity > 0 else AlertColors.LOW_SEVERITY
      text_width = int(content_rect.width - (AlertConstants.ALERT_INSET * 2))
      wrapped_lines = wrap_text(font, alert_data.text, AlertConstants.FONT_SIZE, text_width)
      line_count = len(wrapped_lines)
      text_height = line_count * (AlertConstants.FONT_SIZE + 5)
      alert_item_height = max(text_height + (AlertConstants.ALERT_INSET * 2), AlertConstants.ALERT_HEIGHT)

      alert_rect = rl.Rectangle(
        content_rect.x + 10,
        content_rect.y + y_offset,
        content_rect.width - 30,
        alert_item_height,
      )

      roundness = AlertConstants.BORDER_RADIUS / min(alert_rect.height, alert_rect.width)
      rl.draw_rectangle_rounded(alert_rect, roundness, 10, bg_color)

      text_x = alert_rect.x + AlertConstants.ALERT_INSET
      text_y = alert_rect.y + AlertConstants.ALERT_INSET

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
      self.release_notes = self.params.get("UpdaterNewReleaseNotes")
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
