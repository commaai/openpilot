from __future__ import annotations

import functools
import math
import numpy as np
import raylib_nb as rl
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from openpilot.common.params import Params
from openpilot.common.realtime import drop_realtime
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.selfdrived.alertmanager import OFFROAD_ALERTS
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.mici.widgets.button import GreyBigButton, BigPillButton
from openpilot.selfdrive.ui.mici.widgets.dialog import BigConfirmationDialog
from openpilot.selfdrive.ui.mici.widgets.qr import QR
from openpilot.common.hardware import HARDWARE
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.system.ui.widgets.scroller import NavScroller
from openpilot.system.ui.lib.application import gui_app, FontWeight, TextAlignment, TextAlignmentVertical
from openpilot.system.ui.lib.multilang import tr

REFRESH_INTERVAL = 5.0  # seconds


# Card background: translucent black squircle with inner shadows, like the Figma design
CARD_RADIUS = 42
CARD_CORNER_EXPONENT = 2.4  # superellipse corners, 2 would be circular
CARD_FILL_ALPHA = 0.8
# inner shadows drawn in order over the fill: (gray, opacity, spread, blur sigma)
CARD_SHADOWS = [(1.0, 0.36, 3.9, 7.9)]
CARD_SHADOWS_PRESSED = [(1.0, 0.75, 7.1, 16.8), (0.0, 0.77, 1.4, 4.3)]
CARD_SLICE = 60  # corners and edges are drawn from this much of a nine-slice texture, the center is stretched


def _squircle(size: int, inset: float, px: float) -> np.ndarray:
  """Coverage of a size x size card shape inset by `inset`, rendered at px pixels per point with 4x4 supersampling"""
  ss, n = 4, round(size * px)
  c = (np.arange(n * ss) + 0.5) / (ss * px)
  r = CARD_RADIUS - inset
  corner = np.clip(np.maximum(CARD_RADIUS - c, c - (size - CARD_RADIUS)) / r, 0, None) ** CARD_CORNER_EXPONENT
  inside = (c >= inset) & (c <= size - inset)
  mask = (corner[None, :] + corner[:, None] <= 1) & inside[None, :] & inside[:, None]
  return mask.reshape(n, ss, n, ss).mean(axis=(1, 3))


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
  """Gaussian blur, with everything beyond the edges taken as 1"""
  k = math.ceil(3 * sigma)
  x = np.arange(-k, k + 1)
  kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
  kernel /= kernel.sum()
  img = np.pad(img, k, constant_values=1.0)
  for axis in (0, 1):
    img = np.apply_along_axis(np.convolve, axis, img, kernel, mode='same')
  return img[k:-k, k:-k]


@functools.cache
def _card_texture(pressed: bool) -> rl.Texture:
  """Nine-slice texture of the card background, see draw_card"""
  size = 2 * CARD_SLICE + 1
  px = gui_app._scale
  shape = _squircle(size, 0, px)
  color, alpha = np.zeros_like(shape), np.full_like(shape, CARD_FILL_ALPHA)  # premultiplied gray, alpha
  for gray, opacity, spread, sigma in (CARD_SHADOWS_PRESSED if pressed else CARD_SHADOWS):
    shadow = opacity * _blur(1 - _squircle(size, spread, px), sigma * px)
    color = gray * shadow + color * (1 - shadow)
    alpha = shadow + alpha * (1 - shadow)
  alpha *= shape

  rgba = np.empty((*shape.shape, 4), dtype=np.uint8)
  rgba[..., :3] = np.clip(np.rint(255 * color / np.maximum(alpha, 1e-6)), 0, 255)[..., None]
  rgba[..., 3] = np.clip(np.rint(255 * alpha), 0, 255)
  texture = rl.load_texture_from_image(rl.Image(None, rgba.shape[1], rgba.shape[0], 1, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8))
  rl.update_texture(texture, rgba)
  rl.set_texture_filter(texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
  rl.set_texture_wrap(texture, rl.TextureWrap.TEXTURE_WRAP_CLAMP)
  texture.width = texture.height = size
  return texture


def _slices(start: float, length: float) -> list[tuple[float, float, float, float]]:
  """(texture start, texture length, start, length) of the near edge, stretched middle, and far edge along one axis"""
  s = min(CARD_SLICE, length / 2)
  return [(0, s, start, s), (CARD_SLICE, 1, start + s, length - 2 * s), (2 * CARD_SLICE + 1 - s, s, start + length - s, s)]


def draw_card(rect: rl.Rectangle, pressed: bool = False):
  """Draw an alert card background, sized to rect"""
  texture = _card_texture(pressed)
  for sy, sh, dy, dh in _slices(rect.y, rect.height):
    for sx, sw, dx, dw in _slices(rect.x, rect.width):
      if dw > 0 and dh > 0:
        rl.draw_texture_pro(texture, rl.Rectangle(sx, sy, sw, sh), rl.Rectangle(dx, dy, dw, dh), rl.Vector2(0, 0), 0.0, rl.WHITE)


@dataclass
class AlertData:
  key: str
  text: str
  severity: int
  visible: bool = False
  icon: str | None = None


class AlertItem(Widget):
  # TODO: click should always go somewhere: home or specific settings pane
  """Individual alert item widget, a card sized to fit its text."""
  ALERT_WIDTH = 520
  NOTIFICATION_HEIGHT = 240
  ALERT_PADDING = 28
  ICON_SIZE = 64
  ICON_MARGIN = 12
  TEXT_COLOR = rl.Color(255, 255, 255, int(255 * 0.9))
  TITLE_BODY_SPACING = 24

  def __init__(self, alert_data: AlertData):
    super().__init__()
    self.alert_data = alert_data

    # Load alert icons
    self._icon_orange = gui_app.texture("icons_mici/offroad_alerts/orange_warning.png", self.ICON_SIZE, self.ICON_SIZE)
    self._icon_red = gui_app.texture("icons_mici/offroad_alerts/red_warning.png", self.ICON_SIZE, self.ICON_SIZE)
    self._icon_green = gui_app.texture("icons_mici/offroad_alerts/green_wheel.png", self.ICON_SIZE, self.ICON_SIZE)
    self._custom_icon = gui_app.texture(alert_data.icon, self.ICON_SIZE, self.ICON_SIZE) if alert_data.icon else None

    if self._custom_icon is not None:
      self._icon = self._custom_icon
    elif alert_data.severity == -1:
      self._icon = self._icon_green
    elif alert_data.severity > 0:
      self._icon = self._icon_red
    else:
      self._icon = self._icon_orange

    self._title_label = UnifiedLabel(text="", font_size=32, font_weight=FontWeight.SEMI_BOLD, text_color=self.TEXT_COLOR,
                                     alignment=TextAlignment.LEFT,
                                     alignment_vertical=TextAlignmentVertical.TOP, line_height=0.95)

    self._body_label = UnifiedLabel(text="", font_size=28, font_weight=FontWeight.ROMAN, text_color=self.TEXT_COLOR,
                                    alignment=TextAlignment.LEFT,
                                    alignment_vertical=TextAlignmentVertical.BOTTOM, line_height=0.95)

    self._title_text = ""
    self._body_text = ""
    self._text_offset = 0.0  # centers text shorter than the icon

    self._update_content()

  @property
  def icon(self) -> rl.Texture:
    return self._icon

  def _split_text(self, text: str) -> tuple[str, str]:
    """Split text into title (first sentence) and body (remaining text)."""
    # Find the end of the first sentence (period, exclamation, or question mark followed by space or end)
    match = re.search(r'[.!?](?:\s+|$)', text)
    if match:
      # Found a sentence boundary - split at the end of the sentence
      title = text[:match.start()].strip()
      body = text[match.end():].strip()
      return title, body
    else:
      # No sentence boundary found, return full text as title
      return text.strip(), ""

  def _update_content(self):
    """Update text and calculate height."""
    if not self.alert_data.visible or not self.alert_data.text:
      self.set_visible(False)
      return

    self.set_visible(True)

    # Split text into title and body
    self._title_text, self._body_text = self._split_text(self.alert_data.text)

    # Calculate text width (alert width minus padding and icon space on right)
    title_width = self.ALERT_WIDTH - (self.ALERT_PADDING * 2) - self.ICON_SIZE - self.ICON_MARGIN
    body_width = self.ALERT_WIDTH - (self.ALERT_PADDING * 2)

    # Update labels
    self._title_label.set_text(self._title_text)
    self._body_label.set_text(self._body_text)

    # Calculate content height
    title_height = self._title_label.get_content_height(title_width) if self._title_text else 0
    body_height = self._body_label.get_content_height(body_width) if self._body_text else 0
    spacing = self.TITLE_BODY_SPACING if (self._title_text and self._body_text) else 0
    total_text_height = title_height + spacing + body_height

    # Fit the card to the text, and the icon
    self._text_offset = max(self.ICON_SIZE - total_text_height, 0) / 2
    height = math.ceil(max(total_text_height, self.ICON_SIZE) + self.ALERT_PADDING * 2)
    self.set_rect(rl.Rectangle(0, 0, self.ALERT_WIDTH, height))

  def update_alert_data(self, alert_data: AlertData):
    """Update alert data and refresh display."""
    self.alert_data = alert_data
    self._update_content()

  def _render(self, _):
    if not self.alert_data.visible or not self.alert_data.text:
      return

    draw_card(self._rect, self.is_pressed)

    # Calculate text area (left side, avoiding icon on right)
    title_width = self.ALERT_WIDTH - (self.ALERT_PADDING * 2) - self.ICON_SIZE - self.ICON_MARGIN
    body_width = self.ALERT_WIDTH - (self.ALERT_PADDING * 2)
    text_x = self._rect.x + self.ALERT_PADDING
    text_y = self._rect.y + self.ALERT_PADDING + self._text_offset

    # Draw title label
    if self._title_text:
      title_rect = rl.Rectangle(
        text_x,
        text_y,
        title_width,
        self._title_label.get_content_height(title_width),
      )
      self._title_label.render(title_rect)
      text_y += title_rect.height + self.TITLE_BODY_SPACING

    # Draw body label
    if self._body_text:
      body_rect = rl.Rectangle(
        text_x,
        text_y,
        body_width,
        self._rect.y + self._rect.height - self.ALERT_PADDING - self._text_offset - text_y,
      )
      self._body_label.render(body_rect)

    # Draw warning icon on the right side
    icon_x = self._rect.x + self.ALERT_WIDTH - self.ALERT_PADDING - self.ICON_SIZE
    icon_y = self._rect.y + self.ALERT_PADDING
    rl.draw_texture_ex(self._icon, rl.Vector2(icon_x, icon_y), 0.0, 1.0, rl.WHITE)


class NotificationItem(Widget):
  def __init__(self, notification: dict):
    super().__init__()
    self.alert_data = AlertData(key="", text="", severity=notification.get("severity", 0), visible=True)
    self.notification = notification
    self.icon = gui_app.texture(notification["icon"], AlertItem.ICON_SIZE, AlertItem.ICON_SIZE)
    self._image = gui_app.texture(notification["image"])
    self._menu = notification["menu"]
    if self._menu:
      self.set_click_callback(self._show_menu)
    self.set_rect(rl.Rectangle(0, 0, AlertItem.ALERT_WIDTH, AlertItem.NOTIFICATION_HEIGHT))

  def _show_menu(self):
    scroller = NavScroller()

    def dismiss():
      scroller.dismiss()
      ui_state.notifications.dismiss(self.notification["id"])

    for card in self._menu:
      if "qr" in card:
        scroller._scroller.add_widget(QR(card["qr"]))
      elif "title" in card:
        icon = gui_app.texture("icons_mici/setup/small_slider/slider_arrow.png", 64, 56, flip_x=True) if card.get("icon") == "arrow" else None
        scroller._scroller.add_widget(GreyBigButton(card["title"], card["text"], icon))
      elif card["button"] == "dismiss":
        button = BigPillButton("dismiss")
        button.set_click_callback(dismiss)
        scroller._scroller.add_widget(button)
      else:
        cloudlog.warning(f"Unknown notification menu card: {card}")
    gui_app.push_widget(scroller)

  def _render(self, _):
    draw_card(self._rect, self.is_pressed)
    image_pos = rl.Vector2(self._rect.x + (self._rect.width - self._image.width) / 2, self._rect.y + (self._rect.height - self._image.height) / 2)
    rl.draw_texture_ex(self._image, image_pos, 0.0, 1.0, rl.WHITE)
    icon_pos = rl.Vector2(self._rect.x + self._rect.width - AlertItem.ALERT_PADDING - self.icon.width, self._rect.y + AlertItem.ALERT_PADDING)
    rl.draw_texture_ex(self.icon, icon_pos, 0.0, 1.0, rl.WHITE)


class MiciOffroadAlerts(NavScroller):
  def __init__(self):
    super().__init__(horizontal=False, spacing=12, pad=0)
    self.params = Params()
    self.sorted_alerts: list[AlertData] = []
    self.alert_items: list[AlertItem] = []
    self._notifications: list[dict] = []
    self._notification_items: list[NotificationItem] = []

    # Create empty state label
    self._empty_label = UnifiedLabel(tr("no alerts"), 65, FontWeight.DISPLAY, rl.WHITE,
                                     alignment=TextAlignment.CENTER,
                                     alignment_vertical=TextAlignmentVertical.MIDDLE)

    # Build initial alert list
    self._build_alerts()

    # Start param thread
    self._pending_params: dict | None = None
    self._params_thread = threading.Thread(target=self._params_worker, daemon=True)
    self._params_thread.start()

    # refresh every frame, shown or not, for the home pill count
    gui_app.add_nav_stack_tick(self._refresh_state)

  def active_alerts(self) -> int:
    return sum(alert.visible for alert in self.sorted_alerts) + len(self._notification_items)

  def highest_severity_icon(self) -> rl.Texture | None:
    item = max((item for item in self._notification_items + self.alert_items if item.alert_data.visible),
               key=lambda item: (item.alert_data.severity, bool(item.alert_data.icon)), default=None)
    return item.icon if item is not None else None

  def set_pairing_callback(self, callback: Callable[[], None]):
    for alert_item in self.alert_items:
      if alert_item.alert_data.key == "Offroad_Pairing":
        alert_item.set_click_callback(callback)

  def _build_alerts(self):
    """Build sorted list of alerts from OFFROAD_ALERTS."""
    self.sorted_alerts = []

    # Add UpdateAvailable alert at the top (severity = -1 to indicate special handling)
    update_alert_data = AlertData(key="UpdateAvailable", text="", severity=-1)
    self.sorted_alerts.append(update_alert_data)
    update_alert_item = AlertItem(update_alert_data)
    update_alert_item.set_click_callback(lambda: HARDWARE.reboot())
    self.alert_items.append(update_alert_item)
    self._scroller.add_widget(update_alert_item)

    # Add regular alerts sorted by severity
    for key, config in sorted(OFFROAD_ALERTS.items(), key=lambda x: x[1].get("severity", 0), reverse=True):
      severity = config.get("severity", 0)
      alert_data = AlertData(key=key, text="", severity=severity, icon=config.get("icon"))
      self.sorted_alerts.append(alert_data)

      # Create alert item widget
      alert_item = AlertItem(alert_data)
      if key == "Offroad_ExcessiveActuation":
        alert_item.set_click_callback(self._confirm_uninstall)
      self.alert_items.append(alert_item)
      self._scroller.add_widget(alert_item)

  def _confirm_uninstall(self):
    icon = gui_app.texture("icons_mici/settings/device/uninstall.png", 64, 64)
    gui_app.push_widget(BigConfirmationDialog("slide to\nuninstall", icon, lambda: self.params.put_bool("DoUninstall", True, block=True),
                                              exit_on_confirm=False))

  def _params_worker(self):
    drop_realtime()
    while True:
      self._pending_params = ({"UpdaterNewDescription": self.params.get("UpdaterNewDescription")} |
                              {alert_data.key: self.params.get(alert_data.key) for alert_data in self.sorted_alerts})
      time.sleep(REFRESH_INTERVAL)

  def _sort_items(self):
    self._scroller.items.sort(key=lambda w: -w.alert_data.severity)

  def _refresh(self, pending_params: dict) -> int:
    """Refresh alerts from params and return active count."""
    active_count = 0

    # Handle UpdateAvailable alert specially
    update_available = pending_params["UpdateAvailable"]
    update_alert_data = next((alert_data for alert_data in self.sorted_alerts if alert_data.key == "UpdateAvailable"), None)

    if update_alert_data:
      if update_available:
        version_string = ""

        # Get new version description and parse version and date
        new_desc = pending_params["UpdaterNewDescription"] or ""
        if new_desc:
          # format: "version / branch / commit / date"
          parts = new_desc.split(" / ")
          if len(parts) > 3:
            version, date = parts[0], parts[3]
            version_string = f"\nopenpilot {version}, {date}\n"

        update_alert_data.text = f"Update available {version_string}. Click to update. Read the release notes at blog.comma.ai."
        update_alert_data.visible = True
        active_count += 1
      else:
        update_alert_data.text = ""
        update_alert_data.visible = False

    # Handle regular alerts
    for alert_data in self.sorted_alerts:
      if alert_data.key == "UpdateAvailable":
        continue  # Skip, already handled above

      text = ""
      alert_json = pending_params[alert_data.key]

      if alert_json:
        text = alert_json.get("text", "").replace("%1", alert_json.get("extra", ""))

      if text and not alert_data.visible:
        # Bump newly visible alerts to the top, severity sort keeps it at the top of its category
        widget = next(w for w in self._scroller.items if w.alert_data is alert_data)
        self._scroller.items.remove(widget)
        self._scroller.items.insert(0, widget)
      alert_data.text = text
      alert_data.visible = bool(text)

      if alert_data.visible:
        active_count += 1

    # Update alert items (they reference the same alert_data objects)
    for alert_item in self.alert_items:
      alert_item.update_alert_data(alert_item.alert_data)

    self._sort_items()

    return active_count

  def _refresh_notifications(self):
    notifications = ui_state.notifications.active
    if notifications == self._notifications:
      return
    self._notifications = notifications

    for item in self._notification_items:
      self._scroller.items.remove(item)
    self._notification_items = [NotificationItem(n) for n in notifications]
    self._scroller.add_widgets(self._notification_items)
    self._sort_items()

  def _refresh_state(self):
    self._refresh_notifications()

    # Refresh alerts when thread updates params
    pending_params = self._pending_params
    if pending_params is not None:
      self._refresh(pending_params)
      self._pending_params = None

  def _render(self, rect: rl.Rectangle):
    """Render the alerts scroller or empty state."""
    if self.active_alerts() == 0:
      self._empty_label.render(rect)
    else:
      self._scroller.render(rect)
