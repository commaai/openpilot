import numpy as np
import pyray as rl
from enum import IntEnum
from dataclasses import dataclass
from cereal import messaging, log
from openpilot.system.ui.lib.application import DEFAULT_FPS
from openpilot.system.ui.lib.application import gui_app, FontWeight


# Alert sizes
AlertSize = log.SelfdriveState.AlertSize
AlertStatus = log.SelfdriveState.AlertStatus


ALERT_COLORS = {
  AlertStatus.normal: rl.Color(0x81, 0x86, 0x8C, 255),  # Gray
  AlertStatus.userPrompt: rl.Color(0xFE, 0x8C, 0x34, 255),  # Orange
  AlertStatus.critical: rl.Color(0xC9, 0x22, 0x31, 255),  # Red
}


@dataclass
class Alert:
  text1: str = ""
  text2: str = ""
  alert_type: str = ""
  size: log.SelfdriveState.AlertSize = AlertSize.none
  status: log.SelfdriveState.AlertStatus = AlertStatus.normal

  def is_equal(self, other):
    """Check if alerts are equal"""
    return (
      self.text1 == other.text1
      and self.text2 == other.text2
      and self.alert_type == other.alert_type
      and self.size == other.size
      and self.status == other.status
    )


class AlertRenderer:
  def __init__(self):
    self.alert = Alert()
    self.started_frame = 0
    self.font_regular = gui_app.font(FontWeight.NORMAL)
    self.font_bold = gui_app.font(FontWeight.BOLD)
    self.font_metrics_cache = {}

  def clear(self):
    """Reset the alert"""
    self.alert = Alert()

  def update_state(self, sm, started_frame):
    """Update alert state based on SubMaster data"""
    a = self.get_alert(sm, started_frame)
    if not self.alert.is_equal(a):
      self.alert = a

  def get_alert(self, sm, started_frame):
    """Get the current alert based on selfdrive state"""
    # Default empty alert
    started_frame = 0
    a = Alert()

    if not sm.valid['selfdriveState']:
      return a

    # Get selfdrive state
    ss = sm['selfdriveState']
    selfdrive_frame = sm.recv_frame['selfdriveState']

    # Don't show old alerts
    alert_status = ss.alertStatus
    status_value = int(alert_status) if isinstance(alert_status, int) else alert_status.raw
    if True:  # selfdrive_frame >= started_frame:
      a = Alert(
        text1=ss.alertText1,
        text2=ss.alertText2,
        alert_type=ss.alertType,
        size=ss.alertSize,
        status=status_value,
      )

    # Handle selfdrive timeout
    if not sm.updated['selfdriveState'] and (sm.frame - started_frame) > 5 * DEFAULT_FPS:
      SELFDRIVE_STATE_TIMEOUT = 5
      ss_missing = (np.uint64(rl.get_time() * 1e9) - sm.recv_time['selfdriveState']) / 1e9

      if selfdrive_frame < started_frame:
        # Car started but selfdrive not seen
        a = Alert(
          text1="openpilot Unavailable",
          text2="Waiting to start",
          alert_type="selfdriveWaiting",
          size=AlertSize.mid,
          status=AlertStatus.normal,
        )
      elif ss_missing > SELFDRIVE_STATE_TIMEOUT:
        # Selfdrive lagging or died
        if ss.enabled and (ss_missing - SELFDRIVE_STATE_TIMEOUT) < 10:
          a = Alert(
            text1="TAKE CONTROL IMMEDIATELY",
            text2="System Unresponsive",
            alert_type="selfdriveUnresponsive",
            size=AlertSize.full,
            status=AlertStatus.critical,
          )
        else:
          a = Alert(
            text1="System Unresponsive",
            text2="Reboot Device",
            alert_type="selfdriveUnresponsivePermanent",
            size=AlertSize.mid,
            status=AlertStatus.normal,
          )

    return a

  def draw(self, rect: rl.Rectangle, sm: messaging.SubMaster):
    """Draw the alert within the specified rectangle"""
    self.update_state(sm, sm.recv_frame['selfdriveState'])

    # Convert capnp enum to Python enum if needed
    alert_size = self.alert.size
    if hasattr(alert_size, 'raw'):
      alert_size = alert_size.raw

    if alert_size == AlertSize.none:
      return

    # Match C++ alert height calculations exactly
    alert_heights = {AlertSize.small: 271, AlertSize.mid: 420, AlertSize.full: rect.height}
    h = alert_heights.get(alert_size, 0)

    # Calculate margins and radius
    margin = 40
    radius = 30

    if alert_size == AlertSize.full:
      margin = 0
      radius = 0

    # Create alert rectangle within the provided rectangle
    alert_rect = rl.Rectangle(
      rect.x + margin, rect.y + rect.height - h + margin, rect.width - margin * 2, h - margin * 2
    )

    # Get alert status
    alert_status = self.alert.status
    if hasattr(alert_status, 'raw'):
      alert_status = alert_status.raw

    # Draw background with gradient
    color = ALERT_COLORS.get(alert_status, ALERT_COLORS[AlertStatus.normal])

    # Draw main background
    rl.draw_rectangle_rounded(alert_rect, radius / alert_rect.width, 10, color)

    # Draw gradient overlay - match C++ gradient exactly
    rl.draw_rectangle_gradient_v(
      int(alert_rect.x),
      int(alert_rect.y),
      int(alert_rect.width),
      int(alert_rect.height),
      rl.Color(0, 0, 0, 13),  # 0.05 alpha at top (0.05 * 255 = ~13)
      rl.Color(0, 0, 0, 89),  # 0.35 alpha at bottom (0.35 * 255 = ~89)
    )

    # Center point for text positioning
    center_x = rect.x + rect.width / 2
    center_y = alert_rect.y + alert_rect.height / 2

    # Match C++ text rendering precisely
    if alert_size == AlertSize.small:
      font_size = 74
      text_width = self._measure_text(self.font_bold, self.alert.text1, font_size).x
      # Center text exactly as in C++
      rl.draw_text_ex(
        self.font_bold,
        self.alert.text1,
        rl.Vector2(center_x - text_width / 2, center_y - font_size / 2),
        font_size,
        0,
        rl.WHITE,
      )

    elif alert_size == AlertSize.mid:
      # Main text (title)
      font_size1 = 88
      text1_width = rl.measure_text_ex(self.font_bold, self.alert.text1, font_size1, 0).x
      text1_y = center_y - 125  # Exact offset from C++

      rl.draw_text_ex(
        self.font_bold, self.alert.text1, rl.Vector2(center_x - text1_width / 2, text1_y), font_size1, 0, rl.WHITE
      )

      # Subtitle text
      font_size2 = 66
      text2_width = rl.measure_text_ex(self.font_regular, self.alert.text2, font_size2, 0).x
      text2_y = center_y + 21  # Exact offset from C++

      rl.draw_text_ex(
        self.font_regular, self.alert.text2, rl.Vector2(center_x - text2_width / 2, text2_y), font_size2, 0, rl.WHITE
      )

    elif alert_size == AlertSize.full:
      # Determine font size based on text length
      is_long = len(self.alert.text1) > 15
      font_size1 = 132 if is_long else 177
      text1_y = alert_rect.y + (240 if is_long else 270)  # Match C++ positioning

      # Handle text wrapping for title
      wrapped_text1 = self._wrap_text(self.alert.text1, rect.width - 100, font_size1, self.font_bold)
      text1_height = self._get_text_height(wrapped_text1, font_size1, self.font_bold)

      # Draw title text
      for i, line in enumerate(wrapped_text1):
        line_width = rl.measure_text_ex(self.font_bold, line, font_size1, 0).x
        line_y = text1_y + i * font_size1
        rl.draw_text_ex(self.font_bold, line, rl.Vector2(center_x - line_width / 2, line_y), font_size1, 0, rl.WHITE)

      # Subtitle
      font_size2 = 88
      text2_y = rect.y + rect.height - (361 if is_long else 420)  # Match C++ positioning

      # Handle text wrapping for subtitle
      wrapped_text2 = self._wrap_text(self.alert.text2, rect.width - 100, font_size2, self.font_regular)

      # Draw subtitle text
      for i, line in enumerate(wrapped_text2):
        line_width = rl.measure_text_ex(self.font_regular, line, font_size2, 0).x
        line_y = text2_y + i * font_size2
        rl.draw_text_ex(self.font_regular, line, rl.Vector2(center_x - line_width / 2, line_y), font_size2, 0, rl.WHITE)

  def _wrap_text(self, text, max_width, font_size, font):
    """Wrap text to fit within max width"""
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
      test_line = current_line + " " + word if current_line else word
      text_width = rl.measure_text_ex(font, test_line, font_size, 0).x

      if text_width <= max_width:
        current_line = test_line
      else:
        if current_line:
          lines.append(current_line)
        current_line = word

    if current_line:
      lines.append(current_line)

    return lines

  def _get_text_height(self, lines, font_size, font):
    """Calculate total height of wrapped text"""
    return len(lines) * font_size

  def _measure_text(self, font: rl.Font, text: str, font_size: int) -> rl.Vector2:
    """Get text metrics for a given text and font weight with caching"""
    key = (text, font_size)
    if key not in self.font_metrics_cache:
      metrics = rl.measure_text_ex(font, text, font_size, 0)
      self.font_metrics_cache[key] = metrics

    return self.font_metrics_cache[key]
