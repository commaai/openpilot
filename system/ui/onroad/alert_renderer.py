import numpy as np
import pyray as rl
from dataclasses import dataclass
from cereal import messaging, log
from openpilot.system.ui.lib.application import gui_app, FontWeight

# Constants
ALERT_COLORS = {
  log.SelfdriveState.AlertStatus.normal: rl.Color(0, 0, 0, 150),  # Black
  log.SelfdriveState.AlertStatus.userPrompt: rl.Color(0xFE, 0x8C, 0x34, 100),  # Orange
  log.SelfdriveState.AlertStatus.critical: rl.Color(0xC9, 0x22, 0x31, 150),  # Red
}

ALERT_HEIGHTS = {
  log.SelfdriveState.AlertSize.small: 271,
  log.SelfdriveState.AlertSize.mid: 420,
}

SELFDRIVE_STATE_TIMEOUT = 5  # Seconds
SELFDRIVE_UNRESPONSIVE_TIMEOUT = 10  # Seconds


@dataclass
class Alert:
  text1: str = ""
  text2: str = ""
  alert_type: str = ""
  size: log.SelfdriveState.AlertSize = log.SelfdriveState.AlertSize.none
  status: log.SelfdriveState.AlertStatus = log.SelfdriveState.AlertStatus.normal

  def is_equal(self, other: 'Alert') -> bool:
    """Check if two alerts are equal."""
    return (
      self.text1 == other.text1
      and self.text2 == other.text2
      and self.alert_type == other.alert_type
      and self.size == other.size
      and self.status == other.status
    )


class AlertRenderer:
  def __init__(self):
    """Initialize the alert renderer."""
    self.alert: Alert = Alert()
    self.started_frame: int = 0
    self.font_regular: rl.Font = gui_app.font(FontWeight.NORMAL)
    self.font_bold: rl.Font = gui_app.font(FontWeight.BOLD)
    self.font_metrics_cache: dict[tuple[str, int, str], rl.Vector2] = {}

  def clear(self) -> None:
    """Reset the alert to its default state."""
    self.alert = Alert()

  def update_state(self, sm: messaging.SubMaster, started_frame: int) -> None:
    """Update alert state based on SubMaster data."""
    self.started_frame = started_frame
    new_alert = self.get_alert(sm)
    if not self.alert.is_equal(new_alert):
      self.alert = new_alert

  def get_alert(self, sm: messaging.SubMaster) -> Alert:
    """Generate the current alert based on selfdrive state."""
    if not sm.valid['selfdriveState']:
      return Alert()

    ss = sm['selfdriveState']
    selfdrive_frame = sm.recv_frame['selfdriveState']
    alert_status = self._get_enum_value(ss.alertStatus, log.SelfdriveState.AlertStatus)

    # Return current alert if selfdrive state is recent
    if selfdrive_frame >= self.started_frame:
      return Alert(
        text1=ss.alertText1,
        text2=ss.alertText2,
        alert_type=ss.alertType,
        size=self._get_enum_value(ss.alertSize, log.SelfdriveState.AlertSize),
        status=alert_status,
      )

    # Handle selfdrive timeout
    ss_missing = (np.uint64(rl.get_time() * 1e9) - sm.recv_time['selfdriveState']) / 1e9
    if selfdrive_frame < self.started_frame:
      return Alert(
        text1="openpilot Unavailable",
        text2="Waiting to start",
        alert_type="selfdriveWaiting",
        size=log.SelfdriveState.AlertSize.mid,
        status=log.SelfdriveState.AlertStatus.normal,
      )
    elif ss_missing > SELFDRIVE_STATE_TIMEOUT:
      if ss.enabled and (ss_missing - SELFDRIVE_STATE_TIMEOUT) < SELFDRIVE_UNRESPONSIVE_TIMEOUT:
        return Alert(
          text1="TAKE CONTROL IMMEDIATELY",
          text2="System Unresponsive",
          alert_type="selfdriveUnresponsive",
          size=log.SelfdriveState.AlertSize.full,
          status=log.SelfdriveState.AlertStatus.critical,
        )
      return Alert(
        text1="System Unresponsive",
        text2="Reboot Device",
        alert_type="selfdriveUnresponsivePermanent",
        size=log.SelfdriveState.AlertSize.mid,
        status=log.SelfdriveState.AlertStatus.normal,
      )

    return Alert()

  def draw(self, rect: rl.Rectangle, sm: messaging.SubMaster) -> None:
    """Render the alert within the specified rectangle."""
    self.update_state(sm, sm.recv_frame['selfdriveState'])
    alert_size = self._get_enum_value(self.alert.size, log.SelfdriveState.AlertSize)
    if alert_size == log.SelfdriveState.AlertSize.none:
      return

    # Calculate alert rectangle
    margin = 0 if alert_size == log.SelfdriveState.AlertSize.full else 40
    radius = 0 if alert_size == log.SelfdriveState.AlertSize.full else 30
    height = ALERT_HEIGHTS.get(alert_size, rect.height)
    alert_rect = rl.Rectangle(
      rect.x + margin,
      rect.y + rect.height - height + margin,
      rect.width - margin * 2,
      height - margin * 2,
    )

    # Draw background
    alert_status = self._get_enum_value(self.alert.status, log.SelfdriveState.AlertStatus)
    color = ALERT_COLORS.get(alert_status, ALERT_COLORS[log.SelfdriveState.AlertStatus.normal])
    if alert_size != log.SelfdriveState.AlertSize.full:
      roundness = radius / (min(alert_rect.width, alert_rect.height) / 2)
      rl.draw_rectangle_rounded(alert_rect, roundness, 10, color)
    else:
      rl.draw_rectangle_rec(alert_rect, color)

    # Draw text
    center_x = rect.x + rect.width / 2
    center_y = alert_rect.y + alert_rect.height / 2
    self._draw_text(alert_size, alert_rect, center_x, center_y)

  def _draw_text(
    self, alert_size: log.SelfdriveState.AlertSize, alert_rect: rl.Rectangle, center_x: float, center_y: float
  ) -> None:
    """Draw text based on alert size."""
    if alert_size == log.SelfdriveState.AlertSize.small:
      font_size = 74
      text_width = self._measure_text(self.font_bold, self.alert.text1, font_size, 'bold').x
      rl.draw_text_ex(
        self.font_bold,
        self.alert.text1,
        rl.Vector2(center_x - text_width / 2, center_y - font_size / 2),
        font_size,
        0,
        rl.WHITE,
      )
    elif alert_size == log.SelfdriveState.AlertSize.mid:
      font_size1 = 88
      text1_width = self._measure_text(self.font_bold, self.alert.text1, font_size1, 'bold').x
      rl.draw_text_ex(
        self.font_bold,
        self.alert.text1,
        rl.Vector2(center_x - text1_width / 2, center_y - 125),
        font_size1,
        0,
        rl.WHITE,
      )
      font_size2 = 66
      text2_width = self._measure_text(self.font_regular, self.alert.text2, font_size2, 'regular').x
      rl.draw_text_ex(
        self.font_regular,
        self.alert.text2,
        rl.Vector2(center_x - text2_width / 2, center_y + 21),
        font_size2,
        0,
        rl.WHITE,
      )
    elif alert_size == log.SelfdriveState.AlertSize.full:
      is_long = len(self.alert.text1) > 15
      font_size1 = 132 if is_long else 177
      text1_y = alert_rect.y + (240 if is_long else 270)
      wrapped_text1 = self._wrap_text(self.alert.text1, alert_rect.width - 100, font_size1, self.font_bold)
      for i, line in enumerate(wrapped_text1):
        line_width = self._measure_text(self.font_bold, line, font_size1, 'bold').x
        rl.draw_text_ex(
          self.font_bold,
          line,
          rl.Vector2(center_x - line_width / 2, text1_y + i * font_size1),
          font_size1,
          0,
          rl.WHITE,
        )
      font_size2 = 88
      text2_y = alert_rect.y + alert_rect.height - (361 if is_long else 420)
      wrapped_text2 = self._wrap_text(self.alert.text2, alert_rect.width - 100, font_size2, self.font_regular)
      for i, line in enumerate(wrapped_text2):
        line_width = self._measure_text(self.font_regular, line, font_size2, 'regular').x
        rl.draw_text_ex(
          self.font_regular,
          line,
          rl.Vector2(center_x - line_width / 2, text2_y + i * font_size2),
          font_size2,
          0,
          rl.WHITE,
        )

  def _wrap_text(self, text: str, max_width: float, font_size: int, font: rl.Font) -> list[str]:
    """Wrap text to fit within max width."""
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
      test_line = f"{current_line} {word}" if current_line else word
      if self._measure_text(font, test_line, font_size, 'bold' if font == self.font_bold else 'regular').x <= max_width:
        current_line = test_line
      else:
        if current_line:
          lines.append(current_line)
        current_line = word
    if current_line:
      lines.append(current_line)
    return lines

  def _measure_text(self, font: rl.Font, text: str, font_size: int, font_type: str) -> rl.Vector2:
    """Measure text dimensions with caching."""
    key = (text, font_size, font_type)
    if key not in self.font_metrics_cache:
      self.font_metrics_cache[key] = rl.measure_text_ex(font, text, font_size, 0)
    return self.font_metrics_cache[key]

  @staticmethod
  def _get_enum_value(enum_value, enum_type: type):
    """Safely convert capnp enum to Python enum value."""
    return enum_value.raw if hasattr(enum_value, 'raw') else enum_value
