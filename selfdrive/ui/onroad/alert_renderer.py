import time
import pyray as rl
from dataclasses import dataclass
from cereal import messaging, log
from openpilot.system.hardware import TICI
from openpilot.system.ui.lib.application import gui_app, FontWeight, DEFAULT_FPS
from openpilot.system.ui.lib.label import gui_text_box
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.selfdrive.ui.ui_state import ui_state


ALERT_MARGIN = 40
ALERT_PADDING = 60
ALERT_LINE_SPACING = 45
ALERT_BORDER_RADIUS = 30

ALERT_FONT_SMALL = 66
ALERT_FONT_MEDIUM = 74
ALERT_FONT_BIG = 88

SELFDRIVE_STATE_TIMEOUT = 5  # Seconds
SELFDRIVE_UNRESPONSIVE_TIMEOUT = 10  # Seconds


# Constants
ALERT_COLORS = {
  log.SelfdriveState.AlertStatus.normal: rl.Color(0, 0, 0, 235),  # Black
  log.SelfdriveState.AlertStatus.userPrompt: rl.Color(0xFE, 0x8C, 0x34, 235),  # Orange
  log.SelfdriveState.AlertStatus.critical: rl.Color(0xC9, 0x22, 0x31, 235),  # Red
}


@dataclass
class Alert:
  text1: str = ""
  text2: str = ""
  size: int = 0
  status: int = 0


# Pre-defined alert instances
ALERT_STARTUP_PENDING = Alert(
  text1="openpilot Unavailable",
  text2="Waiting to start",
  size=log.SelfdriveState.AlertSize.mid,
  status=log.SelfdriveState.AlertStatus.normal,
)

ALERT_CRITICAL_TIMEOUT = Alert(
  text1="TAKE CONTROL IMMEDIATELY",
  text2="System Unresponsive",
  size=log.SelfdriveState.AlertSize.full,
  status=log.SelfdriveState.AlertStatus.critical,
)

ALERT_CRITICAL_REBOOT = Alert(
  text1="System Unresponsive",
  text2="Reboot Device",
  size=log.SelfdriveState.AlertSize.full,
  status=log.SelfdriveState.AlertStatus.critical,
)


class AlertRenderer:
  def __init__(self):
    self.font_regular: rl.Font = gui_app.font(FontWeight.NORMAL)
    self.font_bold: rl.Font = gui_app.font(FontWeight.BOLD)

  def get_alert(self, sm: messaging.SubMaster) -> Alert | None:
    """Generate the current alert based on selfdrive state."""
    ss = sm['selfdriveState']

    # Check if selfdriveState messages have stopped arriving
    if not sm.updated['selfdriveState']:
      recv_frame = sm.recv_frame['selfdriveState']
      if (sm.frame - recv_frame) > 5 * DEFAULT_FPS:
        # Check if waiting to start
        if recv_frame < ui_state.started_frame:
          return ALERT_STARTUP_PENDING

        # Handle selfdrive timeout
        if TICI:
          ss_missing = time.monotonic() - sm.recv_time['selfdriveState']
          if ss_missing > SELFDRIVE_STATE_TIMEOUT:
            if ss.enabled and (ss_missing - SELFDRIVE_STATE_TIMEOUT) < SELFDRIVE_UNRESPONSIVE_TIMEOUT:
              return ALERT_CRITICAL_TIMEOUT
            return ALERT_CRITICAL_REBOOT

    # No alert if size is none
    if ss.alertSize == 0:
      return None

    # Return current alert
    return Alert(text1=ss.alertText1, text2=ss.alertText2, size=ss.alertSize, status=ss.alertStatus)

  def draw(self, rect: rl.Rectangle, sm: messaging.SubMaster) -> None:
    alert = self.get_alert(sm)
    if not alert:
      return

    alert_rect = self._get_alert_rect(rect, alert.size)
    self._draw_background(alert_rect, alert)

    text_rect = rl.Rectangle(
      alert_rect.x + ALERT_PADDING,
      alert_rect.y + ALERT_PADDING,
      alert_rect.width - 2 * ALERT_PADDING,
      alert_rect.height - 2 * ALERT_PADDING
    )
    self._draw_text(text_rect, alert)

  def _get_alert_rect(self, rect: rl.Rectangle, size: int) -> rl.Rectangle:
    if size == log.SelfdriveState.AlertSize.full:
      return rect

    height = (ALERT_FONT_MEDIUM + 2 * ALERT_PADDING if size == log.SelfdriveState.AlertSize.small else
             ALERT_FONT_BIG + ALERT_LINE_SPACING + ALERT_FONT_SMALL + 2 * ALERT_PADDING)

    return rl.Rectangle(
      rect.x + ALERT_MARGIN,
      rect.y + rect.height - ALERT_MARGIN - height,
      rect.width - 2 * ALERT_MARGIN,
      height
    )

  def _draw_background(self, rect: rl.Rectangle, alert: Alert) -> None:
    color = ALERT_COLORS.get(alert.status, ALERT_COLORS[log.SelfdriveState.AlertStatus.normal])

    if alert.size != log.SelfdriveState.AlertSize.full:
      roundness = ALERT_BORDER_RADIUS / (min(rect.width, rect.height) / 2)
      rl.draw_rectangle_rounded(rect, roundness, 10, color)
    else:
      rl.draw_rectangle_rec(rect, color)

  def _draw_text(self, rect: rl.Rectangle, alert: Alert) -> None:
    if alert.size == log.SelfdriveState.AlertSize.small:
      self._draw_centered(alert.text1, rect, self.font_bold, ALERT_FONT_MEDIUM)

    elif alert.size == log.SelfdriveState.AlertSize.mid:
      self._draw_centered(alert.text1, rect, self.font_bold, ALERT_FONT_BIG, center_y=False)
      rect.y += ALERT_FONT_BIG + ALERT_LINE_SPACING
      self._draw_centered(alert.text2, rect, self.font_regular, ALERT_FONT_SMALL, center_y=False)

    else:
      is_long = len(alert.text1) > 15
      font_size1 = 132 if is_long else 177
      align_ment = rl.GuiTextAlignment.TEXT_ALIGN_CENTER
      vertical_align = rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE
      text_rect = rl.Rectangle(rect.x, rect.y, rect.width, rect.height // 2)

      gui_text_box(text_rect, alert.text1, font_size1, alignment=align_ment, alignment_vertical=vertical_align, font_weight=FontWeight.BOLD)
      text_rect.y = rect.y + rect.height // 2
      gui_text_box(text_rect, alert.text2, ALERT_FONT_BIG, alignment=align_ment)

  def _draw_centered(self, text, rect, font, font_size, center_y=True, color=rl.WHITE) -> None:
    text_size = measure_text_cached(font, text, font_size)
    x = rect.x + (rect.width - text_size.x) / 2
    y = rect.y + ((rect.height - text_size.y) / 2 if center_y else 0)
    rl.draw_text_ex(font, text, rl.Vector2(x, y), font_size, 0, color)
