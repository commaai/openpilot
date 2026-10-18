from __future__ import annotations

import pyray as rl

from openpilot.selfdrive.ui.tss3_oracle_runner import STAGE_LABELS, Tss3OracleRunner
from openpilot.selfdrive.ui.ui_state import device
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import Button, ButtonStyle

BACKGROUND = rl.Color(20, 20, 20, 255)
MUTED = rl.Color(170, 170, 170, 255)
ACCENT = rl.Color(70, 91, 234, 255)
SUCCESS = rl.Color(51, 171, 76, 255)
ERROR = rl.Color(226, 44, 44, 255)


class Tss3OracleBringupDialog(Widget):
  """Comma-three page for the shared TSS3 RAM-oracle startup flow."""

  def __init__(self):
    super().__init__()
    self._runner = Tss3OracleRunner()
    self._status = self._runner.snapshot()
    self._action_button = self._child(
      Button("CANCEL BRINGUP", click_callback=gui_app.pop_widget, font_size=42, button_style=ButtonStyle.DANGER, border_radius=55)
    )

  def show_event(self):
    super().show_event()
    device.set_override_interactive_timeout(300)

  def hide_event(self):
    self._runner.request_cancel()
    device.set_override_interactive_timeout(None)
    super().hide_event()

  def _update_state(self):
    self._status = self._runner.snapshot()
    stage = self._status["stage"]
    if stage in ("arming", "armed"):
      self._action_button.set_text("CANCEL BRINGUP")
      self._action_button.set_button_style(ButtonStyle.DANGER)
    else:
      self._action_button.set_text("CLOSE")
      self._action_button.set_button_style(ButtonStyle.NORMAL)

  @staticmethod
  def _draw_centered(font: rl.Font, text: str, y: float, size: int, color: rl.Color, rect: rl.Rectangle) -> float:
    measured = measure_text_cached(font, text, size)
    rl.draw_text_ex(font, text, rl.Vector2(rect.x + (rect.width - measured.x) / 2, y), size, 0, color)
    return measured.y

  @staticmethod
  def _draw_wrapped(font: rl.Font, text: str, x: float, y: float, width: int, size: int, color: rl.Color, max_lines: int) -> float:
    lines = wrap_text(font, text, size, width)
    if len(lines) > max_lines:
      lines = lines[:max_lines]
      lines[-1] = lines[-1].rstrip(".") + "…"
    rl.draw_text_ex(font, "\n".join(lines), rl.Vector2(x, y), size, 0, color)
    return len(lines) * size

  def _render(self, rect: rl.Rectangle):
    rl.draw_rectangle_rec(rect, BACKGROUND)

    status = self._status
    stage = status["stage"]
    progress = status["progress"]
    color = ERROR if status["error"] else SUCCESS if status["done"] else ACCENT
    title_font = gui_app.font(FontWeight.BOLD)
    body_font = gui_app.font(FontWeight.NORMAL)
    margin = 110
    content_width = int(rect.width - 2 * margin)

    y = rect.y + 65
    y += self._draw_centered(title_font, status["title"], y, 68, rl.WHITE, rect) + 30
    stage_text = f"{progress}%  |  {STAGE_LABELS.get(stage, stage)}"
    y += self._draw_centered(body_font, stage_text, y, 44, color, rect) + 28

    progress_rect = rl.Rectangle(rect.x + margin, y, content_width, 28)
    rl.draw_rectangle_rounded(progress_rect, 1.0, 12, rl.Color(57, 57, 57, 255))
    if progress > 0:
      filled = rl.Rectangle(progress_rect.x, progress_rect.y, progress_rect.width * progress / 100, progress_rect.height)
      rl.draw_rectangle_rounded(filled, 1.0, 12, color)
    y += 72

    y += self._draw_wrapped(body_font, status["detail"], rect.x + margin, y, content_width, 43, rl.WHITE, 5) + 52
    self._draw_wrapped(
      body_font,
      "RAM-only startup path. No EPS flash writes. Keep the vehicle in Park. "
      + "Cancellation is cooperative before PROGRAMMING; after that, closing this page lets bringup finish safely.",
      rect.x + margin,
      y,
      content_width,
      35,
      MUTED,
      3,
    )

    button_width = min(720, content_width)
    button_rect = rl.Rectangle(rect.x + (rect.width - button_width) / 2, rect.y + rect.height - 170, button_width, 110)
    self._action_button.render(button_rect)
