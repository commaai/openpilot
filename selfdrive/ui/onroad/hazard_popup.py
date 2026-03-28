import time
import pyray as rl
from collections.abc import Callable
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import Button, ButtonStyle

POPUP_TIMEOUT = 15.0  # seconds until auto-dismiss

POPUP_WIDTH = 1400
POPUP_HEIGHT = 520
MARGIN = 50
BUTTON_HEIGHT = 150
TIMER_BAR_HEIGHT = 14
TITLE_FONT_SIZE = 110

OVERLAY_COLOR = rl.Color(0, 0, 0, 160)
PANEL_COLOR = rl.Color(20, 20, 20, 235)
TIMER_BAR_BG_COLOR = rl.Color(70, 70, 70, 255)
TIMER_BAR_FILL_COLOR = rl.Color(230, 140, 30, 255)
BORDER_RADIUS = 20


class HazardPopup(Widget):
  """
  Onroad popup that asks the driver to confirm a detected road hazard.

  Shows "Hazard" text, Yes/No buttons, and a 15-second countdown bar.
  Auto-dismisses on timeout.

  Set a response_callback before pushing:
      popup.set_response_callback(fn)

  The callback receives (answer: str, latency_s: float) where answer is
  one of "yes", "no", or "timeout".
  """

  def __init__(self):
    super().__init__()
    self._response_callback: Callable[[str, float], None] | None = None
    self._start_time: float = 0.0

    self._no_button = self._child(Button("No", self._handle_no, button_style=ButtonStyle.NORMAL))
    self._yes_button = self._child(Button("Yes", self._handle_yes, button_style=ButtonStyle.DANGER))

  def set_response_callback(self, callback: Callable[[str, float], None] | None) -> None:
    self._response_callback = callback

  def show_event(self):
    super().show_event()
    self._start_time = time.monotonic()

  # ── private ────────────────────────────────────────────────────────────────

  def _elapsed(self) -> float:
    return time.monotonic() - self._start_time

  def _fire_response(self, answer: str) -> None:
    if self._response_callback is not None:
      self._response_callback(answer, self._elapsed())

  def _handle_yes(self):
    self._fire_response("yes")
    gui_app.pop_widget()

  def _handle_no(self):
    self._fire_response("no")
    gui_app.pop_widget()

  def _render(self, rect: rl.Rectangle):
    elapsed = self._elapsed()
    progress = max(0.0, 1.0 - elapsed / POPUP_TIMEOUT)

    # Auto-dismiss when timer expires
    if progress <= 0.0:
      self._fire_response("timeout")
      gui_app.pop_widget()
      return

    # Dim overlay behind the panel
    rl.draw_rectangle_rec(rect, OVERLAY_COLOR)

    # Center the panel on screen
    panel_x = rect.x + (rect.width - POPUP_WIDTH) / 2
    panel_y = rect.y + (rect.height - POPUP_HEIGHT) / 2
    panel_rect = rl.Rectangle(panel_x, panel_y, POPUP_WIDTH, POPUP_HEIGHT)

    roundness = BORDER_RADIUS / (min(POPUP_WIDTH, POPUP_HEIGHT) / 2)
    rl.draw_rectangle_rounded(panel_rect, roundness, 10, PANEL_COLOR)

    self._render_title(panel_x, panel_y)
    self._render_buttons(panel_x, panel_y)
    self._render_timer_bar(panel_x, panel_y, progress)

  def _render_title(self, panel_x: float, panel_y: float):
    font = gui_app.font(FontWeight.BOLD)
    text = "Hazard"
    text_size = measure_text_cached(font, text, TITLE_FONT_SIZE)

    # Vertically centered in the space above the buttons
    content_top = panel_y + MARGIN
    content_bottom = panel_y + POPUP_HEIGHT - MARGIN - BUTTON_HEIGHT - MARGIN - TIMER_BAR_HEIGHT - MARGIN
    title_y = content_top + (content_bottom - content_top - text_size.y) / 2

    title_x = panel_x + (POPUP_WIDTH - text_size.x) / 2
    rl.draw_text_ex(font, text, rl.Vector2(int(title_x), int(title_y)), TITLE_FONT_SIZE, 0, rl.WHITE)

  def _render_buttons(self, panel_x: float, panel_y: float):
    button_y = panel_y + POPUP_HEIGHT - MARGIN - TIMER_BAR_HEIGHT - MARGIN - BUTTON_HEIGHT
    button_w = (POPUP_WIDTH - 3 * MARGIN) / 2

    no_rect = rl.Rectangle(panel_x + MARGIN, button_y, button_w, BUTTON_HEIGHT)
    yes_rect = rl.Rectangle(panel_x + POPUP_WIDTH - button_w - MARGIN, button_y, button_w, BUTTON_HEIGHT)

    self._no_button.render(no_rect)
    self._yes_button.render(yes_rect)

  def _render_timer_bar(self, panel_x: float, panel_y: float, progress: float):
    bar_x = panel_x + MARGIN
    bar_y = panel_y + POPUP_HEIGHT - MARGIN - TIMER_BAR_HEIGHT
    bar_w = POPUP_WIDTH - 2 * MARGIN

    # Background track
    rl.draw_rectangle_rounded(rl.Rectangle(bar_x, bar_y, bar_w, TIMER_BAR_HEIGHT), 0.5, 10, TIMER_BAR_BG_COLOR)

    # Shrinking fill
    fill_w = bar_w * progress
    if fill_w >= 2:
      rl.draw_rectangle_rounded(rl.Rectangle(bar_x, bar_y, fill_w, TIMER_BAR_HEIGHT), 0.5, 10, TIMER_BAR_FILL_COLOR)
