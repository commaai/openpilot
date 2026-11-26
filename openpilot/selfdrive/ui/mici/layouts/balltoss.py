import json
from pathlib import Path
import time
import raylib_nb as rl

from openpilot.system.ui.lib.application import FontWeight, TextAlignment, TextAlignmentVertical, gui_app
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import gui_label

DISPLAY_PATH = Path("/tmp/balltoss_display.json")
GREEN = rl.Color(81, 255, 0, 255)
RED = rl.Color(255, 48, 48, 255)
DIM = rl.Color(150, 150, 150, 255)
DARK = rl.Color(30, 30, 30, 255)
IDLE_TEXT = rl.Color(210, 210, 210, 255)


class BalltossDisplay(Widget):
  def __init__(self):
    super().__init__()
    self._state: dict = {}
    self._last_read = 0.
    self._last_mtime = 0
    self.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
    gui_app.push_widget(self)

  @property
  def covers_screen(self) -> bool:
    return True

  def _refresh(self) -> None:
    now = time.monotonic()
    if now - self._last_read < .25:
      return
    self._last_read = now
    try:
      mtime = DISPLAY_PATH.stat().st_mtime_ns
      if mtime != self._last_mtime:
        self._state = json.loads(DISPLAY_PATH.read_text())
        self._last_mtime = mtime
    except (OSError, ValueError):
      pass

  def _render(self, rect: rl.Rectangle) -> None:
    self._refresh()
    rl.draw_rectangle_rec(rect, rl.BLACK)
    pad = 24
    w = rect.width - 2 * pad
    now = time.time()  # noqa: TID251
    state = self._state if now - self._state.get("updated_at", 0) < 30 else {}
    player = state.get("player")
    ends_at = state.get("ends_at") or 0

    down = state.get("down")
    if down:
      rl.draw_rectangle(0, 0, int(rect.width), 8, RED)
      gui_label(rl.Rectangle(pad, 20, w, 36), "BALLTOSS", 28, RED, FontWeight.BOLD, TextAlignment.LEFT, TextAlignmentVertical.MIDDLE)
      gui_label(rl.Rectangle(pad, 74, w, 80), "DOWN", 64, RED, FontWeight.BOLD, TextAlignment.LEFT, TextAlignmentVertical.MIDDLE)
      gui_label(rl.Rectangle(pad, 174, w, 44), str(down)[:34], 24, rl.WHITE, FontWeight.NORMAL, TextAlignment.LEFT, TextAlignmentVertical.MIDDLE)
    elif player and now < ends_at:
      rl.draw_rectangle(0, 0, int(rect.width), 6, GREEN)
      rem = max(0, int(ends_at - now))
      gui_label(rl.Rectangle(pad, 20, w * 0.6, 36), "PLAYING", 28, GREEN, FontWeight.BOLD, TextAlignment.LEFT, TextAlignmentVertical.MIDDLE)
      gui_label(rl.Rectangle(pad, 20, w, 36), f"{rem // 60}:{rem % 60:02d}", 30, rl.WHITE,
                FontWeight.SEMI_BOLD, TextAlignment.RIGHT, TextAlignmentVertical.MIDDLE)
      gui_label(rl.Rectangle(pad, 74, w, 80), str(player), 64, rl.WHITE, FontWeight.BOLD, TextAlignment.LEFT, TextAlignmentVertical.MIDDLE)
      score = state.get("score")
      score_text = f"Score {score:.3f}" if isinstance(score, (int, float)) else "balltoss"
      gui_label(rl.Rectangle(pad, 174, w, 44), score_text, 34, GREEN if (score or 0) > 0 else DIM,
                FontWeight.SEMI_BOLD, TextAlignment.LEFT, TextAlignmentVertical.MIDDLE)
    else:
      rl.draw_rectangle(0, 0, int(rect.width), 6, DARK)
      gui_label(rl.Rectangle(pad, 20, w, 36), "BALLTOSS", 28, DIM, FontWeight.BOLD, TextAlignment.LEFT, TextAlignmentVertical.MIDDLE)
      gui_label(rl.Rectangle(pad, 76, w, 78), "IDLE", 60, IDLE_TEXT, FontWeight.BOLD, TextAlignment.LEFT, TextAlignmentVertical.MIDDLE)
      nxt_player, nxt_start = state.get("next_player"), state.get("next_start")
      if nxt_player and nxt_start and nxt_start > now:
        wait = int(nxt_start - now)
        eta = f"{wait // 60}m" if wait >= 90 else f"{wait}s"
        footer = f"Next: {nxt_player} in {eta}"
      else:
        footer = "balltoss.comma.life"
      gui_label(rl.Rectangle(pad, 174, w, 44), footer, 28, DIM, FontWeight.NORMAL, TextAlignment.LEFT, TextAlignmentVertical.MIDDLE)
