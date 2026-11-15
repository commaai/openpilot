import threading
import raylib_nb as rl
from collections.abc import Callable
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget

SPINNER_SIZE = 120
SPACING = 30
FONT_SIZE = 48
DEGREES_PER_SECOND = 360.0


class PowerScreen(Widget):
  """Full screen spinner that takes all input while the device reboots or powers off."""

  def __init__(self, text: str):
    super().__init__()
    self._text = text
    self._comma = gui_app.texture("images/spinner_comma.png", SPINNER_SIZE, SPINNER_SIZE)
    self._track = gui_app.texture("images/spinner_track.png", SPINNER_SIZE, SPINNER_SIZE, alpha_premultiply=True)
    self._rotation = 0.0

  @property
  def covers_screen(self) -> bool:
    return True

  def _render(self, rect: rl.Rectangle):
    font = gui_app.font(FontWeight.DISPLAY)
    text_size = measure_text_cached(font, self._text, FONT_SIZE)
    x = rect.x + (rect.width - SPINNER_SIZE - SPACING - text_size.x) / 2
    center = rl.Vector2(x + SPINNER_SIZE / 2, rect.y + rect.height / 2)

    self._rotation = (self._rotation + DEGREES_PER_SECOND * rl.get_frame_time()) % 360.0
    rl.draw_texture_pro(self._track, rl.Rectangle(0, 0, SPINNER_SIZE, SPINNER_SIZE), rl.Rectangle(center.x, center.y, SPINNER_SIZE, SPINNER_SIZE),
                        rl.Vector2(SPINNER_SIZE / 2, SPINNER_SIZE / 2), self._rotation, rl.WHITE)
    rl.draw_texture_v(self._comma, rl.Vector2(x, center.y - SPINNER_SIZE / 2), rl.WHITE)
    rl.draw_text_ex(font, self._text, rl.Vector2(x + SPINNER_SIZE + SPACING, center.y - text_size.y / 2), FONT_SIZE, 0, rl.WHITE)


def power_action(text: str, action: Callable[[], None]):
  # show the spinner right away, the action blocks (params fsync, sudo) and the UI is killed soon after
  gui_app.push_widget(PowerScreen(text))
  threading.Thread(target=action, daemon=True).start()
