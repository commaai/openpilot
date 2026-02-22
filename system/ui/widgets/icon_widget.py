import pyray as rl
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget


class ImageIcon(Widget):
  def __init__(self, image_path: str, size: tuple[int, int]):
    super().__init__()
    self._texture = gui_app.texture(image_path, size[0], size[1])
    self.set_rect(rl.Rectangle(0, 0, float(size[0]), float(size[1])))
    self.set_enabled(False)

  def _render(self, _) -> None:
    rl.draw_texture_ex(self._texture, rl.Vector2(self._rect.x, self._rect.y), 0.0, 1.0, rl.WHITE)
