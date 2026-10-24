import pyray as rl

from openpilot.common.swaglog import cloudlog
from openpilot.system.ui.lib.application import FontWeight
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import Label
from openpilot.common.qrcode import make_texture


class QR(Widget):
  def __init__(self, url: str, width: int = 170):
    super().__init__()
    self._url = url

    self._texture = self._generate_qr_code()

    self.set_rect(rl.Rectangle(0, 0, width, width))
    self._error = Label("QR Code Error", font_size=30, font_weight=FontWeight.BOLD, text_color=rl.RED)

  def _generate_qr_code(self):
    try:
      return make_texture(self._url, inverted=True)
    except Exception as e:
      cloudlog.warning(f"QR code generation failed: {e}")
      return None

  def _render(self, rect: rl.Rectangle):
    if not self._texture:
      self._error.render(rect)
      return
    scale = rect.height / self._texture.height
    pos = rl.Vector2(round(rect.x), round(rect.y))
    rl.draw_texture_ex(self._texture, pos, 0.0, scale, rl.WHITE)

  def __del__(self):
    if self._texture and self._texture.id != 0:
      rl.unload_texture(self._texture)
