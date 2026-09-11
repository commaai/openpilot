from collections.abc import Callable
import pyray as rl
import time

from openpilot.common.swaglog import cloudlog
from openpilot.system.ui.lib.application import FontWeight
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import Label
from openpilot.common.qrcode import make_texture

MIN_PIXELS_PER_MODULE = 3

class QR(Widget):
  def __init__(self, get_url: Callable[[], str], refresh_interval: int | None = None):
    super().__init__()
    self._get_url = get_url
    self._refresh_interval = refresh_interval
    self._last_pairing_qr_generation = float("-inf")

    self._texture = self._generate_qr_code()
    modules = self._texture.width // 10
    size = 170 if 170 / modules >= MIN_PIXELS_PER_MODULE else 240

    self.set_rect(rl.Rectangle(0, 0, size, size))
    self._error = Label("QR Code Error", font_size=30, font_weight=FontWeight.BOLD, text_color=rl.RED)

  def _check_qr_refresh(self) -> None:
    current_time = time.monotonic()
    if current_time - self._last_pairing_qr_generation >= self._refresh_interval:
      if self._texture and self._texture.id != 0:
        rl.unload_texture(self._texture)
      self._texture = self._generate_qr_code()
      self._last_pairing_qr_generation = current_time

  def _generate_qr_code(self):
    try:
      return make_texture(self._get_url(), inverted=True)
    except Exception as e:
      cloudlog.warning(f"QR code generation failed: {e}")
      return None

  def _render(self, rect: rl.Rectangle):
    if self._refresh_interval:
      self._check_qr_refresh()
    if not self._texture:
      self._error.render(rect)
      return
    scale = rect.height / self._texture.height
    pos = rl.Vector2(round(rect.x), round(rect.y))
    rl.draw_texture_ex(self._texture, pos, 0.0, scale, rl.WHITE)

  def __del__(self):
    if self._texture and self._texture.id != 0:
      rl.unload_texture(self._texture)