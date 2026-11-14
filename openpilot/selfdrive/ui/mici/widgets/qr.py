import threading
from collections.abc import Callable

import numpy as np
import raylib_nb as rl

from openpilot.common.swaglog import cloudlog
from openpilot.system.ui.lib.application import FontWeight
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import Label
from openpilot.common.qrcode import make_pixels, pixels_texture


class QR(Widget):
  def __init__(self, url: str | Callable[[], str], width: int = 170):
    super().__init__()
    self._texture: rl.Texture | None = None
    self._pixels: np.ndarray | None = None
    self._failed = False

    self.set_rect(rl.Rectangle(0, 0, width, width))
    self._error = Label("QR Code Error", font_size=30, font_weight=FontWeight.BOLD, text_color=rl.RED)
    self.set_url(url)

  def set_url(self, url: str | Callable[[], str]):
    # making a code takes ~50 ms, a dropped frame, so it's made in the background. so is the url, if it's a function
    def generate():
      try:
        self._pixels = make_pixels(url() if callable(url) else url, inverted=True, module_size=4)
      except Exception as e:
        cloudlog.warning(f"QR code generation failed: {e}")
        self._failed = True
    threading.Thread(target=generate, daemon=True).start()

  def _render(self, rect: rl.Rectangle):
    if self._pixels is not None:
      pixels, self._pixels = self._pixels, None
      self._unload()
      self._texture = pixels_texture(pixels)
      self._failed = False

    if not self._texture:
      if self._failed:
        self._error.render(rect)
      return
    scale = rect.height / self._texture.height
    pos = rl.Vector2(round(rect.x), round(rect.y))
    rl.draw_texture_ex(self._texture, pos, 0.0, scale, rl.WHITE)

  def _unload(self):
    if self._texture and self._texture.id != 0:
      rl.unload_texture(self._texture)
    self._texture = None

  def __del__(self):
    self._unload()
