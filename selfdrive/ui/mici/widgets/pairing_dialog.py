import pyray as rl
import qrcode
import numpy as np
import time

from openpilot.common.api import Api
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.widgets import NavWidget
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.widgets.label import MiciLabel


class PairingDialog(NavWidget):
  """Dialog for device pairing with QR code."""

  QR_REFRESH_INTERVAL = 300  # 5 minutes in seconds

  def __init__(self):
    super().__init__()
    self.set_back_callback(lambda: gui_app.set_modal_overlay(None))
    self._params = Params()
    self._qr_texture: rl.Texture | None = None
    self._last_qr_generation = float("-inf")

    self._txt_pair = gui_app.texture("icons_mici/settings/device/pair.png", 33, 60)
    self._pair_label = MiciLabel("pair with comma connect", 48, font_weight=FontWeight.BOLD,
                                 color=rl.Color(255, 255, 255, int(255 * 0.9)), line_height=40, wrap_text=True)

  def _get_pairing_url(self) -> str:
    try:
      dongle_id = self._params.get("DongleId") or ""
      token = Api(dongle_id).get_token({'pair': True})
    except Exception as e:
      cloudlog.warning(f"Failed to get pairing token: {e}")
      token = ""
    return f"https://connect.comma.ai/?pair={token}"

  def _generate_qr_code(self) -> None:
    try:
      qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=0)
      qr.add_data(self._get_pairing_url())
      qr.make(fit=True)

      pil_img = qr.make_image(fill_color="white", back_color="black").convert('RGBA')
      img_array = np.array(pil_img, dtype=np.uint8)

      if self._qr_texture and self._qr_texture.id != 0:
        rl.unload_texture(self._qr_texture)

      rl_image = rl.Image()
      rl_image.data = rl.ffi.cast("void *", img_array.ctypes.data)
      rl_image.width = pil_img.width
      rl_image.height = pil_img.height
      rl_image.mipmaps = 1
      rl_image.format = rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8

      self._qr_texture = rl.load_texture_from_image(rl_image)
    except Exception as e:
      cloudlog.warning(f"QR code generation failed: {e}")
      self._qr_texture = None

  def _check_qr_refresh(self) -> None:
    current_time = time.monotonic()
    if current_time - self._last_qr_generation >= self.QR_REFRESH_INTERVAL:
      self._generate_qr_code()
      self._last_qr_generation = current_time

  def _update_state(self):
    super()._update_state()
    if ui_state.prime_state.is_paired():
      self._playing_dismiss_animation = True

  def _render(self, rect: rl.Rectangle) -> int:
    self._check_qr_refresh()

    self._render_qr_code()

    label_x = self._rect.x + 8 + self._rect.height + 24
    self._pair_label.set_width(int(self._rect.width - label_x))
    self._pair_label.set_position(label_x, self._rect.y + 16)
    self._pair_label.render()

    rl.draw_texture_ex(self._txt_pair, rl.Vector2(label_x, self._rect.y + self._rect.height - self._txt_pair.height - 16),
                       0.0, 1.0, rl.Color(255, 255, 255, int(255 * 0.35)))

    return -1

  def _render_qr_code(self) -> None:
    if not self._qr_texture:
      error_font = gui_app.font(FontWeight.BOLD)
      rl.draw_text_ex(
        error_font, "QR Code Error", rl.Vector2(self._rect.x + 20, self._rect.y + self._rect.height // 2 - 15), 30, 0.0, rl.RED
      )
      return

    scale = self._rect.height / self._qr_texture.height
    pos = rl.Vector2(self._rect.x + 8, self._rect.y)
    rl.draw_texture_ex(self._qr_texture, pos, 0.0, scale, rl.WHITE)

  def __del__(self):
    if self._qr_texture and self._qr_texture.id != 0:
      rl.unload_texture(self._qr_texture)


if __name__ == "__main__":
  gui_app.init_window("pairing device")
  pairing = PairingDialog()
  try:
    for _ in gui_app.render():
      result = pairing.render(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
      if result != -1:
        break
  finally:
    del pairing
