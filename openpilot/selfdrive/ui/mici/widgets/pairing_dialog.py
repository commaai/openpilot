import pyray as rl
import time

from openpilot.common.api import Api
from openpilot.common.qrcode import make_texture
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.widgets.nav_widget import NavWidget
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.widgets.label import UnifiedLabel


class PairingDialog(NavWidget):
  """Dialog for device pairing with QR code."""

  QR_REFRESH_INTERVAL = 300  # 5 minutes in seconds

  def __init__(self):
    super().__init__()
    self._params = Params()
    self._qr_texture: rl.Texture | None = None
    self._last_qr_generation = float("-inf")

    self._txt_pair = gui_app.texture("icons_mici/settings/device/pair.png", 33, 60)
    self._pair_label = UnifiedLabel("pair with comma connect", font_size=48, font_weight=FontWeight.BOLD, line_height=0.8)

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
      if self._qr_texture and self._qr_texture.id != 0:
        rl.unload_texture(self._qr_texture)
      self._qr_texture = make_texture(self._get_pairing_url(), inverted=True)
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
    if ui_state.prime_state.is_paired() and not self.is_dismissing:
      self.dismiss()

  def _render(self, rect: rl.Rectangle):
    self._check_qr_refresh()

    self._render_qr_code()

    label_x = self._rect.x + 8 + self._rect.height + 24
    self._pair_label.set_max_width(int(self._rect.width - label_x))
    self._pair_label.set_position(label_x, self._rect.y + 16)
    self._pair_label.render()

    rl.draw_texture_ex(self._txt_pair, rl.Vector2(label_x, self._rect.y + self._rect.height - self._txt_pair.height - 16),
                       0.0, 1.0, rl.Color(255, 255, 255, int(255 * 0.35)))

  def _render_qr_code(self) -> None:
    if not self._qr_texture:
      error_font = gui_app.font(FontWeight.BOLD)
      rl.draw_text_ex(
        error_font, "QR Code Error", rl.Vector2(self._rect.x + 20, self._rect.y + self._rect.height // 2 - 15), 30, 0.0, rl.RED
      )
      return

    scale = self._rect.height / self._qr_texture.height
    pos = rl.Vector2(round(self._rect.x + 8), round(self._rect.y))
    rl.draw_texture_ex(self._qr_texture, pos, 0.0, scale, rl.WHITE)

  def __del__(self):
    if self._qr_texture and self._qr_texture.id != 0:
      rl.unload_texture(self._qr_texture)


if __name__ == "__main__":
  gui_app.init_window("pairing device")
  pairing = PairingDialog()
  gui_app.push_widget(pairing)
  try:
    for _ in gui_app.render():
      pass
  finally:
    del pairing
