import pyray as rl
import qrcode
import numpy as np
import time

from openpilot.common.api import Api
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.lib.text_measure import measure_text_cached


class PairingDialog:
  """Dialog for device pairing with QR code."""

  QR_REFRESH_INTERVAL = 300  # 5 minutes in seconds

  def __init__(self):
    self.params = Params()
    self.qr_texture: rl.Texture | None = None
    self.last_qr_generation = 0

  def _get_pairing_url(self) -> str:
    try:
      dongle_id = self.params.get("DongleId") or ""
      token = Api(dongle_id).get_token()
    except Exception as e:
      cloudlog.warning(f"Failed to get pairing token: {e}")
      token = ""
    return f"https://connect.comma.ai/setup?token={token}"

  def _generate_qr_code(self) -> None:
    try:
      qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
      qr.add_data(self._get_pairing_url())
      qr.make(fit=True)

      pil_img = qr.make_image(fill_color="black", back_color="white").convert('RGBA')
      img_array = np.array(pil_img, dtype=np.uint8)

      if self.qr_texture and self.qr_texture.id != 0:
        rl.unload_texture(self.qr_texture)

      rl_image = rl.Image()
      rl_image.data = rl.ffi.cast("void *", img_array.ctypes.data)
      rl_image.width = pil_img.width
      rl_image.height = pil_img.height
      rl_image.mipmaps = 1
      rl_image.format = rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8

      self.qr_texture = rl.load_texture_from_image(rl_image)
    except Exception as e:
      cloudlog.warning(f"QR code generation failed: {e}")
      self.qr_texture = None

  def _check_qr_refresh(self) -> None:
    current_time = time.monotonic()
    if current_time - self.last_qr_generation >= self.QR_REFRESH_INTERVAL:
      self._generate_qr_code()
      self.last_qr_generation = current_time

  def render(self, rect: rl.Rectangle) -> int:
    rl.clear_background(rl.Color(224, 224, 224, 255))

    self._check_qr_refresh()

    margin = 70
    content_rect = rl.Rectangle(rect.x + margin, rect.y + margin, rect.width - 2 * margin, rect.height - 2 * margin)
    y = content_rect.y

    # Close button
    close_size = 80
    close_icon = gui_app.texture("icons/close.png", close_size, close_size)
    close_rect = rl.Rectangle(content_rect.x, y, close_size, close_size)

    mouse_pos = rl.get_mouse_position()
    is_hover = rl.check_collision_point_rec(mouse_pos, close_rect)
    is_pressed = rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT)
    is_released = rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT)

    color = rl.Color(180, 180, 180, 150) if (is_hover and is_pressed) else rl.WHITE
    rl.draw_texture(close_icon, int(content_rect.x), int(y), color)

    if (is_hover and is_released) or rl.is_key_pressed(rl.KeyboardKey.KEY_ESCAPE):
      return 1

    y += close_size + 40

    # Title
    title = "Pair your device to your comma account"
    title_font = gui_app.font(FontWeight.NORMAL)
    left_width = int(content_rect.width * 0.5 - 15)

    title_wrapped = wrap_text(title_font, title, 75, left_width)
    rl.draw_text_ex(title_font, "\n".join(title_wrapped), rl.Vector2(content_rect.x, y), 75, 0.0, rl.BLACK)
    y += len(title_wrapped) * 75 + 60

    # Two columns: instructions and QR code
    remaining_height = content_rect.height - (y - content_rect.y)
    right_width = content_rect.width // 2 - 20

    # Instructions
    self._render_instructions(rl.Rectangle(content_rect.x, y, left_width, remaining_height))

    # QR code
    qr_size = min(right_width, content_rect.height) - 40
    qr_x = content_rect.x + left_width + 40 + (right_width - qr_size) // 2
    qr_y = content_rect.y
    self._render_qr_code(rl.Rectangle(qr_x, qr_y, qr_size, qr_size))

    return -1

  def _render_instructions(self, rect: rl.Rectangle) -> None:
    instructions = [
      "Go to https://connect.comma.ai on your phone",
      "Click \"add new device\" and scan the QR code on the right",
      "Bookmark connect.comma.ai to your home screen to use it like an app",
    ]

    font = gui_app.font(FontWeight.BOLD)
    y = rect.y

    for i, text in enumerate(instructions):
      circle_radius = 25
      circle_x = rect.x + circle_radius + 15
      text_x = rect.x + circle_radius * 2 + 40
      text_width = rect.width - (circle_radius * 2 + 40)

      wrapped = wrap_text(font, text, 47, int(text_width))
      text_height = len(wrapped) * 47
      circle_y = y + text_height // 2

      # Circle and number
      rl.draw_circle(int(circle_x), int(circle_y), circle_radius, rl.Color(70, 70, 70, 255))
      number = str(i + 1)
      number_width = measure_text_cached(font, number, 30).x
      rl.draw_text(number, int(circle_x - number_width // 2), int(circle_y - 15), 30, rl.WHITE)

      # Text
      rl.draw_text_ex(font, "\n".join(wrapped), rl.Vector2(text_x, y), 47, 0.0, rl.BLACK)
      y += text_height + 50

  def _render_qr_code(self, rect: rl.Rectangle) -> None:
    if not self.qr_texture:
      rl.draw_rectangle_rounded(rect, 0.1, 20, rl.Color(240, 240, 240, 255))
      error_font = gui_app.font(FontWeight.BOLD)
      rl.draw_text_ex(
        error_font, "QR Code Error", rl.Vector2(rect.x + 20, rect.y + rect.height // 2 - 15), 30, 0.0, rl.RED
      )
      return

    source = rl.Rectangle(0, 0, self.qr_texture.width, self.qr_texture.height)
    rl.draw_texture_pro(self.qr_texture, source, rect, rl.Vector2(0, 0), 0, rl.WHITE)

  def __del__(self):
    if self.qr_texture and self.qr_texture.id != 0:
      rl.unload_texture(self.qr_texture)


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
