from __future__ import annotations
import pyray as rl
import time

from openpilot.common.api import Api
from openpilot.common.qrcode import make_texture
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.selfdrive.ui.mici.widgets.button import GreyBigButton
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import Label
from openpilot.system.ui.widgets.scroller import NavScroller

class QRWidget(Widget):
  def __init__(self, texture: rl.Texture | None = None):
    super().__init__()
    self._texture = texture
    self.set_rect(rl.Rectangle(0, 0, 208, 208))
    self._error = Label("QR Code Error", font_size=30, font_weight=FontWeight.BOLD, text_color=rl.RED)

  def _render(self, rect: rl.Rectangle):
    if not self._texture:
      self._error.render(rect)
      return
    scale = rect.height / self._texture.height
    pos = rl.Vector2(round(rect.x + 8), round(rect.y))
    rl.draw_texture_ex(self._texture, pos, 0.0, scale, rl.WHITE)

class PrimeScroller(NavScroller):
  """Dialog for device pairing/prime with QR code."""

  QR_REFRESH_INTERVAL = 300  # 5 minutes in seconds

  def __init__(self):
    super().__init__()
    self._params = Params()

    can_claim_trial = ui_state.prime_state.can_claim_prime_trial()
    self.initial_is_paired = ui_state.prime_state.is_paired()

    # pairing components
    self._pairing_qr_texture: rl.Texture | None = None
    self._last_pairing_qr_generation = float("-inf")
    self._pairing_icon = gui_app.texture("icons_mici/offroad_alerts/green_settings.png", 64, 64)
    self._pairing_info = GreyBigButton("scan to pair device", "connect.comma.ai", self._pairing_icon)

    # prime management components
    self._prime_qr_texture = self._generate_qr_code(self._get_prime_url())
    self._prime_icon = gui_app.texture("icons_mici/offroad_alerts/green_cell.png", 64, 64)
    self._phone_icon = gui_app.texture("icons_mici/settings/device/phone.png", 85, 64)
    self._prime_adverts = [
      GreyBigButton(
        "scan to claim prime trial" if can_claim_trial else "scan to open prime settings",
        "try it for 30 days" if can_claim_trial else "upgrade to prime",
        self._prime_icon
      ),
      GreyBigButton("", "prime adds 24/7 LTE and 1 year of cloud storage in connect.",),
      GreyBigButton("", "prime lets you view live video and GPS location remotely in connect.",),
      GreyBigButton("", "prime also includes commacare extended device warranty.",),
    ]
    self._prime_management = GreyBigButton(
      "scan to open prime settings",
      f"subscribed {"(full)" if ui_state.prime_state.is_full_prime() else "(lite)" }",
      self._phone_icon
    )

    if ui_state.prime_state.is_paired():
      self._scroller.add_widgets([
        QRWidget(self._prime_qr_texture),
        *([self._prime_management] if ui_state.prime_state.is_prime() else self._prime_adverts),
      ])
      if not ui_state.prime_state.is_prime(): self._scroller.add_widget(QRWidget(self._prime_qr_texture))
    else:
      self._scroller.add_widgets([
        QRWidget(self._pairing_qr_texture),
        self._pairing_info
      ])

  def _generate_qr_code(self, url):
    try:
      return make_texture(url, inverted=True)
    except Exception as e:
      cloudlog.warning(f"QR code generation failed: {e}")
      return None

  def _get_pairing_url(self) -> str:
    try:
      dongle_id = self._params.get("DongleId") or ""
      token = Api(dongle_id).get_token({'pair': True})
    except Exception as e:
      cloudlog.warning(f"Failed to get pairing token: {e}")
      token = ""
    return f"https://connect.comma.ai/?pair={token}"

  def _check_qr_refresh(self) -> None:
    current_time = time.monotonic()
    if current_time - self._last_pairing_qr_generation >= self.QR_REFRESH_INTERVAL:
      if self._pairing_qr_texture and self._pairing_qr_texture.id != 0:
        rl.unload_texture(self._pairing_qr_texture)
      self._pairing_qr_texture = self._generate_qr_code(self._get_pairing_url())
      self._last_pairing_qr_generation = current_time

  def _get_prime_url(self) -> str:
    if dongle_id := self._params.get("DongleId"):
      return f"https://connect.comma.ai/{dongle_id}/prime"

  def _update_state(self):
    super()._update_state()
    if not self.initial_is_paired and ui_state.prime_state.is_paired() and not self.is_dismissing:
      self.dismiss()

  def _render(self, rect: rl.Rectangle):
    if not ui_state.prime_state.is_paired():
      self._check_qr_refresh()
    super()._render(rect)

  def __del__(self):
    if self._pairing_qr_texture and self._pairing_qr_texture.id != 0:
      rl.unload_texture(self._pairing_qr_texture)
    if self._prime_qr_texture and self._prime_qr_texture.id != 0:
      rl.unload_texture(self._prime_qr_texture)


if __name__ == "__main__":
  gui_app.init_window("pairing device")
  pairing = PrimeScroller()
  gui_app.push_widget(pairing)
  try:
    for _ in gui_app.render():
      pass
  finally:
    del pairing
