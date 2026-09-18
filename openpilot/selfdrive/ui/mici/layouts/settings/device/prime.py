import pyray as rl
import time

from openpilot.common.api import Api
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.selfdrive.ui.mici.widgets.button import BigButton, GreyBigButton
from openpilot.selfdrive.ui.mici.widgets.info import InfoLayoutMici
from openpilot.selfdrive.ui.mici.widgets.qr import QR
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets.scroller import NavScroller


class PairingInfoLayout(InfoLayoutMici):
  def __init__(self):
    super().__init__("account", "", "status", "")
    self._commacare_badge = gui_app.texture("icons_mici/settings/device/commacare.png", 27, 32)
    self._provider_icons = {provider: gui_app.texture(f"icons_mici/settings/device/{provider}.png", 32, 32)
                           for provider in ("github", "google", "apple")}

  def _update_state(self):
    super()._update_state()
    self.subtext1.set_text(ui_state.prime_state.get_pairing_account())
    self.subtext2.set_text(self._get_prime_status())
    self._provider_icon = self._provider_icons.get(ui_state.prime_state.get_pairing_provider())
    self._show_commacare = ui_state.prime_state.has_commacare()

  @staticmethod
  def _get_prime_status() -> str:
    if ui_state.prime_state.is_prime():
      return "prime" if ui_state.prime_state.is_full_prime() else "prime lite"
    return "not subscribed"

  def _layout(self):
    super()._layout()
    provider_offset = self._provider_icon.width + 14 if self._provider_icon else 0
    commacare_offset = self._commacare_badge.width + 14 if self._show_commacare else 0
    self.subtext1.set_position(self.subtext1.rect.x + provider_offset, self.subtext1.rect.y)
    self.subtext1.set_max_width(int(self._rect.width - provider_offset - (40 if self._provider_icon else 20)))
    self.subtext2.set_position(self.subtext2.rect.x + commacare_offset, self.subtext2.rect.y)
    self.subtext2.set_max_width(int(self._rect.width - commacare_offset - (40 if self._show_commacare else 20)))

  def _render(self, rect: rl.Rectangle):
    super()._render(rect)

    if self._provider_icon:
      icon_pos = rl.Vector2(self._rect.x + 20, self.subtext1.rect.y + (self.subtext1.rect.height - self._provider_icon.height) / 2)
      rl.draw_texture_v(self._provider_icon, icon_pos, self._subheader_color)
    if self._show_commacare:
      icon_pos = rl.Vector2(self._rect.x + 20, self.subtext2.rect.y + (self.subtext2.rect.height - self._commacare_badge.height) / 2 + 4)
      rl.draw_texture_v(self._commacare_badge, icon_pos, rl.WHITE)


class PrimeManagementScroller(NavScroller):
  """Prime subscription management and upgrade information."""

  def __init__(self):
    super().__init__()
    self._params = Params()

    can_claim_trial = ui_state.prime_state.can_claim_prime_trial()

    self._prime_icon = gui_app.texture("icons_mici/settings/device/green_cell.png", 64, 64)
    self._phone_icon = gui_app.texture("icons_mici/settings/device/phone.png", 85, 64)
    self._prime_adverts = [
      QR(self._get_prime_url()),
      GreyBigButton(
        "try prime for\n30 days" if can_claim_trial else "upgrade to prime",
        "scan to claim trial" if can_claim_trial else "scan to manage\nprime status",
        self._prime_icon
      ),
      GreyBigButton("", "prime adds 24/7 LTE to device and 1 year of cloud storage in connect.",),
      GreyBigButton("", "prime lets you view live video and GPS location remotely in connect.",),
      GreyBigButton("", "prime also includes commacare extended device warranty.",),
    ]
    self._prime_management = [
      QR(self._get_prime_url()),
      GreyBigButton(
        "manage prime",
        "scan to open\ndevice prime settings",
        self._phone_icon
      ),
    ]
    self._scroller.add_widgets(self._prime_management if ui_state.prime_state.is_prime() else self._prime_adverts)

  def _get_prime_url(self) -> str:
    if dongle_id := self._params.get("DongleId"):
      return f"https://connect.comma.ai/{dongle_id}/prime"
    return "https://connect.comma.ai"

  def _update_state(self):
    super()._update_state()
    if not ui_state.prime_state.is_paired():
      self.dismiss()


class PrimeScroller(NavScroller):
  """Pairing flow or paired account information with access to prime management."""

  QR_REFRESH_INTERVAL = 300  # 5 minutes in seconds

  def __init__(self):
    super().__init__()
    self._params = Params()
    self.initial_is_paired = ui_state.prime_state.is_paired()
    self._last_pairing_qr_generation = float("-inf")

    self._manage_icon = gui_app.texture("icons_mici/settings/comma_icon.png", 33, 60)
    if not ui_state.prime_state.is_prime():
      title = "claim\nprime trial" if ui_state.prime_state.can_claim_prime_trial() else "upgrade\nto prime"
      self._manage_prime = BigButton(title, icon=self._manage_icon)
    else:
      self._manage_prime = BigButton("manage prime", icon=self._manage_icon)
    self._manage_prime.set_click_callback(lambda: gui_app.push_widget(PrimeManagementScroller()))

    if self.initial_is_paired:
      self._scroller.add_widgets([
        PairingInfoLayout(),
        self._manage_prime,
      ])
    else:
      self._qr = QR(self._get_pairing_url())
      self._scroller._show_scroll_indicator = False
      self._scroller.add_widgets([
        self._qr,
        GreyBigButton("finish setup", "scan to pair device\nwith connect",
                      gui_app.texture("icons_mici/settings/device/green_settings.png", 64, 64)),
        GreyBigButton("", "connect lets you review recent driving footage and bookmark events."),
      ])

  def _get_pairing_url(self) -> str:
    try:
      dongle_id = self._params.get("DongleId") or ""
      token = Api(dongle_id).get_token({'pair': True})
    except Exception as e:
      cloudlog.warning(f"Failed to get pairing token: {e}")
      token = ""
    return f"https://connect.comma.ai/?pair={token}"

  def _update_layout_rects(self):
    super()._update_layout_rects()
    if not self.initial_is_paired:
      self._qr.set_rect(rl.Rectangle(self._qr.rect.x, self._qr.rect.y, self._rect.height, self._rect.height))

  def _render(self, rect: rl.Rectangle):
    if not self.initial_is_paired:
      current_time = time.monotonic()
      if current_time - self._last_pairing_qr_generation >= self.QR_REFRESH_INTERVAL:
        self._qr._url = self._get_pairing_url()
        if self._qr._texture and self._qr._texture.id != 0:
          rl.unload_texture(self._qr._texture)
        self._qr._texture = self._qr._generate_qr_code()
        self._last_pairing_qr_generation = current_time
    super()._render(rect)

  def _update_state(self):
    super()._update_state()
    if self.initial_is_paired != ui_state.prime_state.is_paired() and gui_app.get_active_widget() is self:
      self.dismiss()


if __name__ == "__main__":
  gui_app.init_window("pairing device")
  pairing = PrimeScroller()
  gui_app.push_widget(pairing)
  try:
    for _ in gui_app.render():
      pass
  finally:
    del pairing
