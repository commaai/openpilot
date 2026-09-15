import pyray as rl

from openpilot.common.api import Api
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.selfdrive.ui.mici.widgets.button import BigButton, GreyBigButton
from openpilot.selfdrive.ui.mici.widgets.qr import QR
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.system.ui.widgets.scroller import NavScroller


class PairingInfoLayout(Widget):
  def __init__(self):
    super().__init__()
    self._commacare_badge = gui_app.texture("icons_mici/settings/device/commacare.png", 24, 29)
    subheader_color = rl.Color(255, 255, 255, int(255 * 0.9 * 0.65))
    self._labels = [
      UnifiedLabel("paired with", 48, max_width=340, font_weight=FontWeight.DISPLAY, wrap_text=False),
      UnifiedLabel(ui_state.prime_state.get_pairing_account, 32, max_width=340,
                   text_color=subheader_color, font_weight=FontWeight.ROMAN, wrap_text=False, scroll=True),
      UnifiedLabel("status", 48, max_width=340, font_weight=FontWeight.DISPLAY, wrap_text=False),
      UnifiedLabel(self._get_prime_status, 32, max_width=304,
                   text_color=subheader_color, font_weight=FontWeight.ROMAN, wrap_text=False),
    ]
    self.set_rect(rl.Rectangle(0, 0, 360, 180))

  @staticmethod
  def _get_prime_status() -> str:
    if ui_state.prime_state.is_prime():
      return "prime" if ui_state.prime_state.is_full_prime() else "prime lite"
    return "not subscribed"

  def _render(self, _):
    show_commacare = ui_state.prime_state.has_commacare()
    for label, y_offset in zip(self._labels, (-10, 68 - 25, 114 - 30, 161 - 25), strict=True):
      badge_offset = self._commacare_badge.width + 12 if show_commacare and label is self._labels[3] else 0
      label.set_position(self._rect.x + 20 + badge_offset, self._rect.y + y_offset)
      label.render()

    if show_commacare:
      label = self._labels[3]
      badge_pos = rl.Vector2(self._rect.x + 20, label.rect.y + (label.rect.height - self._commacare_badge.height) / 2)
      rl.draw_texture_v(self._commacare_badge, badge_pos, rl.WHITE)


class PrimeManagementScroller(NavScroller):
  """Prime subscription management and upgrade information."""

  def __init__(self):
    super().__init__()
    self._params = Params()

    can_claim_trial = ui_state.prime_state.can_claim_prime_trial()

    # prime management components
    self._prime_icon = gui_app.texture("icons_mici/settings/device/green_cell.png", 64, 64)
    self._phone_icon = gui_app.texture("icons_mici/settings/device/phone.png", 85, 64)
    self._prime_adverts = [
      QR(self._get_prime_url),
      GreyBigButton(
        "try prime for\n30 days" if can_claim_trial else "upgrade to prime",
        "scan to claim trial" if can_claim_trial else "scan to manage\nprime status",
        self._prime_icon
      ),
      GreyBigButton("", "prime adds 24/7 LTE and 1 year of cloud storage in connect.",),
      GreyBigButton("", "prime lets you view live video and GPS location remotely in connect.",),
      GreyBigButton("", "prime also includes commacare extended device warranty.",),
    ]
    self._prime_management = [
      QR(self._get_prime_url),
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


class PrimeScroller(NavScroller):
  """Pairing flow or paired account information with access to prime management."""

  QR_REFRESH_INTERVAL = 300  # 5 minutes in seconds

  def __init__(self):
    super().__init__()
    self._params = Params()
    self.initial_is_paired = ui_state.prime_state.is_paired()

    self._manage_icon = gui_app.texture("icons_mici/settings/comma_icon.png", 33, 60)
    if not ui_state.prime_state.is_prime():
      subtitle = "claim prime trial" if ui_state.prime_state.can_claim_prime_trial() else "upgrade to prime"
      self._manage_prime = BigButton("prime", subtitle, icon=self._manage_icon)
    else:
      self._manage_prime = BigButton("manage prime", icon=self._manage_icon)
    self._manage_prime.set_click_callback(lambda: gui_app.push_widget(PrimeManagementScroller()))

    if self.initial_is_paired:
      self._scroller.add_widgets([
        PairingInfoLayout(),
        self._manage_prime,
      ])
    else:
      self._scroller._show_scroll_indicator = False
      self._scroller.add_widgets([
        QR(self._get_pairing_url, refresh_interval=self.QR_REFRESH_INTERVAL),
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

  def _update_state(self):
    super()._update_state()
    if not self.initial_is_paired and ui_state.prime_state.is_paired() and not self.is_dismissing:
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
