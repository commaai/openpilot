from openpilot.common.api import Api
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.selfdrive.ui.mici.widgets.button import GreyBigButton
from openpilot.selfdrive.ui.mici.widgets.qr import QR
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets.scroller import NavScroller

class PrimeScroller(NavScroller):
  """Dialog for device pairing/prime with QR code."""

  QR_REFRESH_INTERVAL = 300  # 5 minutes in seconds

  def __init__(self):
    super().__init__()
    self._params = Params()

    can_claim_trial = ui_state.prime_state.can_claim_prime_trial()
    self.initial_is_paired = ui_state.prime_state.is_paired()

    # pairing components
    self._pairing_icon = gui_app.texture("icons_mici/settings/device/green_settings.png", 64, 64)
    self._pairing_info = GreyBigButton("finish setup", "scan to pair device\nwith connect", self._pairing_icon)
    self._connect_info = GreyBigButton("", "connect lets you review recent driving footage and bookmark events.")

    # prime management components
    self._prime_icon = gui_app.texture("icons_mici/settings/device/green_cell.png", 64, 64)
    self._phone_icon = gui_app.texture("icons_mici/settings/device/phone.png", 85, 64)
    self._prime_adverts = [
      GreyBigButton(
        "try prime for\n30 days" if can_claim_trial else "upgrade to prime",
        "scan to claim trial" if can_claim_trial else "scan to manage\nprime status",
        self._prime_icon
      ),
      GreyBigButton("", "prime adds 24/7 LTE and 1 year of cloud storage in connect.",),
      GreyBigButton("", "prime lets you view live video and GPS location remotely in connect.",),
      GreyBigButton("", "prime also includes commacare extended device warranty.",),
    ]
    self._prime_management = GreyBigButton(
      f"subscribed {"(full)" if ui_state.prime_state.is_full_prime() else "(lite)" }",
      "scan to open\nprime settings",
      self._phone_icon
    )

    if ui_state.prime_state.is_paired():
      self._scroller.add_widgets([
        QR(self._get_prime_url),
        *([self._prime_management] if ui_state.prime_state.is_prime() else self._prime_adverts),
      ])
      if not ui_state.prime_state.is_prime(): self._scroller.add_widget(QR(self._get_prime_url))
    else:
      self._scroller._show_scroll_indicator = False
      self._scroller.add_widgets([
        QR(self._get_pairing_url, refresh_interval=self.QR_REFRESH_INTERVAL),
        self._pairing_info,
        self._connect_info
      ])

  def _get_pairing_url(self) -> str:
    try:
      dongle_id = self._params.get("DongleId") or ""
      token = Api(dongle_id).get_token({'pair': True})
    except Exception as e:
      cloudlog.warning(f"Failed to get pairing token: {e}")
      token = ""
    return f"https://connect.comma.ai/?pair={token}"

  def _get_prime_url(self) -> str:
    if dongle_id := self._params.get("DongleId"):
      return f"https://connect.comma.ai/{dongle_id}/prime"

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
