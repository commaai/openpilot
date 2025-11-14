import pyray as rl

from openpilot.system.ui.widgets.scroller import Scroller
from openpilot.selfdrive.ui.mici.widgets.button import BigButton
from openpilot.selfdrive.ui.mici.layouts.settings.toggles import TogglesLayoutMici
from openpilot.selfdrive.ui.mici.layouts.settings.network import NetworkLayoutMici
from openpilot.selfdrive.ui.mici.layouts.settings.device import DeviceLayoutMici, PairBigButton
from openpilot.selfdrive.ui.mici.layouts.settings.developer import DeveloperLayoutMici
from openpilot.selfdrive.ui.mici.layouts.settings.firehose import FirehoseLayoutMici
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import NavWidget


class SettingsLayout(NavWidget):
  def __init__(self):
    super().__init__()

    toggles_btn = BigButton("toggles", "", "icons_mici/settings/toggles_icon.png")
    toggles_btn.set_click_callback(lambda: gui_app.stack.push(TogglesLayoutMici(back_callback=lambda: gui_app.stack.pop())))
    network_btn = BigButton("network", "", "icons_mici/settings/network/wifi_strength_full.png")
    network_btn.set_click_callback(lambda: gui_app.stack.push(NetworkLayoutMici(back_callback=lambda: gui_app.stack.pop())))
    device_btn = BigButton("device", "", "icons_mici/settings/device_icon.png")
    device_btn.set_click_callback(lambda: gui_app.stack.push(DeviceLayoutMici(back_callback=lambda: gui_app.stack.pop())))
    developer_btn = BigButton("developer", "", "icons_mici/settings/developer_icon.png")
    developer_btn.set_click_callback(lambda: gui_app.stack.push(DeveloperLayoutMici(back_callback=lambda: gui_app.stack.pop())))
    firehose_btn = BigButton("firehose", "", "icons_mici/settings/comma_icon.png")
    firehose_btn.set_click_callback(lambda: gui_app.stack.push(FirehoseLayoutMici(back_callback=lambda: gui_app.stack.pop())))

    self._scroller = Scroller([
      toggles_btn,
      network_btn,
      device_btn,
      PairBigButton(),
      #BigDialogButton("manual", "", "icons_mici/settings/manual_icon.png", "Check out the mici user\nmanual at comma.ai/setup"),
      firehose_btn,
      developer_btn,
    ], snap_items=False)

  def _render(self, rect: rl.Rectangle):
    self._scroller.render(rect)
