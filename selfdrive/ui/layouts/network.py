from openpilot.system.ui.lib.widget import Widget
from openpilot.system.ui.lib.wifi_manager import WifiManagerWrapper
from openpilot.system.ui.widgets.network import WifiManagerUI


class NetworkLayout(Widget):
  def __init__(self):
    super().__init__()
    self.wifi_manager = WifiManagerWrapper()
    self.wifi_ui = WifiManagerUI(self.wifi_manager)

  def _render(self):
    self.wifi_ui.render(self._rect)

  def shutdown(self):
    self.wifi_manager.shutdown()
