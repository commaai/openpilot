import pyray as rl
from openpilot.system.ui.lib.wifi_manager import WifiManagerWrapper
from openpilot.system.ui.widgets.network import WifiManagerUI


class NetworkLayout:
  def __init__(self):
    self.wifi_manager = WifiManagerWrapper()
    self.wifi_ui = WifiManagerUI(self.wifi_manager)

  def render(self, rect: rl.Rectangle):
    self.wifi_ui.render(rect)

  @property
  def require_full_screen(self):
    return self.wifi_ui.require_full_screen

  def shutdown(self):
    self.wifi_manager.shutdown()
