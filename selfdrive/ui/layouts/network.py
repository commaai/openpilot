import pyray as rl
from openpilot.system.ui.lib.wifi_manager import WifiManagerWrapper
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.network import WifiManagerUI
import time


class NetworkLayout(Widget):
  def __init__(self):
    super().__init__()
    self.wifi_manager = WifiManagerWrapper()
    self.wifi_manager.connect()
    self.wifi_manager.start()
    time.sleep(2)
    # self.wifi_ui = WifiManagerUI(self.wifi_manager)
    # self.wifi_manager.shutdown()

  def _render(self, rect: rl.Rectangle):
    self.wifi_ui.render(rect)

  def __del__(self):
    self.wifi_manager.shutdown()
