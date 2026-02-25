#!/usr/bin/env python3
"""
Minimal UI app to reproduce the scroller overlay stuck bug.

Bug: Connect to a network (triggers move animation), quickly swipe back,
then tap Wi-Fi again. The dark overlay stays permanently.

Run from repo root: python system/ui/scroller_overlay_repro.py

Steps to reproduce:
1. Tap "Wi-Fi"
2. Tap any network to connect
3. Swipe back quickly
4. Tap "Send Callback" (fires networks_updated on hidden wifi panel)
5. Tap "Wi-Fi" again -> overlay stuck

To verify it's the callback: skip step 4 and the overlay should be fine.
"""
import pyray as rl
from dataclasses import dataclass

from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets.nav_widget import NavWidget
from openpilot.system.ui.widgets.scroller import Scroller
from openpilot.system.ui.lib.wifi_manager import Network, SecurityType, ConnectStatus
from openpilot.selfdrive.ui.mici.layouts.settings.network.wifi_ui import WifiUIMici
from openpilot.selfdrive.ui.mici.widgets.button import BigButton


@dataclass
class WifiState:
  ssid: str | None = None
  prev_ssid: str | None = None
  status: int = 0  # ConnectStatus.DISCONNECTED


class FakeWifiManager:
  """Simulates WifiManager. Queues networks_updated to fire while wifi panel is hidden."""

  def __init__(self):
    self.wifi_state = WifiState()
    self._networks: list[Network] = []
    self.ipv4_address: str = ""
    self._saved_ssids: set[str] = set()
    self._networks_updated: list = []
    self._pending_callback: list[Network] | None = None  # Fire this when tick runs

  @property
  def networks(self) -> list[Network]:
    return self._networks

  @property
  def connecting_to_ssid(self) -> str | None:
    return self.wifi_state.ssid if self.wifi_state.status == ConnectStatus.CONNECTING else None

  @property
  def connected_ssid(self) -> str | None:
    return self.wifi_state.ssid if self.wifi_state.status == ConnectStatus.CONNECTED else None

  def is_connection_saved(self, ssid: str) -> bool:
    return ssid in self._saved_ssids

  def add_callbacks(self, **kwargs):
    if "networks_updated" in kwargs:
      self._networks_updated.append(kwargs["networks_updated"])

  def set_active(self, active: bool):
    pass

  def forget_connection(self, ssid: str):
    pass

  def connect_to_network(self, ssid: str, password: str, hidden: bool = False):
    self.wifi_state = WifiState(ssid=ssid, status=ConnectStatus.CONNECTING)

  def activate_connection(self, ssid: str, block: bool = False):
    self.wifi_state = WifiState(ssid=ssid, status=ConnectStatus.CONNECTING)

  def process_callbacks(self):
    """Called every frame from nav_stack_tick. Fires pending networks_updated (simulates late callback)."""
    if self._pending_callback is not None:
      nets = self._pending_callback
      self._pending_callback = None
      for cb in self._networks_updated:
        cb(nets)

  def fire_networks_updated_while_hidden(self, networks: list[Network]):
    """Queue a callback to fire on next process_callbacks (simulates scan completing while user is on main panel)."""
    self._pending_callback = networks


class MainPanel(NavWidget):
  """Main panel with Wi-Fi button and a manual trigger for the callback."""

  def __init__(self, wifi_manager: FakeWifiManager, wifi_ui: WifiUIMici):
    super().__init__()
    self._wifi_manager = wifi_manager
    self._wifi_ui = wifi_ui

    self._wifi_btn = BigButton("Wi-Fi", "tap to open")
    self._wifi_btn.set_click_callback(self._on_wifi_click)

    self._trigger_btn = BigButton("Send Callback", "fire networks_updated")
    self._trigger_btn.set_click_callback(self._on_trigger_click)

    self._scroller = Scroller([self._wifi_btn, self._trigger_btn])

    self.set_back_callback(gui_app.pop_widget)

  def _on_wifi_click(self):
    gui_app.push_widget(self._wifi_ui)

  def _on_trigger_click(self):
    """Manually fire networks_updated on the hidden wifi panel (simulates late scan callback)."""
    self._wifi_manager.wifi_state = WifiState(ssid="Alpha", status=ConnectStatus.CONNECTING)
    for cb in self._wifi_manager._networks_updated:
      cb(self._wifi_manager.networks)

  def show_event(self):
    super().show_event()
    self._scroller.show_event()

  def hide_event(self):
    super().hide_event()
    self._scroller.hide_event()

  def _render(self, rect: rl.Rectangle):
    self._scroller.render(rect)


def main():
  gui_app.init_window("Scroller Overlay Repro", fps=60)

  nets = [
    Network(ssid="Alpha", strength=80, security_type=SecurityType.OPEN, is_tethering=False),
    Network(ssid="Bravo", strength=60, security_type=SecurityType.OPEN, is_tethering=False),
    Network(ssid="Charlie", strength=40, security_type=SecurityType.OPEN, is_tethering=False),
  ]

  wm = FakeWifiManager()
  wm._networks = nets

  wifi_ui = WifiUIMici(wm)
  wifi_ui._on_network_updated(nets)  # Seed initial networks

  main_panel = MainPanel(wm, wifi_ui)
  wifi_ui.set_back_callback(gui_app.pop_widget)

  gui_app.push_widget(main_panel)

  for _ in gui_app.render():
    pass

  gui_app.close()


if __name__ == "__main__":
  main()
