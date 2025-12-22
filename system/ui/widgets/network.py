from enum import IntEnum

import pyray as rl
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr, tr_lazy
from openpilot.system.ui.lib.wifi_manager import WifiManager, SecurityType, Network, MeteredType
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.widgets.keyboard import Keyboard
from openpilot.system.ui.widgets.label import gui_label, Align
from openpilot.system.ui.widgets.scroller_tici import Scroller
from openpilot.system.ui.widgets.list_view import ButtonAction, ListItem, MultipleButtonAction, ToggleAction, button_item, text_item
from openpilot.system.ui.widgets.network_item import NetworkItem, ITEM_HEIGHT, UIState

# These are only used for AdvancedNetworkSettings, standalone apps just need WifiManagerUI
try:
  from openpilot.common.params import Params
  from openpilot.selfdrive.ui.ui_state import ui_state
  from openpilot.selfdrive.ui.lib.prime_state import PrimeType
except Exception:
  Params = None
  ui_state = None  # type: ignore
  PrimeType = None  # type: ignore

NM_DEVICE_STATE_NEED_AUTH = 60
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 64


class PanelType(IntEnum):
  WIFI = 0
  ADVANCED = 1


class NavButton(Widget):
  def __init__(self, text: str):
    super().__init__()
    self.text = text
    self.set_rect(rl.Rectangle(0, 0, 400, 100))

  def _render(self, _):
    color = rl.Color(74, 74, 74, 255) if self.is_pressed else rl.Color(57, 57, 57, 255)
    rl.draw_rectangle_rounded(self._rect, 0.6, 10, color)
    gui_label(self.rect, self.text, font_size=60, align=Align.CENTER)


class NetworkUI(Widget):
  def __init__(self, wifi_manager: WifiManager):
    super().__init__()
    self._wifi_manager = wifi_manager
    self._current_panel: PanelType = PanelType.WIFI
    self._wifi_panel = WifiManagerUI(wifi_manager)
    self._advanced_panel = AdvancedNetworkSettings(wifi_manager)
    self._nav_button = NavButton(tr("Advanced"))
    self._nav_button.set_click_callback(self._cycle_panel)

  def show_event(self):
    self._set_current_panel(PanelType.WIFI)
    self._wifi_panel.show_event()

  def hide_event(self):
    self._wifi_panel.hide_event()

  def _cycle_panel(self):
    if self._current_panel == PanelType.WIFI:
      self._set_current_panel(PanelType.ADVANCED)
    else:
      self._set_current_panel(PanelType.WIFI)

  def _render(self, _):
    # subtract button
    content_rect = rl.Rectangle(self._rect.x, self._rect.y + self._nav_button.rect.height + 40,
                                self._rect.width, self._rect.height - self._nav_button.rect.height - 40)
    if self._current_panel == PanelType.WIFI:
      self._nav_button.text = tr("Advanced")
      self._nav_button.set_position(self._rect.x + self._rect.width - self._nav_button.rect.width, self._rect.y + 20)
      self._wifi_panel.render(content_rect)
    else:
      self._nav_button.text = tr("Back")
      self._nav_button.set_position(self._rect.x, self._rect.y + 20)
      self._advanced_panel.render(content_rect)

    self._nav_button.render()

  def _set_current_panel(self, panel: PanelType):
    self._current_panel = panel


class AdvancedNetworkSettings(Widget):
  def __init__(self, wifi_manager: WifiManager):
    super().__init__()
    self._wifi_manager = wifi_manager
    self._wifi_manager.add_callbacks(networks_updated=self._on_network_updated)
    self._params = Params()

    self._keyboard = Keyboard(max_text_size=MAX_PASSWORD_LENGTH, min_text_size=MIN_PASSWORD_LENGTH, show_password_toggle=True)

    # Tethering
    self._tethering_action = ToggleAction(initial_state=False)
    tethering_btn = ListItem(tr_lazy("Enable Tethering"), action_item=self._tethering_action, callback=self._toggle_tethering)

    # Edit tethering password
    self._tethering_password_action = ButtonAction(tr_lazy("EDIT"))
    tethering_password_btn = ListItem(tr_lazy("Tethering Password"), action_item=self._tethering_password_action, callback=self._edit_tethering_password)

    # Roaming toggle
    roaming_enabled = self._params.get_bool("GsmRoaming")
    self._roaming_action = ToggleAction(initial_state=roaming_enabled)
    self._roaming_btn = ListItem(tr_lazy("Enable Roaming"), action_item=self._roaming_action, callback=self._toggle_roaming)

    # Cellular metered toggle
    cellular_metered = self._params.get_bool("GsmMetered")
    self._cellular_metered_action = ToggleAction(initial_state=cellular_metered)
    self._cellular_metered_btn = ListItem(tr_lazy("Cellular Metered"),
                                          description=tr_lazy("Prevent large data uploads when on a metered cellular connection"),
                                          action_item=self._cellular_metered_action, callback=self._toggle_cellular_metered)

    # APN setting
    self._apn_btn = button_item(tr_lazy("APN Setting"), tr_lazy("EDIT"), callback=self._edit_apn)

    # Wi-Fi metered toggle
    self._wifi_metered_action = MultipleButtonAction([tr_lazy("default"), tr_lazy("metered"), tr_lazy("unmetered")], 255, 0,
                                                     callback=self._toggle_wifi_metered)
    wifi_metered_btn = ListItem(tr_lazy("Wi-Fi Network Metered"), description=tr_lazy("Prevent large data uploads when on a metered Wi-Fi connection"),
                                action_item=self._wifi_metered_action)

    items: list[Widget] = [
      tethering_btn,
      tethering_password_btn,
      text_item(tr_lazy("IP Address"), lambda: self._wifi_manager.ipv4_address),
      self._roaming_btn,
      self._apn_btn,
      self._cellular_metered_btn,
      wifi_metered_btn,
      button_item(tr_lazy("Hidden Network"), tr_lazy("CONNECT"), callback=self._connect_to_hidden_network),
    ]

    self._scroller = Scroller(items, line_separator=True, spacing=0)

    # Set initial config
    metered = self._params.get_bool("GsmMetered")
    self._wifi_manager.update_gsm_settings(roaming_enabled, self._params.get("GsmApn") or "", metered)

  def _on_network_updated(self, networks: list[Network]):
    self._tethering_action.set_enabled(True)
    self._tethering_action.set_state(self._wifi_manager.is_tethering_active())
    self._tethering_password_action.set_enabled(True)

    if self._wifi_manager.is_tethering_active() or self._wifi_manager.ipv4_address == "":
      self._wifi_metered_action.set_enabled(False)
      self._wifi_metered_action.selected_button = 0
    elif self._wifi_manager.ipv4_address != "":
      metered = self._wifi_manager.current_network_metered
      self._wifi_metered_action.set_enabled(True)
      self._wifi_metered_action.selected_button = int(metered) if metered in (MeteredType.UNKNOWN, MeteredType.YES, MeteredType.NO) else 0

  def _toggle_tethering(self):
    checked = self._tethering_action.get_state()
    self._tethering_action.set_enabled(False)
    if checked:
      self._wifi_metered_action.set_enabled(False)
    self._wifi_manager.set_tethering_active(checked)

  def _toggle_roaming(self):
    roaming_state = self._roaming_action.get_state()
    self._params.put_bool("GsmRoaming", roaming_state)
    self._wifi_manager.update_gsm_settings(roaming_state, self._params.get("GsmApn") or "", self._params.get_bool("GsmMetered"))

  def _edit_apn(self):
    def update_apn(result):
      if result != 1:
        return

      apn = self._keyboard.text.strip()
      if apn == "":
        self._params.remove("GsmApn")
      else:
        self._params.put("GsmApn", apn)

      self._wifi_manager.update_gsm_settings(self._params.get_bool("GsmRoaming"), apn, self._params.get_bool("GsmMetered"))

    current_apn = self._params.get("GsmApn") or ""
    self._keyboard.reset(min_text_size=0)
    self._keyboard.set_title(tr("Enter APN"), tr("leave blank for automatic configuration"))
    self._keyboard.set_text(current_apn)
    gui_app.set_modal_overlay(self._keyboard, update_apn)

  def _toggle_cellular_metered(self):
    metered = self._cellular_metered_action.get_state()
    self._params.put_bool("GsmMetered", metered)
    self._wifi_manager.update_gsm_settings(self._params.get_bool("GsmRoaming"), self._params.get("GsmApn") or "", metered)

  def _toggle_wifi_metered(self, metered):
    metered_type = {0: MeteredType.UNKNOWN, 1: MeteredType.YES, 2: MeteredType.NO}.get(metered, MeteredType.UNKNOWN)
    self._wifi_metered_action.set_enabled(False)
    self._wifi_manager.set_current_network_metered(metered_type)

  def _connect_to_hidden_network(self):
    def connect_hidden(result):
      if result != 1:
        return

      ssid = self._keyboard.text
      if not ssid:
        return

      def enter_password(result):
        password = self._keyboard.text
        if password == "":
          # connect without password
          self._wifi_manager.connect_to_network(ssid, "", hidden=True)
          return

        self._wifi_manager.connect_to_network(ssid, password, hidden=True)

      self._keyboard.reset(min_text_size=0)
      self._keyboard.set_title(tr("Enter password"), tr("for \"{}\"").format(ssid))
      gui_app.set_modal_overlay(self._keyboard, enter_password)

    self._keyboard.reset(min_text_size=1)
    self._keyboard.set_title(tr("Enter SSID"), "")
    gui_app.set_modal_overlay(self._keyboard, connect_hidden)

  def _edit_tethering_password(self):
    def update_password(result):
      if result != 1:
        return

      password = self._keyboard.text
      self._wifi_manager.set_tethering_password(password)
      self._tethering_password_action.set_enabled(False)

    self._keyboard.reset(min_text_size=MIN_PASSWORD_LENGTH)
    self._keyboard.set_title(tr("Enter new tethering password"), "")
    self._keyboard.set_text(self._wifi_manager.tethering_password)
    gui_app.set_modal_overlay(self._keyboard, update_password)

  def _update_state(self):
    self._wifi_manager.process_callbacks()

    # If not using prime SIM, show GSM settings and enable IPv4 forwarding
    show_cell_settings = ui_state.prime_state.get_type() in (PrimeType.NONE, PrimeType.LITE)
    self._wifi_manager.set_ipv4_forward(show_cell_settings)
    self._roaming_btn.set_visible(show_cell_settings)
    self._apn_btn.set_visible(show_cell_settings)
    self._cellular_metered_btn.set_visible(show_cell_settings)

  def _render(self, _):
    self._scroller.render(self._rect)


class WifiManagerUI(Widget):
  def __init__(self, wifi_manager: WifiManager):
    super().__init__()
    self._wifi_manager = wifi_manager
    self._network_items: dict[str, NetworkItem] = {}
    self._scroller = Scroller([], line_separator=True, spacing=0)
    self.keyboard = Keyboard(max_text_size=MAX_PASSWORD_LENGTH, min_text_size=MIN_PASSWORD_LENGTH, show_password_toggle=True)
    self._wifi_manager.add_callbacks(need_auth=self._on_need_auth,
                                     activated=self._clear_state,
                                     forgotten=self._clear_state,
                                     networks_updated=self._on_network_updated,
                                     disconnected=self._clear_state)

  def show_event(self):
    # start/stop scanning when widget is visible
    self._wifi_manager.set_active(True)

  def hide_event(self):
    self._wifi_manager.set_active(False)

  def _update_layout_rects(self) -> None:
    for item in self._network_items.values():
      item.set_rect(rl.Rectangle(0, 0, self._rect.width, ITEM_HEIGHT))

  def _update_state(self):
    self._wifi_manager.process_callbacks()

  def _on_network_updated(self, networks: list[Network]):
    new_items = {}
    for net in networks:
      item = self._network_items.get(net.ssid) or NetworkItem(
        net, self._networks_buttons_callback, self._forget_networks_buttons_callback
      )
      item.network = net
      item.set_rect(rl.Rectangle(0, 0, self._rect.width, ITEM_HEIGHT))
      new_items[net.ssid] = item

    self._network_items = new_items
    self._scroller.set_widgets(list(self._network_items.values()))

  def _render(self, rect: rl.Rectangle):
    if not self._network_items:
      gui_label(rect, tr("Scanning Wi-Fi networks..."), 72, align=Align.CENTER)
      return
    self._scroller.render(self._rect)

  def _auth_required(self, network: Network, wrong: bool = False):
    self.keyboard.set_title(tr("Wrong password") if wrong else tr("Enter password"), tr("for \"{}\"").format(network.ssid))
    self.keyboard.reset(min_text_size=MIN_PASSWORD_LENGTH)
    gui_app.set_modal_overlay(self.keyboard, lambda res: res == 1 and self.connect_to_network(network, self.keyboard.text))

  def _networks_buttons_callback(self, network):
    if not network.is_saved and network.security_type != SecurityType.OPEN:
      self._auth_required(network)
    elif not network.is_connected:
      self.connect_to_network(network)

  def _forget_networks_buttons_callback(self, network):
    dlg = ConfirmDialog(tr("Forget Wi-Fi Network \"{}\"?").format(network.ssid), tr("Forget"), tr("Cancel"))
    gui_app.set_modal_overlay(dlg, lambda res: res == 1 and self.forget_network(network))

  def connect_to_network(self, network: Network, password=''):
    if item := self._network_items.get(network.ssid):
      item.state = UIState.CONNECTING
      if network.is_saved and not password:
        self._wifi_manager.activate_connection(network.ssid)
      else:
        self._wifi_manager.connect_to_network(network.ssid, password)

  def forget_network(self, network: Network):
    if item := self._network_items.get(network.ssid):
      item.state = UIState.FORGETTING
      self._wifi_manager.forget_connection(network.ssid)

  def _on_need_auth(self, ssid):
    if item := self._network_items.get(ssid):
      self._auth_required(item.network, wrong=True)

  def _clear_state(self):
    for item in self._network_items.values():
      item.state = UIState.IDLE


def main():
  gui_app.init_window("Wi-Fi Manager")
  wifi_ui = WifiManagerUI(WifiManager())

  for _ in gui_app.render():
    wifi_ui.render(rl.Rectangle(50, 50, gui_app.width - 100, gui_app.height - 100))

  gui_app.close()


if __name__ == "__main__":
  main()
