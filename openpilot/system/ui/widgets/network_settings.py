from openpilot.system.ui.widgets.network import WifiManagerUI
from enum import IntEnum

import raylib_nb as rl
from openpilot.system.ui.lib.application import gui_app, TextAlignment
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.wifi_manager import WifiManager, Network, MeteredType
from openpilot.system.ui.widgets import DialogResult, Widget
from openpilot.system.ui.widgets.keyboard import Keyboard
from openpilot.system.ui.widgets.label import gui_label
from openpilot.system.ui.widgets.scroller_tici import Scroller
from openpilot.system.ui.widgets.list_view import ButtonAction, ListItem, MultipleButtonAction, ToggleAction, button_item, text_item

NM_DEVICE_STATE_NEED_AUTH = 60
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 64
ITEM_HEIGHT = 160
ICON_SIZE = 50

STRENGTH_ICONS = [
  "icons/wifi_strength_low.png",
  "icons/wifi_strength_medium.png",
  "icons/wifi_strength_high.png",
  "icons/wifi_strength_full.png",
]


class PanelType(IntEnum):
  WIFI = 0
  ADVANCED = 1


class UIState(IntEnum):
  IDLE = 0
  CONNECTING = 1
  NEEDS_AUTH = 2
  SHOW_FORGET_CONFIRM = 3
  FORGETTING = 4


class NavButton(Widget):
  def __init__(self, text: str):
    super().__init__()
    self.text = text
    self.set_rect(rl.Rectangle(0, 0, 400, 100))

  def _render(self, _):
    color = rl.Color(74, 74, 74, 255) if self.is_pressed else rl.Color(57, 57, 57, 255)
    rl.draw_rectangle_rounded(self._rect, 0.6, 10, color)
    gui_label(self.rect, self.text, font_size=60, alignment=TextAlignment.CENTER)


class NetworkUI(Widget):
  def __init__(self, wifi_manager: WifiManager):
    super().__init__()
    self._wifi_manager = wifi_manager
    self._current_panel: PanelType = PanelType.WIFI
    self._wifi_panel = self._child(WifiManagerUI(wifi_manager))
    self._advanced_panel = self._child(AdvancedNetworkSettings(wifi_manager))
    self._nav_button = self._child(NavButton(tr("Advanced")))
    self._nav_button.set_click_callback(self._cycle_panel)

  def show_event(self):
    super().show_event()
    self._set_current_panel(PanelType.WIFI)

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
    # AdvancedNetworkSettings needs the full openpilot environment, standalone apps just use WifiManagerUI
    from openpilot.common.params import Params
    from openpilot.selfdrive.ui.ui_state import ui_state
    from openpilot.selfdrive.ui.lib.prime_state import PrimeType
    super().__init__()
    self._wifi_manager = wifi_manager
    self._wifi_manager.add_callbacks(networks_updated=self._on_network_updated)
    self._params = Params()
    self._prime_state = ui_state.prime_state
    self._cell_prime_types = (PrimeType.NONE, PrimeType.LITE)

    self._keyboard = Keyboard(max_text_size=MAX_PASSWORD_LENGTH, min_text_size=MIN_PASSWORD_LENGTH, show_password_toggle=True)

    # Tethering
    self._tethering_action = ToggleAction(initial_state=False)
    tethering_btn = ListItem(lambda: tr("Enable Tethering"), action_item=self._tethering_action, callback=self._toggle_tethering)

    # Edit tethering password
    self._tethering_password_action = ButtonAction(lambda: tr("EDIT"))
    tethering_password_btn = ListItem(lambda: tr("Tethering Password"), action_item=self._tethering_password_action, callback=self._edit_tethering_password)

    # Roaming toggle
    roaming_enabled = self._params.get_bool("GsmRoaming")
    self._roaming_action = ToggleAction(initial_state=roaming_enabled)
    self._roaming_btn = ListItem(lambda: tr("Enable Roaming"), action_item=self._roaming_action, callback=self._toggle_roaming)

    # Cellular metered toggle
    cellular_metered = self._params.get_bool("GsmMetered")
    self._cellular_metered_action = ToggleAction(initial_state=cellular_metered)
    self._cellular_metered_btn = ListItem(lambda: tr("Cellular Metered"),
                                          description=lambda: tr("Prevent large data uploads when on a metered cellular connection"),
                                          action_item=self._cellular_metered_action, callback=self._toggle_cellular_metered)

    # APN setting
    self._apn_btn = button_item(lambda: tr("APN Setting"), lambda: tr("EDIT"), callback=self._edit_apn)

    # Wi-Fi metered toggle
    self._wifi_metered_action = MultipleButtonAction([lambda: tr("default"), lambda: tr("metered"), lambda: tr("unmetered")], 255, 0,
                                                     callback=self._toggle_wifi_metered)
    wifi_metered_btn = ListItem(lambda: tr("Wi-Fi Network Metered"), description=lambda: tr("Prevent large data uploads when on a metered Wi-Fi connection"),
                                action_item=self._wifi_metered_action)

    items: list[Widget] = [
      tethering_btn,
      tethering_password_btn,
      text_item(lambda: tr("IP Address"), lambda: self._wifi_manager.ipv4_address),
      self._roaming_btn,
      self._apn_btn,
      self._cellular_metered_btn,
      wifi_metered_btn,
      button_item(lambda: tr("Hidden Network"), lambda: tr("CONNECT"), callback=self._connect_to_hidden_network),
    ]

    self._scroller = Scroller(items, line_separator=True, spacing=0)

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
    self._params.put_bool("GsmRoaming", self._roaming_action.get_state(), block=True)

  def _edit_apn(self):
    def update_apn(result: DialogResult):
      if result != DialogResult.CONFIRM:
        return

      apn = self._keyboard.text.strip()
      if apn == "":
        self._params.remove("GsmApn")
      else:
        self._params.put("GsmApn", apn, block=True)

    current_apn = self._params.get("GsmApn") or ""
    self._keyboard.reset(min_text_size=0)
    self._keyboard.set_title(tr("Enter APN"), tr("leave blank for automatic configuration"))
    self._keyboard.set_text(current_apn)
    self._keyboard.set_callback(update_apn)
    gui_app.push_widget(self._keyboard)

  def _toggle_cellular_metered(self):
    self._params.put_bool("GsmMetered", self._cellular_metered_action.get_state(), block=True)

  def _toggle_wifi_metered(self, metered):
    metered_type = {0: MeteredType.UNKNOWN, 1: MeteredType.YES, 2: MeteredType.NO}.get(metered, MeteredType.UNKNOWN)
    self._wifi_metered_action.set_enabled(False)
    self._wifi_manager.set_current_network_metered(metered_type)

  def _connect_to_hidden_network(self):
    def connect_hidden(result: DialogResult):
      if result != DialogResult.CONFIRM:
        return

      ssid = self._keyboard.text
      if not ssid:
        return

      def enter_password(result: DialogResult):
        if result != DialogResult.CONFIRM:
          return

        password = self._keyboard.text
        if password == "":
          # connect without password
          self._wifi_manager.connect_to_network(ssid, "", hidden=True)
          return

        self._wifi_manager.connect_to_network(ssid, password, hidden=True)

      self._keyboard.reset(min_text_size=0)
      self._keyboard.set_title(tr("Enter password"), tr("for \"{}\"").format(ssid))
      self._keyboard.set_callback(enter_password)
      gui_app.push_widget(self._keyboard)

    self._keyboard.reset(min_text_size=1)
    self._keyboard.set_title(tr("Enter SSID"), "")
    self._keyboard.set_callback(connect_hidden)
    gui_app.push_widget(self._keyboard)

  def _edit_tethering_password(self):
    def update_password(result: DialogResult):
      if result != DialogResult.CONFIRM:
        return

      password = self._keyboard.text
      self._wifi_manager.set_tethering_password(password)
      self._tethering_password_action.set_enabled(False)

    self._keyboard.reset(min_text_size=MIN_PASSWORD_LENGTH)
    self._keyboard.set_title(tr("Enter new tethering password"), "")
    self._keyboard.set_text(self._wifi_manager.tethering_password)
    self._keyboard.set_callback(update_password)
    gui_app.push_widget(self._keyboard)

  def _update_state(self):
    self._wifi_manager.process_callbacks()

    # If not using prime SIM, show GSM settings and enable IPv4 forwarding
    show_cell_settings = self._prime_state.get_type() in self._cell_prime_types
    self._wifi_manager.set_ipv4_forward(show_cell_settings)
    self._roaming_btn.set_visible(show_cell_settings)
    self._apn_btn.set_visible(show_cell_settings)
    self._cellular_metered_btn.set_visible(show_cell_settings)

  def _render(self, _):
    self._scroller.render(self._rect)


