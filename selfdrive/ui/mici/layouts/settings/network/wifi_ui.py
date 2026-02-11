import math
import numpy as np
import pyray as rl
from collections.abc import Callable

from openpilot.common.swaglog import cloudlog
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.selfdrive.ui.mici.widgets.dialog import BigMultiOptionDialog, BigInputDialog, BigDialogOptionButton, BigConfirmationDialogV2
from openpilot.system.ui.lib.application import gui_app, MousePos, FontWeight
from openpilot.system.ui.widgets import Widget, NavWidget
from openpilot.system.ui.lib.wifi_manager import WifiManager, Network, SecurityType


def normalize_ssid(ssid: str) -> str:
  return ssid.replace("’", "'")  # for iPhone hotspots


class LoadingAnimation(Widget):
  def _render(self, _):
    cx = int(self._rect.x + 70)
    cy = int(self._rect.y + self._rect.height / 2 - 50)

    y_mag = 20
    anim_scale = 5
    spacing = 28

    for i in range(3):
      x = cx - spacing + i * spacing
      y = int(cy + min(math.sin((rl.get_time() - i * 0.2) * anim_scale) * y_mag, 0))
      alpha = int(np.interp(cy - y, [0, y_mag], [255 * 0.45, 255 * 0.9]))
      rl.draw_circle(x, y, 10, rl.Color(255, 255, 255, alpha))


class WifiIcon(Widget):
  def __init__(self):
    super().__init__()
    self.set_rect(rl.Rectangle(0, 0, 86, 64))

    self._wifi_low_txt = gui_app.texture("icons_mici/settings/network/wifi_strength_low.png", 86, 64)
    self._wifi_medium_txt = gui_app.texture("icons_mici/settings/network/wifi_strength_medium.png", 86, 64)
    self._wifi_full_txt = gui_app.texture("icons_mici/settings/network/wifi_strength_full.png", 86, 64)
    self._lock_txt = gui_app.texture("icons_mici/settings/network/new/lock.png", 22, 32)

    self._network: Network | None = None
    self._scale = 1.0

  def set_current_network(self, network: Network):
    self._network = network

  def set_scale(self, scale: float):
    self._scale = scale

  @staticmethod
  def get_strength_icon_idx(strength: int) -> int:
    return round(strength / 100 * 2)

  def _render(self, _):
    if self._network is None:
      return

    # Determine which wifi strength icon to use
    strength = self.get_strength_icon_idx(self._network.strength)
    if strength == 2:
      strength_icon = self._wifi_full_txt
    elif strength == 1:
      strength_icon = self._wifi_medium_txt
    else:
      strength_icon = self._wifi_low_txt

    icon_x = int(self._rect.x + (self._rect.width - strength_icon.width * self._scale) // 2)
    icon_y = int(self._rect.y + (self._rect.height - strength_icon.height * self._scale) // 2)
    rl.draw_texture_ex(strength_icon, (icon_x, icon_y), 0.0, self._scale, rl.WHITE)

    # Render lock icon at lower right of wifi icon if secured
    if self._network.security_type not in (SecurityType.OPEN, SecurityType.UNSUPPORTED):
      lock_scale = self._scale * 1.1
      lock_x = int(icon_x + 1 + strength_icon.width * self._scale - self._lock_txt.width * lock_scale / 2)
      lock_y = int(icon_y + 1 + strength_icon.height * self._scale - self._lock_txt.height * lock_scale / 2)
      rl.draw_texture_ex(self._lock_txt, (lock_x, lock_y), 0.0, lock_scale, rl.WHITE)


class WifiItem(BigDialogOptionButton):
  LEFT_MARGIN = 20

  def __init__(self, network: Network):
    super().__init__(network.ssid)

    self.set_rect(rl.Rectangle(0, 0, gui_app.width, self.HEIGHT))

    self._selected_txt = gui_app.texture("icons_mici/settings/network/new/wifi_selected.png", 48, 96)

    self._network = network
    self._wifi_icon = WifiIcon()
    self._wifi_icon.set_current_network(network)

  def set_current_network(self, network: Network):
    self._network = network
    self._wifi_icon.set_current_network(network)

  def _render(self, _):
    if self._network.is_connected:
      selected_x = int(self._rect.x - self._selected_txt.width / 2)
      selected_y = int(self._rect.y + (self._rect.height - self._selected_txt.height) / 2)
      rl.draw_texture(self._selected_txt, selected_x, selected_y, rl.WHITE)

    self._wifi_icon.set_scale((1.0 if self._selected else 0.65) * 0.7)
    self._wifi_icon.render(rl.Rectangle(
      self._rect.x + self.LEFT_MARGIN,
      self._rect.y,
      self.SELECTED_HEIGHT,
      self._rect.height
    ))

    if self._selected:
      self._label.set_font_size(self.SELECTED_HEIGHT)
      self._label.set_color(rl.Color(255, 255, 255, int(255 * 0.9)))
      self._label.set_font_weight(FontWeight.DISPLAY)
    else:
      self._label.set_font_size(self.HEIGHT)
      self._label.set_color(rl.Color(255, 255, 255, int(255 * 0.58)))
      self._label.set_font_weight(FontWeight.DISPLAY_REGULAR)

    label_offset = self.LEFT_MARGIN + self._wifi_icon.rect.width + 20
    label_rect = rl.Rectangle(self._rect.x + label_offset, self._rect.y, self._rect.width - label_offset, self._rect.height)
    self._label.set_text(normalize_ssid(self._network.ssid))
    self._label.render(label_rect)


class ConnectButton(Widget):
  def __init__(self):
    super().__init__()
    self._bg_txt = gui_app.texture("icons_mici/settings/network/new/connect_button.png", 410, 100)
    self._bg_pressed_txt = gui_app.texture("icons_mici/settings/network/new/connect_button_pressed.png", 410, 100)
    self._bg_full_txt = gui_app.texture("icons_mici/settings/network/new/full_connect_button.png", 520, 100)
    self._bg_full_pressed_txt = gui_app.texture("icons_mici/settings/network/new/full_connect_button_pressed.png", 520, 100)

    self._full: bool = False

    self._label = UnifiedLabel("", 36, FontWeight.MEDIUM, rl.Color(255, 255, 255, int(255 * 0.9)),
                               alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER,
                               alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE)

  @property
  def full(self) -> bool:
    return self._full

  def set_full(self, full: bool):
    self._full = full
    self.set_rect(rl.Rectangle(0, 0, 520 if self._full else 410, 100))

  def set_label(self, text: str):
    self._label.set_text(text)

  def _render(self, _):
    if self._full:
      bg_txt = self._bg_full_pressed_txt if self.is_pressed and self.enabled else self._bg_full_txt
    else:
      bg_txt = self._bg_pressed_txt if self.is_pressed and self.enabled else self._bg_txt

    rl.draw_texture(bg_txt, int(self._rect.x), int(self._rect.y), rl.WHITE)

    self._label.set_text_color(rl.Color(255, 255, 255, int(255 * 0.9) if self.enabled else int(255 * 0.9 * 0.65)))
    self._label.render(self._rect)


class ForgetButton(Widget):
  HORIZONTAL_MARGIN = 8

  def __init__(self, forget_network: Callable, open_network_manage_page):
    super().__init__()
    self._forget_network = forget_network
    self._open_network_manage_page = open_network_manage_page

    self._bg_txt = gui_app.texture("icons_mici/settings/network/new/forget_button.png", 100, 100)
    self._bg_pressed_txt = gui_app.texture("icons_mici/settings/network/new/forget_button_pressed.png", 100, 100)
    self._trash_txt = gui_app.texture("icons_mici/settings/network/new/trash.png", 35, 42)
    self.set_rect(rl.Rectangle(0, 0, 100 + self.HORIZONTAL_MARGIN * 2, 100))

  def _handle_mouse_release(self, mouse_pos: MousePos):
    super()._handle_mouse_release(mouse_pos)
    dlg = BigConfirmationDialogV2("slide to forget", "icons_mici/settings/network/new/trash.png", red=True,
                                  confirm_callback=self._forget_network)
    gui_app.set_modal_overlay(dlg, callback=self._open_network_manage_page)

  def _render(self, _):
    bg_txt = self._bg_pressed_txt if self.is_pressed else self._bg_txt
    rl.draw_texture(bg_txt, int(self._rect.x + self.HORIZONTAL_MARGIN), int(self._rect.y), rl.WHITE)

    trash_x = int(self._rect.x + (self._rect.width - self._trash_txt.width) // 2)
    trash_y = int(self._rect.y + (self._rect.height - self._trash_txt.height) // 2)
    rl.draw_texture(self._trash_txt, trash_x, trash_y, rl.WHITE)


class NetworkInfoPage(NavWidget):
  def __init__(self, wifi_manager, connect_callback: Callable, forget_callback: Callable, open_network_manage_page: Callable):
    super().__init__()
    self._wifi_manager = wifi_manager

    self.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))

    self._wifi_icon = WifiIcon()
    self._forget_btn = ForgetButton(lambda: forget_callback(self._network.ssid) if self._network is not None else None,
                                    open_network_manage_page)
    self._connect_btn = ConnectButton()
    self._connect_btn.set_click_callback(lambda: connect_callback(self._network.ssid) if self._network is not None else None)

    self._title = UnifiedLabel("", 64, FontWeight.DISPLAY, rl.Color(255, 255, 255, int(255 * 0.9)),
                               alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE, scroll=True)
    self._subtitle = UnifiedLabel("", 36, FontWeight.ROMAN, rl.Color(255, 255, 255, int(255 * 0.9 * 0.65)),
                                  alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE)

    self.set_back_callback(lambda: gui_app.set_modal_overlay(None))

    # State
    self._network: Network | None = None
    self._connecting: Callable[[], str | None] | None = None

  def show_event(self):
    super().show_event()
    self._title.reset_scroll()

  def update_networks(self, networks: dict[str, Network]):
    # update current network from latest scan results
    for ssid, network in networks.items():
      if self._network is not None and ssid == self._network.ssid:
        self.set_current_network(network)
        break
    else:
      # network disappeared, close page
      gui_app.set_modal_overlay(None)

  def _update_state(self):
    super()._update_state()
    # Modal overlays stop main UI rendering, so we need to call here
    self._wifi_manager.process_callbacks()

    if self._network is None:
      return

    self._connect_btn.set_full(not self._network.is_saved and not self._is_connecting)
    if self._is_connecting:
      self._connect_btn.set_label("connecting...")
      self._connect_btn.set_enabled(False)
    elif self._network.is_connected:
      self._connect_btn.set_label("connected")
      self._connect_btn.set_enabled(False)
    elif self._network.security_type == SecurityType.UNSUPPORTED:
      self._connect_btn.set_label("connect")
      self._connect_btn.set_enabled(False)
    else:  # saved or unknown
      self._connect_btn.set_label("connect")
      self._connect_btn.set_enabled(True)

    self._title.set_text(normalize_ssid(self._network.ssid))
    if self._network.security_type == SecurityType.OPEN:
      self._subtitle.set_text("open")
    elif self._network.security_type == SecurityType.UNSUPPORTED:
      self._subtitle.set_text("unsupported")
    else:
      self._subtitle.set_text("secured")

  def set_current_network(self, network: Network):
    self._network = network
    self._wifi_icon.set_current_network(network)

  def set_connecting(self, is_connecting: Callable[[], str | None]):
    self._connecting = is_connecting

  @property
  def _is_connecting(self):
    if self._connecting is None or self._network is None:
      return False
    is_connecting = self._connecting() == self._network.ssid
    return is_connecting

  def _render(self, _):
    self._wifi_icon.render(rl.Rectangle(
      self._rect.x + 32,
      self._rect.y + (self._rect.height - self._connect_btn.rect.height - self._wifi_icon.rect.height) / 2,
      self._wifi_icon.rect.width,
      self._wifi_icon.rect.height,
    ))

    self._title.render(rl.Rectangle(
      self._rect.x + self._wifi_icon.rect.width + 32 + 32,
      self._rect.y + 32 - 16,
      self._rect.width - (self._wifi_icon.rect.width + 32 + 32),
      64,
    ))

    self._subtitle.render(rl.Rectangle(
      self._rect.x + self._wifi_icon.rect.width + 32 + 32,
      self._rect.y + 32 + 64 - 16,
      self._rect.width - (self._wifi_icon.rect.width + 32 + 32),
      48,
    ))

    self._connect_btn.render(rl.Rectangle(
      self._rect.x + 8,
      self._rect.y + self._rect.height - self._connect_btn.rect.height,
      self._connect_btn.rect.width,
      self._connect_btn.rect.height,
    ))

    if not self._connect_btn.full:
      self._forget_btn.render(rl.Rectangle(
        self._rect.x + self._rect.width - self._forget_btn.rect.width,
        self._rect.y + self._rect.height - self._forget_btn.rect.height,
        self._forget_btn.rect.width,
        self._forget_btn.rect.height,
      ))

    return -1


class WifiUIMici(BigMultiOptionDialog):
  # Wait this long after user interacts with widget to update network list
  INACTIVITY_TIMEOUT = 1

  def __init__(self, wifi_manager: WifiManager, back_callback: Callable):
    super().__init__([], None)

    # Set up back navigation
    self.set_back_callback(back_callback)

    self._network_info_page = NetworkInfoPage(wifi_manager, self._connect_to_network, self._forget_network, self._open_network_manage_page)
    self._network_info_page.set_connecting(lambda: self._connecting)

    self._loading_animation = LoadingAnimation()

    self._wifi_manager = wifi_manager
    self._connecting: str | None = None
    self._networks: dict[str, Network] = {}

    # widget state
    self._last_interaction_time = -float('inf')
    self._restore_selection = False

    self._wifi_manager.add_callbacks(
      need_auth=self._on_need_auth,
      activated=self._on_activated,
      forgotten=self._on_forgotten,
      networks_updated=self._on_network_updated,
      disconnected=self._on_disconnected,
    )

  def show_event(self):
    # Call super to prepare scroller; selection scroll is handled dynamically
    super().show_event()
    self._wifi_manager.set_active(True)
    self._last_interaction_time = -float('inf')

  def hide_event(self):
    super().hide_event()
    self._wifi_manager.set_active(False)

  def _open_network_manage_page(self, result=None):
    self._network_info_page.update_networks(self._networks)
    gui_app.set_modal_overlay(self._network_info_page)

  def _forget_network(self, ssid: str):
    network = self._networks.get(ssid)
    if network is None:
      cloudlog.warning(f"Trying to forget unknown network: {ssid}")
      return

    self._wifi_manager.forget_connection(network.ssid)

  def _on_network_updated(self, networks: list[Network]):
    self._networks = {network.ssid: network for network in networks}
    self._update_buttons()
    self._network_info_page.update_networks(self._networks)

  def _update_buttons(self):
    # Don't update buttons while user is actively interacting
    if rl.get_time() - self._last_interaction_time < self.INACTIVITY_TIMEOUT:
      return

    for network in self._networks.values():
      # pop and re-insert to eliminate stuttering on update (prevents position lost for a frame)
      network_button_idx = next((i for i, btn in enumerate(self._scroller._items) if btn.option == network.ssid), None)
      if network_button_idx is not None:
        network_button = self._scroller._items.pop(network_button_idx)
        # Update network on existing button
        network_button.set_current_network(network)
      else:
        network_button = WifiItem(network)

      self._scroller.add_widget(network_button)

    # remove networks no longer present
    self._scroller._items[:] = [btn for btn in self._scroller._items if btn.option in self._networks]

    # try to restore previous selection to prevent jumping from adding/removing/reordering buttons
    self._restore_selection = True

  def _connect_with_password(self, ssid: str, password: str):
    if password:
      self._connecting = ssid
      self._wifi_manager.connect_to_network(ssid, password)
      self._update_buttons()

  def _on_option_selected(self, option: str):
    super()._on_option_selected(option)

    if option in self._networks:
      self._network_info_page.set_current_network(self._networks[option])
      self._open_network_manage_page()

  def _connect_to_network(self, ssid: str):
    network = self._networks.get(ssid)
    if network is None:
      cloudlog.warning(f"Trying to connect to unknown network: {ssid}")
      return

    if network.is_saved:
      self._connecting = network.ssid
      self._wifi_manager.activate_connection(network.ssid)
      self._update_buttons()
    elif network.security_type == SecurityType.OPEN:
      self._connecting = network.ssid
      self._wifi_manager.connect_to_network(network.ssid, "")
      self._update_buttons()
    else:
      self._on_need_auth(network.ssid, False)

  def _on_need_auth(self, ssid, incorrect_password=True):
    hint = "incorrect password..." if incorrect_password else "enter password..."
    dlg = BigInputDialog(hint, "", minimum_length=8,
                         confirm_callback=lambda _password: self._connect_with_password(ssid, _password))
    # go back to the manage network page
    gui_app.set_modal_overlay(dlg, self._open_network_manage_page)

  def _on_activated(self):
    self._connecting = None

  def _on_forgotten(self):
    self._connecting = None

  def _on_disconnected(self):
    self._connecting = None

  def _update_state(self):
    super()._update_state()
    if self.is_pressed:
      self._last_interaction_time = rl.get_time()

  def _render(self, _):
    # Update Scroller layout and restore current selection whenever buttons are updated, before first render
    current_selection = self.get_selected_option()
    if self._restore_selection and current_selection in self._networks:
      self._scroller._layout()
      BigMultiOptionDialog._on_option_selected(self, current_selection)
      self._restore_selection = None

    super()._render(_)

    if not self._networks:
      self._loading_animation.render(self._rect)
