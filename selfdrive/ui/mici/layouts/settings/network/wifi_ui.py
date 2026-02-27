import math
import numpy as np
import pyray as rl
from collections.abc import Callable

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.ui.mici.widgets.dialog import BigInputDialog, BigConfirmationDialogV2
from openpilot.selfdrive.ui.mici.widgets.button import BigButton, LABEL_COLOR
from openpilot.system.ui.lib.application import gui_app, MousePos, FontWeight
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.scroller import NavScroller
from openpilot.system.ui.lib.wifi_manager import WifiManager, Network, SecurityType, normalize_ssid


class LoadingAnimation(Widget):
  HIDE_TIME = 4

  def __init__(self):
    super().__init__()
    self._opacity_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)
    self._opacity_target = 1.0
    self._hide_time = 0.0

  def show_event(self):
    self._opacity_target = 1.0
    self._hide_time = rl.get_time()

  def _render(self, _):
    if rl.get_time() - self._hide_time > self.HIDE_TIME:
      self._opacity_target = 0.0

    self._opacity_filter.update(self._opacity_target)

    if self._opacity_filter.x < 0.01:
      return

    cx = int(self._rect.x + self._rect.width / 2)
    cy = int(self._rect.y + self._rect.height / 2)

    y_mag = 7
    anim_scale = 4
    spacing = 14

    for i in range(3):
      x = cx - spacing + i * spacing
      y = int(cy + min(math.sin((rl.get_time() - i * 0.2) * anim_scale) * y_mag, 0))
      alpha = int(np.interp(cy - y, [0, y_mag], [255 * 0.45, 255 * 0.9]) * self._opacity_filter.x)
      rl.draw_circle(x, y, 5, rl.Color(255, 255, 255, alpha))


class WifiIcon(Widget):
  def __init__(self, network: Network):
    super().__init__()
    self.set_rect(rl.Rectangle(0, 0, 48 + 5, 36 + 5))

    self._wifi_slash_txt = gui_app.texture("icons_mici/settings/network/wifi_strength_slash.png", 48, 42)
    self._wifi_low_txt = gui_app.texture("icons_mici/settings/network/wifi_strength_low.png", 48, 36)
    self._wifi_medium_txt = gui_app.texture("icons_mici/settings/network/wifi_strength_medium.png", 48, 36)
    self._wifi_full_txt = gui_app.texture("icons_mici/settings/network/wifi_strength_full.png", 48, 36)
    self._lock_txt = gui_app.texture("icons_mici/settings/network/new/lock.png", 21, 27)

    self._network: Network = network
    self._network_missing = False  # if network disappeared from scan results

  def update_network(self, network: Network):
    self._network = network

  def set_network_missing(self, missing: bool):
    self._network_missing = missing

  @staticmethod
  def get_strength_icon_idx(strength: int) -> int:
    return round(strength / 100 * 2)

  def _render(self, _):
    # Determine which wifi strength icon to use
    strength = self.get_strength_icon_idx(self._network.strength)
    if self._network_missing:
      strength_icon = self._wifi_slash_txt
    elif strength == 2:
      strength_icon = self._wifi_full_txt
    elif strength == 1:
      strength_icon = self._wifi_medium_txt
    else:
      strength_icon = self._wifi_low_txt

    rl.draw_texture_ex(strength_icon, (self._rect.x, self._rect.y + self._rect.height - strength_icon.height), 0.0, 1.0, rl.WHITE)

    # Render lock icon at lower right of wifi icon if secured
    if self._network.security_type not in (SecurityType.OPEN, SecurityType.UNSUPPORTED):
      lock_x = self._rect.x + self._rect.width - self._lock_txt.width
      lock_y = self._rect.y + self._rect.height - self._lock_txt.height + 6
      rl.draw_texture_ex(self._lock_txt, (lock_x, lock_y), 0.0, 1.0, rl.WHITE)


class WifiButton(BigButton):
  LABEL_PADDING = 98
  LABEL_WIDTH = 402 - 98 - 28  # button width - left padding - right padding
  SUB_LABEL_WIDTH = 402 - BigButton.LABEL_HORIZONTAL_PADDING * 2

  def __init__(self, network: Network, wifi_manager: WifiManager):
    super().__init__(normalize_ssid(network.ssid), scroll=True)

    self._network = network
    self._wifi_manager = wifi_manager

    self._wifi_icon = WifiIcon(network)
    self._forget_btn = ForgetButton(self._forget_network)
    self._check_txt = gui_app.texture("icons_mici/setup/driver_monitoring/dm_check.png", 32, 32)

    # Eager state (not sourced from Network)
    self._network_missing = False
    self._network_forgetting = False
    self._wrong_password = False

  def update_network(self, network: Network):
    self._network = network
    self._wifi_icon.update_network(network)

    # We can assume network is not missing if got new Network
    self._network_missing = False
    self._wifi_icon.set_network_missing(False)
    if self._is_connected or self._is_connecting:
      self._wrong_password = False

  def _forget_network(self):
    if self._network_forgetting:
      return

    self._network_forgetting = True
    self._forget_btn.set_visible(False)
    self._wifi_manager.forget_connection(self._network.ssid)

  def on_forgotten(self):
    self._network_forgetting = False
    self._forget_btn.set_visible(True)

  def set_network_missing(self, missing: bool):
    self._network_missing = missing
    self._wifi_icon.set_network_missing(missing)

  def set_wrong_password(self):
    self._wrong_password = True
    self.trigger_shake()

  @property
  def network(self) -> Network:
    return self._network

  @property
  def _show_forget_btn(self):
    if self._network.is_tethering:
      return False

    return (self._is_saved and not self._wrong_password) or self._is_connecting

  def _handle_mouse_release(self, mouse_pos: MousePos):
    if self._show_forget_btn and rl.check_collision_point_rec(mouse_pos, self._forget_btn.rect):
      return
    super()._handle_mouse_release(mouse_pos)

  def _get_label_font_size(self):
    return 48

  def _draw_content(self, btn_y: float):
    self._label.set_color(LABEL_COLOR)
    label_rect = rl.Rectangle(self._rect.x + self.LABEL_PADDING, btn_y + self.LABEL_VERTICAL_PADDING,
                              self.LABEL_WIDTH, self._rect.height - self.LABEL_VERTICAL_PADDING * 2)
    self._label.render(label_rect)

    if self.value:
      sub_label_x = self._rect.x + self.LABEL_HORIZONTAL_PADDING
      label_y = btn_y + self._rect.height - self.LABEL_VERTICAL_PADDING
      sub_label_w = self.SUB_LABEL_WIDTH - (self._forget_btn.rect.width if self._show_forget_btn else 0)
      sub_label_height = self._sub_label.get_content_height(sub_label_w)

      if self._is_connected and not self._network_forgetting:
        check_y = int(label_y - sub_label_height + (sub_label_height - self._check_txt.height) / 2)
        rl.draw_texture(self._check_txt, int(sub_label_x), check_y, rl.Color(255, 255, 255, int(255 * 0.9 * 0.65)))
        sub_label_x += self._check_txt.width + 14

      sub_label_rect = rl.Rectangle(sub_label_x, label_y - sub_label_height, sub_label_w, sub_label_height)
      self._sub_label.render(sub_label_rect)

    # Wifi icon
    self._wifi_icon.render(rl.Rectangle(
      self._rect.x + 30,
      btn_y + 30,
      self._wifi_icon.rect.width,
      self._wifi_icon.rect.height,
    ))

    # Forget button
    if self._show_forget_btn:
      self._forget_btn.render(rl.Rectangle(
        self._rect.x + self._rect.width - self._forget_btn.rect.width,
        btn_y + self._rect.height - self._forget_btn.rect.height,
        self._forget_btn.rect.width,
        self._forget_btn.rect.height,
      ))

  def set_touch_valid_callback(self, touch_callback: Callable[[], bool]) -> None:
    super().set_touch_valid_callback(lambda: touch_callback() and not self._forget_btn.is_pressed)
    self._forget_btn.set_touch_valid_callback(touch_callback)

  @property
  def _is_saved(self):
    return self._wifi_manager.is_connection_saved(self._network.ssid)

  @property
  def _is_connecting(self):
    return self._wifi_manager.connecting_to_ssid == self._network.ssid

  @property
  def _is_connected(self):
    return self._wifi_manager.connected_ssid == self._network.ssid

  def _update_state(self):
    if any((self._network_missing, self._is_connecting, self._is_connected, self._network_forgetting,
            self._network.security_type == SecurityType.UNSUPPORTED)):
      self.set_enabled(False)
      self._sub_label.set_color(rl.Color(255, 255, 255, int(255 * 0.585)))
      self._sub_label.set_font_weight(FontWeight.ROMAN)

      if self._network_forgetting:
        self.set_value("forgetting...")
      elif self._is_connecting:
        self.set_value("starting..." if self._network.is_tethering else "connecting...")
      elif self._is_connected:
        self.set_value("tethering" if self._network.is_tethering else "connected")
      elif self._network_missing:
        # after connecting/connected since NM will still attempt to connect/stay connected for a while
        self.set_value("not in range")
      else:
        self.set_value("unsupported")

    else:  # saved, wrong password, or unknown
      self.set_value("wrong password" if self._wrong_password else "connect")
      self.set_enabled(True)
      self._sub_label.set_color(rl.Color(255, 255, 255, int(255 * 0.9)))
      self._sub_label.set_font_weight(FontWeight.SEMI_BOLD)


class ForgetButton(Widget):
  MARGIN = 12  # bottom and right

  def __init__(self, forget_network: Callable):
    super().__init__()
    self._forget_network = forget_network

    self._bg_txt = gui_app.texture("icons_mici/settings/network/new/forget_button.png", 84, 84)
    self._bg_pressed_txt = gui_app.texture("icons_mici/settings/network/new/forget_button_pressed.png", 84, 84)
    self._trash_txt = gui_app.texture("icons_mici/settings/network/new/trash.png", 29, 35)
    self.set_rect(rl.Rectangle(0, 0, 84 + self.MARGIN * 2, 84 + self.MARGIN * 2))

  def _handle_mouse_release(self, mouse_pos: MousePos):
    super()._handle_mouse_release(mouse_pos)
    dlg = BigConfirmationDialogV2("slide to forget", "icons_mici/settings/network/new/trash.png", red=True,
                                  confirm_callback=self._forget_network)
    gui_app.push_widget(dlg)

  def _render(self, _):
    bg_txt = self._bg_pressed_txt if self.is_pressed else self._bg_txt
    rl.draw_texture_ex(bg_txt, (self._rect.x + (self._rect.width - self._bg_txt.width) / 2,
                                self._rect.y + (self._rect.height - self._bg_txt.height) / 2), 0, 1.0, rl.WHITE)

    trash_x = self._rect.x + (self._rect.width - self._trash_txt.width) / 2
    trash_y = self._rect.y + (self._rect.height - self._trash_txt.height) / 2
    rl.draw_texture_ex(self._trash_txt, (trash_x, trash_y), 0, 1.0, rl.WHITE)


class WifiUIMici(NavScroller):
  def __init__(self, wifi_manager: WifiManager):
    super().__init__()

    # Set up back navigation
    self.set_back_callback(gui_app.pop_widget)

    self._loading_animation = LoadingAnimation()

    self._wifi_manager = wifi_manager
    self._networks: dict[str, Network] = {}

    self._wifi_manager.add_callbacks(
      need_auth=self._on_need_auth,
      forgotten=self._on_forgotten,
      networks_updated=self._on_network_updated,
    )

  def show_event(self):
    # Clear scroller items and update from latest scan results
    super().show_event()
    self._loading_animation.show_event()
    self._wifi_manager.set_active(True)
    self._scroller.items.clear()
    # trigger button update on latest sorted networks
    self._on_network_updated(self._wifi_manager.networks)

  def _on_network_updated(self, networks: list[Network]):
    self._networks = {network.ssid: network for network in networks}
    self._update_buttons()

  def _update_buttons(self):
    # Update existing buttons, add new ones to the end
    existing = {btn.network.ssid: btn for btn in self._scroller.items if isinstance(btn, WifiButton)}

    for network in self._networks.values():
      if network.ssid in existing:
        existing[network.ssid].update_network(network)
      else:
        btn = WifiButton(network, self._wifi_manager)
        btn.set_click_callback(lambda ssid=network.ssid: self._connect_to_network(ssid))
        self._scroller.add_widget(btn)

    # Mark networks no longer in scan results (display handled by _update_state)
    for btn in self._scroller.items:
      if isinstance(btn, WifiButton) and btn.network.ssid not in self._networks:
        btn.set_network_missing(True)

  def _connect_with_password(self, ssid: str, password: str):
    self._wifi_manager.connect_to_network(ssid, password)
    self._move_network_to_front(ssid, scroll=True)

  def _connect_to_network(self, ssid: str):
    network = self._networks.get(ssid)
    if network is None:
      cloudlog.warning(f"Trying to connect to unknown network: {ssid}")
      return

    if self._wifi_manager.is_connection_saved(network.ssid):
      self._wifi_manager.activate_connection(network.ssid)
    elif network.security_type == SecurityType.OPEN:
      self._wifi_manager.connect_to_network(network.ssid, "")
    else:
      self._on_need_auth(network.ssid, False)
      return

    self._move_network_to_front(ssid, scroll=True)

  def _on_need_auth(self, ssid, incorrect_password=True):
    if incorrect_password:
      for btn in self._scroller.items:
        if isinstance(btn, WifiButton) and btn.network.ssid == ssid:
          btn.set_wrong_password()
          break
      return

    dlg = BigInputDialog("enter password...", "", minimum_length=8,
                         confirm_callback=lambda _password: self._connect_with_password(ssid, _password))
    gui_app.push_widget(dlg)

  def _on_forgotten(self, ssid):
    # For eager UI forget
    for btn in self._scroller.items:
      if isinstance(btn, WifiButton) and btn.network.ssid == ssid:
        btn.on_forgotten()

  def _move_network_to_front(self, ssid: str | None, scroll: bool = False):
    # Move connecting/connected network to the front with animation
    front_btn_idx = next((i for i, btn in enumerate(self._scroller.items)
                          if isinstance(btn, WifiButton) and
                          btn.network.ssid == ssid), None) if ssid else None

    if front_btn_idx is not None and front_btn_idx > 0:
      self._scroller.move_item(front_btn_idx, 0)

      if scroll:
        # Scroll to the new position of the network
        self._scroller.scroll_to(self._scroller.scroll_panel.get_offset(), smooth=True)

  def _update_state(self):
    super()._update_state()

    self._move_network_to_front(self._wifi_manager.wifi_state.ssid)

    # Show loading animation near end
    max_scroll = max(self._scroller.content_size - self._scroller.rect.width, 1)
    progress = -self._scroller.scroll_panel.get_offset() / max_scroll
    if progress > 0.8 or len(self._scroller.items) <= 1:
      self._loading_animation.show_event()

  def _render(self, _):
    super()._render(self._rect)

    anim_w = 90
    anim_x = self._rect.x + self._rect.width - anim_w
    anim_y = self._rect.y + self._rect.height - 25 + 2
    self._loading_animation.render(rl.Rectangle(anim_x, anim_y, anim_w, 20))
