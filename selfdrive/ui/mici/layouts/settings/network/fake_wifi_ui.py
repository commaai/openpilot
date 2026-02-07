import pyray as rl
from collections.abc import Callable

from openpilot.system.ui.widgets.scroller import Scroller
from openpilot.selfdrive.ui.mici.widgets.dialog import BigInputDialog, BigConfirmationDialogV2
from openpilot.system.ui.lib.application import gui_app, FontWeight, MousePos
from openpilot.system.ui.widgets import Widget, NavWidget
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.common.filter_simple import BounceFilter

# Dimensions matching BigButton
WIFI_BUTTON_WIDTH = 402
WIFI_BUTTON_HEIGHT = 180
WIFI_BUTTON_CORNER_RADIUS = 36

# Padding
TOP_LEFT_PADDING = 30
BOTTOM_LEFT_PADDING = 30
BOTTOM_RIGHT_PADDING = 15

# Font sizes
SSID_FONT_SIZE = 42
STATUS_FONT_SIZE = 36


class ForgetButton(Widget):
  """Small forget button for the wifi button."""

  def __init__(self, forget_callback: Callable):
    super().__init__()
    self._forget_callback = forget_callback
    self._click_handled = False  # Flag to indicate click was handled

    self._bg_txt = gui_app.texture("icons_mici/settings/network/new/forget_button.png", 100, 100)
    self._bg_pressed_txt = gui_app.texture("icons_mici/settings/network/new/forget_button_pressed.png", 100, 100)
    self._trash_txt = gui_app.texture("icons_mici/settings/network/new/trash.png", 35, 42)
    self.set_rect(rl.Rectangle(0, 0, 100, 100))

  def _handle_mouse_release(self, mouse_pos: MousePos):
    super()._handle_mouse_release(mouse_pos)
    self._click_handled = True  # Mark that we handled this click
    if self._forget_callback:
      self._forget_callback()

  def clear_click_handled(self):
    """Reset the click handled flag. Call at start of each frame."""
    self._click_handled = False

  def _render(self, _):
    bg_txt = self._bg_pressed_txt if self.is_pressed else self._bg_txt
    rl.draw_texture(bg_txt, int(self._rect.x), int(self._rect.y), rl.WHITE)

    trash_x = int(self._rect.x + (self._rect.width - self._trash_txt.width) // 2)
    trash_y = int(self._rect.y + (self._rect.height - self._trash_txt.height) // 2)
    rl.draw_texture(self._trash_txt, trash_x, trash_y, rl.WHITE)


class FakeWifiButton(Widget):
  """Custom wifi button with signal icon, SSID, connect/disconnect label, and forget button."""

  PRESSED_SCALE = 1.07

  def __init__(self, ssid: str, on_connect: Callable, on_forget: Callable):
    super().__init__()
    self.set_rect(rl.Rectangle(0, 0, WIFI_BUTTON_WIDTH, WIFI_BUTTON_HEIGHT))

    self._ssid = ssid
    self._on_connect = on_connect
    self._on_forget = on_forget

    # State
    self._is_connected = False
    self._is_saved = False

    # Scale animation filter (matches BigButton)
    self._scale_filter = BounceFilter(1.0, 0.1, 1 / gui_app.target_fps)

    # Load button background textures (402x180, matching BigButton)
    self._txt_default_bg = gui_app.texture("icons_mici/buttons/button_rectangle.png", 402, 180)
    self._txt_pressed_bg = gui_app.texture("icons_mici/buttons/button_rectangle_hover.png", 402, 180)

    # Load wifi icon
    self._wifi_icon = gui_app.texture("icons_mici/settings/network/wifi_strength_full.png", 64, 47)

    # Fonts
    self._ssid_font = gui_app.font(FontWeight.SEMI_BOLD)
    self._status_font = gui_app.font(FontWeight.MEDIUM)

    # Forget button (only shown when saved)
    self._forget_btn = ForgetButton(self._handle_forget)

  def set_connected(self, connected: bool):
    self._is_connected = connected
    if connected:
      self._is_saved = True  # Connecting saves the network

  def set_saved(self, saved: bool):
    self._is_saved = saved
    if not saved:
      self._is_connected = False  # Forgetting also disconnects

  def _handle_forget(self):
    # Only call the callback - state changes happen in the confirmation dialog's callback
    if self._on_forget:
      self._on_forget(self._ssid)

  def _handle_mouse_release(self, mouse_pos: MousePos):
    # Don't trigger connect if forget button handled this click
    # The forget button processes its events during render(), before this is called
    if self._forget_btn._click_handled:
      return

    # Toggle connection state and call callback
    if self._on_connect:
      self._on_connect(self._ssid)

  def _render(self, _):
    # Clear forget button's click flag at the start of each frame
    self._forget_btn.clear_click_handled()

    # Draw button background texture with scale animation
    txt_bg = self._txt_pressed_bg if self.is_pressed else self._txt_default_bg
    scale = self._scale_filter.update(self.PRESSED_SCALE if self.is_pressed else 1.0)
    btn_x = self._rect.x + (self._rect.width * (1 - scale)) / 2
    btn_y = self._rect.y + (self._rect.height * (1 - scale)) / 2
    rl.draw_texture_ex(txt_bg, (btn_x, btn_y), 0, scale, rl.WHITE)

    # Draw wifi signal icon (top left with padding)
    icon_x = int(self._rect.x + TOP_LEFT_PADDING)
    icon_y = int(self._rect.y + TOP_LEFT_PADDING)
    rl.draw_texture(self._wifi_icon, icon_x, icon_y, rl.WHITE)

    # Draw SSID name (inline with icon, with ellipsis if too long)
    ssid_x = icon_x + self._wifi_icon.width + 15
    ssid_y = icon_y + (self._wifi_icon.height - SSID_FONT_SIZE) // 2

    # Calculate max width for SSID (leave space for padding on right)
    max_ssid_width = self._rect.width - (ssid_x - self._rect.x) - TOP_LEFT_PADDING

    # Truncate SSID with ellipsis if needed
    display_ssid = self._ssid
    ssid_size = measure_text_cached(self._ssid_font, display_ssid, SSID_FONT_SIZE)
    if ssid_size.x > max_ssid_width:
      while len(display_ssid) > 1:
        display_ssid = display_ssid[:-1]
        test_text = display_ssid + "..."
        test_size = measure_text_cached(self._ssid_font, test_text, SSID_FONT_SIZE)
        if test_size.x <= max_ssid_width:
          display_ssid = test_text
          break

    ssid_color = rl.Color(255, 255, 255, int(255 * 0.9))
    rl.draw_text_ex(self._ssid_font, display_ssid, rl.Vector2(ssid_x, ssid_y), SSID_FONT_SIZE, 0, ssid_color)

    # Draw connect/disconnect label (bottom left with padding)
    status_text = "disconnect" if self._is_connected else "connect"
    status_x = int(self._rect.x + BOTTOM_LEFT_PADDING)
    status_y = int(self._rect.y + self._rect.height - BOTTOM_LEFT_PADDING - STATUS_FONT_SIZE)
    status_color = rl.Color(255, 255, 255, int(255 * 0.65))
    rl.draw_text_ex(self._status_font, status_text, rl.Vector2(status_x, status_y), STATUS_FONT_SIZE, 0, status_color)

    # Draw forget button (bottom right, shown when network is saved)
    if self._is_saved:
      forget_x = self._rect.x + self._rect.width - BOTTOM_RIGHT_PADDING - self._forget_btn.rect.width
      forget_y = self._rect.y + self._rect.height - BOTTOM_RIGHT_PADDING - self._forget_btn.rect.height
      self._forget_btn.render(rl.Rectangle(forget_x, forget_y, self._forget_btn.rect.width, self._forget_btn.rect.height))


class FakeWifiUI(NavWidget):
  """A fake wifi selector for testing the new horizontal scroll layout."""

  def __init__(self, back_callback: Callable):
    super().__init__()
    self.set_back_callback(back_callback)

    # Create 20 fake networks
    self._network_buttons: list[FakeWifiButton] = []
    for i in range(1, 21):
      network_name = f"unifi{i}"
      btn = FakeWifiButton(
        network_name,
        on_connect=self._on_network_clicked,
        on_forget=self._on_network_forget
      )
      self._network_buttons.append(btn)

    self._scroller = Scroller(self._network_buttons, snap_items=False, scroll_bar_margin=True)

  def _reorder_networks(self):
    """Reorder network list so connected network is first, then saved networks, then others."""
    # Sort: connected first, then saved (not connected), then unsaved
    connected = [btn for btn in self._network_buttons if btn._is_connected]
    saved_not_connected = [btn for btn in self._network_buttons if btn._is_saved and not btn._is_connected]
    unsaved = [btn for btn in self._network_buttons if not btn._is_saved and not btn._is_connected]
    new_order = connected + saved_not_connected + unsaved

    # Update scroller's items list
    self._scroller._items[:] = new_order

  def _on_network_clicked(self, network_name: str):
    """Handle network button click - show password dialog or toggle connection."""
    # Find the button for this network
    btn = next((b for b in self._network_buttons if b._ssid == network_name), None)
    if btn is None:
      return

    if btn._is_connected:
      # Disconnect
      btn.set_connected(False)
      self._reorder_networks()
      print(f"Disconnected from {network_name}")
    elif btn._is_saved:
      # Saved network - reconnect without password
      for b in self._network_buttons:
        b.set_connected(False)
      btn.set_connected(True)
      self._reorder_networks()
      print(f"Reconnected to saved network {network_name}")
    else:
      # Show password dialog to connect
      def on_password_entered(password: str):
        if password:
          # Disconnect all other networks first
          for b in self._network_buttons:
            b.set_connected(False)
          # Connect to this network
          btn.set_connected(True)
          self._reorder_networks()
          print(f"Connected to {network_name} with password: {password}")

      dlg = BigInputDialog(
        f"enter password...",
        "",
        minimum_length=8,
        confirm_callback=on_password_entered
      )
      gui_app.set_modal_overlay(dlg)

  def _on_network_forget(self, network_name: str):
    """Handle forget button click."""
    def do_forget():
      btn = next((b for b in self._network_buttons if b._ssid == network_name), None)
      if btn:
        btn.set_saved(False)
        self._reorder_networks()
        print(f"Forgot network {network_name}")

    dlg = BigConfirmationDialogV2("slide to forget", "icons_mici/settings/network/new/trash.png", red=True,
                                  confirm_callback=do_forget)
    gui_app.set_modal_overlay(dlg)

  def show_event(self):
    super().show_event()
    self._scroller.show_event()

  def _render(self, rect: rl.Rectangle):
    self._scroller.render(rect)
