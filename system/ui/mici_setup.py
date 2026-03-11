#!/usr/bin/env python3
import os
import re
import ssl
import threading
import time
import urllib.request
import urllib.error
from urllib.parse import urlparse
from collections.abc import Callable

import pyray as rl

from cereal import log
from openpilot.common.filter_simple import BounceFilter
from openpilot.system.hardware import HARDWARE, TICI
from openpilot.common.realtime import config_realtime_process, set_core_affinity
from openpilot.common.swaglog import cloudlog
from openpilot.common.time_helpers import system_time_valid
from openpilot.common.utils import run_cmd
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.wifi_manager import WifiManager, ConnectStatus
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.nav_widget import NavWidget
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.system.ui.widgets.scroller import Scroller, NavScroller, ITEM_SPACING
from openpilot.system.ui.widgets.slider import LargerSlider
from openpilot.selfdrive.ui.mici.layouts.settings.network import WifiNetworkButton
from openpilot.selfdrive.ui.mici.layouts.settings.network.wifi_ui import WifiUIMici
from openpilot.selfdrive.ui.mici.widgets.dialog import BigInputDialog, BigConfirmationCircleButton
from openpilot.selfdrive.ui.mici.widgets.button import BigButton

NetworkType = log.DeviceState.NetworkType

OPENPILOT_URL = "https://openpilot.comma.ai"
USER_AGENT = f"AGNOSSetup-{HARDWARE.get_os_version()}"

INSTALLER_DESTINATION_PATH = "/tmp/installer"
INSTALLER_URL_PATH = "/tmp/installer_url"


class NetworkConnectivityMonitor:
  def __init__(self, should_check: Callable[[], bool] | None = None):
    self.network_connected = threading.Event()
    self.wifi_connected = threading.Event()
    self.recheck_event = threading.Event()
    self._should_check = should_check or (lambda: True)
    self._stop_event = threading.Event()
    self._last_timesyncd_restart = 0.0
    self._thread: threading.Thread | None = None

  def start(self):
    self._stop_event.clear()
    if self._thread is None or not self._thread.is_alive():
      self._thread = threading.Thread(target=self._run, daemon=True)
      self._thread.start()

  def stop(self):
    if self._thread is not None:
      self._stop_event.set()
      self._thread.join()
      self._thread = None

  def reset(self):
    self.network_connected.clear()
    self.wifi_connected.clear()

  def invalidate(self):
    self.recheck_event.set()
    self.reset()

  def _run(self):
    while not self._stop_event.is_set():
      if self._should_check():
        try:
          request = urllib.request.Request(OPENPILOT_URL, method="HEAD")
          urllib.request.urlopen(request, timeout=2.0)

          # Discard stale result if invalidated during request
          if self.recheck_event.is_set():
            self.recheck_event.clear()
            continue

          self.network_connected.set()
          if HARDWARE.get_network_type() == NetworkType.wifi:
            self.wifi_connected.set()
        except urllib.error.URLError as e:
          if (isinstance(e.reason, ssl.SSLCertVerificationError) and
              not system_time_valid() and
              time.monotonic() - self._last_timesyncd_restart > 5):
            self._last_timesyncd_restart = time.monotonic()
            run_cmd(["sudo", "systemctl", "restart", "systemd-timesyncd"])
          self.reset()
        except Exception:
          self.reset()
      else:
        self.reset()

      if self._stop_event.wait(timeout=1.0):
        break


class StartPage(Widget):
  def __init__(self):
    super().__init__()

    self._title = UnifiedLabel("start", 64, text_color=rl.Color(255, 255, 255, int(255 * 0.9)),
                               font_weight=FontWeight.DISPLAY, alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER,
                               alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE)

    self._start_bg_txt = gui_app.texture("icons_mici/setup/start_button.png", 500, 224, keep_aspect_ratio=False)
    self._start_bg_pressed_txt = gui_app.texture("icons_mici/setup/start_button_pressed.png", 500, 224, keep_aspect_ratio=False)
    self._scale_filter = BounceFilter(1.0, 0.1, 1 / gui_app.target_fps)
    self._click_delay = 0.075

  def _render(self, rect: rl.Rectangle):
    scale = self._scale_filter.update(1.07 if self.is_pressed else 1.0)
    base_draw_x = rect.x + (rect.width - self._start_bg_txt.width) / 2
    base_draw_y = rect.y + (rect.height - self._start_bg_txt.height) / 2
    draw_x = base_draw_x + (self._start_bg_txt.width * (1 - scale)) / 2
    draw_y = base_draw_y + (self._start_bg_txt.height * (1 - scale)) / 2
    texture = self._start_bg_pressed_txt if self.is_pressed else self._start_bg_txt
    rl.draw_texture_ex(texture, (draw_x, draw_y), 0, scale, rl.WHITE)

    self._title.render(rl.Rectangle(rect.x, rect.y + (draw_y - base_draw_y), rect.width, rect.height))


class SoftwareSelectionPage(NavWidget):
  def __init__(self, use_openpilot_callback: Callable,
               use_custom_software_callback: Callable):
    super().__init__()

    self._openpilot_slider = self._child(LargerSlider("slide to install\nopenpilot", use_openpilot_callback))
    self._openpilot_slider.set_enabled(lambda: self.enabled and not self.is_dismissing)
    self._custom_software_slider = self._child(LargerSlider("slide to install\ncustom software", use_custom_software_callback, green=False, shimmer_offset=0.4))
    self._custom_software_slider.set_enabled(lambda: self.enabled and not self.is_dismissing)

  def show_event(self):
    super().show_event()
    self._nav_bar._alpha = 0.0

  def _update_state(self):
    super()._update_state()
    if self.is_dismissing:
      self.reset()

  def reset(self):
    self._openpilot_slider.reset()
    self._custom_software_slider.reset()

  def _render(self, rect: rl.Rectangle):
    self._openpilot_slider.set_opacity(1.0 - self._custom_software_slider.slider_percentage)
    self._custom_software_slider.set_opacity(1.0 - self._openpilot_slider.slider_percentage)

    openpilot_rect = rl.Rectangle(
      rect.x + (rect.width - self._openpilot_slider.rect.width) / 2,
      rect.y,
      self._openpilot_slider.rect.width,
      rect.height / 2,
    )
    self._openpilot_slider.render(openpilot_rect)

    custom_software_rect = rl.Rectangle(
      rect.x + (rect.width - self._custom_software_slider.rect.width) / 2,
      rect.y + rect.height / 2,
      self._custom_software_slider.rect.width,
      rect.height / 2,
    )
    self._custom_software_slider.render(custom_software_rect)


class CustomSoftwareWarningPage(NavScroller):
  def __init__(self, continue_callback: Callable, back_callback: Callable):
    super().__init__()
    self.set_back_callback(back_callback)

    self._continue_button = BigPillButton("next")
    self._continue_button.set_click_callback(continue_callback)

    self._scroller.add_widgets([
      GreyBigButton("caution: installing\n3rd party software", "swipe down to go back",
                    gui_app.texture("icons_mici/setup/warning.png", 64, 58)),
      GreyBigButton("", "• It has not been tested by comma."),
      GreyBigButton("", "• It may not comply with safety standards."),
      GreyBigButton("", "• It may damage your device and/or vehicle."),
      GreyBigButton("how to restore to a\nfactory state later", "https://flash.comma.ai",
                    gui_app.texture("icons_mici/setup/restore.png", 64, 64)),
      self._continue_button,
    ])


# TODO: unifi with updater's progress page
class DownloadingPage(NavWidget):
  def __init__(self):
    super().__init__()

    self._title_label = UnifiedLabel("downloading...", 64, text_color=rl.Color(255, 255, 255, int(255 * 0.9)),
                                     font_weight=FontWeight.DISPLAY)
    self._progress_label = UnifiedLabel("", 132, text_color=rl.Color(255, 255, 255, int(255 * 0.9 * 0.65)),
                                        font_weight=FontWeight.ROMAN, alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_BOTTOM)
    self._progress = 0

  def _back_enabled(self) -> bool:
    return False

  def show_event(self):
    super().show_event()
    self._nav_bar._alpha = 0.0  # not dismissable
    self.set_progress(0)

  def set_progress(self, progress: int):
    self._progress = progress
    self._progress_label.set_text(f"{progress}%")

  def _render(self, rect: rl.Rectangle):
    rl.draw_rectangle_rec(rect, rl.BLACK)
    self._title_label.render(rl.Rectangle(
      rect.x + 12,
      rect.y + 2,
      rect.width,
      64,
    ))

    self._progress_label.render(rl.Rectangle(
      rect.x + 12,
      rect.y + 18,
      rect.width,
      rect.height,
    ))


class FailedPage(NavScroller):
  def __init__(self, retry_callback: Callable | None, title: str = "download failed",
               description: str | None = None, icon: str = "icons_mici/setup/warning.png"):
    super().__init__()
    self.set_back_callback(retry_callback)

    self._reason_card = GreyBigButton("", "")
    self._reason_card.set_visible(False)

    self._scroller.add_widgets([
      GreyBigButton(title, description or "swipe down to go\nback and try again",
                    gui_app.texture(icon, 64, 58)),
      self._reason_card,
      BigConfirmationCircleButton("reboot\ndevice", gui_app.texture("icons_mici/settings/device/reboot.png", 64, 70),
                                  HARDWARE.reboot, exit_on_confirm=False),
    ])

  def set_reason(self, reason: str):
    if reason:
      self._reason_card.set_value(reason)
      self._reason_card.set_visible(True)
    else:
      self._reason_card.set_visible(False)


class GreyBigButton(BigButton):
  """Users should manage newlines with this class themselves"""

  LABEL_HORIZONTAL_PADDING = 30

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.set_touch_valid_callback(lambda: False)

    self._rect.width = 476

    self._label.set_font_size(36)
    self._label.set_font_weight(FontWeight.BOLD)
    self._label.set_line_height(1.0)

    self._sub_label.set_font_size(36)
    self._sub_label.set_text_color(rl.Color(255, 255, 255, int(255 * 0.9)))
    self._sub_label.set_font_weight(FontWeight.DISPLAY_REGULAR)
    self._sub_label.set_alignment_vertical(rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE if not self._label.text else
                                           rl.GuiTextAlignmentVertical.TEXT_ALIGN_BOTTOM)
    self._sub_label.set_line_height(0.95)

  @property
  def LABEL_VERTICAL_PADDING(self):
    return BigButton.LABEL_VERTICAL_PADDING if self._label.text else 18

  def _width_hint(self) -> int:
    return int(self._rect.width - self.LABEL_HORIZONTAL_PADDING * 2)

  def _get_label_font_size(self):
    return 36

  def _render(self, _):
    rl.draw_rectangle_rounded(self._rect, 0.4, 10, rl.Color(255, 255, 255, int(255 * 0.15)))
    self._draw_content(self._rect.y)


class BigPillButton(BigButton):
  def __init__(self, *args, green: bool = False, disabled_background: bool = False, **kwargs):
    self._green = green
    self._disabled_background = disabled_background
    super().__init__(*args, **kwargs)

    self._label.set_font_size(48)
    self._label.set_alignment(rl.GuiTextAlignment.TEXT_ALIGN_CENTER)
    self._label.set_alignment_vertical(rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE)

  def _load_images(self):
    if self._green:
      self._txt_default_bg = gui_app.texture("icons_mici/setup/start_button.png", 402, 180)
      self._txt_pressed_bg = gui_app.texture("icons_mici/setup/start_button_pressed.png", 402, 180)
    else:
      self._txt_default_bg = gui_app.texture("icons_mici/setup/continue.png", 402, 180)
      self._txt_pressed_bg = gui_app.texture("icons_mici/setup/continue_pressed.png", 402, 180)
    self._txt_disabled_bg = gui_app.texture("icons_mici/setup/continue_disabled.png", 402, 180)

  def set_green(self, green: bool):
    if self._green != green:
      self._green = green
      self._load_images()

  def _update_label_layout(self):
    # Don't change label text size
    pass

  def _handle_background(self) -> tuple[rl.Texture, float, float, float]:
    txt_bg, btn_x, btn_y, scale = super()._handle_background()

    if self._disabled_background:
      txt_bg = self._txt_disabled_bg
    return txt_bg, btn_x, btn_y, scale


class NetworkSetupPageBase(Scroller):
  def __init__(self, network_monitor: NetworkConnectivityMonitor, continue_callback: Callable[[bool], None],
               disable_connect_hint: bool = False):
    super().__init__()

    self._wifi_manager = WifiManager()
    self._wifi_manager.set_active(True)
    self._network_monitor = network_monitor
    self._custom_software = False
    self._wifi_ui = WifiUIMici(self._wifi_manager)

    self._connect_button = GreyBigButton("connect to\ninternet", "swipe down to go back",
                                         gui_app.texture("icons_mici/setup/small_slider/slider_arrow.png", 64, 56, flip_x=True))
    self._connect_button.set_visible(not disable_connect_hint)

    self._wifi_button = WifiNetworkButton(self._wifi_manager)
    self._wifi_button.set_click_callback(lambda: gui_app.push_widget(self._wifi_ui))

    self._prev_has_internet = False
    self._prev_wifi_connected = False
    self._pending_has_internet_scroll: float | None = None  # stores time to use as delay
    self._pending_continue_grow_animation = False
    self._pending_wifi_grow_animation = False

    def on_waiting_click():
      offset = (self._wifi_button.rect.x + self._wifi_button.rect.width / 2) - (self._rect.x + self._rect.width / 2)
      self._scroller.scroll_to(offset, smooth=True, block_interaction=True)
      # trigger grow when wifi button in view
      self._pending_wifi_grow_animation = True

    self._waiting_button = BigPillButton("connect to\ncontinue", disabled_background=True)
    self._waiting_button.set_click_callback(on_waiting_click)
    self._continue_button = BigPillButton("install openpilot", green=True)
    self._continue_button.set_click_callback(lambda: continue_callback(self._custom_software))

    self._scroller.add_widgets([
      self._connect_button,
      self._wifi_button,
      self._continue_button,
      self._waiting_button,
    ])

    gui_app.add_nav_stack_tick(self._nav_stack_tick)

  def show_event(self):
    super().show_event()
    # make sure we populate strength and ip immediately if already have wifi
    self._wifi_manager.set_active(True)
    self._prev_has_internet = self._has_internet
    self._prev_wifi_connected = self._wifi_manager.wifi_state.status == ConnectStatus.CONNECTED
    self._pending_has_internet_scroll = None
    self._pending_continue_grow_animation = False
    self._pending_wifi_grow_animation = False

    if self._prev_has_internet or self._prev_wifi_connected:
      self.set_shown_callback(lambda: self._scroll_to_end_and_grow())

  @property
  def _has_internet(self) -> bool:
    network_changing = self._wifi_ui.any_network_forgetting or self._wifi_manager.wifi_state.status == ConnectStatus.CONNECTING
    if network_changing:
      self._network_monitor.invalidate()

    has_internet = (self._network_monitor.network_connected.is_set() and
                    not network_changing and
                    not self._network_monitor.recheck_event.is_set())
    return has_internet

  def _nav_stack_tick(self):
    # Only run tick when this page or its WiFi UI is on the stack
    if gui_app.get_active_widget() is not self and not gui_app.widget_in_stack(self._wifi_ui):
      self._wifi_manager.process_callbacks()
      return

    # Check network state before processing callbacks so forgetting flag
    # is still set on the frame the forgotten callback fires
    has_internet = self._has_internet
    wifi_connected = self._wifi_manager.wifi_state.status == ConnectStatus.CONNECTED

    self._continue_button.set_visible(has_internet)
    self._waiting_button.set_visible(not has_internet)

    # TODO: fire show/hide events on visibility changes
    if not has_internet:
      self._pending_continue_grow_animation = False
      self._waiting_button.set_text("waiting for\ninternet..." if wifi_connected else "connect to\ncontinue")

    self._wifi_manager.process_callbacks()

    # Dismiss WiFi UI and scroll on WiFi connect or internet gain
    if (has_internet and not self._prev_has_internet) or (wifi_connected and not self._prev_wifi_connected):
      # TODO: cancel if connect is transient
      self._pending_has_internet_scroll = rl.get_time()

    self._prev_has_internet = has_internet
    self._prev_wifi_connected = wifi_connected

    if self._pending_has_internet_scroll is not None:
      # Scrolls over to continue button, then grows once in view
      elapsed = rl.get_time() - self._pending_has_internet_scroll
      if elapsed > 0.7 or gui_app.get_active_widget() is self:  # instant scroll + grow if not popping
        # Animate WifiUi down first before scroll
        self._pending_has_internet_scroll = None
        gui_app.pop_widgets_to(self, self._scroll_to_end_and_grow)

  def _scroll_to_end_and_grow(self):
    self._scroller._layout()
    end_offset = -(self._scroller.content_size - self._rect.width)
    remaining = self._scroller.scroll_panel.get_offset() - end_offset
    self._scroller.scroll_to(remaining, smooth=True, block_interaction=True)
    self._pending_continue_grow_animation = True

  def set_custom_software(self, custom_software: bool):
    self._custom_software = custom_software
    self._continue_button.set_text("install openpilot" if not custom_software else "choose software")
    self._continue_button.set_green(not custom_software)

  def _update_state(self):
    super()._update_state()

    if self._pending_continue_grow_animation:
      btn_right = self._continue_button.rect.x + self._continue_button.rect.width
      visible_right = self._rect.x + self._rect.width
      if btn_right < visible_right + 50:
        self._pending_continue_grow_animation = False
        self._continue_button.trigger_grow_animation()

    if self._pending_wifi_grow_animation and abs(self._wifi_button.rect.x - ITEM_SPACING) < 50:
      self._pending_wifi_grow_animation = False
      self._wifi_button.trigger_grow_animation()


class NetworkSetupPage(NetworkSetupPageBase, NavScroller):
  def __init__(self, network_monitor: NetworkConnectivityMonitor, continue_callback: Callable[[bool], None],
               back_callback: Callable[[], None] | None):
    super().__init__(network_monitor, continue_callback)
    self.set_back_callback(back_callback)


class Setup(Widget):
  def __init__(self):
    super().__init__()
    self.download_url = ""
    self.download_progress = 0
    self.download_thread = None
    self._download_failed_reason: str | None = None

    self._network_monitor = NetworkConnectivityMonitor()
    self._network_monitor.start()

    def getting_started_button_callback():
      gui_app.push_widget(self._software_selection_page)

    self._start_page = StartPage()
    self._start_page.set_click_callback(getting_started_button_callback)
    self._start_page.set_enabled(lambda: self.enabled)  # for nav stack

    self._network_setup_page = NetworkSetupPage(self._network_monitor, self._network_setup_continue_callback, self._pop_to_software_selection)

    self._software_selection_page = SoftwareSelectionPage(self._push_network_setup, lambda: gui_app.push_widget(self._custom_software_warning_page))

    self._download_failed_page = FailedPage(self._pop_to_software_selection, icon="icons_mici/setup/red_warning.png")

    self._custom_software_warning_page = CustomSoftwareWarningPage(lambda: self._push_network_setup(True), self._pop_to_software_selection)

    self._downloading_page = DownloadingPage()

    gui_app.add_nav_stack_tick(self._nav_stack_tick)

  def _nav_stack_tick(self):
    self._downloading_page.set_progress(self.download_progress)

    if self._download_failed_reason is not None:
      reason = self._download_failed_reason
      self._download_failed_reason = None
      self._download_failed_page.set_reason(reason)
      gui_app.pop_widgets_to(self._software_selection_page, lambda: gui_app.push_widget(self._download_failed_page))

  def _render(self, rect: rl.Rectangle):
    self._start_page.render(rect)

  def close(self):
    self._network_monitor.stop()

  def _pop_to_software_selection(self):
    # reset sliders after dismiss completes
    gui_app.pop_widgets_to(self._software_selection_page, self._software_selection_page.reset)

  def _push_network_setup(self, custom_software: bool = False):
    # to fire the correct continue callback later
    self._network_setup_page.set_custom_software(custom_software)
    gui_app.push_widget(self._network_setup_page)

  def _network_setup_continue_callback(self, custom_software: bool):
    if not custom_software:
      self._download(OPENPILOT_URL)
    else:
      def handle_keyboard_result(text):
        url = text.strip()
        if url:
          self._download(url)

      keyboard = BigInputDialog("custom software URL...", confirm_callback=handle_keyboard_result, auto_return_to_letters="./")
      gui_app.push_widget(keyboard)

  def _download(self, url: str):
    # autocomplete incomplete URLs
    if re.match("^([^/.]+)/([^/]+)$", url):
      url = f"https://installer.comma.ai/{url}"

    parsed = urlparse(url, scheme='https')
    self.download_url = (urlparse(f"https://{url}") if not parsed.netloc else parsed).geturl()
    self.download_progress = 0

    def start_download():
      self.download_thread = threading.Thread(target=self._download_thread, daemon=True)
      self.download_thread.start()

    self._downloading_page.set_shown_callback(start_download)
    gui_app.push_widget(self._downloading_page)

  def _download_thread(self):
    try:
      import tempfile

      fd, tmpfile = tempfile.mkstemp(prefix="installer_")

      headers = {"User-Agent": USER_AGENT,
                 "X-openpilot-serial": HARDWARE.get_serial(),
                 "X-openpilot-device-type": HARDWARE.get_device_type()}
      req = urllib.request.Request(self.download_url, headers=headers)

      with open(tmpfile, 'wb') as f, urllib.request.urlopen(req, timeout=30) as response:
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        block_size = 8192

        while True:
          buffer = response.read(block_size)
          if not buffer:
            break

          downloaded += len(buffer)
          f.write(buffer)

          if total_size:
            self.download_progress = int(downloaded * 100 / total_size)

      is_elf = False
      with open(tmpfile, 'rb') as f:
        header = f.read(4)
        is_elf = header == b'\x7fELF'

      if not is_elf:
        self._download_failed_reason = "No custom software found at this URL: " + self.download_url.replace("https://", "", 1)
        return

      # NOTE: currently unused, for future logging
      with open(INSTALLER_URL_PATH, "w") as f:
        f.write(self.download_url)

      # AGNOS might try to execute the installer before this process exits.
      # Therefore, important to close the fd before renaming the installer.
      os.close(fd)
      os.rename(tmpfile, INSTALLER_DESTINATION_PATH)

      # give time for installer UI to take over
      time.sleep(0.1)
      gui_app.request_close()

    except urllib.error.HTTPError as e:
      if e.code == 409:
        self._download_failed_reason = "Incompatible openpilot version."
    except Exception:
      self._download_failed_reason = "Invalid URL: " + self.download_url.replace("https://", "", 1)


def main():
  config_realtime_process(0, 51)
  # attempt to affine. AGNOS will start setup with all cores, should only fail when manually launching with screen off
  if TICI:
    try:
      set_core_affinity([5])
    except OSError:
      cloudlog.exception("Failed to set core affinity for setup process")

  try:
    gui_app.init_window("Setup")
    setup = Setup()
    gui_app.push_widget(setup)
    for _ in gui_app.render():
      pass
    setup.close()
  except Exception as e:
    print(f"Setup error: {e}")
  finally:
    gui_app.close()


if __name__ == "__main__":
  main()
