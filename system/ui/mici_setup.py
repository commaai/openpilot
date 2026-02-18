#!/usr/bin/env python3
from abc import abstractmethod
import os
import re
import threading
import time
import urllib.request
import urllib.error
from urllib.parse import urlparse
from enum import IntEnum
import shutil
from collections.abc import Callable

import pyray as rl

from cereal import log
from openpilot.common.utils import run_cmd
from openpilot.system.hardware import HARDWARE
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.wifi_manager import WifiManager
from openpilot.system.ui.lib.scroll_panel2 import GuiScrollPanel2
from openpilot.system.ui.widgets import Widget, DialogResult
from openpilot.system.ui.widgets.button import (IconButton, SmallButton, WideRoundedButton, SmallerRoundedButton,
                                                SmallCircleIconButton, WidishRoundedButton, SmallRedPillButton,
                                                FullRoundedButton)
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.system.ui.widgets.slider import LargerSlider, SmallSlider
from openpilot.selfdrive.ui.mici.layouts.settings.network import WifiUIMici
from openpilot.selfdrive.ui.mici.widgets.dialog import BigInputDialog

NetworkType = log.DeviceState.NetworkType

OPENPILOT_URL = "https://openpilot.comma.ai"
USER_AGENT = f"AGNOSSetup-{HARDWARE.get_os_version()}"

CONTINUE_PATH = "/data/continue.sh"
TMP_CONTINUE_PATH = "/data/continue.sh.new"
INSTALL_PATH = "/data/openpilot"
VALID_CACHE_PATH = "/data/.openpilot_cache"
INSTALLER_SOURCE_PATH = "/usr/comma/installer"
INSTALLER_DESTINATION_PATH = "/tmp/installer"
INSTALLER_URL_PATH = "/tmp/installer_url"

CONTINUE = """#!/usr/bin/env bash

cd /data/openpilot
exec ./launch_openpilot.sh
"""


class NetworkConnectivityMonitor:
  def __init__(self, should_check: Callable[[], bool] | None = None, check_interval: float = 1.0):
    self.network_connected = threading.Event()
    self.wifi_connected = threading.Event()
    self._should_check = should_check or (lambda: True)
    self._check_interval = check_interval
    self._stop_event = threading.Event()
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

  def _run(self):
    while not self._stop_event.is_set():
      if self._should_check():
        try:
          request = urllib.request.Request(OPENPILOT_URL, method="HEAD")
          urllib.request.urlopen(request, timeout=1.0)
          self.network_connected.set()
          if HARDWARE.get_network_type() == NetworkType.wifi:
            self.wifi_connected.set()
        except Exception:
          self.reset()
      else:
        self.reset()

      if self._stop_event.wait(timeout=self._check_interval):
        break


class SetupState(IntEnum):
  GETTING_STARTED = 0
  NETWORK_SETUP = 1
  NETWORK_SETUP_CUSTOM_SOFTWARE = 2
  SOFTWARE_SELECTION = 3
  CUSTOM_SOFTWARE = 4
  DOWNLOADING = 5
  DOWNLOAD_FAILED = 6
  CUSTOM_SOFTWARE_WARNING = 7


class StartPage(Widget):
  def __init__(self):
    super().__init__()

    self._title = UnifiedLabel("start", 64, text_color=rl.Color(255, 255, 255, int(255 * 0.9)),
                               font_weight=FontWeight.DISPLAY, alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER,
                               alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE)

    self._start_bg_txt = gui_app.texture("icons_mici/setup/green_button.png", 520, 224)
    self._start_bg_pressed_txt = gui_app.texture("icons_mici/setup/green_button_pressed.png", 520, 224)

  def _render(self, rect: rl.Rectangle):
    draw_x = rect.x + (rect.width - self._start_bg_txt.width) / 2
    draw_y = rect.y + (rect.height - self._start_bg_txt.height) / 2
    texture = self._start_bg_pressed_txt if self.is_pressed else self._start_bg_txt
    rl.draw_texture(texture, int(draw_x), int(draw_y), rl.WHITE)

    self._title.render(rect)


class SoftwareSelectionPage(Widget):
  def __init__(self, use_openpilot_callback: Callable,
               use_custom_software_callback: Callable):
    super().__init__()

    self._openpilot_slider = LargerSlider("slide to use\nopenpilot", use_openpilot_callback)
    self._custom_software_slider = LargerSlider("slide to use\ncustom software", use_custom_software_callback, green=False)

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


class TermsHeader(Widget):
  def __init__(self, text: str, icon_texture: rl.Texture):
    super().__init__()

    self._title = UnifiedLabel(text, 36, text_color=rl.Color(255, 255, 255, int(255 * 0.9)),
                               font_weight=FontWeight.BOLD, alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE,
                               line_height=0.8)
    self._icon_texture = icon_texture

    self.set_rect(rl.Rectangle(0, 0, gui_app.width - 16 * 2, self._icon_texture.height))

  def set_title(self, text: str):
    self._title.set_text(text)

  def set_icon(self, icon_texture: rl.Texture):
    self._icon_texture = icon_texture

  def _render(self, _):
    rl.draw_texture_ex(self._icon_texture, rl.Vector2(self._rect.x, self._rect.y),
                       0.0, 1.0, rl.WHITE)

    # May expand outside parent rect
    title_content_height = self._title.get_content_height(int(self._rect.width - self._icon_texture.width - 16))
    title_rect = rl.Rectangle(
      self._rect.x + self._icon_texture.width + 16,
      self._rect.y + (self._rect.height - title_content_height) / 2,
      self._rect.width - self._icon_texture.width - 16,
      title_content_height,
    )
    self._title.render(title_rect)


class TermsPage(Widget):
  ITEM_SPACING = 20

  def __init__(self, continue_callback: Callable, back_callback: Callable | None = None,
               back_text: str = "back", continue_text: str = "accept"):
    super().__init__()

    # TODO: use Scroller
    self._scroll_panel = GuiScrollPanel2(horizontal=False)

    self._continue_text = continue_text
    self._continue_slider: bool = continue_text in ("reboot", "power off")
    self._continue_button: WideRoundedButton | FullRoundedButton | SmallSlider
    if self._continue_slider:
      self._continue_button = SmallSlider(continue_text, confirm_callback=continue_callback)
      self._scroll_panel.set_enabled(lambda: not self._continue_button.is_pressed)
    elif back_callback is not None:
      self._continue_button = WideRoundedButton(continue_text)
    else:
      self._continue_button = FullRoundedButton(continue_text)
    self._continue_button.set_enabled(False)
    self._continue_button.set_opacity(0.0)
    self._continue_button.set_touch_valid_callback(self._scroll_panel.is_touch_valid)
    if not self._continue_slider:
      self._continue_button.set_click_callback(continue_callback)

    self._enable_back = back_callback is not None
    self._back_button = SmallButton(back_text)
    self._back_button.set_opacity(0.0)
    self._back_button.set_touch_valid_callback(self._scroll_panel.is_touch_valid)
    self._back_button.set_click_callback(back_callback)

    self._scroll_down_indicator = IconButton(gui_app.texture("icons_mici/setup/scroll_down_indicator.png", 64, 78))
    self._scroll_down_indicator.set_enabled(False)

  def reset(self):
    self._scroll_panel.set_offset(0)
    self._continue_button.set_enabled(False)
    self._continue_button.set_opacity(0.0)
    self._back_button.set_enabled(False)
    self._back_button.set_opacity(0.0)
    self._scroll_down_indicator.set_opacity(1.0)

  def show_event(self):
    super().show_event()
    self.reset()

  @property
  @abstractmethod
  def _content_height(self):
    pass

  @property
  def _scrolled_down_offset(self):
    return -self._content_height + (self._continue_button.rect.height + 16 + 30)

  @abstractmethod
  def _render_content(self, scroll_offset):
    pass

  def _render(self, _):
    scroll_offset = round(self._scroll_panel.update(self._rect, self._content_height + self._continue_button.rect.height + 16))

    if scroll_offset <= self._scrolled_down_offset:
      # don't show back if not enabled
      if self._enable_back:
        self._back_button.set_enabled(True)
        self._back_button.set_opacity(1.0, smooth=True)
      self._continue_button.set_enabled(True)
      self._continue_button.set_opacity(1.0, smooth=True)
      self._scroll_down_indicator.set_opacity(0.0, smooth=True)
    else:
      self._back_button.set_enabled(False)
      self._back_button.set_opacity(0.0, smooth=True)
      self._continue_button.set_enabled(False)
      self._continue_button.set_opacity(0.0, smooth=True)
      self._scroll_down_indicator.set_opacity(1.0, smooth=True)

    # Render content
    self._render_content(scroll_offset)

    # black gradient at top and bottom for scrolling content
    rl.draw_rectangle_gradient_v(int(self._rect.x), int(self._rect.y),
                                 int(self._rect.width), 20, rl.BLACK, rl.BLANK)
    rl.draw_rectangle_gradient_v(int(self._rect.x), int(self._rect.y + self._rect.height - 20),
                                 int(self._rect.width), 20, rl.BLANK, rl.BLACK)

    # fade out back button as slider is moved
    if self._continue_slider and scroll_offset <= self._scrolled_down_offset:
      self._back_button.set_opacity(1.0 - self._continue_button.slider_percentage)
      self._back_button.set_visible(self._continue_button.slider_percentage < 0.99)

    self._back_button.render(rl.Rectangle(
      self._rect.x + 8,
      self._rect.y + self._rect.height - self._back_button.rect.height,
      self._back_button.rect.width,
      self._back_button.rect.height,
    ))

    continue_x = self._rect.x + 8
    if self._enable_back:
      continue_x = self._rect.x + self._rect.width - self._continue_button.rect.width - 8
    if self._continue_slider:
      continue_x += 8
    self._continue_button.render(rl.Rectangle(
      continue_x,
      self._rect.y + self._rect.height - self._continue_button.rect.height,
      self._continue_button.rect.width,
      self._continue_button.rect.height,
    ))

    self._scroll_down_indicator.render(rl.Rectangle(
      self._rect.x + self._rect.width - self._scroll_down_indicator.rect.width - 8,
      self._rect.y + self._rect.height - self._scroll_down_indicator.rect.height - 8,
      self._scroll_down_indicator.rect.width,
      self._scroll_down_indicator.rect.height,
    ))


class CustomSoftwareWarningPage(TermsPage):
  def __init__(self, continue_callback: Callable, back_callback: Callable):
    super().__init__(continue_callback, back_callback)

    self._title_header = TermsHeader("use caution installing\n3rd party software",
                                     gui_app.texture("icons_mici/setup/warning.png", 66, 60))
    self._body = UnifiedLabel("• It has not been tested by comma.\n" +
                              "• It may not comply with relevant safety standards.\n" +
                              "• It may cause damage to your device and/or vehicle.\n", 36, text_color=rl.Color(255, 255, 255, int(255 * 0.9)),
                              font_weight=FontWeight.ROMAN)

    self._restore_header = TermsHeader("how to backup &\nrestore", gui_app.texture("icons_mici/setup/restore.png", 60, 60))
    self._restore_body = UnifiedLabel("To restore your device to a factory state later, use https://flash.comma.ai",
                                      36, text_color=rl.Color(255, 255, 255, int(255 * 0.9)),
                                      font_weight=FontWeight.ROMAN)

  @property
  def _content_height(self):
    return self._restore_body.rect.y + self._restore_body.rect.height - self._scroll_panel.get_offset()

  def _render_content(self, scroll_offset):
    self._title_header.set_position(self._rect.x + 16, self._rect.y + 8 + scroll_offset)
    self._title_header.render()

    body_rect = rl.Rectangle(
      self._rect.x + 8,
      self._title_header.rect.y + self._title_header.rect.height + self.ITEM_SPACING,
      self._rect.width - 50,
      self._body.get_content_height(int(self._rect.width - 50)),
    )
    self._body.render(body_rect)

    self._restore_header.set_position(self._rect.x + 16, self._body.rect.y + self._body.rect.height + self.ITEM_SPACING)
    self._restore_header.render()

    self._restore_body.render(rl.Rectangle(
      self._rect.x + 8,
      self._restore_header.rect.y + self._restore_header.rect.height + self.ITEM_SPACING,
      self._rect.width - 50,
      self._restore_body.get_content_height(int(self._rect.width - 50)),
    ))


class DownloadingPage(Widget):
  def __init__(self):
    super().__init__()

    self._title_label = UnifiedLabel("downloading", 64, text_color=rl.Color(255, 255, 255, int(255 * 0.9)),
                                     font_weight=FontWeight.DISPLAY)
    self._progress_label = UnifiedLabel("", 128, text_color=rl.Color(255, 255, 255, int(255 * 0.9 * 0.35)),
                                        font_weight=FontWeight.ROMAN, alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_BOTTOM)
    self._progress = 0

  def set_progress(self, progress: int):
    self._progress = progress
    self._progress_label.set_text(f"{progress}%")

  def _render(self, rect: rl.Rectangle):
    self._title_label.render(rl.Rectangle(
      rect.x + 20,
      rect.y + 10,
      rect.width,
      64,
    ))

    self._progress_label.render(rl.Rectangle(
      rect.x + 20,
      rect.y + 20,
      rect.width,
      rect.height,
    ))


class FailedPage(Widget):
  def __init__(self, reboot_callback: Callable, retry_callback: Callable, title: str = "download failed"):
    super().__init__()

    self._title_label = UnifiedLabel(title, 64, text_color=rl.Color(255, 255, 255, int(255 * 0.9)),
                                     font_weight=FontWeight.DISPLAY)
    self._reason_label = UnifiedLabel("", 36, text_color=rl.Color(255, 255, 255, int(255 * 0.9 * 0.65)),
                                      font_weight=FontWeight.ROMAN)

    self._reboot_button = SmallRedPillButton("reboot")
    self._reboot_button.set_click_callback(reboot_callback)

    self._retry_button = WideRoundedButton("retry")
    self._retry_button.set_click_callback(retry_callback)

  def set_reason(self, reason: str):
    self._reason_label.set_text(reason)

  def _render(self, rect: rl.Rectangle):
    self._title_label.render(rl.Rectangle(
      rect.x + 8,
      rect.y + 10,
      rect.width,
      64,
    ))

    self._reason_label.render(rl.Rectangle(
      rect.x + 8,
      rect.y + 10 + 64,
      rect.width,
      36,
    ))

    self._reboot_button.render(rl.Rectangle(
      rect.x + 8,
      rect.y + rect.height - self._reboot_button.rect.height,
      self._reboot_button.rect.width,
      self._reboot_button.rect.height,
    ))

    self._retry_button.render(rl.Rectangle(
      rect.x + 8 + self._reboot_button.rect.width + 8,
      rect.y + rect.height - self._retry_button.rect.height,
      self._retry_button.rect.width,
      self._retry_button.rect.height,
    ))


class NetworkSetupState(IntEnum):
  MAIN = 0
  WIFI_PANEL = 1


class NetworkSetupPage(Widget):
  def __init__(self, wifi_manager, continue_callback: Callable, back_callback: Callable):
    super().__init__()
    self._wifi_ui = WifiUIMici(wifi_manager, back_callback=lambda: self.set_state(NetworkSetupState.MAIN))

    self._no_wifi_txt = gui_app.texture("icons_mici/settings/network/wifi_strength_slash.png", 58, 50)
    self._wifi_full_txt = gui_app.texture("icons_mici/settings/network/wifi_strength_full.png", 58, 50)
    self._waiting_text = "waiting for internet..."
    self._network_header = TermsHeader(self._waiting_text, self._no_wifi_txt)

    back_txt = gui_app.texture("icons_mici/setup/back_new.png", 37, 32)
    self._back_button = SmallCircleIconButton(back_txt)
    self._back_button.set_click_callback(back_callback)

    self._wifi_button = SmallerRoundedButton("wifi")
    self._wifi_button.set_click_callback(lambda: self.set_state(NetworkSetupState.WIFI_PANEL))

    self._continue_button = WidishRoundedButton("continue")
    self._continue_button.set_enabled(False)
    self._continue_button.set_click_callback(continue_callback)

    self._state = NetworkSetupState.MAIN
    self._prev_has_internet = False

  def set_state(self, state: NetworkSetupState):
    if self._state == NetworkSetupState.WIFI_PANEL and state != NetworkSetupState.WIFI_PANEL:
      self._wifi_ui.hide_event()
    self._state = state
    if state == NetworkSetupState.WIFI_PANEL:
      self._wifi_ui.show_event()

  def set_has_internet(self, has_internet: bool):
    if has_internet:
      self._network_header.set_title("connected to internet")
      self._network_header.set_icon(self._wifi_full_txt)
      self._continue_button.set_enabled(True)
    else:
      self._network_header.set_title(self._waiting_text)
      self._network_header.set_icon(self._no_wifi_txt)
      self._continue_button.set_enabled(False)

    if has_internet and not self._prev_has_internet:
      self.set_state(NetworkSetupState.MAIN)
    self._prev_has_internet = has_internet

  def show_event(self):
    super().show_event()
    self._state = NetworkSetupState.MAIN

  def hide_event(self):
    super().hide_event()
    if self._state == NetworkSetupState.WIFI_PANEL:
      self._wifi_ui.hide_event()

  def _render(self, _):
    if self._state == NetworkSetupState.MAIN:
      self._network_header.render(rl.Rectangle(
        self._rect.x + 16,
        self._rect.y + 16,
        self._rect.width - 32,
        self._network_header.rect.height,
      ))

      self._back_button.render(rl.Rectangle(
        self._rect.x + 8,
        self._rect.y + self._rect.height - self._back_button.rect.height,
        self._back_button.rect.width,
        self._back_button.rect.height,
      ))

      self._wifi_button.render(rl.Rectangle(
        self._rect.x + 8 + self._back_button.rect.width + 10,
        self._rect.y + self._rect.height - self._wifi_button.rect.height,
        self._wifi_button.rect.width,
        self._wifi_button.rect.height,
      ))

      self._continue_button.render(rl.Rectangle(
        self._rect.x + self._rect.width - self._continue_button.rect.width - 8,
        self._rect.y + self._rect.height - self._continue_button.rect.height,
        self._continue_button.rect.width,
        self._continue_button.rect.height,
      ))
    else:
      self._wifi_ui.render(self._rect)


class Setup(Widget):
  def __init__(self):
    super().__init__()
    self.state = SetupState.GETTING_STARTED
    self.failed_url = ""
    self.failed_reason = ""
    self.download_url = ""
    self.download_progress = 0
    self.download_thread = None
    self._wifi_manager = WifiManager()
    self._wifi_manager.set_active(True)
    self._network_monitor = NetworkConnectivityMonitor()
    self._network_monitor.start()
    self._prev_has_internet = False
    gui_app.set_modal_overlay_tick(self._modal_overlay_tick)

    self._start_page = StartPage()
    self._start_page.set_click_callback(self._getting_started_button_callback)

    self._network_setup_page = NetworkSetupPage(self._wifi_manager, self._network_setup_continue_button_callback,
                                                self._network_setup_back_button_callback)

    self._software_selection_page = SoftwareSelectionPage(self._software_selection_continue_button_callback,
                                                          self._software_selection_custom_software_button_callback)

    self._download_failed_page = FailedPage(HARDWARE.reboot, self._download_failed_startover_button_callback)

    self._custom_software_warning_page = CustomSoftwareWarningPage(self._software_selection_custom_software_continue,
                                                                   self._custom_software_warning_back_button_callback)

    self._downloading_page = DownloadingPage()

  def _modal_overlay_tick(self):
    has_internet = self._network_monitor.network_connected.is_set()
    if has_internet and not self._prev_has_internet:
      gui_app.set_modal_overlay(None)
    self._prev_has_internet = has_internet

  def _update_state(self):
    self._wifi_manager.process_callbacks()

  def _set_state(self, state: SetupState):
    self.state = state
    if self.state == SetupState.SOFTWARE_SELECTION:
      self._software_selection_page.reset()
    elif self.state == SetupState.CUSTOM_SOFTWARE_WARNING:
      self._custom_software_warning_page.reset()

    if self.state in (SetupState.NETWORK_SETUP, SetupState.NETWORK_SETUP_CUSTOM_SOFTWARE):
      self._network_setup_page.show_event()
      self._network_monitor.reset()
    else:
      self._network_setup_page.hide_event()

  def _render(self, rect: rl.Rectangle):
    if self.state == SetupState.GETTING_STARTED:
      self._start_page.render(rect)
    elif self.state in (SetupState.NETWORK_SETUP, SetupState.NETWORK_SETUP_CUSTOM_SOFTWARE):
      self.render_network_setup(rect)
    elif self.state == SetupState.SOFTWARE_SELECTION:
      self._software_selection_page.render(rect)
    elif self.state == SetupState.CUSTOM_SOFTWARE_WARNING:
      self._custom_software_warning_page.render(rect)
    elif self.state == SetupState.CUSTOM_SOFTWARE:
      self.render_custom_software()
    elif self.state == SetupState.DOWNLOADING:
      self.render_downloading(rect)
    elif self.state == SetupState.DOWNLOAD_FAILED:
      self._download_failed_page.render(rect)

  def _custom_software_warning_back_button_callback(self):
    self._set_state(SetupState.SOFTWARE_SELECTION)

  def _getting_started_button_callback(self):
    self._set_state(SetupState.SOFTWARE_SELECTION)

  def _software_selection_continue_button_callback(self):
    self.use_openpilot()

  def _software_selection_custom_software_button_callback(self):
    self._set_state(SetupState.CUSTOM_SOFTWARE_WARNING)

  def _software_selection_custom_software_continue(self):
    self._set_state(SetupState.NETWORK_SETUP_CUSTOM_SOFTWARE)

  def _download_failed_startover_button_callback(self):
    self._set_state(SetupState.GETTING_STARTED)

  def _network_setup_back_button_callback(self):
    self._set_state(SetupState.SOFTWARE_SELECTION)

  def _network_setup_continue_button_callback(self):
    if self.state == SetupState.NETWORK_SETUP:
      self.download(OPENPILOT_URL)
    elif self.state == SetupState.NETWORK_SETUP_CUSTOM_SOFTWARE:
      self._set_state(SetupState.CUSTOM_SOFTWARE)

  def close(self):
    self._network_monitor.stop()

  def render_network_setup(self, rect: rl.Rectangle):
    has_internet = self._network_monitor.network_connected.is_set()
    self._prev_has_internet = has_internet
    self._network_setup_page.set_has_internet(has_internet)
    self._network_setup_page.render(rect)

  def render_downloading(self, rect: rl.Rectangle):
    self._downloading_page.set_progress(self.download_progress)
    self._downloading_page.render(rect)

  def render_custom_software(self):
    def handle_keyboard_result(text):
      url = text.strip()
      if url:
        self.download(url)

    def handle_keyboard_exit(result):
      if result == DialogResult.CANCEL:
        self._set_state(SetupState.SOFTWARE_SELECTION)

    keyboard = BigInputDialog("custom software URL...", confirm_callback=handle_keyboard_result)
    gui_app.set_modal_overlay(keyboard, callback=handle_keyboard_exit)

  def use_openpilot(self):
    if os.path.isdir(INSTALL_PATH) and os.path.isfile(VALID_CACHE_PATH):
      os.remove(VALID_CACHE_PATH)
      with open(TMP_CONTINUE_PATH, "w") as f:
        f.write(CONTINUE)
      run_cmd(["chmod", "+x", TMP_CONTINUE_PATH])
      shutil.move(TMP_CONTINUE_PATH, CONTINUE_PATH)
      shutil.copyfile(INSTALLER_SOURCE_PATH, INSTALLER_DESTINATION_PATH)

      # give time for installer UI to take over
      time.sleep(0.1)
      gui_app.request_close()
    else:
      self._set_state(SetupState.NETWORK_SETUP)

  def download(self, url: str):
    # autocomplete incomplete URLs
    if re.match("^([^/.]+)/([^/]+)$", url):
      url = f"https://installer.comma.ai/{url}"

    parsed = urlparse(url, scheme='https')
    self.download_url = (urlparse(f"https://{url}") if not parsed.netloc else parsed).geturl()

    self._set_state(SetupState.DOWNLOADING)

    self.download_thread = threading.Thread(target=self._download_thread, daemon=True)
    self.download_thread.start()

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
            self._downloading_page.set_progress(self.download_progress)

      is_elf = False
      with open(tmpfile, 'rb') as f:
        header = f.read(4)
        is_elf = header == b'\x7fELF'

      if not is_elf:
        self.download_failed(self.download_url, "No custom software found at this URL.")
        return

      # AGNOS might try to execute the installer before this process exits.
      # Therefore, important to close the fd before renaming the installer.
      os.close(fd)
      os.rename(tmpfile, INSTALLER_DESTINATION_PATH)

      with open(INSTALLER_URL_PATH, "w") as f:
        f.write(self.download_url)

      # give time for installer UI to take over
      time.sleep(0.1)
      gui_app.request_close()

    except urllib.error.HTTPError as e:
      if e.code == 409:
        error_msg = "Incompatible openpilot version"
        self.download_failed(self.download_url, error_msg)
    except Exception:
      error_msg = "Invalid URL"
      self.download_failed(self.download_url, error_msg)

  def download_failed(self, url: str, reason: str):
    self.failed_url = url
    self.failed_reason = reason
    self._download_failed_page.set_reason(reason)
    self._set_state(SetupState.DOWNLOAD_FAILED)


def main():
  try:
    gui_app.init_window("Setup")
    setup = Setup()
    for should_render in gui_app.render():
      if should_render:
        setup.render(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
    setup.close()
  except Exception as e:
    print(f"Setup error: {e}")
  finally:
    gui_app.close()


if __name__ == "__main__":
  main()
