#!/usr/bin/env python3
import numpy as np
import traceback
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
from openpilot.common.realtime import config_realtime_process, set_core_affinity
from openpilot.common.swaglog import cloudlog
from openpilot.common.utils import run_cmd
from openpilot.system.hardware import HARDWARE, TICI
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.wifi_manager import WifiManager
from openpilot.system.ui.lib.scroll_panel2 import GuiScrollPanel2
from openpilot.system.ui.widgets import Widget, DialogResult
from openpilot.system.ui.widgets.nav_widget import NavWidget
from openpilot.system.ui.widgets.button import (IconButton, SmallButton, WideRoundedButton, SmallerRoundedButton,
                                                SmallCircleIconButton, WidishRoundedButton, SmallRedPillButton,
                                                FullRoundedButton)
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.system.ui.widgets.scroller import Scroller, ITEM_SPACING
from openpilot.system.ui.widgets.slider import LargerSlider, SmallSlider
from openpilot.selfdrive.ui.mici.layouts.settings.network import WifiNetworkButton, WifiUIMici
from openpilot.selfdrive.ui.mici.widgets.dialog import BigInputDialog
from openpilot.selfdrive.ui.mici.widgets.button import BigButton

from selfdrive.ui.mici.widgets.button import BigCircleButton
from selfdrive.ui.mici.widgets.dialog import BigConfirmationDialogV2

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
  def __init__(self, should_check: Callable[[], bool] | None = None):
    self.network_connected = threading.Event()
    self.wifi_connected = threading.Event()
    self._should_check = should_check or (lambda: True)
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
          urllib.request.urlopen(request, timeout=2.0)
          # time.sleep(3)
          self.network_connected.set()
          if HARDWARE.get_network_type() == NetworkType.wifi:
            self.wifi_connected.set()
        except Exception:
          self.reset()
      else:
        self.reset()

      if self._stop_event.wait(timeout=1.0):
        break


class SetupState(IntEnum):
  START = 0
  SOFTWARE_SELECTION = 1
  DOWNLOADING = 2


class StartPage(Widget):
  def __init__(self):
    super().__init__()

    self._title = UnifiedLabel("start", 64, text_color=rl.Color(255, 255, 255, int(255 * 0.9)),
                               font_weight=FontWeight.DISPLAY, alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER,
                               alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE)

    self._start_bg_txt = gui_app.texture("icons_mici/setup/start_button.png", 500, 224, keep_aspect_ratio=False)
    self._start_bg_pressed_txt = gui_app.texture("icons_mici/setup/start_button_pressed.png", 500, 224, keep_aspect_ratio=False)

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

    self._openpilot_slider = LargerSlider("slide to install\nopenpilot", use_openpilot_callback)
    self._openpilot_slider.set_enabled(lambda: self.enabled)
    self._custom_software_slider = LargerSlider("slide to install\nother software", use_custom_software_callback, green=False)
    self._custom_software_slider.set_enabled(lambda: self.enabled)

  def reset(self):
    print('RESETTING SLIDERS')
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


class CustomSoftwareWarningPage(NavWidget):
  def __init__(self, continue_callback: Callable, back_callback: Callable):
    super().__init__()
    self.set_back_callback(back_callback)

    def show_confirm_dialog():
      gui_app.push_widget(BigConfirmationDialogV2("I want to\ncontinue", "icons_mici/setup/driver_monitoring/dm_check.png",
                                                  confirm_callback=continue_callback))

    self._continue_button = BigCircleButton("icons_mici/setup/driver_monitoring/dm_check.png")
    self._continue_button.set_click_callback(show_confirm_dialog)

    def show_back_dialog():
      gui_app.push_widget(BigConfirmationDialogV2("I want to\ngo back", "icons_mici/setup/cancel.png", confirm_callback=back_callback))

    self._back_button = BigCircleButton("icons_mici/setup/cancel.png")
    self._back_button.set_click_callback(show_back_dialog)

    self._scroller = Scroller([
      GreyBigButton("use caution", "you are installing\n3rd party software",
                    gui_app.texture("icons_mici/setup/warning.png", 64, 58), wide=True),
      GreyBigButton("", "• It has not been tested by comma.\n" +
                    "• It may not comply with relevant safety standards.", wide=True),
      GreyBigButton("", "• It may cause damage to your device and/or vehicle.\n" +
                    "• You are fully responsible for your device.", wide=True),
      GreyBigButton("to restore to a\nfactory state later", "https://flash.comma.ai",
                    gui_app.texture("icons_mici/setup/restore.png", 64, 64), wide=True),
      self._continue_button,
      self._back_button,
    ])

  def hide_event(self):
    super().hide_event()
    self._scroller.hide_event()

  def show_event(self):
    super().show_event()
    self._scroller.show_event()

  def _render(self, _):
    self._scroller.render(self._rect)


class DownloadingPage(Widget):
  def __init__(self):
    super().__init__()

    self._title_label = UnifiedLabel("downloading...", 64, text_color=rl.Color(255, 255, 255, int(255 * 0.9)),
                                     font_weight=FontWeight.DISPLAY)
    self._progress_label = UnifiedLabel("", 128, text_color=rl.Color(255, 255, 255, int(255 * 0.9 * 0.65)),
                                        font_weight=FontWeight.ROMAN, alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_BOTTOM)
    self._progress = 0

  def set_progress(self, progress: int):
    self._progress = progress
    self._progress_label.set_text(f"{progress}%")

  def _render(self, rect: rl.Rectangle):
    self._title_label.render(rl.Rectangle(
      rect.x + 12,
      rect.y + 0,
      rect.width,
      64,
    ))

    self._progress_label.render(rl.Rectangle(
      rect.x + 12,
      rect.y + 16,
      rect.width,
      rect.height,
    ))


class FailedPage(NavWidget):
  def __init__(self, reboot_callback: Callable, retry_callback: Callable, title: str = "download failed"):
    super().__init__()
    self.set_back_callback(gui_app.pop_widget)

    self._title_label = UnifiedLabel(title, 64, text_color=rl.Color(255, 255, 255, int(255 * 0.9)),
                                     font_weight=FontWeight.DISPLAY)
    self._reason_label = UnifiedLabel("", 36, text_color=rl.Color(255, 255, 255, int(255 * 0.9 * 0.65)),
                                      font_weight=FontWeight.ROMAN)

    self._reboot_button = SmallRedPillButton("reboot")
    self._reboot_button.set_click_callback(reboot_callback)
    self._reboot_button.set_enabled(lambda: self.enabled)  # for nav stack

    self._retry_button = WideRoundedButton("retry")
    self._retry_button.set_click_callback(gui_app.pop_widget)
    self._retry_button.set_enabled(lambda: self.enabled)  # for nav stack

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


class GreyBigButton(BigButton):
  """Users should manage newlines with this class themselves"""

  LABEL_HORIZONTAL_PADDING = 30

  def __init__(self, *args, wide: bool = False, **kwargs):
    super().__init__(*args, **kwargs)
    self.set_touch_valid_callback(lambda: False)

    wide = True

    if wide:
      self._rect.width = 476

    self._label.set_font_size(36)
    self._label.set_font_weight(FontWeight.BOLD)
    self._label.set_line_height(1.0)

    self._sub_label.set_font_size(32)
    self._sub_label.set_text_color(rl.Color(255, 255, 255, int(255 * 0.9)))
    self._sub_label.set_font_weight(FontWeight.DISPLAY_REGULAR)
    self._sub_label.set_alignment_vertical(rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE if not self._label.text else
                                           rl.GuiTextAlignmentVertical.TEXT_ALIGN_BOTTOM)
    self._sub_label.set_line_height(0.95)

  def _width_hint(self) -> int:
    return int(self._rect.width - self.LABEL_HORIZONTAL_PADDING * 2)

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

  def _load_images(self):
    if self._green:
      self._txt_default_bg = gui_app.texture("icons_mici/setup/start_button.png", 402, 180)
      self._txt_pressed_bg = gui_app.texture("icons_mici/setup/start_button_pressed.png", 402, 180)
    else:
      self._txt_default_bg = gui_app.texture("icons_mici/setup/continue.png", 402, 180)
      self._txt_pressed_bg = gui_app.texture("icons_mici/setup/continue_pressed.png", 402, 180)
    self._txt_disabled_bg = gui_app.texture("icons_mici/setup/continue_disabled.png", 402, 180)


class NetworkSetupPage(NavWidget):
  def __init__(self, network_monitor: NetworkConnectivityMonitor, continue_callback: Callable, back_callback: Callable):
    super().__init__()
    self.set_back_callback(back_callback)

    self._wifi_manager = WifiManager()
    self._wifi_manager.set_active(True)
    self._network_monitor = network_monitor
    self._custom_software = False
    self._prev_has_internet = False
    self._wifi_ui = WifiUIMici(self._wifi_manager)

    self._connect_button = GreyBigButton("connect to\ninternet", "or swipe down to go back",
                                         gui_app.texture("icons_mici/setup/small_slider/slider_arrow.png", 64, 56, flip_x=True))

    self._wifi_button = WifiNetworkButton(self._wifi_manager)
    self._wifi_button.set_click_callback(lambda: gui_app.push_widget(self._wifi_ui))

    self._pending_has_internet_scroll = None
    self._pending_grow_animation = False
    self._pending_shake = False

    def on_waiting_click():
      offset = (self._wifi_button.rect.x + self._wifi_button.rect.width / 2) - (self._rect.x + self._rect.width / 2)
      self._scroller.scroll_to(offset, smooth=True, block=True)
      self._pending_shake = True

    def on_continue_click():
      # if not self._custom_software:
      #   gui_app.pop_widget()
      continue_callback(self._custom_software)

    self._waiting_button = BigPillButton("waiting for\ninternet...", disabled_background=True)
    self._waiting_button.set_click_callback(on_waiting_click)
    self._continue_button = BigPillButton("install openpilot", green=True)
    self._continue_button.set_click_callback(on_continue_click)

    self._scroller = Scroller([
      self._connect_button,
      self._wifi_button,
      self._continue_button,
      self._waiting_button,
    ])

    # set up position for invisible items so that scroll_to works
    # self._scroller._layout()

    gui_app.set_nav_stack_tick(self._nav_stack_tick)

  def set_custom_software(self, custom_software: bool):
    self._custom_software = custom_software

    # "download\n& install" if self._custom_software else "continue", green=not self._custom_software
    # if self._custom_software:
    #   self._continue_button.set_text("choose custom software")
    # else:
    #   self._continue_button.set_text("install\nopenpilot")
    self._continue_button.set_text("install openpilot" if not custom_software else "choose software")
    self._continue_button.set_green(not custom_software)

  def show_event(self):
    super().show_event()
    self._scroller.show_event()
    self._prev_has_internet = False
    print('SHOW EVENT')
    # self._network_monitor.reset()

  def hide_event(self):
    super().hide_event()
    self._scroller.hide_event()

  def _nav_stack_tick(self):
    self._wifi_manager.process_callbacks()

    # has_internet = self._network_monitor.network_connected.is_set()
    # if has_internet and not self._prev_has_internet and gui_app.get_active_widget() == self:
    #   gui_app.pop_widgets_to(self)
    #   end_offset = -(self._scroller.content_size - self._rect.width)
    #   remaining = self._scroller.scroll_panel.get_offset() - end_offset
    #   self._scroller.scroll_to(remaining, smooth=True, block=True)
    #   self._pending_grow_animation = True
    #
    #   self._prev_has_internet = has_internet

  def _update_state(self):
    super()._update_state()

    if self._pending_grow_animation:
      btn_right = self._continue_button.rect.x + self._continue_button.rect.width
      visible_right = self._rect.x + self._rect.width
      if btn_right < visible_right + 50:
        self._pending_grow_animation = False
        self._continue_button.trigger_grow_animation()

    if self._pending_shake and abs(self._wifi_button.rect.x - ITEM_SPACING) < 50:
      self._pending_shake = False
      self._wifi_button.trigger_shake()

    if self._network_monitor.network_connected.is_set():
      self._continue_button.set_visible(True)
      self._waiting_button.set_visible(False)
    else:
      self._continue_button.set_visible(False)
      self._waiting_button.set_visible(True)

    # print('content', self._scroller.content_size, 'rect', self._rect.width)
    # print('offset', self._scroller.scroll_panel.get_offset())

    # This intentionally doesn't trigger pop when in keyboard or forget dialog
    has_internet = self._network_monitor.network_connected.is_set()
    if has_internet and not self._prev_has_internet:  # and gui_app.get_active_widget() == self:
      self._pending_has_internet_scroll = rl.get_time()
    self._prev_has_internet = has_internet

    # print('offset', self._scroller.scroll_panel.get_offset(), has_internet)

    if self._pending_has_internet_scroll is not None:
      elapsed = rl.get_time() - self._pending_has_internet_scroll
      if elapsed > 0.5:
        self._pending_has_internet_scroll = None
        # print('SCROLLING OVER')
        gui_app.pop_widgets_to(self)

        # ensure layout is up to date for scroll_to
        self._scroller._layout()
        end_offset = -(self._scroller.content_size - self._rect.width)
        remaining = self._scroller.scroll_panel.get_offset() - end_offset
        self._scroller.scroll_to(remaining, smooth=True, block=True)
        self._pending_grow_animation = True

  def _render(self, _):
    self._scroller.render(self._rect)


class Setup(Widget):
  def __init__(self):
    super().__init__()
    self.state = SetupState.START
    self.failed_url = ""
    self.failed_reason = ""
    self.download_url = ""
    self.download_progress = 0
    self.download_thread = None

    self._network_monitor = NetworkConnectivityMonitor()
    self._network_monitor.start()

    self._start_page = StartPage()
    self._start_page.set_click_callback(lambda: self._set_state(SetupState.SOFTWARE_SELECTION))
    self._start_page.set_enabled(lambda: self.enabled)  # for nav stack

    self._network_setup_page = NetworkSetupPage(self._network_monitor, self._network_setup_continue_callback, self._back_to_software_selection)
    # TODO: change these to touch_valid
    # self._network_setup_page.set_enabled(lambda: self.enabled)  # for nav stack

    self._software_selection_page = SoftwareSelectionPage(self._software_selection_continue_button_callback,
                                                          self._software_selection_custom_software_button_callback)
    self._software_selection_page.set_enabled(lambda: self.enabled)  # for nav stack

    self._download_failed_page = FailedPage(HARDWARE.reboot, lambda: self._set_state(SetupState.START))

    self._custom_software_warning_page = CustomSoftwareWarningPage(lambda: self._push_network_setup(True), self._back_to_software_selection)
    # self._custom_software_warning_page.set_enabled(lambda: self.enabled)  # for nav stack

    self._downloading_page = DownloadingPage()

  def _back_to_software_selection(self):
    # pop and reset sliders
    gui_app.pop_widgets_to(self)
    self._set_state(SetupState.SOFTWARE_SELECTION)

  def _update_state(self):
    pass
    # self._wifi_manager.process_callbacks()

    # self._network_setup_page.set_has_internet(self._network_monitor.network_connected.is_set())
    # self._network_setup_page.render(rect)

  def _set_state(self, state: SetupState):
    print('SETTING STATE', state)
    self.state = state
    if self.state == SetupState.SOFTWARE_SELECTION:
      self._software_selection_page.reset()

  def _push_network_setup(self, custom_software: bool = False):
    # self._network_setup_page.show_event()
    # TODO: move this to network setup page's show event and remove from Setup
    # self._network_monitor.reset()
    # self._network_setup_page.set_has_internet(False)
    self._network_setup_page.set_custom_software(custom_software)  # to fire the correct continue callback after continuing

    print('SHOWING NETWORK SETUP PAGE')
    # gui_app.set_modal_overlay(self._network_setup_page)
    gui_app.pop_widgets_to(self)
    gui_app.push_widget(self._network_setup_page)
    # self._set_state(SetupState.SOFTWARE_SELECTION)

  def _render(self, rect: rl.Rectangle):
    # print('state', repr(self.state))
    if self.state == SetupState.START:
      self._start_page.render(rect)
    elif self.state == SetupState.SOFTWARE_SELECTION:
      self._software_selection_page.render(rect)
    elif self.state == SetupState.DOWNLOADING:
      self.render_downloading(rect)

  def _software_selection_continue_button_callback(self):
    self.use_openpilot()

  def _software_selection_custom_software_button_callback(self):
    gui_app.push_widget(self._custom_software_warning_page)

  def _network_setup_continue_callback(self, custom_software: bool):
    if not custom_software:
      self.download(OPENPILOT_URL)
    else:
      def handle_keyboard_result(text):
        url = text.strip()
        if url:
          self.download(url)

      keyboard = BigInputDialog("custom software URL", confirm_callback=handle_keyboard_result)
      gui_app.push_widget(keyboard)

  def close(self):
    self._network_monitor.stop()

  # def render_network_setup(self, rect: rl.Rectangle):
  #   # gui_app.set_modal_overlay(self._network_setup_page)
  #   has_internet = self._network_monitor.network_connected.is_set()
  #   self._network_setup_page.set_has_internet(has_internet)
  #   self._network_setup_page.render(rect)

  def render_downloading(self, rect: rl.Rectangle):
    self._downloading_page.set_progress(self.download_progress)
    self._downloading_page.render(rect)

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
      self._push_network_setup()

  def download(self, url: str):
    gui_app.pop_widgets_to(self)

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
    gui_app.push_widget(self._download_failed_page)
    self._set_state(SetupState.START)


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
    traceback.print_exc()
    print(f"Setup error: {e}")
  finally:
    gui_app.close()


if __name__ == "__main__":
  main()
