#!/usr/bin/env python3
import os
import re
import threading
import time
import urllib.request
from urllib.parse import urlparse
from enum import IntEnum
import shutil

import pyray as rl

from cereal import log
from openpilot.common.run import run_cmd
from openpilot.system.hardware import HARDWARE
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import Button, ButtonStyle, ButtonRadio
from openpilot.system.ui.widgets.keyboard import Keyboard
from openpilot.system.ui.widgets.label import Label, TextAlignment
from openpilot.system.ui.widgets.network import WifiManagerUI, WifiManagerWrapper

NetworkType = log.DeviceState.NetworkType

MARGIN = 50
TITLE_FONT_SIZE = 116
TITLE_FONT_WEIGHT = FontWeight.MEDIUM
NEXT_BUTTON_WIDTH = 310
BODY_FONT_SIZE = 96
BUTTON_HEIGHT = 160
BUTTON_SPACING = 50

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

class SetupState(IntEnum):
  LOW_VOLTAGE = 0
  GETTING_STARTED = 1
  NETWORK_SETUP = 2
  SOFTWARE_SELECTION = 3
  CUSTOM_SOFTWARE = 4
  DOWNLOADING = 5
  DOWNLOAD_FAILED = 6
  CUSTOM_SOFTWARE_WARNING = 7


class Setup(Widget):
  def __init__(self):
    super().__init__()
    self.state = SetupState.GETTING_STARTED
    self.network_check_thread = None
    self.network_connected = threading.Event()
    self.wifi_connected = threading.Event()
    self.stop_network_check_thread = threading.Event()
    self.failed_url = ""
    self.failed_reason = ""
    self.download_url = ""
    self.download_progress = 0
    self.download_thread = None
    self.wifi_manager = WifiManagerWrapper()
    self.wifi_ui = WifiManagerUI(self.wifi_manager)
    self.keyboard = Keyboard()
    self.selected_radio = None
    self.warning = gui_app.texture("icons/warning.png", 150, 150)
    self.checkmark = gui_app.texture("icons/circled_check.png", 100, 100)

    self._low_voltage_title_label = Label("WARNING: Low Voltage", TITLE_FONT_SIZE, FontWeight.MEDIUM, TextAlignment.LEFT, text_color=rl.Color(255, 89, 79, 255))
    self._low_voltage_body_label = Label("Power your device in a car with a harness or proceed at your own risk.", BODY_FONT_SIZE,
                                         text_alignment=TextAlignment.LEFT)
    self._low_voltage_continue_button = Button("Continue", self._low_voltage_continue_button_callback)
    self._low_voltage_poweroff_button = Button("Power Off", HARDWARE.shutdown)

    self._getting_started_button = Button("", self._getting_started_button_callback, button_style=ButtonStyle.PRIMARY, border_radius=0)
    self._getting_started_title_label = Label("Getting Started", TITLE_FONT_SIZE, FontWeight.BOLD, TextAlignment.LEFT)
    self._getting_started_body_label = Label("Before we get on the road, let's finish installation and cover some details.",
                                             BODY_FONT_SIZE, text_alignment=TextAlignment.LEFT)

    self._software_selection_openpilot_button = ButtonRadio("openpilot", self.checkmark, font_size=BODY_FONT_SIZE, text_padding=80)
    self._software_selection_custom_software_button = ButtonRadio("Custom Software", self.checkmark, font_size=BODY_FONT_SIZE, text_padding=80)
    self._software_selection_continue_button = Button("Continue", self._software_selection_continue_button_callback,
                                                      button_style=ButtonStyle.PRIMARY)
    self._software_selection_continue_button.set_enabled(False)
    self._software_selection_back_button = Button("Back", self._software_selection_back_button_callback)
    self._software_selection_title_label = Label("Choose Software to Use", TITLE_FONT_SIZE, FontWeight.BOLD, TextAlignment.LEFT)

    self._download_failed_reboot_button = Button("Reboot device", HARDWARE.reboot)
    self._download_failed_startover_button = Button("Start over", self._download_failed_startover_button_callback, button_style=ButtonStyle.PRIMARY)
    self._download_failed_title_label = Label("Download Failed", TITLE_FONT_SIZE, FontWeight.BOLD, TextAlignment.LEFT)
    self._download_failed_url_label = Label("", 64, FontWeight.NORMAL, TextAlignment.LEFT)
    self._download_failed_body_label = Label("", BODY_FONT_SIZE, text_alignment=TextAlignment.LEFT)

    self._network_setup_back_button = Button("Back", self._network_setup_back_button_callback)
    self._network_setup_continue_button = Button("Waiting for internet", self._network_setup_continue_button_callback,
                                                 button_style=ButtonStyle.PRIMARY)
    self._network_setup_continue_button.set_enabled(False)
    self._network_setup_title_label = Label("Connect to Wi-Fi", TITLE_FONT_SIZE, FontWeight.BOLD, TextAlignment.LEFT)

    self._custom_software_warning_continue_button = Button("Continue", self._custom_software_warning_continue_button_callback)
    self._custom_software_warning_back_button = Button("Back", self._custom_software_warning_back_button_callback)
    self._custom_software_warning_title_label = Label("WARNING: Custom Software", 100, FontWeight.BOLD, TextAlignment.LEFT, text_color=rl.Color(255,89,79,255),
                                                      text_padding=60)
    self._custom_software_warning_body_label = Label("Use caution when installing third-party software. Third-party software has not been tested by comma,"
                                              + " and may cause damage to your device and/or vehicle.\n\nIf you'd like to proceed, use https://flash.comma.ai "
                                              + "to restore your device to a factory state later.",
                                             85, text_alignment=TextAlignment.LEFT, text_padding=60)
    self._downloading_body_label = Label("Downloading...", TITLE_FONT_SIZE, FontWeight.MEDIUM)

    try:
      with open("/sys/class/hwmon/hwmon1/in1_input") as f:
        voltage = float(f.read().strip()) / 1000.0
        if voltage < 7:
          self.state = SetupState.LOW_VOLTAGE
    except (FileNotFoundError, ValueError):
      self.state = SetupState.LOW_VOLTAGE

  def _render(self, rect: rl.Rectangle):
    if self.state == SetupState.LOW_VOLTAGE:
      self.render_low_voltage(rect)
    elif self.state == SetupState.GETTING_STARTED:
      self.render_getting_started(rect)
    elif self.state == SetupState.NETWORK_SETUP:
      self.render_network_setup(rect)
    elif self.state == SetupState.SOFTWARE_SELECTION:
      self.render_software_selection(rect)
    elif self.state == SetupState.CUSTOM_SOFTWARE_WARNING:
      self.render_custom_software_warning(rect)
    elif self.state == SetupState.CUSTOM_SOFTWARE:
      self.render_custom_software()
    elif self.state == SetupState.DOWNLOADING:
      self.render_downloading(rect)
    elif self.state == SetupState.DOWNLOAD_FAILED:
      self.render_download_failed(rect)

  def _low_voltage_continue_button_callback(self):
    self.state = SetupState.GETTING_STARTED

  def _custom_software_warning_back_button_callback(self):
    self.state = SetupState.SOFTWARE_SELECTION

  def _custom_software_warning_continue_button_callback(self):
    self.state = SetupState.NETWORK_SETUP
    self.stop_network_check_thread.clear()
    self.start_network_check()

  def _getting_started_button_callback(self):
    self.state = SetupState.SOFTWARE_SELECTION

  def _software_selection_back_button_callback(self):
    self.state = SetupState.GETTING_STARTED

  def _software_selection_continue_button_callback(self):
    if self._software_selection_openpilot_button.selected:
      self.use_openpilot()
    else:
      self.state = SetupState.CUSTOM_SOFTWARE_WARNING

  def _download_failed_startover_button_callback(self):
    self.state = SetupState.GETTING_STARTED

  def _network_setup_back_button_callback(self):
    self.state = SetupState.SOFTWARE_SELECTION

  def _network_setup_continue_button_callback(self):
    self.stop_network_check_thread.set()
    if self._software_selection_openpilot_button.selected:
      self.download(OPENPILOT_URL)
    else:
      self.state = SetupState.CUSTOM_SOFTWARE

  def render_low_voltage(self, rect: rl.Rectangle):
    rl.draw_texture(self.warning, int(rect.x + 150), int(rect.y + 110), rl.WHITE)

    self._low_voltage_title_label.render(rl.Rectangle(rect.x + 150, rect.y + 110 + 150 + 100, rect.width - 500 - 150, TITLE_FONT_SIZE))
    self._low_voltage_body_label.render(rl.Rectangle(rect.x + 150, rect.y + 110 + 150 + 150, rect.width - 500, BODY_FONT_SIZE * 3))

    button_width = (rect.width - MARGIN * 3) / 2
    button_y = rect.height - MARGIN - BUTTON_HEIGHT
    self._low_voltage_poweroff_button.render(rl.Rectangle(rect.x + MARGIN, button_y, button_width, BUTTON_HEIGHT))
    self._low_voltage_continue_button.render(rl.Rectangle(rect.x + MARGIN * 2 + button_width, button_y, button_width, BUTTON_HEIGHT))

  def render_getting_started(self, rect: rl.Rectangle):
    self._getting_started_title_label.render(rl.Rectangle(rect.x + 165, rect.y + 280, rect.width - 265, TITLE_FONT_SIZE))
    self._getting_started_body_label.render(rl.Rectangle(rect.x + 165, rect.y + 280 + TITLE_FONT_SIZE, rect.width - 500, BODY_FONT_SIZE * 3))

    btn_rect = rl.Rectangle(rect.width - NEXT_BUTTON_WIDTH, 0, NEXT_BUTTON_WIDTH, rect.height)
    self._getting_started_button.render(btn_rect)
    triangle = gui_app.texture("images/button_continue_triangle.png", 54, int(btn_rect.height))
    rl.draw_texture_v(triangle, rl.Vector2(btn_rect.x + btn_rect.width / 2 - triangle.width / 2, btn_rect.height / 2 - triangle.height / 2), rl.WHITE)

  def check_network_connectivity(self):
    while not self.stop_network_check_thread.is_set():
      if self.state == SetupState.NETWORK_SETUP:
        try:
          urllib.request.urlopen(OPENPILOT_URL, timeout=2)
          self.network_connected.set()
          if HARDWARE.get_network_type() == NetworkType.wifi:
            self.wifi_connected.set()
          else:
            self.wifi_connected.clear()
        except Exception:
          self.network_connected.clear()
      time.sleep(1)

  def start_network_check(self):
    if self.network_check_thread is None or not self.network_check_thread.is_alive():
      self.network_check_thread = threading.Thread(target=self.check_network_connectivity, daemon=True)
      self.network_check_thread.start()

  def close(self):
    if self.network_check_thread is not None:
      self.stop_network_check_thread.set()
      self.network_check_thread.join()

  def render_network_setup(self, rect: rl.Rectangle):
    self._network_setup_title_label.render(rl.Rectangle(rect.x + MARGIN, rect.y + MARGIN, rect.width - MARGIN * 2, TITLE_FONT_SIZE))

    wifi_rect = rl.Rectangle(rect.x + MARGIN, rect.y + TITLE_FONT_SIZE + MARGIN + 25, rect.width - MARGIN * 2,
                             rect.height - TITLE_FONT_SIZE - 25 - BUTTON_HEIGHT - MARGIN * 3)
    rl.draw_rectangle_rounded(wifi_rect, 0.05, 10, rl.Color(51, 51, 51, 255))
    wifi_content_rect = rl.Rectangle(wifi_rect.x + MARGIN, wifi_rect.y, wifi_rect.width - MARGIN * 2, wifi_rect.height)
    self.wifi_ui.render(wifi_content_rect)

    button_width = (rect.width - BUTTON_SPACING - MARGIN * 2) / 2
    button_y = rect.height - BUTTON_HEIGHT - MARGIN

    self._network_setup_back_button.render(rl.Rectangle(rect.x + MARGIN, button_y, button_width, BUTTON_HEIGHT))

    # Check network connectivity status
    continue_enabled = self.network_connected.is_set()
    self._network_setup_continue_button.set_enabled(continue_enabled)
    continue_text = ("Continue" if self.wifi_connected.is_set() else "Continue without Wi-Fi") if continue_enabled else "Waiting for internet"
    self._network_setup_continue_button.set_text(continue_text)
    self._network_setup_continue_button.render(rl.Rectangle(rect.x + MARGIN + button_width + BUTTON_SPACING, button_y, button_width, BUTTON_HEIGHT))

  def render_software_selection(self, rect: rl.Rectangle):
    self._software_selection_title_label.render(rl.Rectangle(rect.x + MARGIN, rect.y + MARGIN, rect.width - MARGIN * 2, TITLE_FONT_SIZE))

    radio_height = 230
    radio_spacing = 30

    self._software_selection_continue_button.set_enabled(False)

    openpilot_rect = rl.Rectangle(rect.x + MARGIN, rect.y + TITLE_FONT_SIZE + MARGIN * 2, rect.width - MARGIN * 2, radio_height)
    self._software_selection_openpilot_button.render(openpilot_rect)

    if self._software_selection_openpilot_button.selected:
      self._software_selection_continue_button.set_enabled(True)
      self._software_selection_custom_software_button.selected = False

    custom_rect = rl.Rectangle(rect.x + MARGIN, rect.y + TITLE_FONT_SIZE + MARGIN * 2 + radio_height + radio_spacing, rect.width - MARGIN * 2, radio_height)
    self._software_selection_custom_software_button.render(custom_rect)

    if self._software_selection_custom_software_button.selected:
      self._software_selection_continue_button.set_enabled(True)
      self._software_selection_openpilot_button.selected = False

    button_width = (rect.width - BUTTON_SPACING - MARGIN * 2) / 2
    button_y = rect.height - BUTTON_HEIGHT - MARGIN

    self._software_selection_back_button.render(rl.Rectangle(rect.x + MARGIN, button_y, button_width, BUTTON_HEIGHT))
    self._software_selection_continue_button.render(rl.Rectangle(rect.x + MARGIN + button_width + BUTTON_SPACING, button_y, button_width, BUTTON_HEIGHT))

  def render_downloading(self, rect: rl.Rectangle):
    self._downloading_body_label.render(rl.Rectangle(rect.x, rect.y + rect.height / 2 - TITLE_FONT_SIZE / 2, rect.width, TITLE_FONT_SIZE))

  def render_download_failed(self, rect: rl.Rectangle):
    self._download_failed_title_label.render(rl.Rectangle(rect.x + 117, rect.y + 185, rect.width - 117, TITLE_FONT_SIZE))
    self._download_failed_url_label.set_text(self.failed_url)
    self._download_failed_url_label.render(rl.Rectangle(rect.x + 117, rect.y + 185 + TITLE_FONT_SIZE + 67, rect.width - 117 - 100, 64))

    self._download_failed_body_label.set_text(self.failed_reason)
    self._download_failed_body_label.render(rl.Rectangle(rect.x + 117, rect.y, rect.width - 117 - 100, rect.height))

    button_width = (rect.width - BUTTON_SPACING - MARGIN * 2) / 2
    button_y = rect.height - BUTTON_HEIGHT - MARGIN
    self._download_failed_reboot_button.render(rl.Rectangle(rect.x + MARGIN, button_y, button_width, BUTTON_HEIGHT))
    self._download_failed_startover_button.render(rl.Rectangle(rect.x + MARGIN + button_width + BUTTON_SPACING, button_y, button_width, BUTTON_HEIGHT))

  def render_custom_software_warning(self, rect: rl.Rectangle):
    self._custom_software_warning_title_label.render(rl.Rectangle(rect.x + 50, rect.y + 150, rect.width - 265, TITLE_FONT_SIZE))
    self._custom_software_warning_body_label.render(rl.Rectangle(rect.x + 50, rect.y + 200 , rect.width - 50, BODY_FONT_SIZE * 3))

    button_width = (rect.width - MARGIN * 3) / 2
    button_y = rect.height - MARGIN - BUTTON_HEIGHT
    self._custom_software_warning_back_button.render(rl.Rectangle(rect.x + MARGIN, button_y, button_width, BUTTON_HEIGHT))
    self._custom_software_warning_continue_button.render(rl.Rectangle(rect.x + MARGIN * 2 + button_width, button_y, button_width, BUTTON_HEIGHT))

  def render_custom_software(self):
    def handle_keyboard_result(result):
      # Enter pressed
      if result == 1:
        url = self.keyboard.text
        self.keyboard.clear()
        if url:
          self.download(url)

      # Cancel pressed
      elif result == 0:
        self.state = SetupState.SOFTWARE_SELECTION

    self.keyboard.reset()
    self.keyboard.set_title("Enter URL", "for Custom Software")
    gui_app.set_modal_overlay(self.keyboard, callback=handle_keyboard_result)

  def use_openpilot(self):
    if os.path.isdir(INSTALL_PATH) and os.path.isfile(VALID_CACHE_PATH):
      os.remove(VALID_CACHE_PATH)
      with open(TMP_CONTINUE_PATH, "w") as f:
        f.write(CONTINUE)
      run_cmd(["chmod", "+x", TMP_CONTINUE_PATH])
      shutil.move(TMP_CONTINUE_PATH, CONTINUE_PATH)
      shutil.copyfile(INSTALLER_SOURCE_PATH, INSTALLER_DESTINATION_PATH)

      # give time for installer UI to take over
      time.sleep(1)
      gui_app.request_close()
    else:
      self.state = SetupState.NETWORK_SETUP
      self.stop_network_check_thread.clear()
      self.start_network_check()

  def download(self, url: str):
    # autocomplete incomplete URLs
    if re.match("^([^/.]+)/([^/]+)$", url):
      url = f"https://installer.comma.ai/{url}"

    parsed = urlparse(url, scheme='https')
    self.download_url = (urlparse(f"https://{url}") if not parsed.netloc else parsed).geturl()

    self.state = SetupState.DOWNLOADING

    self.download_thread = threading.Thread(target=self._download_thread, daemon=True)
    self.download_thread.start()

  def _download_thread(self):
    try:
      import tempfile

      fd, tmpfile = tempfile.mkstemp(prefix="installer_")

      headers = {"User-Agent": USER_AGENT, "X-openpilot-serial": HARDWARE.get_serial()}
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
        self.download_failed(self.download_url, "No custom software found at this URL.")
        return

      # AGNOS might try to execute the installer before this process exits.
      # Therefore, important to close the fd before renaming the installer.
      os.close(fd)
      os.rename(tmpfile, INSTALLER_DESTINATION_PATH)

      with open(INSTALLER_URL_PATH, "w") as f:
        f.write(self.download_url)

      # give time for installer UI to take over
      time.sleep(5)
      gui_app.request_close()

    except Exception:
      error_msg = "Ensure the entered URL is valid, and the device's internet connection is good."
      self.download_failed(self.download_url, error_msg)

  def download_failed(self, url: str, reason: str):
    self.failed_url = url
    self.failed_reason = reason
    self.state = SetupState.DOWNLOAD_FAILED


def main():
  try:
    gui_app.init_window("Setup", 20)
    setup = Setup()
    for _ in gui_app.render():
      setup.render(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
    setup.close()
  except Exception as e:
    print(f"Setup error: {e}")
  finally:
    gui_app.close()


if __name__ == "__main__":
  main()
