#!/usr/bin/env python3
import os
import re
import threading
import time
import urllib.request
from enum import IntEnum
import pyray as rl

from cereal import log
from openpilot.system.hardware import HARDWARE
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.button import gui_button, ButtonStyle
from openpilot.system.ui.lib.label import gui_label, gui_text_box
from openpilot.system.ui.widgets.network import WifiManagerUI, WifiManagerWrapper
from openpilot.system.ui.widgets.keyboard import Keyboard

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


class SetupState(IntEnum):
  LOW_VOLTAGE = 0
  GETTING_STARTED = 1
  NETWORK_SETUP = 2
  SOFTWARE_SELECTION = 3
  CUSTOM_URL = 4
  DOWNLOADING = 5
  DOWNLOAD_FAILED = 6


class Setup:
  def __init__(self):
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

    try:
      with open("/sys/class/hwmon/hwmon1/in1_input") as f:
        voltage = float(f.read().strip()) / 1000.0
        if voltage < 7:
          self.state = SetupState.LOW_VOLTAGE
    except (FileNotFoundError, ValueError):
      self.state = SetupState.LOW_VOLTAGE

  def render(self, rect: rl.Rectangle):
    if self.state == SetupState.LOW_VOLTAGE:
      self.render_low_voltage(rect)
    elif self.state == SetupState.GETTING_STARTED:
      self.render_getting_started(rect)
    elif self.state == SetupState.NETWORK_SETUP:
      self.render_network_setup(rect)
    elif self.state == SetupState.SOFTWARE_SELECTION:
      self.render_software_selection(rect)
    elif self.state == SetupState.CUSTOM_URL:
      self.render_custom_url()
    elif self.state == SetupState.DOWNLOADING:
      self.render_downloading(rect)
    elif self.state == SetupState.DOWNLOAD_FAILED:
      self.render_download_failed(rect)

  def render_low_voltage(self, rect: rl.Rectangle):
    rl.draw_texture(self.warning, int(rect.x + 150), int(rect.y + 110), rl.WHITE)

    title_rect = rl.Rectangle(rect.x + 150, rect.y + 110 + 150 + 100, rect.width - 500 - 150, TITLE_FONT_SIZE)
    gui_label(title_rect, "WARNING: Low Voltage", TITLE_FONT_SIZE, rl.Color(255, 89, 79, 255), FontWeight.MEDIUM)

    body_rect = rl.Rectangle(rect.x + 150, rect.y + 110 + 150 + 100 + TITLE_FONT_SIZE + 25, rect.width - 500 - 150, BODY_FONT_SIZE * 3)
    gui_text_box(body_rect, "Power your device in a car with a harness or proceed at your own risk.", BODY_FONT_SIZE)

    button_width = (rect.width - MARGIN * 3) / 2
    button_y = rect.height - MARGIN - BUTTON_HEIGHT

    if gui_button(rl.Rectangle(rect.x + MARGIN, button_y, button_width, BUTTON_HEIGHT), "Power off"):
      HARDWARE.shutdown()

    if gui_button(rl.Rectangle(rect.x + MARGIN * 2 + button_width, button_y, button_width, BUTTON_HEIGHT), "Continue"):
      self.state = SetupState.GETTING_STARTED

  def render_getting_started(self, rect: rl.Rectangle):
    title_rect = rl.Rectangle(rect.x + 165, rect.y + 280, rect.width - 265, TITLE_FONT_SIZE)
    gui_label(title_rect, "Getting Started", TITLE_FONT_SIZE, font_weight=FontWeight.MEDIUM)

    desc_rect = rl.Rectangle(rect.x + 165, rect.y + 280 + TITLE_FONT_SIZE + 90, rect.width - 500, BODY_FONT_SIZE * 3)
    gui_text_box(desc_rect, "Before we get on the road, let's finish installation and cover some details.", BODY_FONT_SIZE)

    btn_rect = rl.Rectangle(rect.width - NEXT_BUTTON_WIDTH, 0, NEXT_BUTTON_WIDTH, rect.height)

    ret = gui_button(btn_rect, "", button_style=ButtonStyle.PRIMARY, border_radius=0)
    triangle = gui_app.texture("images/button_continue_triangle.png", 54, int(btn_rect.height))
    rl.draw_texture_v(triangle, rl.Vector2(btn_rect.x + btn_rect.width / 2 - triangle.width / 2, btn_rect.height / 2 - triangle.height / 2), rl.WHITE)

    if ret:
      self.state = SetupState.NETWORK_SETUP
      self.wifi_manager.request_scan()
      self.start_network_check()

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
    if self.wifi_ui.require_full_screen:
      self.wifi_ui.render(rect)
      return

    title_rect = rl.Rectangle(rect.x + MARGIN, rect.y + MARGIN, rect.width - MARGIN * 2, TITLE_FONT_SIZE)
    gui_label(title_rect, "Connect to Wi-Fi", TITLE_FONT_SIZE, font_weight=FontWeight.MEDIUM)

    wifi_rect = rl.Rectangle(rect.x + MARGIN, rect.y + TITLE_FONT_SIZE + MARGIN + 25, rect.width - MARGIN * 2,
                             rect.height - TITLE_FONT_SIZE - 25 - BUTTON_HEIGHT - MARGIN * 3)
    rl.draw_rectangle_rounded(wifi_rect, 0.05, 10, rl.Color(51, 51, 51, 255))
    wifi_content_rect = rl.Rectangle(wifi_rect.x + MARGIN, wifi_rect.y, wifi_rect.width - MARGIN * 2, wifi_rect.height)
    self.wifi_ui.render(wifi_content_rect)

    button_width = (rect.width - BUTTON_SPACING - MARGIN * 2) / 2
    button_y = rect.height - BUTTON_HEIGHT - MARGIN

    if gui_button(rl.Rectangle(rect.x + MARGIN, button_y, button_width, BUTTON_HEIGHT), "Back"):
      self.state = SetupState.GETTING_STARTED

    # Check network connectivity status
    continue_enabled = self.network_connected.is_set()
    continue_text = ("Continue" if self.wifi_connected.is_set() else "Continue without Wi-Fi") if continue_enabled else "Waiting for internet"

    if gui_button(
      rl.Rectangle(rect.x + MARGIN + button_width + BUTTON_SPACING, button_y, button_width, BUTTON_HEIGHT),
      continue_text,
      button_style=ButtonStyle.PRIMARY if continue_enabled else ButtonStyle.NORMAL,
      is_enabled=continue_enabled,
    ):
      self.state = SetupState.SOFTWARE_SELECTION
      self.stop_network_check_thread.set()

  def render_software_selection(self, rect: rl.Rectangle):
    title_rect = rl.Rectangle(rect.x + MARGIN, rect.y + MARGIN, rect.width - MARGIN * 2, TITLE_FONT_SIZE)
    gui_label(title_rect, "Choose Software to Install", TITLE_FONT_SIZE, font_weight=FontWeight.MEDIUM)

    radio_height = 230
    radio_spacing = 30

    openpilot_rect = rl.Rectangle(rect.x + MARGIN, rect.y + TITLE_FONT_SIZE + MARGIN * 2, rect.width - MARGIN * 2, radio_height)
    openpilot_selected = self.selected_radio == "openpilot"

    rl.draw_rectangle_rounded(openpilot_rect, 0.1, 10, rl.Color(70, 91, 234, 255) if openpilot_selected else rl.Color(79, 79, 79, 255))
    gui_label(rl.Rectangle(openpilot_rect.x + 100, openpilot_rect.y, openpilot_rect.width - 200, radio_height), "openpilot", BODY_FONT_SIZE)

    if openpilot_selected:
      checkmark_pos = rl.Vector2(openpilot_rect.x + openpilot_rect.width - 100 - self.checkmark.width,
                                 openpilot_rect.y + radio_height / 2 - self.checkmark.height / 2)
      rl.draw_texture_v(self.checkmark, checkmark_pos, rl.WHITE)

    custom_rect = rl.Rectangle(rect.x + MARGIN, rect.y + TITLE_FONT_SIZE + MARGIN * 2 + radio_height + radio_spacing, rect.width - MARGIN * 2, radio_height)
    custom_selected = self.selected_radio == "custom"

    rl.draw_rectangle_rounded(custom_rect, 0.1, 10, rl.Color(70, 91, 234, 255) if custom_selected else rl.Color(79, 79, 79, 255))
    gui_label(rl.Rectangle(custom_rect.x + 100, custom_rect.y, custom_rect.width - 200, radio_height), "Custom Software", BODY_FONT_SIZE)

    if custom_selected:
      checkmark_pos = rl.Vector2(custom_rect.x + custom_rect.width - 100 - self.checkmark.width, custom_rect.y + radio_height / 2 - self.checkmark.height / 2)
      rl.draw_texture_v(self.checkmark, checkmark_pos, rl.WHITE)

    mouse_pos = rl.get_mouse_position()
    if rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT):
      if rl.check_collision_point_rec(mouse_pos, openpilot_rect):
        self.selected_radio = "openpilot"
      elif rl.check_collision_point_rec(mouse_pos, custom_rect):
        self.selected_radio = "custom"

    button_width = (rect.width - BUTTON_SPACING - MARGIN * 2) / 2
    button_y = rect.height - BUTTON_HEIGHT - MARGIN

    if gui_button(rl.Rectangle(rect.x + MARGIN, button_y, button_width, BUTTON_HEIGHT), "Back"):
      self.state = SetupState.NETWORK_SETUP

    continue_enabled = self.selected_radio is not None
    if gui_button(
      rl.Rectangle(rect.x + MARGIN + button_width + BUTTON_SPACING, button_y, button_width, BUTTON_HEIGHT),
      "Continue",
      button_style=ButtonStyle.PRIMARY,
      is_enabled=continue_enabled,
    ):
      if continue_enabled:
        if self.selected_radio == "openpilot":
          self.download(OPENPILOT_URL)
        else:
          self.state = SetupState.CUSTOM_URL

  def render_downloading(self, rect: rl.Rectangle):
    title_rect = rl.Rectangle(rect.x, rect.y + rect.height / 2 - TITLE_FONT_SIZE / 2, rect.width, TITLE_FONT_SIZE)
    gui_label(title_rect, "Downloading...", TITLE_FONT_SIZE, font_weight=FontWeight.MEDIUM, alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER)

  def render_download_failed(self, rect: rl.Rectangle):
    title_rect = rl.Rectangle(rect.x + 117, rect.y + 185, rect.width - 117, TITLE_FONT_SIZE)
    gui_label(title_rect, "Download Failed", TITLE_FONT_SIZE, font_weight=FontWeight.MEDIUM)

    url_rect = rl.Rectangle(rect.x + 117, rect.y + 185 + TITLE_FONT_SIZE + 67, rect.width - 117 - 100, 64)
    gui_label(url_rect, self.failed_url, 64, font_weight=FontWeight.NORMAL)

    error_rect = rl.Rectangle(rect.x + 117, rect.y + 185 + TITLE_FONT_SIZE + 67 + 64 + 48,
                              rect.width - 117 - 100, rect.height - 185 + TITLE_FONT_SIZE + 67 + 64 + 48 - BUTTON_HEIGHT - MARGIN * 2)
    gui_text_box(error_rect, self.failed_reason, BODY_FONT_SIZE)

    button_width = (rect.width - BUTTON_SPACING - MARGIN * 2) / 2
    button_y = rect.height - BUTTON_HEIGHT - MARGIN

    if gui_button(rl.Rectangle(rect.x + MARGIN, button_y, button_width, BUTTON_HEIGHT), "Reboot device"):
      HARDWARE.reboot()

    if gui_button(rl.Rectangle(rect.x + MARGIN + button_width + BUTTON_SPACING, button_y, button_width, BUTTON_HEIGHT), "Start over",
                  button_style=ButtonStyle.PRIMARY):
      self.state = SetupState.GETTING_STARTED

  def render_custom_url(self):
    result = self.keyboard.render("Enter URL", "for Custom Software")

    # Enter pressed
    if result == 1:
      url = self.keyboard.text
      self.keyboard.clear()
      if url:
        self.download(url)

    # Cancel pressed
    elif result == 0:
      self.state = SetupState.SOFTWARE_SELECTION

  def download(self, url: str):
    # autocomplete incomplete URLs
    if re.match("^([^/.]+)/([^/]+)$", url):
      url = f"https://installer.comma.ai/{url}"

    self.download_url = url
    self.state = SetupState.DOWNLOADING

    self.download_thread = threading.Thread(target=self._download_thread, daemon=True)
    self.download_thread.start()

  def _download_thread(self):
    try:
      import tempfile

      _, tmpfile = tempfile.mkstemp(prefix="installer_")

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

      os.rename(tmpfile, "/tmp/installer")
      os.chmod("/tmp/installer", 0o755)

      with open("/tmp/installer_url", "w") as f:
        f.write(self.download_url)

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
    gui_app.init_window("Setup")
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
