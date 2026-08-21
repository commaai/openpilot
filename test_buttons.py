#!/usr/bin/env python3
import os
import time

from openpilot.cereal import messaging
from openpilot.common.hardware import COMMA_HARDWARE
from openpilot.common.realtime import Priority, config_realtime_process, set_core_affinity
from openpilot.system.ui.lib.application import gui_app
# from openpilot.selfdrive.ui.layouts.main import MainLayout
# from openpilot.selfdrive.ui.mici.layouts.main import MiciMainLayout
from openpilot.selfdrive.ui.mici.widgets.button import BigButton, BigMultiToggle, BigToggle, GreyBigButton
from openpilot.system.ui.mici_setup import BigPillButton
from openpilot.system.ui.widgets.scroller import Scroller
from openpilot.selfdrive.ui.ui_state import ui_state

BIG_UI = gui_app.big_ui()


class Font64BigButton(BigButton):
  """SettingsBigButton (settings.py) and PairBigButton (device.py) both bump the title to 64"""

  def _get_label_font_size(self):
    return 64


class MainLayout(Scroller):
  def __init__(self):
    super().__init__()

    # icons at the sizes the UI actually loads them at
    txt_settings = gui_app.texture("icons_mici/settings.png", 64, 64)
    txt_info = gui_app.texture("icons_mici/settings/device/info.png", 64, 64)
    txt_cameras = gui_app.texture("icons_mici/settings/device/cameras.png", 64, 64)
    txt_comma = gui_app.texture("icons_mici/settings/comma_icon.png", 33, 60)
    txt_reboot = gui_app.texture("icons_mici/settings/device/reboot.png", 64, 70)
    txt_update = gui_app.texture("icons_mici/settings/device/update.png", 64, 75)
    txt_up_to_date = gui_app.texture("icons_mici/settings/device/up_to_date.png", 64, 64)
    txt_ssh = gui_app.texture("icons_mici/settings/developer/ssh.png", 56, 64)
    txt_wifi = gui_app.texture("icons_mici/settings/network/wifi_strength_full.png", 64, 47)
    txt_warning = gui_app.texture("icons_mici/setup/warning.png", 64, 64)
    txt_warning_58 = gui_app.texture("icons_mici/setup/warning.png", 64, 58)
    txt_green_dm = gui_app.texture("icons_mici/setup/green_dm.png", 64, 64)
    txt_factory_reset = gui_app.texture("icons_mici/setup/factory_reset.png", 64, 64)

    self._scroller.add_widgets([
      # BigButton: title, icon, no value (settings.py SettingsBigButton)
      Font64BigButton("toggles", "", txt_settings),
      # BigButton: single line title, icon, no value (device.py regulatory info, network_layout.py tethering password)
      BigButton("regulatory info", "", txt_info),
      # BigButton: multi-line title, icon, no value (device.py)
      BigButton("driver\ncamera preview", "", txt_cameras),
      # BigButton: title, value, icon (device.py PairBigButton)
      Font64BigButton("pair", "connect.comma.ai", txt_comma),
      # BigButton: title, value, icon (developer.py SSH keys)
      BigButton("SSH keys", "Not set", txt_ssh),
      # BigButton: title, value, icon (software.py InstallUpdateButton)
      BigButton("install update", "0.10.1 (release-chestnut)", txt_reboot),
      # BigButton: no title, value, icon (software.py CheckUpdateButton, title cleared when a value is set)
      BigButton("", "updater failed\nto respond", txt_update),
      # BigButton: title, value, no icon (network_layout.py apn settings)
      BigButton("apn settings", "edit"),
      # BigButton: no title, value, no icon (wifi_ui.py ScanningButton)
      BigButton("", "searching for networks"),
      # BigButton: title, value, icon, scrolling title (network/__init__.py WifiNetworkButton)
      BigButton("a-really-long-network-name", "192.168.100.100", txt_wifi, scroll=True),
      # BigButton: title, icon, no value, scrolling title (software.py branch picker, current target)
      BigButton("release3-staging", "", txt_up_to_date, scroll=True),
      # BigButton: title only, no icon, scrolling title (software.py branch picker, other branches)
      BigButton("master-ci", "", None, scroll=True),
      # BigButton: title, long value, no icon (software.py TargetBranchButton, value is an arbitrary branch)
      BigButton("target branch", "a-really-long-feature-branch"),

      # BigPillButton: centered middle aligned title, no icon or value (mici_setup.py)
      BigPillButton("next"),
      # BigPillButton: multi-line title, disabled background (mici_setup.py)
      BigPillButton("connect to\ncontinue", disabled_background=True),
      # BigPillButton: green variant (mici_setup.py)
      BigPillButton("install openpilot", green=True),

      # BigToggle: title only, off (developer.py)
      BigToggle("joystick debug mode"),
      # BigToggle: title only, on, wraps to two lines (developer.py)
      BigToggle("longitudinal maneuver mode", initial_state=True),
      # BigMultiToggle: title + the current option as the value (network_layout.py)
      BigMultiToggle("network usage", ["default", "metered", "unmetered"]),
      # BigMultiToggle: longer title (toggles.py BigMultiParamToggle)
      BigMultiToggle("driving personality", ["aggressive", "standard", "relaxed"]),

      # GreyBigButton: multi-line title, single line value, icon (toggles.py, onboarding.py, mici_setup.py headers)
      GreyBigButton("enabling\nexperimental mode", "scroll to continue", txt_warning),
      # GreyBigButton: single line title, single line value, icon (developer.py, onboarding.py)
      GreyBigButton("cabin camera data", "do you want to share video data for training?", txt_green_dm),
      # GreyBigButton: single line title, multi-line value, icon (mici_reset.py, mici_updater.py, onboarding.py)
      GreyBigButton("factory reset", "resetting erases\nall user content & data", txt_factory_reset),
      # GreyBigButton: same shape with the shorter icon (mici_setup.py FailedPage, description defaulted)
      GreyBigButton("download failed", "swipe down to go\nback and try again", txt_warning_58),
      # GreyBigButton: title only (toggles.py section headers)
      GreyBigButton("End-to-End Longitudinal Control"),
      # GreyBigButton: no title, wrapping value, no icon (toggles.py body copy, BigDialog)
      GreyBigButton("", "openpilot will drive as it thinks a human would, including stopping for red lights and stop signs."),
      # GreyBigButton: no title, explicit multi-line value, no icon (mici_reset.py, mici_setup.py reason card)
      GreyBigButton("", "For a deeper reset, go to\nhttps://flash.comma.ai"),
    ])



def main():
  cores = {5, }
  # above plannerd and radard
  config_realtime_process(0, Priority.CTRL_HIGH)

  gui_app.init_window("UI")
  ml = MainLayout()
  gui_app.push_widget(ml)


  for should_render, frame_time, cpu_time in gui_app.render():
    ui_state.update()


if __name__ == "__main__":
  main()
