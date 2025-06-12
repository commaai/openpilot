import pyray as rl
import threading

from openpilot.common.swaglog import cloudlog
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.system.ui.lib.application import DEFAULT_FPS
from openpilot.system.hardware import HARDWARE
from openpilot.selfdrive.ui.ui_state import UIState


# Constants
BACKLIGHT_DT = 0.05
BACKLIGHT_TS = 10.0
BACKLIGHT_OFFROAD = 50.0
UI_FREQ = 20


class DeviceState:
  def __init__(self, ui_state: UIState):
    self._ui_state = ui_state

    # Brightness management
    self.brightness = 0
    self.last_brightness = 0
    self.offroad_brightness = BACKLIGHT_OFFROAD
    self.brightness_filter = FirstOrderFilter(BACKLIGHT_OFFROAD, BACKLIGHT_TS, BACKLIGHT_DT)

    # Wakefulness management
    self.awake = True
    self.ignition_on = False
    self.interactive_timeout = 0

    # Threading for async brightness updates
    self._brightness_thread: threading.Thread | None = None

    # Initialize device state
    self.set_awake(True)
    self.reset_interactive_timeout()

  def update(self):
    # Handle user interaction and update device state
    if rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT):
      self.reset_interactive_timeout()

    self.update_brightness()
    self.update_wakefulness()

  def set_awake(self, on: bool):
    if on != self.awake:
      self.awake = on
      HARDWARE.set_display_power(self.awake)
      cloudlog.debug(f"Device {'awake' if self.awake else 'asleep'}")

  def reset_interactive_timeout(self, timeout: int = -1):
    if timeout == -1:
      timeout = 10 if self._ui_state.ignition else 30
    self.interactive_timeout = timeout * DEFAULT_FPS

  def update_brightness(self):
    clipped_brightness = self.offroad_brightness

    if self._ui_state.started and self._ui_state.light_sensor >= 0:
      clipped_brightness = self._ui_state.light_sensor

      # CIE 1931 color space conversion
      # https://www.photonstophotos.net/GeneralTopics/Exposure/Psychometric_Lightness_and_Gamma.htm
      if clipped_brightness <= 8:
        clipped_brightness = clipped_brightness / 903.3
      else:
        clipped_brightness = pow((clipped_brightness + 16.0) / 116.0, 3.0)

      # Scale back to 10% to 100%
      clipped_brightness = max(10.0, min(100.0, 100.0 * clipped_brightness))

    # Apply filter to smooth brightness changes
    brightness = int(self.brightness_filter.update(clipped_brightness))

    # Turn off brightness when not awake
    if not self.awake:
      brightness = 0

    # Update brightness if changed and not already updating
    if brightness != self.last_brightness:
      if self._brightness_thread is None or not self._brightness_thread.is_alive():
        self._set_brightness_async(brightness)
        self.last_brightness = brightness

  def _set_brightness_async(self, brightness: int):
    cloudlog.debug(f"Setting brightness to: {brightness}")
    self._brightness_thread = threading.Thread(target=lambda: HARDWARE.set_screen_brightness(brightness), daemon=True)
    self._brightness_thread.start()

  def update_wakefulness(self):
    # Detect ignition state changes
    ignition_just_turned_off = not self._ui_state.ignition and self.ignition_on
    self.ignition_on = self._ui_state.ignition
    if ignition_just_turned_off:
      self.reset_interactive_timeout()
    elif self.interactive_timeout > 0:
      self.interactive_timeout -= 1

    # Set awake state based on ignition or interactive timeout
    should_be_awake = self._ui_state.ignition or self.interactive_timeout > 0
    self.set_awake(should_be_awake)

  def get_brightness(self) -> int:
    return self.brightness

  def is_awake(self) -> bool:
    return self.awake

  def has_user_activity(self) -> bool:
    return self.interactive_timeout > 0

  def __del__(self):
    if self._brightness_thread and self._brightness_thread.is_alive():
      self._brightness_thread.join(timeout=1.0)
