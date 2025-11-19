#!/usr/bin/env python3
"""
Cycle through different driver camera brightness/strategy settings with keyboard
Press RIGHT ARROW to go to next setting
Press LEFT ARROW to go to previous setting
Press ESC to exit
"""
import pyray as rl

from msgq.visionipc import VisionStreamType
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.onroad.cameraview import CameraView


class TestCameraView(CameraView):
  """Camera view with adjustable brightness parameter for testing"""
  def __init__(self, name: str, stream_type: VisionStreamType, brightness: float):
    super().__init__(name, stream_type)
    # Override brightness value
    self._brightness_val = rl.ffi.new("float[1]", [brightness])
    self.brightness = brightness


if __name__ == "__main__":
  # Test configurations: (brightness, description)
  configs = [
    (1.0, "1.0 - Normal (no adjustment)"),
    (1.1, "1.1 - Very subtle boost"),
    (1.2, "1.2 - Subtle boost"),
    (1.3, "1.3 - Light boost"),
    (1.4, "1.4 - Moderate boost"),
    (1.5, "1.5 - Medium boost"),
    (1.6, "1.6 - Strong boost"),
    (1.7, "1.7 - Stronger boost"),
    (1.8, "1.8 - Very strong boost"),
    (2.0, "2.0 - Maximum recommended"),
  ]

  print("\n" + "="*60)
  print("Driver Camera Brightness Cycle Test")
  print("="*60)
  print("Controls:")
  print("  TAP RIGHT SIDE - Next setting")
  print("  TAP LEFT SIDE  - Previous setting")
  print("="*60 + "\n")

  gui_app.init_window("Driver Camera Cycle Test")

  # Start with index 4 (1.4)
  current_index = 4
  current_camera = TestCameraView("camerad", VisionStreamType.VISION_STREAM_DRIVER, configs[current_index][0])
  print(f"Starting with: {configs[current_index][1]} ({current_index + 1}/{len(configs)})")

  # Track if we need to recreate camera
  needs_update = False
  new_index = current_index

  for _ in gui_app.render():
    # Check for touch/mouse input
    for event in gui_app.mouse_events:
      if event.left_down:
        # Right side = next, left side = previous
        if event.pos.x > gui_app.width / 2:
          new_index = min(current_index + 1, len(configs) - 1)
          needs_update = True
        else:
          new_index = max(current_index - 1, 0)
          needs_update = True

    # Recreate camera if needed
    if needs_update and new_index != current_index:
      current_index = new_index
      current_camera.close()
      current_camera = TestCameraView("camerad", VisionStreamType.VISION_STREAM_DRIVER, configs[current_index][0])
      brightness, description = configs[current_index]
      print(f"Switched to: {description} ({current_index + 1}/{len(configs)})")
      needs_update = False

    # Render camera view (fullscreen, no overlay)
    current_camera.render(rl.Rectangle(0, 0, gui_app.width, gui_app.height))

  # Cleanup
  current_camera.close()
  print("Done!\n")
