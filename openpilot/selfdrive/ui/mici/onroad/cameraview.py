from functools import cache
import numpy as np

from openpilot.cereal.visionipc import VisionStreamType
from openpilot.selfdrive.ui.onroad.cameraview import CameraView as BaseCameraView, color_curve
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus


@cache
def frame_color(engaged: bool, enhance_driver: bool) -> tuple[np.ndarray, float]:
  """engaged: 20% saturation, 20% more contrast, and the display's gamma. otherwise 85% bright. a driver's face brightened"""
  def curve(c):
    if engaged:
      c = np.clip((c - 0.5) * 1.2 + 0.5, 0, 1)
      c = c ** (1 / 1.28)
    else:
      c = c * 0.85
    if enhance_driver:
      c = np.clip((c + 0.15 - 0.5) * 0.88 + 0.5, 0, 1)
      c = (c * c * (3 - 2 * c)) ** 0.8
    return c

  saturation = 0.2 if engaged else 1.0
  return color_curve(curve), saturation


class CameraView(BaseCameraView):
  def __init__(self, name: str, stream_type: VisionStreamType):
    super().__init__(name, stream_type)
    self._enhance_driver = stream_type == VisionStreamType.VISION_STREAM_CABIN

  def _frame_color(self) -> tuple[np.ndarray, float]:
    return frame_color(ui_state.status != UIStatus.DISENGAGED, self._enhance_driver)
