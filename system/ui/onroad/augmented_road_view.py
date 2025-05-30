import numpy as np
import pyray as rl
from enum import Enum

from cereal import messaging, log
from msgq.visionipc import VisionStreamType
from openpilot.system.ui.onroad.alert_renderer import AlertRenderer
from openpilot.system.ui.onroad.driver_state import DriverStateRenderer
from openpilot.system.ui.onroad.hud_renderer import HudRenderer
from openpilot.system.ui.onroad.model_renderer import ModelRenderer
from openpilot.system.ui.widgets.cameraview import CameraView
from openpilot.system.ui.lib.application import gui_app
from openpilot.common.transformations.camera import DEVICE_CAMERAS, DeviceCameraConfig, view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler


OpState = log.SelfdriveState.OpenpilotState
CALIBRATED = log.LiveCalibrationData.Status.calibrated
DEFAULT_DEVICE_CAMERA = DEVICE_CAMERAS["tici", "ar0231"]
UI_BORDER_SIZE = 30

class BorderStatus(Enum):
  DISENGAGED = rl.Color(0x17, 0x33, 0x49, 0xc8) # Blue for disengaged state
  OVERRIDE = rl.Color(0x91, 0x9b, 0x95, 0xf1)   # Gray for override state
  ENGAGED = rl.Color(0x17, 0x86, 0x44, 0xf1)    # Green for engaged state


class AugmentedRoadView(CameraView):
  def __init__(self, sm: messaging.SubMaster, stream_type: VisionStreamType):
    super().__init__("camerad", stream_type)

    self.sm = sm
    self.stream_type = stream_type
    self.is_wide_camera = stream_type == VisionStreamType.VISION_STREAM_WIDE_ROAD

    self.device_camera: DeviceCameraConfig | None = None
    self.view_from_calib = view_frame_from_device_frame.copy()
    self.view_from_wide_calib = view_frame_from_device_frame.copy()

    self._last_calib_time: float = 0
    self._last_rect_dims = (0.0, 0.0)
    self._cached_matrix: np.ndarray | None = None
    self._content_rect = rl.Rectangle()

    self.model_renderer = ModelRenderer()
    self._hud_renderer = HudRenderer()
    self.alert_renderer = AlertRenderer()
    self.driver_state_renderer = DriverStateRenderer()

  def render(self, rect):
    # Update calibration before rendering
    self._update_calibration()

    # Create inner content area with border padding
    self._content_rect = rl.Rectangle(
      rect.x + UI_BORDER_SIZE,
      rect.y + UI_BORDER_SIZE,
      rect.width - 2 * UI_BORDER_SIZE,
      rect.height - 2 * UI_BORDER_SIZE,
    )

    # Draw colored border based on driving state
    self._draw_border(rect)

    # Enable scissor mode to clip all rendering within content rectangle boundaries
    # This creates a rendering viewport that prevents graphics from drawing outside the border
    rl.begin_scissor_mode(
      int(self._content_rect.x),
      int(self._content_rect.y),
      int(self._content_rect.width),
      int(self._content_rect.height)
    )

    # Render the base camera view
    super().render(rect)

    # Draw all UI overlays
    self.model_renderer.draw(self._content_rect, self.sm)
    self._hud_renderer.draw(self._content_rect, self.sm)
    self.alert_renderer.draw(self._content_rect, self.sm)
    self.driver_state_renderer.draw(self._content_rect, self.sm)

    # Custom UI extension point - add custom overlays here
    # Use self._content_rect for positioning within camera bounds

    # End clipping region
    rl.end_scissor_mode()

  def _draw_border(self, rect: rl.Rectangle):
    state = self.sm["selfdriveState"]
    if state.state in (OpState.preEnabled, OpState.overriding):
      status = BorderStatus.OVERRIDE
    elif state.enabled:
      status = BorderStatus.ENGAGED
    else:
      status = BorderStatus.DISENGAGED

    rl.draw_rectangle_lines_ex(rect, UI_BORDER_SIZE, status.value)

  def _update_calibration(self):
    # Update device camera if not already set
    if not self.device_camera and sm.seen['roadCameraState'] and sm.seen['deviceState']:
      self.device_camera = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['roadCameraState'].sensor))]

    # Check if live calibration data is available and valid
    if not (sm.updated["liveCalibration"] and sm.valid['liveCalibration']):
      return

    calib = self.sm['liveCalibration']
    if len(calib.rpyCalib) != 3 or calib.calStatus != CALIBRATED:
      return

    # Update view_from_calib matrix
    device_from_calib = rot_from_euler(calib.rpyCalib)
    self.view_from_calib = view_frame_from_device_frame @ device_from_calib

    # Update wide calibration if available
    if hasattr(calib, 'wideFromDeviceEuler') and len(calib.wideFromDeviceEuler) == 3:
      wide_from_device = rot_from_euler(calib.wideFromDeviceEuler)
      self.view_from_wide_calib = view_frame_from_device_frame @ wide_from_device @ device_from_calib

  def _calc_frame_matrix(self, rect: rl.Rectangle) -> np.ndarray:
    # Check if we can use cached matrix
    calib_time = self.sm.recv_frame['liveCalibration']
    current_dims = (self._content_rect.width, self._content_rect.height)
    if (self._last_calib_time == calib_time and
        self._last_rect_dims == current_dims and
        self._cached_matrix is not None):
      return self._cached_matrix

    # Get camera configuration
    device_camera = self.device_camera or DEFAULT_DEVICE_CAMERA
    intrinsic = device_camera.ecam.intrinsics if self.is_wide_camera else device_camera.fcam.intrinsics
    calibration = self.view_from_wide_calib if self.is_wide_camera else self.view_from_calib
    zoom = 2.0 if self.is_wide_camera else 1.1

    # Calculate transforms for vanishing point
    inf_point = np.array([1000.0, 0.0, 0.0])
    calib_transform = intrinsic @ calibration
    kep = calib_transform @ inf_point

    # Calculate center points and dimensions
    x, y = self._content_rect.x, self._content_rect.y
    w, h = self._content_rect.width, self._content_rect.height
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    # Calculate max allowed offsets with margins
    margin = 5
    max_x_offset = cx * zoom - w / 2 - margin
    max_y_offset = cy * zoom - h / 2 - margin

    # Calculate and clamp offsets to prevent out-of-bounds issues
    try:
      if abs(kep[2]) > 1e-6:
        x_offset = np.clip((kep[0] / kep[2] - cx) * zoom, -max_x_offset, max_x_offset)
        y_offset = np.clip((kep[1] / kep[2] - cy) * zoom, -max_y_offset, max_y_offset)
      else:
        x_offset, y_offset = 0, 0
    except (ZeroDivisionError, OverflowError):
      x_offset, y_offset = 0, 0

    # Update cache values
    self._last_calib_time = calib_time
    self._last_rect_dims = current_dims
    self._cached_matrix = np.array([
      [zoom * 2 * cx / w, 0, -x_offset / w * 2],
      [0, zoom * 2 * cy / h, -y_offset / h * 2],
      [0, 0, 1.0]
    ])

    video_transform = np.array([
        [zoom, 0.0, (w / 2 + x - x_offset) - (cx * zoom)],
        [0.0, zoom, (h / 2 + y - y_offset) - (cy * zoom)],
        [0.0, 0.0, 1.0]
    ])
    self.model_renderer.set_transform(video_transform @ calib_transform)

    return self._cached_matrix


if __name__ == "__main__":
  gui_app.init_window("OnRoad Camera View")
  sm = messaging.SubMaster(["modelV2", "controlsState", "liveCalibration", "radarState", "deviceState",
    "pandaStates", "carParams", "driverMonitoringState", "carState", "driverStateV2",
    "roadCameraState", "wideRoadCameraState", "managerState", "selfdriveState", "longitudinalPlan"])
  road_camera_view = AugmentedRoadView(sm, VisionStreamType.VISION_STREAM_ROAD)
  try:
    for _ in gui_app.render():
      sm.update(0)
      road_camera_view.render(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
  finally:
    road_camera_view.close()
