import numpy as np
import pyray as rl

from cereal import messaging, log
from msgq.visionipc import VisionStreamType
from openpilot.system.ui.onroad.model_renderer import ModelRenderer
from openpilot.system.ui.widgets.cameraview import CameraView
from openpilot.system.ui.lib.application import gui_app
from openpilot.common.transformations.camera import DEVICE_CAMERAS, DeviceCameraConfig, view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler


CALIBRATED = log.LiveCalibrationData.Status.calibrated
DEFAULT_DEVICE_CAMERA = DEVICE_CAMERAS["tici", "ar0231"]


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

    self.model_renderer = ModelRenderer()

  def render(self, rect):
    # Update calibration before rendering
    self._update_calibration()

    # Render the base camera view
    super().render(rect)

    # TODO: Add road visualization overlays like:
    # - Lane lines and road edges
    # - Path prediction
    # - Lead vehicle indicators
    # - Additional features
    self.model_renderer.draw(rect, self.sm)

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
    calib_time = self.sm.recv_frame.get('liveCalibration', 0)
    current_dims = (rect.width, rect.height)
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
    w, h = current_dims
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
        [zoom, 0.0, (w / 2 - x_offset) - (cx * zoom)],
        [0.0, zoom, (h / 2 - y_offset) - (cy * zoom)],
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
