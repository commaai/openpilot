import pyray as rl
import numpy as np
from openpilot.system.ui.widgets.cameraview import CameraView
from msgq.visionipc import VisionStreamType
from openpilot.system.ui.lib.application import gui_app
from cereal import messaging


VIEW_FROM_DEVICE = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0]
])

FCAM_INTRINSIC_MATRIX = np.array([
    [2648.0, 0.0, 964.0],
    [0.0, 2648.0, 604.0],
    [0.0, 0.0, 1.0]
])

ECAM_INTRINSIC_MATRIX = np.array([
    [567.0, 0.0, 964.0],
    [0.0, 567.0, 604.0],
    [0.0, 0.0, 1.0]
])

class AugmentedRoadView(CameraView):
  def __init__(self, sm: messaging.SubMaster, stream_type: VisionStreamType):
    super().__init__("camerad", stream_type)
    self.stream_type = stream_type
    self.sm = sm

    self.view_from_calib = VIEW_FROM_DEVICE.copy()
    self.view_from_wide_calib = VIEW_FROM_DEVICE.copy()

    self.is_wide_camera = stream_type == VisionStreamType.VISION_STREAM_WIDE_ROAD

  def render(self, rect):
    super().render(rect)

    # TODO: Add road visualization overlays like:
    # - Lane lines and road edges
    # - Path prediction
    # - Lead vehicle indicators

  def calc_frame_matrix(self, rect: rl.Rectangle) -> rl.Matrix:
    self._update_calibration()

    intrinsic = ECAM_INTRINSIC_MATRIX if self.is_wide_camera else FCAM_INTRINSIC_MATRIX
    calibration = self.view_from_wide_calib if self.is_wide_camera else self.view_from_calib
    zoom = 2.0 if self.is_wide_camera else 1.1

    # Calculate transforms
    calib_transform = intrinsic @ calibration
    Kep = calib_transform @ np.array([1000.0, 0.0, 0.0])

    # Calculate offsets
    w, h = rect.width, rect.height
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    # Calculate offsets with clamping
    max_x_offset = cx * zoom - w / 2 - 5
    max_y_offset = cy * zoom - h / 2 - 5

    if Kep[2] != 0:
      x_offset = np.clip((Kep[0] / Kep[2] - cx) * zoom, -max_x_offset, max_x_offset)
      y_offset = np.clip((Kep[1] / Kep[2] - cy) * zoom, -max_y_offset, max_y_offset)
    else:
      x_offset, y_offset = 0, 0

    # Create transform matrix
    matrix = rl.Matrix()
    matrix.m0 = zoom * 2 * cx / w
    matrix.m5 = zoom * 2 * cy / h
    matrix.m3 = -x_offset / w * 2
    matrix.m7 = -y_offset / h * 2
    matrix.m10 = matrix.m15 = 1.0
    return matrix

  def _update_calibration(self):
    if self.sm.valid['liveCalibration']:
      calib = self.sm['liveCalibration']
      if hasattr(calib, 'rpyCalib') and len(calib.rpyCalib) == 3 and calib.calStatus == 'CALIBRATED':
        device_from_calib = self._list2rot(calib.rpyCalib)
        self.view_from_calib = VIEW_FROM_DEVICE @ device_from_calib

        if hasattr(calib, 'wideFromDeviceEuler') and len(calib.wideFromDeviceEuler) == 3:
          wide_from_device = self._list2rot(calib.wideFromDeviceEuler)
          self.view_from_wide_calib = VIEW_FROM_DEVICE @ wide_from_device @ device_from_calib

  def _list2rot(self, rpy):
    """Convert roll, pitch, yaw to rotation matrix"""
    roll, pitch, yaw = rpy

    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cr, sr = np.cos(roll), np.sin(roll)

    rot = np.array(
      [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
      ]
    )

    return rot


if __name__ == "__main__":
  gui_app.init_window("OnRoad Camera View")
  sm = messaging.SubMaster(['liveCalibration'])
  road_camera_view = AugmentedRoadView(sm, VisionStreamType.VISION_STREAM_ROAD)
  try:
    for _ in gui_app.render():
      sm.update(0)
      road_camera_view.render(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
  finally:
    road_camera_view.close()
