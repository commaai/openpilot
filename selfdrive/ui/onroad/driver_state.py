import numpy as np
import pyray as rl
from dataclasses import dataclass
from openpilot.selfdrive.ui.ui_state import ui_state, UI_BORDER_SIZE
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget

# Default 3D coordinates for face keypoints as a NumPy array
DEFAULT_FACE_KPTS_3D = np.array([
  [-5.98, -51.20, 8.00], [-17.64, -49.14, 8.00], [-23.81, -46.40, 8.00], [-29.98, -40.91, 8.00],
  [-32.04, -37.49, 8.00], [-34.10, -32.00, 8.00], [-36.16, -21.03, 8.00], [-36.16, 6.40, 8.00],
  [-35.47, 10.51, 8.00], [-32.73, 19.43, 8.00], [-29.30, 26.29, 8.00], [-24.50, 33.83, 8.00],
  [-19.01, 41.37, 8.00], [-14.21, 46.17, 8.00], [-12.16, 47.54, 8.00], [-4.61, 49.60, 8.00],
  [4.99, 49.60, 8.00], [12.53, 47.54, 8.00], [14.59, 46.17, 8.00], [19.39, 41.37, 8.00],
  [24.87, 33.83, 8.00], [29.67, 26.29, 8.00], [33.10, 19.43, 8.00], [35.84, 10.51, 8.00],
  [36.53, 6.40, 8.00], [36.53, -21.03, 8.00], [34.47, -32.00, 8.00], [32.42, -37.49, 8.00],
  [30.36, -40.91, 8.00], [24.19, -46.40, 8.00], [18.02, -49.14, 8.00], [6.36, -51.20, 8.00],
  [-5.98, -51.20, 8.00],
], dtype=np.float32)

# UI constants
BTN_SIZE = 192
IMG_SIZE = 144
ARC_LENGTH = 133
ARC_THICKNESS_DEFAULT = 6.7
ARC_THICKNESS_EXTEND = 12.0

SCALES_POS = np.array([0.9, 0.4, 0.4], dtype=np.float32)
SCALES_NEG = np.array([0.7, 0.4, 0.4], dtype=np.float32)

ARC_POINT_COUNT = 37  # Number of points in the arc
ARC_ANGLES = np.linspace(0.0, np.pi, ARC_POINT_COUNT, dtype=np.float32)


@dataclass
class ArcData:
  """Data structure for arc rendering parameters."""
  x: float
  y: float
  width: float
  height: float
  thickness: float


class DriverStateRenderer(Widget):
  def __init__(self):
    super().__init__()
    # Initial state with NumPy arrays
    self.face_kpts_draw = DEFAULT_FACE_KPTS_3D.copy()
    self.is_active = False
    self.is_rhd = False
    self.dm_fade_state = 0.0
    self.last_rect: rl.Rectangle = rl.Rectangle(0, 0, 0, 0)
    self.driver_pose_vals = np.zeros(3, dtype=np.float32)
    self.driver_pose_diff = np.zeros(3, dtype=np.float32)
    self.driver_pose_sins = np.zeros(3, dtype=np.float32)
    self.driver_pose_coss = np.zeros(3, dtype=np.float32)
    self.face_keypoints_transformed = np.zeros((DEFAULT_FACE_KPTS_3D.shape[0], 2), dtype=np.float32)
    self.position_x: float = 0.0
    self.position_y: float = 0.0
    self.h_arc_data = None
    self.v_arc_data = None

    # Pre-allocate drawing arrays
    self.face_lines = [rl.Vector2(0, 0) for _ in range(len(DEFAULT_FACE_KPTS_3D))]
    self.h_arc_lines = [rl.Vector2(0, 0) for _ in range(ARC_POINT_COUNT)]
    self.v_arc_lines = [rl.Vector2(0, 0) for _ in range(ARC_POINT_COUNT)]

    # Load the driver face icon
    self.dm_img = gui_app.texture("icons/driver_face.png", IMG_SIZE, IMG_SIZE)

    # Colors
    self.white_color = rl.Color(255, 255, 255, 255)
    self.arc_color = rl.Color(26, 242, 66, 255)
    self.engaged_color = rl.Color(26, 242, 66, 255)
    self.disengaged_color = rl.Color(139, 139, 139, 255)

    self.set_visible(lambda: (ui_state.sm.recv_frame['driverStateV2'] > ui_state.started_frame and
                              ui_state.sm.seen['driverMonitoringState']))

  def _render(self, rect):
    # Set opacity based on active state
    opacity = 0.65 if self.is_active else 0.2

    # Draw background circle
    rl.draw_circle(int(self.position_x), int(self.position_y), BTN_SIZE // 2, rl.Color(0, 0, 0, 70))

    # Draw face icon
    icon_pos = rl.Vector2(self.position_x - self.dm_img.width // 2, self.position_y - self.dm_img.height // 2)
    rl.draw_texture_v(self.dm_img, icon_pos, rl.Color(255, 255, 255, int(255 * opacity)))

    # Draw face outline
    self.white_color.a = int(255 * opacity)
    rl.draw_spline_linear(self.face_lines, len(self.face_lines), 5.2, self.white_color)

    # Set arc color based on engaged state
    self.arc_color = self.engaged_color if ui_state.engaged else self.disengaged_color
    self.arc_color.a = int(0.4 * 255 * (1.0 - self.dm_fade_state))  # Fade out when inactive

    # Draw arcs
    if self.h_arc_data:
      rl.draw_spline_linear(self.h_arc_lines, len(self.h_arc_lines), self.h_arc_data.thickness, self.arc_color)
    if self.v_arc_data:
      rl.draw_spline_linear(self.v_arc_lines, len(self.v_arc_lines), self.v_arc_data.thickness, self.arc_color)

  def _update_state(self):
    """Update the driver monitoring state based on model data"""
    sm = ui_state.sm
    if not sm.updated["driverMonitoringState"]:
      if (self._rect.x != self.last_rect.x or self._rect.y != self.last_rect.y or
          self._rect.width != self.last_rect.width or self._rect.height != self.last_rect.height):
        self._pre_calculate_drawing_elements()
        self.last_rect = self._rect
      return

    # Get monitoring state
    dm_state = sm["driverMonitoringState"]
    self.is_active = dm_state.isActiveMode
    self.is_rhd = dm_state.isRHD

    # Update fade state (smoother transition between active/inactive)
    fade_target = 0.0 if self.is_active else 0.5
    self.dm_fade_state = np.clip(self.dm_fade_state + 0.2 * (fade_target - self.dm_fade_state), 0.0, 1.0)

    # Get driver orientation data from appropriate camera
    driverstate = sm["driverStateV2"]
    driver_data = driverstate.rightDriverData if self.is_rhd else driverstate.leftDriverData
    driver_orient = driver_data.faceOrientation

    # Update pose values with scaling and smoothing
    driver_orient = np.array(driver_orient)
    scales = np.where(driver_orient < 0, SCALES_NEG, SCALES_POS)
    v_this = driver_orient * scales
    self.driver_pose_diff = np.abs(self.driver_pose_vals - v_this)
    self.driver_pose_vals = 0.8 * v_this + 0.2 * self.driver_pose_vals  # Smooth changes

    # Apply fade to rotation and compute sin/cos
    rotation_amount = self.driver_pose_vals * (1.0 - self.dm_fade_state)
    self.driver_pose_sins = np.sin(rotation_amount)
    self.driver_pose_coss = np.cos(rotation_amount)

    # Create rotation matrix for 3D face model
    sin_y, sin_x, sin_z = self.driver_pose_sins
    cos_y, cos_x, cos_z = self.driver_pose_coss
    r_xyz = np.array(
      [
        [cos_x * cos_z, cos_x * sin_z, -sin_x],
        [-sin_y * sin_x * cos_z - cos_y * sin_z, -sin_y * sin_x * sin_z + cos_y * cos_z, -sin_y * cos_x],
        [cos_y * sin_x * cos_z - sin_y * sin_z, cos_y * sin_x * sin_z + sin_y * cos_z, cos_y * cos_x],
      ]
    )

    # Transform face keypoints using vectorized matrix multiplication
    self.face_kpts_draw = DEFAULT_FACE_KPTS_3D @ r_xyz.T
    self.face_kpts_draw[:, 2] = self.face_kpts_draw[:, 2] * (1.0 - self.dm_fade_state) + 8 * self.dm_fade_state

    # Pre-calculate the transformed keypoints
    kp_depth = (self.face_kpts_draw[:, 2] - 8) / 120.0 + 1.0
    self.face_keypoints_transformed = self.face_kpts_draw[:, :2] * kp_depth[:, None]

    # Pre-calculate all drawing elements
    self._pre_calculate_drawing_elements()

  def _pre_calculate_drawing_elements(self):
    """Pre-calculate all drawing elements based on the current rectangle"""
    # Calculate icon position (bottom-left or bottom-right)
    width, height = self._rect.width, self._rect.height
    offset = UI_BORDER_SIZE + BTN_SIZE // 2
    self.position_x = self._rect.x + (width - offset if self.is_rhd else offset)
    self.position_y = self._rect.y + height - offset

    # Pre-calculate the face lines positions
    positioned_keypoints = self.face_keypoints_transformed + np.array([self.position_x, self.position_y])
    for i in range(len(positioned_keypoints)):
      self.face_lines[i].x = positioned_keypoints[i][0]
      self.face_lines[i].y = positioned_keypoints[i][1]

    # Calculate arc dimensions based on head rotation
    delta_x = -self.driver_pose_sins[1] * ARC_LENGTH / 2.0  # Horizontal movement
    delta_y = -self.driver_pose_sins[0] * ARC_LENGTH / 2.0  # Vertical movement

    # Horizontal arc
    h_width = abs(delta_x)
    self.h_arc_data = self._calculate_arc_data(
      delta_x, h_width, self.position_x, self.position_y - ARC_LENGTH / 2,
      self.driver_pose_sins[1], self.driver_pose_diff[1], is_horizontal=True
    )

    # Vertical arc
    v_height = abs(delta_y)
    self.v_arc_data = self._calculate_arc_data(
      delta_y, v_height, self.position_x - ARC_LENGTH / 2, self.position_y,
      self.driver_pose_sins[0], self.driver_pose_diff[0], is_horizontal=False
    )

  def _calculate_arc_data(
    self, delta: float, size: float, x: float, y: float, sin_val: float, diff_val: float, is_horizontal: bool
  ):
    """Calculate arc data and pre-compute arc points."""
    if size <= 0:
      return None

    thickness = ARC_THICKNESS_DEFAULT + ARC_THICKNESS_EXTEND * min(1.0, diff_val * 5.0)
    start_angle = (90 if sin_val > 0 else -90) if is_horizontal else (0 if sin_val > 0 else 180)
    x = min(x + delta, x) if is_horizontal else x
    y = y if is_horizontal else min(y + delta, y)

    arc_data = ArcData(
      x=x,
      y=y,
      width=size if is_horizontal else ARC_LENGTH,
      height=ARC_LENGTH if is_horizontal else size,
      thickness=thickness,
    )

    # Pre-calculate arc points
    angles = ARC_ANGLES + np.deg2rad(start_angle)

    center_x = x + arc_data.width / 2
    center_y = y + arc_data.height / 2
    radius_x = arc_data.width / 2
    radius_y = arc_data.height / 2

    x_coords = center_x + np.cos(angles) * radius_x
    y_coords = center_y + np.sin(angles) * radius_y

    arc_lines = self.h_arc_lines if is_horizontal else self.v_arc_lines
    for i, (x_coord, y_coord) in enumerate(zip(x_coords, y_coords, strict=True)):
      arc_lines[i].x = x_coord
      arc_lines[i].y = y_coord

    return arc_data
