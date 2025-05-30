import numpy as np
import pyray as rl
from openpilot.system.ui.lib.application import gui_app

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
])

# UI constants
UI_BORDER_SIZE = 30
BTN_SIZE = 192
IMG_SIZE = 144
ARC_LENGTH = 133
ARC_THICKNESS_DEFAULT = 6.7
ARC_THICKNESS_EXTEND = 12.0


class DriverStateRenderer:
  def __init__(self):
    # Initial state with NumPy arrays
    self.face_kpts_draw = DEFAULT_FACE_KPTS_3D.copy()
    self.is_visible = False
    self.is_active = False
    self.is_rhd = False
    self.dm_fade_state = 0.0
    self.driver_pose_vals = np.zeros(3)
    self.driver_pose_diff = np.zeros(3)
    self.driver_pose_sins = np.zeros(3)
    self.driver_pose_coss = np.zeros(3)
    self.face_outline: list[rl.Vector2] = []

    # Load the driver face icon
    self.dm_img = gui_app.texture("icons/driver_face.png", IMG_SIZE, IMG_SIZE)

    # Colors
    self.engaged_color = rl.Color(26, 242, 66, 255)
    self.disengaged_color = rl.Color(139, 139, 139, 255)

  def update_state(self, sm):
    """Update the driver monitoring state based on model data"""
    # Quick exit if driver state isn't available
    self.is_visible = (
      sm.seen['driverStateV2'] and sm.seen["driverMonitoringState"] and sm["selfdriveState"].alertSize == 0
    )
    if not self.is_visible or not sm.updated["driverMonitoringState"]:
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
    scales = np.where(driver_orient < 0, [0.7, 0.4, 0.4], [0.9, 0.4, 0.4])
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

  def draw(self, rect, sm):
    """Draw the driver monitoring visualization"""
    # Update state and exit if not visible
    self.update_state(sm)
    if not self.is_visible:
      return

    # Get drawing dimensions
    width, height = rect.width, rect.height

    # Calculate icon position (bottom-left or bottom-right)
    offset = UI_BORDER_SIZE + BTN_SIZE // 2
    x = rect.x + (width - offset if self.is_rhd else offset)
    y = rect.y + height - offset

    # Set opacity based on active state
    opacity = 0.65 if self.is_active else 0.2

    # Draw background circle
    rl.draw_circle(int(x), int(y), BTN_SIZE // 2, rl.Color(0, 0, 0, 70))

    # Draw face icon
    icon_pos = rl.Vector2(x - self.dm_img.width // 2, y - self.dm_img.height // 2)
    rl.draw_texture_v(self.dm_img, icon_pos, rl.Color(255, 255, 255, int(255 * opacity)))

    # Calculate transformed keypoints for drawing
    kp = (self.face_kpts_draw[:, 2] - 8) / 120.0 + 1.0
    keypoints = self.face_kpts_draw[:, :2] * kp[:, None] + np.array([x, y])
    # Draw face outline
    lines = [rl.Vector2(int(keypoints[i][0]), int(keypoints[i][1])) for i in range(len(keypoints))]
    white_color = rl.Color(255, 255, 255, int(255 * opacity))
    rl.draw_spline_linear(lines, len(lines), 5.2, white_color)

    # Get arc color based on engaged state (hardcoded to True for now)
    engaged = True
    arc_color = self.engaged_color if engaged else self.disengaged_color
    arc_color.a = int(0.4 * 255 * (1.0 - self.dm_fade_state))  # Fade out when inactive

    # Draw tracking arcs if head is rotated
    self.draw_tracking_arcs(x, y, arc_color)

  def draw_tracking_arcs(self, x, y, color):
    """Draw horizontal and vertical tracking arcs showing head rotation"""
    # Calculate arc dimensions based on head rotation
    delta_x = -self.driver_pose_sins[1] * ARC_LENGTH / 2.0  # Horizontal movement
    delta_y = -self.driver_pose_sins[0] * ARC_LENGTH / 2.0  # Vertical movement

    # Draw horizontal tracking arc (if head is turned left/right)
    h_width = abs(delta_x)
    if h_width > 0:
      h_thickness = ARC_THICKNESS_DEFAULT + ARC_THICKNESS_EXTEND * min(1.0, self.driver_pose_diff[1] * 5.0)
      h_start_angle = 90 if self.driver_pose_sins[1] > 0 else -90
      h_x = min(x + delta_x, x)
      h_y = y - ARC_LENGTH / 2

      self.draw_arc(h_x, h_y, h_width, ARC_LENGTH, h_start_angle, 180, h_thickness, color)

    # Draw vertical tracking arc (if head is tilted up/down)
    v_height = abs(delta_y)
    if v_height > 0:
      v_thickness = ARC_THICKNESS_DEFAULT + ARC_THICKNESS_EXTEND * min(1.0, self.driver_pose_diff[0] * 5.0)
      v_start_angle = 0 if self.driver_pose_sins[0] > 0 else 180
      v_x = x - ARC_LENGTH / 2
      v_y = min(y + delta_y, y)

      self.draw_arc(v_x, v_y, ARC_LENGTH, v_height, v_start_angle, 180, v_thickness, color)

  def draw_arc(self, x, y, width, height, start_angle, arc_angle, thickness, color):
    """Draw an arc with specified thickness using line segments"""
    # Convert angles to radians and generate angles with NumPy
    start_rad = np.deg2rad(start_angle)
    end_rad = np.deg2rad(start_angle + arc_angle)
    angles = np.linspace(start_rad, end_rad, 37)  # 37 points for 5-degree steps

    # Calculate ellipse center and radii
    center_x = x + width / 2
    center_y = y + height / 2
    radius_x = width / 2
    radius_y = height / 2

    # Compute arc points using vectorized operations
    x_coords = center_x + np.cos(angles) * radius_x
    y_coords = center_y + np.sin(angles) * radius_y

    # Draw connected line segments to form the arc
    for i in range(len(angles) - 1):
      rl.draw_line_ex(
        rl.Vector2(x_coords[i], y_coords[i]),
        rl.Vector2(x_coords[i + 1], y_coords[i + 1]),
        thickness,
        color,
      )
