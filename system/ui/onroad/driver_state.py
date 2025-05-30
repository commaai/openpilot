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
], dtype=np.float32)

# UI constants
UI_BORDER_SIZE = 30
BTN_SIZE = 192
IMG_SIZE = 144
ARC_LENGTH = 133
ARC_THICKNESS_DEFAULT = 6.7
ARC_THICKNESS_EXTEND = 12.0


SCALES_POS = np.array([0.9, 0.4, 0.4], dtype=np.float32)
SCALES_NEG = np.array([0.7, 0.4, 0.4], dtype=np.float32)


class DriverStateRenderer:
  def __init__(self):
    # Initial state with NumPy arrays
    self.face_kpts_draw = DEFAULT_FACE_KPTS_3D.copy()
    self.is_visible = False
    self.is_active = False
    self.is_rhd = False
    self.dm_fade_state = 0.0
    self.state_updated = False
    self.driver_pose_vals = np.zeros(3, dtype=np.float32)
    self.driver_pose_diff = np.zeros(3, dtype=np.float32)
    self.driver_pose_sins = np.zeros(3, dtype=np.float32)
    self.driver_pose_coss = np.zeros(3, dtype=np.float32)
    self.face_keypoints_transformed = np.zeros((DEFAULT_FACE_KPTS_3D.shape[0], 2), dtype=np.float32)

    # Pre-allocate Vector2 arrays for drawing
    self.face_lines = [rl.Vector2(0, 0) for _ in range(len(DEFAULT_FACE_KPTS_3D))]
    self.h_arc_lines = [rl.Vector2(0, 0) for _ in range(37)]  # 37 points for horizontal arc
    self.v_arc_lines = [rl.Vector2(0, 0) for _ in range(37)]  # 37 points for vertical arc

    # Load the driver face icon
    self.dm_img = gui_app.texture("icons/driver_face.png", IMG_SIZE, IMG_SIZE)
    self.render_texture = rl.load_render_texture(BTN_SIZE, BTN_SIZE)
    rl.set_texture_filter(self.render_texture.texture, rl.TextureFilter.TEXTURE_FILTER_ANISOTROPIC_8X)

    # Colors
    self.white_color = rl.Color(255, 255, 255, 255)
    self.arc_color = rl.Color(26, 242, 66, 255)
    self.engaged_color = rl.Color(26, 242, 66, 255)
    self.disengaged_color = rl.Color(139, 139, 139, 255)

  def update_state(self, sm, rect):
    """Update the driver monitoring state based on model data"""
    if  not sm.updated["driverMonitoringState"]:
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
    self.pre_calculate_drawing_elements()
    self._draw_render_texture()
    self.state_updated = True

  def draw(self, rect, sm):
    """Draw the driver monitoring visualization"""

    self.is_visible = (
      sm.seen['driverStateV2'] and sm.seen["driverMonitoringState"] and sm["selfdriveState"].alertSize == 0
    )

    if not self.is_visible or not sm.updated["driverMonitoringState"]:
      return

    self.update_state(sm, rect)
    if not self.is_visible:
      return

    x = rect.x + (rect.width - (UI_BORDER_SIZE + BTN_SIZE) if self.is_rhd else UI_BORDER_SIZE)
    y = rect.y + rect.height - (UI_BORDER_SIZE + BTN_SIZE)

    opacity = 0.65 if self.is_active else 0.2
    # Draw background circle
    rl.draw_circle(int(x + BTN_SIZE//2), int(y + BTN_SIZE//2), BTN_SIZE // 2, rl.Color(0, 0, 0, int(opacity * 255)))

    rl.draw_texture_pro(
      self.render_texture.texture,
      rl.Rectangle(0, BTN_SIZE, BTN_SIZE, -BTN_SIZE),  # Y flipped (negative height)
      rl.Rectangle(x, y, BTN_SIZE, BTN_SIZE),
      rl.Vector2(0, 0),
      0.0,
      rl.WHITE,
    )

  def _draw_render_texture(self):
    rl.begin_texture_mode(self.render_texture)
    rl.clear_background(rl.Color(0, 0, 0, 0))  # Transparent background

    # Center coordinates in texture
    center_x, center_y = BTN_SIZE // 2, BTN_SIZE // 2

    # Draw face icon - centered in texture
    icon_pos = rl.Vector2(center_x - self.dm_img.width // 2, center_y - self.dm_img.height // 2)
    rl.draw_texture_v(self.dm_img, icon_pos, rl.WHITE)  # Draw at full opacity in texture

    # Adjust face lines to be centered in texture
    face_offset_x = center_x - self.position_x
    face_offset_y = center_y - self.position_y

    # Create temporary face lines centered in texture
    temp_face_lines = []
    for i in range(len(self.face_lines)):
      v = rl.Vector2(self.face_lines[i].x + face_offset_x, self.face_lines[i].y + face_offset_y)
      temp_face_lines.append(v)

    # Draw face outline - always white in texture
    rl.draw_spline_linear(temp_face_lines, len(temp_face_lines), 5.2, rl.WHITE)

    # Set arc color (always at full opacity in texture)
    self.arc_color = self.engaged_color if True else self.disengaged_color

    # Draw tracking arcs if pre-calculated
    if self.h_arc_data:
      # Create temporary arc lines centered in texture
      temp_h_arc_lines = []
      for i in range(len(self.h_arc_lines)):
        v = rl.Vector2(self.h_arc_lines[i].x + face_offset_x, self.h_arc_lines[i].y + face_offset_y)
        temp_h_arc_lines.append(v)

      rl.draw_spline_linear(temp_h_arc_lines, len(temp_h_arc_lines), self.h_arc_data["thickness"], self.arc_color)

    if self.v_arc_data:
      # Create temporary arc lines centered in texture
      temp_v_arc_lines = []
      for i in range(len(self.v_arc_lines)):
        v = rl.Vector2(self.v_arc_lines[i].x + face_offset_x, self.v_arc_lines[i].y + face_offset_y)
        temp_v_arc_lines.append(v)

      rl.draw_spline_linear(temp_v_arc_lines, len(temp_v_arc_lines), self.v_arc_data["thickness"], self.arc_color)

    rl.end_texture_mode()

  def pre_calculate_drawing_elements(self):
    """Pre-calculate all drawing elements for rendering to texture"""
    # For texture rendering, we'll use texture center as our reference point
    center_x = BTN_SIZE / 2
    center_y = BTN_SIZE / 2

    # Store original screen position for final texture placement
    offset = BTN_SIZE // 2
    self.position_x = BTN_SIZE - offset if self.is_rhd else offset
    self.position_y = BTN_SIZE - offset

    # Pre-calculate the face lines positions - centered in texture
    # No need to offset by screen position since we're drawing directly in the texture
    positioned_keypoints = self.face_keypoints_transformed + np.array([center_x, center_y])
    for i in range(len(positioned_keypoints)):
      self.face_lines[i].x = positioned_keypoints[i][0]
      self.face_lines[i].y = positioned_keypoints[i][1]

    # Calculate arc dimensions based on head rotation
    delta_x = -self.driver_pose_sins[1] * ARC_LENGTH / 2.0  # Horizontal movement
    delta_y = -self.driver_pose_sins[0] * ARC_LENGTH / 2.0  # Vertical movement

    # Pre-calculate horizontal arc - centered in texture
    h_width = abs(delta_x)
    if h_width > 0:
      h_thickness = ARC_THICKNESS_DEFAULT + ARC_THICKNESS_EXTEND * min(1.0, self.driver_pose_diff[1] * 5.0)
      h_start_angle = 90 if self.driver_pose_sins[1] > 0 else -90
      h_x = min(center_x + delta_x, center_x)
      h_y = center_y - ARC_LENGTH / 2

      self.h_arc_data = {"x": h_x, "y": h_y, "width": h_width, "height": ARC_LENGTH, "thickness": h_thickness}

      # Pre-calculate arc points
      start_rad = np.deg2rad(h_start_angle)
      end_rad = np.deg2rad(h_start_angle + 180)
      angles = np.linspace(start_rad, end_rad, 37)

      arc_center_x = h_x + h_width / 2
      arc_center_y = h_y + ARC_LENGTH / 2
      radius_x = h_width / 2
      radius_y = ARC_LENGTH / 2

      x_coords = arc_center_x + np.cos(angles) * radius_x
      y_coords = arc_center_y + np.sin(angles) * radius_y

      for i in range(len(angles)):
        self.h_arc_lines[i].x = x_coords[i]
        self.h_arc_lines[i].y = y_coords[i]
    else:
      self.h_arc_data = None

    # Pre-calculate vertical arc - centered in texture
    v_height = abs(delta_y)
    if v_height > 0:
      v_thickness = ARC_THICKNESS_DEFAULT + ARC_THICKNESS_EXTEND * min(1.0, self.driver_pose_diff[0] * 5.0)
      v_start_angle = 0 if self.driver_pose_sins[0] > 0 else 180
      v_x = center_x - ARC_LENGTH / 2
      v_y = min(center_y + delta_y, center_y)

      self.v_arc_data = {"x": v_x, "y": v_y, "width": ARC_LENGTH, "height": v_height, "thickness": v_thickness}

      # Pre-calculate arc points
      start_rad = np.deg2rad(v_start_angle)
      end_rad = np.deg2rad(v_start_angle + 180)
      angles = np.linspace(start_rad, end_rad, 37)

      arc_center_x = v_x + ARC_LENGTH / 2
      arc_center_y = v_y + v_height / 2
      radius_x = ARC_LENGTH / 2
      radius_y = v_height / 2

      x_coords = arc_center_x + np.cos(angles) * radius_x
      y_coords = arc_center_y + np.sin(angles) * radius_y

      for i in range(len(angles)):
        self.v_arc_lines[i].x = x_coords[i]
        self.v_arc_lines[i].y = y_coords[i]
    else:
      self.v_arc_data = None
