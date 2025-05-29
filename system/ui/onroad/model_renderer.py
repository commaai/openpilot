import colorsys
import bisect
import numpy as np
import pyray as rl
from cereal import messaging, car
from openpilot.common.params import Params
from openpilot.system.ui.lib.application import DEFAULT_FPS
from openpilot.system.ui.lib.shader_polygon import draw_polygon


CLIP_MARGIN = 500
MIN_DRAW_DISTANCE = 10.0
MAX_DRAW_DISTANCE = 100.0
PATH_COLOR_TRANSITION_DURATION = 0.5  # Seconds for color transition animation
PATH_BLEND_INCREMENT = 1.0 / (PATH_COLOR_TRANSITION_DURATION * DEFAULT_FPS)

THROTTLE_COLORS = [
  rl.Color(13, 248, 122, 102),   # HSLF(148/360, 0.94, 0.51, 0.4)
  rl.Color(114, 255, 92, 89),    # HSLF(112/360, 1.0, 0.68, 0.35)
  rl.Color(114, 255, 92, 0),     # HSLF(112/360, 1.0, 0.68, 0.0)
]

NO_THROTTLE_COLORS = [
  rl.Color(242, 242, 242, 102), # HSLF(148/360, 0.0, 0.95, 0.4)
  rl.Color(242, 242, 242, 89),  # HSLF(112/360, 0.0, 0.95, 0.35)
  rl.Color(242, 242, 242, 0),   # HSLF(112/360, 0.0, 0.95, 0.0)
]


class ModelRenderer:
  def __init__(self):
    self._longitudinal_control = False
    self._experimental_mode = False
    self._blend_factor = 1.0
    self._prev_allow_throttle = True
    self._lane_line_probs = np.zeros(4, dtype=np.float32)
    self._road_edge_stds = np.zeros(2, dtype=np.float32)
    self._path_offset_z = 1.22

    # Initialize empty polygon vertices
    self._track_vertices = np.empty((0, 2), dtype=np.float32)
    self._lane_line_vertices = [np.empty((0, 2), dtype=np.float32) for _ in range(4)]
    self._road_edge_vertices = [np.empty((0, 2), dtype=np.float32) for _ in range(2)]
    self._lead_vertices = [None, None]

    # Transform matrix (3x3 for car space to screen space)
    self._car_space_transform = np.zeros((3, 3))
    self._transform_dirty = True
    self._clip_region = None
    self._rect = None

    # Get longitudinal control setting from car parameters
    car_params = Params().get("CarParams")
    if car_params:
      cp = messaging.log_from_bytes(car_params, car.CarParams)
      self._longitudinal_control = cp.openpilotLongitudinalControl

  def set_transform(self, transform: np.ndarray):
    self._car_space_transform = transform
    self._transform_dirty = True

  def draw(self, rect: rl.Rectangle, sm: messaging.SubMaster):
    # Check if data is up-to-date
    if not sm.valid['modelV2'] or not sm.valid['liveCalibration']:
      return

    # Set up clipping region
    self._rect = rect
    self._clip_region = rl.Rectangle(
      rect.x - CLIP_MARGIN, rect.y - CLIP_MARGIN, rect.width + 2 * CLIP_MARGIN, rect.height + 2 * CLIP_MARGIN
    )

    # Update flags based on car state
    self._experimental_mode = sm['selfdriveState'].experimentalMode
    self._path_offset_z = sm['liveCalibration'].height[0]
    if sm.updated['carParams']:
      self._longitudinal_control = sm['carParams'].openpilotLongitudinalControl

    # Get model and radar data
    model = sm['modelV2']
    radar_state = sm['radarState'] if sm.valid['radarState'] else None
    lead_one = radar_state.leadOne if radar_state else None
    render_lead_indicator = self._longitudinal_control and radar_state is not None

    # Update model data when needed
    if self._transform_dirty or sm.updated['modelV2'] or sm.updated['radarState']:
      self._update_model(model, lead_one)
      if render_lead_indicator:
        self._update_leads(radar_state, model.position)
      self._transform_dirty = False

    # Draw elements
    self._draw_lane_lines()
    self._draw_path(sm, model, rect.height)

    # Draw lead vehicles if available
    if render_lead_indicator and radar_state:
      lead_two = radar_state.leadTwo

      if lead_one and lead_one.status:
        self._draw_lead(lead_one, self._lead_vertices[0], rect)

      if lead_two and lead_two.status and lead_one and (abs(lead_one.dRel - lead_two.dRel) > 3.0):
        self._draw_lead(lead_two, self._lead_vertices[1], rect)

  def _update_leads(self, radar_state, line):
    """Update positions of lead vehicles"""
    leads = [radar_state.leadOne, radar_state.leadTwo]
    for i, lead_data in enumerate(leads):
      if lead_data and lead_data.status:
        d_rel = lead_data.dRel
        y_rel = lead_data.yRel
        idx = self._get_path_length_idx(line, d_rel)
        z = line.z[idx]
        self._lead_vertices[i] = self._map_to_screen(d_rel, -y_rel, z + self._path_offset_z)

  def _update_model(self, model, lead):
    """Update model visualization data based on model message"""
    model_position = model.position

    # Determine max distance to render
    max_distance = np.clip(model_position.x[-1], MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE)

    # Update lane lines
    lane_lines = model.laneLines
    line_probs = model.laneLineProbs
    max_idx = self._get_path_length_idx(lane_lines[0], max_distance)

    for i in range(4):
      self._lane_line_probs[i] = line_probs[i]
      self._lane_line_vertices[i] = self._map_line_to_polygon(
        lane_lines[i], 0.025 * self._lane_line_probs[i], 0, max_idx
      )

    # Update road edges
    road_edges = model.roadEdges
    edge_stds = model.roadEdgeStds

    for i in range(2):
      self._road_edge_stds[i] = edge_stds[i]
      self._road_edge_vertices[i] = self._map_line_to_polygon(road_edges[i], 0.025, 0, max_idx)

    # Update path
    if lead and lead.status:
      lead_d = lead.dRel * 2.0
      max_distance = np.clip(lead_d - min(lead_d * 0.35, 10.0), 0.0, max_distance)
      max_idx = self._get_path_length_idx(model_position, max_distance)

    self._track_vertices = self._map_line_to_polygon(model_position, 0.9, self._path_offset_z, max_idx, False)

  def _draw_lane_lines(self):
    """Draw lane lines and road edges"""
    for i, vertices in enumerate(self._lane_line_vertices):
      # Skip if no vertices
      if vertices.size == 0:
        continue

      # Draw lane line
      alpha = np.clip(self._lane_line_probs[i], 0.0, 0.7)
      color = rl.Color(255, 255, 255, int(alpha * 255))
      draw_polygon(self._rect, vertices, color)

    for i, vertices in enumerate(self._road_edge_vertices):
      # Skip if no vertices
      if vertices.size == 0:
        continue

      # Draw road edge
      alpha = np.clip(1.0 - self._road_edge_stds[i], 0.0, 1.0)
      color = rl.Color(255, 0, 0, int(alpha * 255))
      draw_polygon(self._rect, vertices, color)

  def _draw_path(self, sm, model, height):
    """Draw the path polygon with gradient based on acceleration"""
    if self._track_vertices.size == 0:
      return

    if self._experimental_mode:
      # Draw with acceleration coloring
      acceleration = model.acceleration.x
      max_len = min(len(self._track_vertices) // 2, len(acceleration))

      # Create segments for gradient coloring
      segment_colors = []
      gradient_stops = []

      i = 0
      while i < max_len:
        track_idx = max_len - i - 1  # flip idx to start from bottom right
        track_y = self._track_vertices[track_idx][1]
        if  track_y < 0 or track_y > height:
          i += 1
          continue

        # Calculate color based on acceleration
        lin_grad_point = (height - track_y) / height

        # speed up: 120, slow down: 0
        path_hue = max(min(60 + acceleration[i] * 35, 120), 0)
        path_hue = int(path_hue * 100 + 0.5) / 100

        saturation = min(abs(acceleration[i] * 1.5), 1)
        lightness = self._map_val(saturation, 0.0, 1.0, 0.95, 0.62)
        alpha = self._map_val(lin_grad_point, 0.75 / 2.0, 0.75, 0.4, 0.0)

        # Use HSL to RGB conversion
        color = self._hsla_to_color(path_hue / 360.0, saturation, lightness, alpha)

        # Create quad segment
        gradient_stops.append(lin_grad_point)
        segment_colors.append(color)

        # Skip a point, unless next is last
        i += 1 + (1 if (i + 2) < max_len else 0)

      if len(segment_colors) < 2:
        draw_polygon(self._rect, self._track_vertices, rl.Color(255, 255, 255, 30))
        return

      # Create gradient specification
      gradient = {
        'start': (0.0, 1.0),  # Bottom of path
        'end': (0.0, 0.0),  # Top of path
        'colors': segment_colors,
        'stops': gradient_stops,
      }
      draw_polygon(self._rect, self._track_vertices, gradient=gradient)
    else:
      # Draw with throttle/no throttle gradient
      allow_throttle = sm['longitudinalPlan'].allowThrottle or not self._longitudinal_control

      # Start transition if throttle state changes
      if allow_throttle != self._prev_allow_throttle:
        self._prev_allow_throttle = allow_throttle
        self._blend_factor = max(1.0 - self._blend_factor, 0.0)

      # Update blend factor
      if self._blend_factor < 1.0:
        self._blend_factor = min(self._blend_factor + PATH_BLEND_INCREMENT, 1.0)

      begin_colors = NO_THROTTLE_COLORS if allow_throttle else THROTTLE_COLORS
      end_colors = THROTTLE_COLORS if allow_throttle else NO_THROTTLE_COLORS

      # Blend colors based on transition
      blended_colors = self._blend_colors(begin_colors, end_colors, self._blend_factor)
      gradient = {
        'start': (0.0, 1.0),  # Bottom of path
        'end': (0.0, 0.0),  # Top of path
        'colors': blended_colors,
        'stops': [0.0, 0.5, 1.0],
      }
      draw_polygon(self._rect, self._track_vertices, gradient=gradient)

  def _draw_lead(self, lead_data, vd, rect):
    """Draw lead vehicle indicator"""
    if not vd:
      return

    speed_buff = 10.0
    lead_buff = 40.0
    d_rel = lead_data.dRel
    v_rel = lead_data.vRel

    # Calculate fill alpha
    fill_alpha = 0
    if d_rel < lead_buff:
      fill_alpha = 255 * (1.0 - (d_rel / lead_buff))
      if v_rel < 0:
        fill_alpha += 255 * (-1 * (v_rel / speed_buff))
      fill_alpha = min(fill_alpha, 255)

    # Calculate size and position
    sz = np.clip((25 * 30) / (d_rel / 3 + 30), 15.0, 30.0) * 2.35
    x = np.clip(vd[0], 0.0, rect.width - sz / 2)
    y = min(vd[1], rect.height - sz * 0.6)

    g_xo = sz / 5
    g_yo = sz / 10

    # Draw glow
    glow = [(x + (sz * 1.35) + g_xo, y + sz + g_yo), (x, y - g_yo), (x - (sz * 1.35) - g_xo, y + sz + g_yo)]
    rl.draw_triangle_fan(glow, len(glow), rl.Color(218, 202, 37, 255))

    # Draw chevron
    chevron = [(x + (sz * 1.25), y + sz), (x, y), (x - (sz * 1.25), y + sz)]
    rl.draw_triangle_fan(chevron, len(chevron), rl.Color(201, 34, 49, int(fill_alpha)))

  @staticmethod
  def _get_path_length_idx(line, path_height):
    """Get the index corresponding to the given path height"""
    return bisect.bisect_right(line.x, path_height) - 1

  def _map_to_screen(self, in_x, in_y, in_z):
    """Project a point in car space to screen space"""
    input_pt = np.array([in_x, in_y, in_z])
    pt = self._car_space_transform @ input_pt

    if abs(pt[2]) < 1e-6:
      return None

    x = pt[0] / pt[2]
    y = pt[1] / pt[2]

    clip = self._clip_region
    if x < clip.x or x > clip.x + clip.width or y < clip.y or y > clip.y + clip.height:
      return None

    return (x, y)

  def _map_line_to_polygon(self, line, y_off, z_off, max_idx, allow_invert=True)-> np.ndarray:
    """Convert a 3D line to a 2D polygon for drawing"""
    line_x = line.x
    line_y = line.y
    line_z = line.z

    left_points: list[tuple[float, float]] = []
    right_points: list[tuple[float, float]] = []

    for i in range(max_idx + 1):
      # Skip points with negative x (behind camera)
      if line_x[i] < 0:
        continue

      left = self._map_to_screen(line_x[i], line_y[i] - y_off, line_z[i] + z_off)
      right = self._map_to_screen(line_x[i], line_y[i] + y_off, line_z[i] + z_off)

      if left and right:
        # Check for inversion when going over hills
        if not allow_invert and left_points and left[1] > left_points[-1][1]:
          continue

        left_points.append(left)
        right_points.append(right)

    if not left_points or not right_points:
      return np.empty((0, 2), dtype=np.float32)

    return np.array(left_points + right_points[::-1], dtype=np.float32)

  @staticmethod
  def _map_val(x, x0, x1, y0, y1):
    x = max(x0, min(x, x1))
    ra = x1 - x0
    rb = y1 - y0
    return (x - x0) * rb / ra + y0 if ra != 0 else y0

  @staticmethod
  def _hsla_to_color(h, s, l, a):
    """Convert HSLA color to Raylib Color using colorsys module"""
    # colorsys uses HLS format (Hue, Lightness, Saturation)
    r, g, b = colorsys.hls_to_rgb(h, l, s)

    # Ensure values are in valid range
    r_val = max(0, min(255, int(r * 255)))
    g_val = max(0, min(255, int(g * 255)))
    b_val = max(0, min(255, int(b * 255)))
    a_val = max(0, min(255, int(a * 255)))

    return rl.Color(r_val, g_val, b_val, a_val)

  @staticmethod
  def _blend_colors(begin_colors, end_colors, t):
    if t >= 1.0:
      return end_colors
    if t <= 0.0:
      return begin_colors

    inv_t = 1.0 - t
    return [rl.Color(
      int(inv_t * start.r + t * end.r),
      int(inv_t * start.g + t * end.g),
      int(inv_t * start.b + t * end.b),
      int(inv_t * start.a + t * end.a)
    ) for start, end in zip(begin_colors, end_colors, strict=True)]
