import colorsys
import math
import numpy as np
import pyray as rl
from cereal import messaging, car
from dataclasses import dataclass, field
from openpilot.common.params import Params
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.locationd.calibrationd import HEIGHT_INIT
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.selfdrive.ui.mici.onroad import blend_colors
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.shader_polygon import draw_polygon, Gradient
from openpilot.system.ui.widgets import Widget

CLIP_MARGIN = 500
MIN_DRAW_DISTANCE = 10.0
MAX_DRAW_DISTANCE = 100.0

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

LANE_LINE_COLORS = {
  UIStatus.DISENGAGED: rl.Color(200, 200, 200, 255),
  UIStatus.OVERRIDE: rl.Color(255, 255, 255, 255),
  UIStatus.ENGAGED: rl.Color(0, 255, 64, 255),
}


@dataclass
class ModelPoints:
  raw_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
  projected_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))


@dataclass
class LeadVehicle:
  glow: list[float] = field(default_factory=list)
  chevron: list[float] = field(default_factory=list)
  fill_alpha: int = 0
  # 3D projected diamond points for road-plane indicator
  diamond: list[tuple[float, float]] = field(default_factory=list)
  # Car-sized underglow rectangle projected on road surface
  underglow: list[tuple[float, float]] = field(default_factory=list)
  # NFS-style trail arrows along path to lead
  trail_arrows: list[list[tuple[float, float]]] = field(default_factory=list)
  # 3D-projected sky arrow (tip, left, right) floating above lead
  sky_arrow: list[tuple[float, float]] = field(default_factory=list)
  d_rel: float = 0.0
  v_rel: float = 0.0


class ModelRenderer(Widget):
  def __init__(self):
    super().__init__()
    self._longitudinal_control = False
    self._experimental_mode = False
    self._blend_filter = FirstOrderFilter(1.0, 0.25, 1 / gui_app.target_fps)
    self._prev_allow_throttle = True
    self._lane_line_probs = np.zeros(4, dtype=np.float32)
    self._road_edge_stds = np.zeros(2, dtype=np.float32)
    self._lead_vehicles = [LeadVehicle(), LeadVehicle()]
    self._path_offset_z = HEIGHT_INIT[0]
    self._lead_glow_filter = FirstOrderFilter(0.0, 0.15, 1 / gui_app.target_fps)
    self._lead_pulse_filter = FirstOrderFilter(0.0, 0.2, 1 / gui_app.target_fps)

    # Lead visualization styles: cycle through on click
    # 0=all, 1=diamond+underglow, 2=arrows+sky, 3=rings+brackets, 4=sky only, 5=diamond only
    self._lead_style = 4  # default to sky arrow only
    self.LEAD_STYLE_COUNT = 6

    # Initialize ModelPoints objects
    self._path = ModelPoints()
    self._lane_lines = [ModelPoints() for _ in range(4)]
    self._road_edges = [ModelPoints() for _ in range(2)]
    self._acceleration_x = np.empty((0,), dtype=np.float32)

    self._acceleration_x_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)
    self._acceleration_x_filter2 = FirstOrderFilter(0.0, 1, 1 / gui_app.target_fps)

    self._torque_filter = FirstOrderFilter(0, 0.1, 1 / gui_app.target_fps)
    self._ll_color_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)
    self._path_flow_distance = 0.0  # accumulated distance for arrow flow

    # Transform matrix (3x3 for car space to screen space)
    self._car_space_transform = np.zeros((3, 3), dtype=np.float32)
    self._transform_dirty = True
    self._clip_region = None

    self._exp_gradient = Gradient(
      start=(0.0, 1.0),  # Bottom of path
      end=(0.0, 0.0),  # Top of path
      colors=[],
      stops=[],
    )

    # Get longitudinal control setting from car parameters
    if car_params := Params().get("CarParams"):
      cp = messaging.log_from_bytes(car_params, car.CarParams)
      self._longitudinal_control = cp.openpilotLongitudinalControl

  def set_transform(self, transform: np.ndarray):
    self._car_space_transform = transform.astype(np.float32)
    self._transform_dirty = True

  def _render(self, rect: rl.Rectangle):
    sm = ui_state.sm

    self._torque_filter.update(-ui_state.sm['carOutput'].actuatorsOutput.torque)

    # Check if data is up-to-date
    if (sm.recv_frame["liveCalibration"] < ui_state.started_frame or
        sm.recv_frame["modelV2"] < ui_state.started_frame):
      return

    # Set up clipping region
    self._clip_region = rl.Rectangle(
      rect.x - CLIP_MARGIN, rect.y - CLIP_MARGIN, rect.width + 2 * CLIP_MARGIN, rect.height + 2 * CLIP_MARGIN
    )

    # Update state
    self._experimental_mode = sm['selfdriveState'].experimentalMode

    live_calib = sm['liveCalibration']
    self._path_offset_z = live_calib.height[0] if live_calib.height else HEIGHT_INIT[0]

    if sm.updated['carParams']:
      self._longitudinal_control = sm['carParams'].openpilotLongitudinalControl

    model = sm['modelV2']
    radar_state = sm['radarState'] if sm.valid['radarState'] else None
    lead_one = radar_state.leadOne if radar_state else None
    render_lead_indicator = self._longitudinal_control and radar_state is not None

    # Update model data when needed
    model_updated = sm.updated['modelV2']
    if model_updated or sm.updated['radarState'] or self._transform_dirty:
      if model_updated:
        self._update_raw_points(model)

      path_x_array = self._path.raw_points[:, 0]
      if path_x_array.size == 0:
        return

      self._update_model(lead_one, path_x_array)
      if render_lead_indicator:
        self._update_leads(radar_state, path_x_array)
      self._transform_dirty = False

    # Draw elements (hide when disengaged)
    if ui_state.status != UIStatus.DISENGAGED:
      self._draw_lane_lines()
      self._draw_path(sm)

    if render_lead_indicator and radar_state:
      self._draw_lead_indicator(radar_state)

  def _update_raw_points(self, model):
    """Update raw 3D points from model data"""
    self._path.raw_points = np.array([model.position.x, model.position.y, model.position.z], dtype=np.float32).T

    for i, lane_line in enumerate(model.laneLines):
      self._lane_lines[i].raw_points = np.array([lane_line.x, lane_line.y, lane_line.z], dtype=np.float32).T

    for i, road_edge in enumerate(model.roadEdges):
      self._road_edges[i].raw_points = np.array([road_edge.x, road_edge.y, road_edge.z], dtype=np.float32).T

    self._lane_line_probs = np.array(model.laneLineProbs, dtype=np.float32)
    self._road_edge_stds = np.array(model.roadEdgeStds, dtype=np.float32)
    self._acceleration_x = np.array(model.acceleration.x, dtype=np.float32)

  def _update_leads(self, radar_state, path_x_array):
    """Update positions of lead vehicles with 3D projected diamond on road surface"""
    self._lead_vehicles = [LeadVehicle(), LeadVehicle()]
    leads = [radar_state.leadOne, radar_state.leadTwo]

    for i, lead_data in enumerate(leads):
      if lead_data and lead_data.status:
        d_rel, y_rel, v_rel = lead_data.dRel, lead_data.yRel, lead_data.vRel
        idx = self._get_path_length_idx(path_x_array, d_rel)

        # Get z-coordinate from path at the lead vehicle position
        z = self._path.raw_points[idx, 2] if idx < len(self._path.raw_points) else 0.0
        road_z = z + self._path_offset_z

        # Project a diamond shape on the road plane in 3D car space
        # Diamond sits flat on the road at the lead's position
        half_w = 2.0  # half-width in meters
        half_l = float(np.clip(d_rel * 0.18, 3.5, 10.0))  # half-length, bigger for visibility at distance

        # 4 points of the diamond rotated 90°: left, front, right, back (in car space)
        diamond_3d = [
          (d_rel, -y_rel - half_l, road_z),          # left tip
          (d_rel + half_w, -y_rel, road_z),           # front
          (d_rel, -y_rel + half_l, road_z),           # right tip
          (d_rel - half_w, -y_rel, road_z),           # back
        ]

        diamond_2d = []
        for pt3 in diamond_3d:
          pt2 = self._map_to_screen(*pt3)
          if pt2:
            diamond_2d.append(pt2)

        # Project car-sized underglow rectangle on road surface
        car_len = 7.0   # generous car length for visual impact
        car_w = 2.5     # generous car width
        underglow_3d = [
          (d_rel + car_len * 0.5, -y_rel - car_w / 2, road_z),   # front-left
          (d_rel + car_len * 0.5, -y_rel + car_w / 2, road_z),   # front-right
          (d_rel - car_len * 0.5, -y_rel + car_w / 2, road_z),   # back-right
          (d_rel - car_len * 0.5, -y_rel - car_w / 2, road_z),   # back-left
        ]
        underglow_2d = []
        for pt3 in underglow_3d:
          pt2 = self._map_to_screen(*pt3)
          if pt2:
            underglow_2d.append(pt2)

        # Project NFS-style chevron arrows flat on road plane
        trail_arrows_2d = []
        if d_rel > 8:
          num_arrows = int(np.clip(d_rel / 12, 2, 5))
          arrow_len = 2.5   # forward length in meters
          arrow_w = 1.2     # half-width of wings
          for a_i in range(num_arrows):
            frac = 0.25 + 0.55 * a_i / max(num_arrows - 1, 1)
            arrow_d = d_rel * frac
            a_idx = self._get_path_length_idx(path_x_array, arrow_d)
            az = self._path.raw_points[a_idx, 2] + self._path_offset_z if a_idx < len(self._path.raw_points) else road_z
            arrow_y = -y_rel * frac
            # 4-point chevron on road plane: tip, right wing, notch (back indent), left wing
            tip = self._map_to_screen(arrow_d + arrow_len, arrow_y, az)
            rw = self._map_to_screen(arrow_d, arrow_y + arrow_w, az)
            notch = self._map_to_screen(arrow_d + arrow_len * 0.35, arrow_y, az)
            lw = self._map_to_screen(arrow_d, arrow_y - arrow_w, az)
            if tip and rw and notch and lw:
              trail_arrows_2d.append([tip, rw, notch, lw])

        # Project sky arrow in 3D: float above lead at avg car height (~1.5m) + 0.5m offset
        # Smaller Z = higher above road in this coordinate system
        # Push back ~2.3m (half avg car length) since radar reports rear bumper
        sky_d = d_rel + 2.3
        sky_z = road_z - 2.5  # 2.5m above road surface
        sky_arrow_2d = []
        sky_tip = self._map_to_screen(sky_d, -y_rel, sky_z + 0.975)  # tip points down toward car
        sky_left = self._map_to_screen(sky_d, -y_rel - 0.9, sky_z)
        sky_right = self._map_to_screen(sky_d, -y_rel + 0.9, sky_z)
        if sky_tip and sky_left and sky_right:
          sky_arrow_2d = [sky_tip, sky_left, sky_right]

        # Also get the center point for glow/fallback
        center = self._map_to_screen(d_rel, -y_rel, road_z)

        if len(diamond_2d) >= 3:
          lv = LeadVehicle(diamond=diamond_2d, underglow=underglow_2d, trail_arrows=trail_arrows_2d,
                           sky_arrow=sky_arrow_2d, d_rel=d_rel, v_rel=v_rel)
          # Keep legacy chevron for fallback
          if center:
            lv_legacy = self._update_lead_vehicle(d_rel, v_rel, center, self._rect)
            lv.chevron = lv_legacy.chevron
            lv.glow = lv_legacy.glow
            lv.fill_alpha = lv_legacy.fill_alpha
          self._lead_vehicles[i] = lv
        elif center:
          lv = self._update_lead_vehicle(d_rel, v_rel, center, self._rect)
          lv.d_rel = d_rel
          lv.v_rel = v_rel
          self._lead_vehicles[i] = lv

  def _update_model(self, lead, path_x_array):
    """Update model visualization data based on model message"""
    max_distance = np.clip(path_x_array[-1], MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE)
    max_idx = self._get_path_length_idx(self._lane_lines[0].raw_points[:, 0], max_distance)

    # Update lane lines using raw points
    line_width_factor = 0.12
    for i, lane_line in enumerate(self._lane_lines):
      if i in (1, 2):
        line_width_factor = 0.16
      lane_line.projected_points = self._map_line_to_polygon(
        lane_line.raw_points, line_width_factor * self._lane_line_probs[i], 0.0, max_idx
      )

    # Update road edges using raw points
    for road_edge in self._road_edges:
      road_edge.projected_points = self._map_line_to_polygon(road_edge.raw_points, line_width_factor, 0.0, max_idx)

    # Update path using raw points
    if lead and lead.status:
      lead_d = lead.dRel * 2.0
      max_distance = np.clip(lead_d - min(lead_d * 0.35, 10.0), 0.0, max_distance)

    soon_acceleration = self._acceleration_x[len(self._acceleration_x) // 4] if len(self._acceleration_x) > 0 else 0
    self._acceleration_x_filter.update(soon_acceleration)
    self._acceleration_x_filter2.update(soon_acceleration)

    # make path width wider/thinner when initially braking/accelerating
    if self._experimental_mode and False:
      high_pass_acceleration = self._acceleration_x_filter.x - self._acceleration_x_filter2.x
      y_off = np.interp(high_pass_acceleration, [-1, 0, 1], [0.9 * 2, 0.9, 0.9 / 2])
    else:
      y_off = 0.9

    max_idx = self._get_path_length_idx(path_x_array, max_distance)
    self._path.projected_points = self._map_line_to_polygon(
      self._path.raw_points, y_off, self._path_offset_z, max_idx, allow_invert=False
    )

    self._update_experimental_gradient()

  def _update_experimental_gradient(self):
    """Pre-calculate experimental mode gradient colors"""
    if not self._experimental_mode:
      return

    max_len = min(len(self._path.projected_points) // 2, len(self._acceleration_x))

    segment_colors = []
    gradient_stops = []

    i = 0
    while i < max_len:
      # Some points (screen space) are out of frame (rect space)
      track_y = self._path.projected_points[i][1]
      if track_y < self._rect.y or track_y > (self._rect.y + self._rect.height):
        i += 1
        continue

      # Calculate color based on acceleration (0 is bottom, 1 is top)
      lin_grad_point = 1 - (track_y - self._rect.y) / self._rect.height

      # speed up: 120, slow down: 0
      path_hue = np.clip(60 + self._acceleration_x[i] * 35, 0, 120)

      saturation = min(abs(self._acceleration_x[i] * 1.5), 1)
      lightness = np.interp(saturation, [0.0, 1.0], [0.95, 0.62])
      alpha = np.interp(lin_grad_point, [0.75 / 2.0, 0.75], [0.4, 0.0])

      # Use HSL to RGB conversion
      color = self._hsla_to_color(path_hue / 360.0, saturation, lightness, alpha)

      gradient_stops.append(lin_grad_point)
      segment_colors.append(color)

      # Skip a point, unless next is last
      i += 1 + (1 if (i + 2) < max_len else 0)

    # Store the gradient in the path object
    self._exp_gradient.colors = segment_colors
    self._exp_gradient.stops = gradient_stops

  def _update_lead_vehicle(self, d_rel, v_rel, point, rect):
    speed_buff, lead_buff = 10.0, 40.0

    # Calculate fill alpha
    fill_alpha = 0
    if d_rel < lead_buff:
      fill_alpha = 255 * (1.0 - (d_rel / lead_buff))
      if v_rel < 0:
        fill_alpha += 255 * (-1 * (v_rel / speed_buff))
      fill_alpha = min(fill_alpha, 255)

    # Calculate size and position
    sz = np.clip((25 * 30) / (d_rel / 3 + 30), 15.0, 30.0) * 1
    x = np.clip(point[0], 0.0, rect.width - sz / 2)
    y = min(point[1], rect.height - sz * 0.6)

    g_xo = sz / 5
    g_yo = sz / 10

    glow = [(x + (sz * 1.35) + g_xo, y + sz + g_yo), (x, y - g_yo), (x - (sz * 1.35) - g_xo, y + sz + g_yo)]
    chevron = [(x + (sz * 1.25), y + sz), (x, y), (x - (sz * 1.25), y + sz)]

    return LeadVehicle(glow=glow, chevron=chevron, fill_alpha=int(fill_alpha))

  def _get_ll_color(self, prob: float, adjacent: bool, left: bool):
    alpha = np.clip(prob, 0.0, 0.7)
    if adjacent:
      _base_color = LANE_LINE_COLORS.get(ui_state.status, LANE_LINE_COLORS[UIStatus.DISENGAGED])
      color = rl.Color(_base_color.r, _base_color.g, _base_color.b, int(alpha * 255))

      # turn adjacent lls orange if torque is high
      torque = self._torque_filter.x
      high_torque = abs(torque) > 0.6
      if high_torque and (left == (torque > 0)):
        color = blend_colors(
          color,
          rl.Color(255, 115, 0, int(alpha * 255)),  # orange
          np.interp(abs(torque), [0.6, 0.8], [0.0, 1.0])
        )
    else:
      color = rl.Color(255, 255, 255, int(alpha * 255))

    if ui_state.status == UIStatus.DISENGAGED:
      color = rl.Color(0, 0, 0, int(alpha * 255))

    return color

  def _draw_lane_lines(self):
    """Draw lane lines and road edges"""
    """Two closest lines should be green (lane line or road edges)"""
    for i, lane_line in enumerate(self._lane_lines):
      if lane_line.projected_points.size == 0:
        continue

      color = self._get_ll_color(float(self._lane_line_probs[i]), i in (1, 2), i in (0, 1))
      # Subtle pulse on adjacent lane lines when engaged
      if ui_state.status != UIStatus.DISENGAGED and i in (1, 2):
        pulse = 1.0 + 0.1 * math.sin(rl.get_time() * 3.0 + i * 0.5)
        color = rl.Color(color.r, color.g, color.b, min(255, int(color.a * pulse)))
      draw_polygon(self._rect, lane_line.projected_points, color)

    for i, road_edge in enumerate(self._road_edges):
      if road_edge.projected_points.size == 0:
        continue

      # if closest lane lines are not confident, make road edges green
      color = self._get_ll_color(float(1.0 - self._road_edge_stds[i]), float(self._lane_line_probs[i + 1]) < 0.25, i == 0)
      draw_polygon(self._rect, road_edge.projected_points, color)

  def _draw_path(self, sm):
    """Draw path as NFS-style chevron arrows on the road plane."""
    raw = self._path.raw_points
    if len(raw) < 4:
      return

    allow_throttle = sm['longitudinalPlan'].allowThrottle or not self._longitudinal_control
    self._blend_filter.update(int(allow_throttle))

    if ui_state.status == UIStatus.DISENGAGED:
      if self._path.projected_points.size:
        draw_polygon(self._rect, self._path.projected_points, rl.Color(0, 0, 0, 90))
      return

    # HACK: force chill mode for tuning
    exp_mode = False  # self._experimental_mode

    z_off = self._path_offset_z
    accel_data = self._acceleration_x
    v_ego = float(sm['carState'].vEgo)

    # Path x-coordinates for interpolation
    path_x = raw[:, 0]
    path_y = raw[:, 1]
    path_z = raw[:, 2]
    max_x = float(path_x[-1]) if len(path_x) > 0 else 0

    # Arrow parameters
    arrow_len = 3.375  # 35% longer in forward direction on road plane
    arrow_w = 1.375  # halfway between 0.9 and 1.85

    # Accumulate actual distance traveled for smooth flow
    dt = rl.get_frame_time()
    self._path_flow_distance += v_ego * dt

    # Uniform spacing for smooth flow, perspective naturally thins at distance
    spacing = 6.0
    phase = self._path_flow_distance % spacing
    max_arrow_dist = min(max_x - arrow_len, 110.0)
    arrow_x = np.arange(3.0 - phase, max_arrow_dist, spacing)
    arrow_x = arrow_x[arrow_x >= 2.0]

    for ax in arrow_x:
      if ax < 2.0 or ax > max_x - arrow_len:
        continue

      # Interpolate path y and z at this arrow position
      y = float(np.interp(ax, path_x, path_y))
      z = float(np.interp(ax, path_x, path_z))
      az = z + z_off

      # Also interpolate acceleration for experimental color
      accel_val = 0.0
      if exp_mode and len(accel_data) > 0:
        accel_val = float(np.interp(ax, path_x[:len(accel_data)], accel_data))

      # Project 4-point chevron on road plane
      tip = self._map_to_screen(ax + arrow_len, y, az)
      rw = self._map_to_screen(ax, y + arrow_w, az)
      notch = self._map_to_screen(ax + arrow_len * 0.35, y, az)
      lw = self._map_to_screen(ax, y - arrow_w, az)

      if not (tip and rw and notch and lw):
        continue

      # Distance fade — mostly opaque, gentle fade only at far end
      dist_fade = float(np.clip(1.0 - ax / 150, 0.6, 1.0))
      # Fade in near car (arrows appearing from under car)
      near_fade = float(np.clip((ax - 2.0) / 3.0, 0.0, 1.0))
      base_alpha = dist_fade * near_fade

      # Color based on mode
      if exp_mode:
        path_hue = float(np.clip(60 + accel_val * 35, 0, 120))
        sat = min(abs(accel_val * 1.5), 1.0)
        rgb = colorsys.hls_to_rgb(path_hue / 360.0, 0.55, sat)
        alpha = int(255 * base_alpha)
        ac = rl.Color(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), alpha)
      else:
        blend = self._blend_filter.x
        r = int(np.interp(blend, [0, 1], [242, 13]))
        g = int(np.interp(blend, [0, 1], [242, 248]))
        b = int(np.interp(blend, [0, 1], [242, 122]))
        alpha = int(255 * base_alpha)
        ac = rl.Color(r, g, b, alpha)

      tv = rl.Vector2(tip[0], tip[1])
      rv = rl.Vector2(rw[0], rw[1])
      nv = rl.Vector2(notch[0], notch[1])
      lv = rl.Vector2(lw[0], lw[1])

      rl.draw_triangle(tv, rv, nv, ac)
      rl.draw_triangle(tv, nv, rv, ac)
      rl.draw_triangle(tv, nv, lv, ac)
      rl.draw_triangle(tv, lv, nv, ac)

      ec = rl.Color(min(255, ac.r + 60), min(255, ac.g + 60), min(255, ac.b + 60), int(alpha * 0.4))
      rl.draw_line_ex(tv, rv, 1.5, ec)
      rl.draw_line_ex(tv, lv, 1.5, ec)
      rl.draw_line_ex(rv, nv, 1.0, ec)
      rl.draw_line_ex(lv, nv, 1.0, ec)

  def _draw_lead_indicator(self, radar_state):
    for lead_idx, lead in enumerate(self._lead_vehicles):
      if not lead.diamond and not lead.chevron:
        if lead_idx == 0:
          self._lead_glow_filter.update(0.0)
        continue

      d_rel = lead.d_rel
      v_rel = lead.v_rel
      is_primary = lead_idx == 0

      # Color based on distance: far=green, medium=yellow, close=red
      if d_rel > 25:
        color = rl.Color(0, 200, 120, 255)
      elif d_rel > 12:
        f = (d_rel - 12) / 13
        color = blend_colors(rl.Color(255, 180, 0, 255), rl.Color(0, 200, 120, 255), f)
      else:
        f = d_rel / 12
        color = blend_colors(rl.Color(255, 40, 30, 255), rl.Color(255, 180, 0, 255), f)

      if is_primary:
        proximity = float(np.clip(1.0 - d_rel / 40.0, 0.0, 1.0))
        closing = float(np.clip(-v_rel / 10.0, 0.0, 1.0))
        intensity = max(proximity, closing)
        self._lead_glow_filter.update(intensity)

      alpha_mult = 1.0 if is_primary else 0.4
      base_alpha = float(np.clip(0.5 + self._lead_glow_filter.x * 0.5, 0.3, 1.0)) * alpha_mult

      # style: 0=all, 1=diamond+underglow, 2=arrows+sky, 3=rings+brackets, 4=sky only, 5=diamond only
      s = self._lead_style

      # Underglow (styles 0, 1)
      if s in (0, 1) and lead.underglow and len(lead.underglow) >= 3:
        ug_pts = lead.underglow
        ug_pulse = 0.85 + 0.15 * math.sin(rl.get_time() * 3.0)
        ug_alpha = int(base_alpha * 60 * ug_pulse)
        ug_color = rl.Color(color.r, color.g, color.b, ug_alpha)
        for j in range(1, len(ug_pts) - 1):
          rl.draw_triangle(
            rl.Vector2(ug_pts[0][0], ug_pts[0][1]),
            rl.Vector2(ug_pts[j][0], ug_pts[j][1]),
            rl.Vector2(ug_pts[j + 1][0], ug_pts[j + 1][1]),
            ug_color,
          )
        ug_edge_alpha = int(base_alpha * 100 * ug_pulse)
        ug_edge_color = rl.Color(color.r, color.g, color.b, ug_edge_alpha)
        for j in range(len(ug_pts)):
          p0 = ug_pts[j]
          p1 = ug_pts[(j + 1) % len(ug_pts)]
          rl.draw_line_ex(rl.Vector2(p0[0], p0[1]), rl.Vector2(p1[0], p1[1]), 1.5, ug_edge_color)

      # Trail arrows — NFS chevrons flat on road plane (styles 0, 2)
      if s in (0, 2) and is_primary and lead.trail_arrows:
        t = rl.get_time()
        for a_i, arrow in enumerate(lead.trail_arrows):
          tip, rw, notch, lw = arrow
          frac = (a_i + 1) / (len(lead.trail_arrows) + 1)
          arrow_pulse = 0.6 + 0.4 * math.sin(t * 4.0 + a_i * 1.5)
          arrow_alpha = int(base_alpha * 160 * frac * arrow_pulse)
          ac = rl.Color(0, 180, 255, arrow_alpha)
          tv = rl.Vector2(tip[0], tip[1])
          rv = rl.Vector2(rw[0], rw[1])
          nv = rl.Vector2(notch[0], notch[1])
          lv = rl.Vector2(lw[0], lw[1])
          # Right half of chevron: tip → right wing → notch
          rl.draw_triangle(tv, rv, nv, ac)
          rl.draw_triangle(tv, nv, rv, ac)
          # Left half of chevron: tip → notch → left wing
          rl.draw_triangle(tv, nv, lv, ac)
          rl.draw_triangle(tv, lv, nv, ac)
          # Edge outlines
          ec = rl.Color(100, 220, 255, int(arrow_alpha * 0.7))
          rl.draw_line_ex(tv, rv, 2.0, ec)
          rl.draw_line_ex(tv, lv, 2.0, ec)
          rl.draw_line_ex(rv, nv, 1.5, ec)
          rl.draw_line_ex(lv, nv, 1.5, ec)

      if lead.diamond and len(lead.diamond) >= 3:
        pts = lead.diamond
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)

        # Diamond fill + edges (styles 0, 1, 5)
        if s in (0, 1, 5):
          if is_primary:
            closing_val = float(np.clip(-v_rel / 10.0, 0.0, 1.0))
            pulse = 1.0
            if closing_val > 0.3:
              pulse = 0.8 + 0.2 * (0.5 + 0.5 * math.sin(rl.get_time() * 6))
            glow_r = max(abs(pts[0][0] - pts[2][0]), abs(pts[1][1] - pts[3][1])) * 0.6 * pulse if len(pts) >= 4 else 20
            glow_alpha = int(base_alpha * 50 * self._lead_glow_filter.x)
            rl.draw_circle(int(cx), int(cy), glow_r, rl.Color(color.r, color.g, color.b, glow_alpha))

          fill_alpha = int(base_alpha * 200)
          fill_color = rl.Color(color.r, color.g, color.b, fill_alpha)
          for j in range(1, len(pts) - 1):
            rl.draw_triangle(
              rl.Vector2(pts[0][0], pts[0][1]),
              rl.Vector2(pts[j][0], pts[j][1]),
              rl.Vector2(pts[j + 1][0], pts[j + 1][1]),
              fill_color,
            )

          edge_alpha = int(base_alpha * 255)
          edge_color = rl.Color(min(255, color.r + 80), min(255, color.g + 80), min(255, color.b + 80), edge_alpha)
          for j in range(len(pts)):
            p0 = pts[j]
            p1 = pts[(j + 1) % len(pts)]
            rl.draw_line_ex(rl.Vector2(p0[0], p0[1]), rl.Vector2(p1[0], p1[1]), 4.0, edge_color)

        # Pulsing rings (styles 0, 3)
        if s in (0, 3) and is_primary:
          diamond_w = abs(pts[1][0] - pts[3][0]) if len(pts) >= 4 else 40
          diamond_h = abs(pts[0][1] - pts[2][1]) if len(pts) >= 4 else 40
          max_ring_r = max(diamond_w, diamond_h) * 0.7
          t = rl.get_time()
          for ring_i in range(3):
            phase = (t * 0.7 + ring_i * 0.5) % 1.5
            ring_r = max(5, max_ring_r * (0.3 + 0.7 * phase / 1.5))
            ring_alpha = int(base_alpha * 100 * (1.0 - phase / 1.5))
            if ring_alpha > 2:
              ring_color = rl.Color(color.r, color.g, color.b, ring_alpha)
              rl.draw_ring(rl.Vector2(cx, cy), max(0, ring_r - 2), ring_r + 2, 0, 360, 24, ring_color)

        # Target brackets (styles 0, 3)
        if s in (0, 3) and is_primary:
          bracket_size = float(np.clip(30 * 30 / (d_rel + 10), 15, 50))
          bracket_thick = 3.0
          bracket_gap = bracket_size * 0.8
          bracket_alpha = int(base_alpha * 220)
          bc = rl.Color(min(255, color.r + 60), min(255, color.g + 60), min(255, color.b + 60), bracket_alpha)
          for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            bx = cx + dx * bracket_gap
            by = cy + dy * bracket_gap
            rl.draw_line_ex(rl.Vector2(bx, by), rl.Vector2(bx - dx * bracket_size * 0.4, by), bracket_thick, bc)
            rl.draw_line_ex(rl.Vector2(bx, by), rl.Vector2(bx, by - dy * bracket_size * 0.4), bracket_thick, bc)

        # NFS floating sky arrow — 3D projected above lead (styles 0, 2, 4)
        if s in (0, 2, 4) and is_primary and len(lead.sky_arrow) == 3:
          sky_tip, sky_lw, sky_rw = lead.sky_arrow
          sky_pulse = 0.8 + 0.2 * math.sin(rl.get_time() * 3.0)
          sky_alpha = int(base_alpha * 255 * sky_pulse)
          sky_fill = rl.Color(0, 220, 255, sky_alpha)
          tv = rl.Vector2(sky_tip[0], sky_tip[1])
          lv = rl.Vector2(sky_lw[0], sky_lw[1])
          rv = rl.Vector2(sky_rw[0], sky_rw[1])
          rl.draw_triangle(tv, lv, rv, sky_fill)
          rl.draw_triangle(tv, rv, lv, sky_fill)
          sky_edge = rl.Color(180, 255, 255, int(max(0, sky_alpha * 0.8)))
          rl.draw_line_ex(tv, lv, 3.0, sky_edge)
          rl.draw_line_ex(tv, rv, 3.0, sky_edge)
          rl.draw_line_ex(lv, rv, 2.0, sky_edge)

      elif lead.chevron:
        # Fallback: 2D chevron
        pts = lead.chevron
        chevron_alpha = int(base_alpha * 255)
        chevron_color = rl.Color(color.r, color.g, color.b, chevron_alpha)
        rl.draw_triangle(
          rl.Vector2(pts[0][0], pts[0][1]),
          rl.Vector2(pts[1][0], pts[1][1]),
          rl.Vector2(pts[2][0], pts[2][1]),
          chevron_color,
        )

  @staticmethod
  def _get_path_length_idx(pos_x_array: np.ndarray, path_height: float) -> int:
    """Get the index corresponding to the given path height"""
    if len(pos_x_array) == 0:
      return 0
    indices = np.where(pos_x_array <= path_height)[0]
    return indices[-1] if indices.size > 0 else 0

  def _map_to_screen(self, in_x, in_y, in_z):
    """Project a point in car space to screen space"""
    input_pt = np.array([in_x, in_y, in_z])
    pt = self._car_space_transform @ input_pt

    if abs(pt[2]) < 1e-6:
      return None

    x, y = pt[0] / pt[2], pt[1] / pt[2]

    clip = self._clip_region
    if not (clip.x <= x <= clip.x + clip.width and clip.y <= y <= clip.y + clip.height):
      return None

    return (x, y)

  def _map_line_to_polygon(self, line: np.ndarray, y_off: float, z_off: float, max_idx: int, allow_invert: bool = True) -> np.ndarray:
    """Convert 3D line to 2D polygon for rendering."""
    if line.shape[0] == 0:
      return np.empty((0, 2), dtype=np.float32)

    # Slice points and filter non-negative x-coordinates
    points = line[:max_idx + 1]
    points = points[points[:, 0] >= 0]
    if points.shape[0] == 0:
      return np.empty((0, 2), dtype=np.float32)

    N = points.shape[0]
    # Generate left and right 3D points in one array using broadcasting
    offsets = np.array([[0, -y_off, z_off], [0, y_off, z_off]], dtype=np.float32)
    points_3d = points[None, :, :] + offsets[:, None, :]  # Shape: 2xNx3
    points_3d = points_3d.reshape(2 * N, 3)  # Shape: (2*N)x3

    # Transform all points to projected space in one operation
    proj = self._car_space_transform @ points_3d.T  # Shape: 3x(2*N)
    proj = proj.reshape(3, 2, N)
    left_proj = proj[:, 0, :]
    right_proj = proj[:, 1, :]

    # Filter points where z is sufficiently large
    valid_proj = (np.abs(left_proj[2]) >= 1e-6) & (np.abs(right_proj[2]) >= 1e-6)
    if not np.any(valid_proj):
      return np.empty((0, 2), dtype=np.float32)

    # Compute screen coordinates
    left_screen = left_proj[:2, valid_proj] / left_proj[2, valid_proj][None, :]
    right_screen = right_proj[:2, valid_proj] / right_proj[2, valid_proj][None, :]

    # Define clip region bounds
    clip = self._clip_region
    x_min, x_max = clip.x, clip.x + clip.width
    y_min, y_max = clip.y, clip.y + clip.height

    # Filter points within clip region
    left_in_clip = (
      (left_screen[0] >= x_min) & (left_screen[0] <= x_max) &
      (left_screen[1] >= y_min) & (left_screen[1] <= y_max)
    )
    right_in_clip = (
      (right_screen[0] >= x_min) & (right_screen[0] <= x_max) &
      (right_screen[1] >= y_min) & (right_screen[1] <= y_max)
    )
    both_in_clip = left_in_clip & right_in_clip

    if not np.any(both_in_clip):
      return np.empty((0, 2), dtype=np.float32)

    # Select valid and clipped points
    left_screen = left_screen[:, both_in_clip]
    right_screen = right_screen[:, both_in_clip]

    # Handle Y-coordinate inversion on hills
    if not allow_invert and left_screen.shape[1] > 1:
      y = left_screen[1, :]  # y-coordinates
      keep = y == np.minimum.accumulate(y)
      if not np.any(keep):
        return np.empty((0, 2), dtype=np.float32)
      left_screen = left_screen[:, keep]
      right_screen = right_screen[:, keep]

    return np.vstack((left_screen.T, right_screen[:, ::-1].T)).astype(np.float32)

  @staticmethod
  def _hsla_to_color(h, s, l, a):
    rgb = colorsys.hls_to_rgb(h, l, s)
    return rl.Color(
      int(rgb[0] * 255),
      int(rgb[1] * 255),
      int(rgb[2] * 255),
      int(a * 255)
    )

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
