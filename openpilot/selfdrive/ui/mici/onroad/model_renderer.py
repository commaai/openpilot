import colorsys
import numpy as np
import pyray as rl
from openpilot.cereal import messaging, log
from opendbc.car.structs import car
from dataclasses import dataclass, field
from openpilot.common.params import Params
from openpilot.selfdrive.controls.radard import RADAR_TO_CAMERA
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.locationd.calibrationd import HEIGHT_INIT
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.selfdrive.ui.mici.onroad import blend_colors
from openpilot.selfdrive.ui.mici.onroad.stop_bar import HeldStopBar
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.shader_polygon import draw_polygon, Gradient
from openpilot.system.ui.widgets import Widget

CLIP_MARGIN = 500
MIN_DRAW_DISTANCE = 10.0
MAX_DRAW_DISTANCE = 100.0

# Road-plane footprint in meters; shared by both lead markers.
LEAD_BAR_WIDTH = 1.8
# Rounded mean overall length of 2025 Corolla, RAV4, CR-V, Civic, and Camry.
LEAD_BAR_DEPTH = 4.7
LEAD_BAR_MAX_DEPTH = 6.0
LEAD_BAR_MIN_HEIGHT = 8.0

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
  points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))
  distance: float = 0.0
  opacity: float = 0.8
  visibility: FirstOrderFilter = field(default_factory=lambda: FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps))
  opacity_filter: FirstOrderFilter = field(default_factory=lambda: FirstOrderFilter(0.5, 0.1, 1 / gui_app.target_fps, initialized=False))


@dataclass
class VisionLeadPosition:
  present: bool = False
  dRel: float = 0.0
  yRel: float = 0.0
  radar: bool = False
  radarTrackId: int = -1


@dataclass
class LeadBarSmoothing:
  identity: tuple
  distance: float
  lateral: float
  lateral_filter: FirstOrderFilter = field(default_factory=lambda: FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps, initialized=False))
  heading_filter: FirstOrderFilter = field(default_factory=lambda: FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps, initialized=False))


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
    self._lead_bar_smoothing = [None, None]
    self._stop_bar = HeldStopBar()
    self._path_offset_z = HEIGHT_INIT[0]

    # Initialize ModelPoints objects
    self._path = ModelPoints()
    self._lane_lines = [ModelPoints() for _ in range(4)]
    self._road_edges = [ModelPoints() for _ in range(2)]
    self._acceleration_x = np.empty((0,), dtype=np.float32)

    self._acceleration_x_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)
    self._acceleration_x_filter2 = FirstOrderFilter(0.0, 1, 1 / gui_app.target_fps)

    self._torque_filter = FirstOrderFilter(0, 0.1, 1 / gui_app.target_fps)
    self._ll_color_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)

    # 3x3 car space -> rect-origin space (draw methods add rect.x/y)
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
    if (sm.recv_frame["extrinsicsCalibration"] < ui_state.started_frame or
        sm.recv_frame["modelV2"] < ui_state.started_frame):
      self._lead_bar_smoothing = [None, None]
      self._lead_vehicles = [LeadVehicle(), LeadVehicle()]
      self._stop_bar.reset()
      return

    # Set up clipping region
    self._clip_region = rl.Rectangle(
      rect.x - CLIP_MARGIN, rect.y - CLIP_MARGIN, rect.width + 2 * CLIP_MARGIN, rect.height + 2 * CLIP_MARGIN
    )

    # Update state
    self._experimental_mode = sm['selfdriveState'].experimentalMode

    extrinsics_calibration = sm['extrinsicsCalibration']
    self._path_offset_z = extrinsics_calibration.height[0] if extrinsics_calibration.height else HEIGHT_INIT[0]

    if sm.updated['carParams']:
      self._longitudinal_control = sm['carParams'].openpilotLongitudinalControl

    model = sm['modelV2']
    radar_fresh = sm.valid['radarState'] and sm.alive['radarState'] and sm.recv_frame['radarState'] >= ui_state.started_frame
    radar_state = sm['radarState'] if radar_fresh else None
    lead_one = radar_state.leadOne if radar_state else None
    use_vision = self._use_vision_leads(sm)
    render_lead_indicator = (self._longitudinal_control and ui_state.status != UIStatus.DISENGAGED and
                             sm.valid['modelV2'] and sm.alive['modelV2'])
    if not render_lead_indicator:
      self._lead_vehicles = [LeadVehicle(), LeadVehicle()]
      self._lead_bar_smoothing = [None, None]

    # Update model data when needed
    model_updated = sm.updated['modelV2']
    if model_updated or sm.updated['radarState'] or self._transform_dirty:
      if model_updated:
        self._update_raw_points(model)

      path_x_array = self._path.raw_points[:, 0]
      if path_x_array.size == 0:
        self._lead_vehicles = [LeadVehicle(), LeadVehicle()]
        self._lead_bar_smoothing = [None, None]
        self._stop_bar.reset()
        return

      self._update_model(lead_one, path_x_array)
      self._transform_dirty = False

    # Advance visual filters at the UI frame rate, including between model updates.
    if render_lead_indicator:
      self._update_leads(radar_state, self._path.raw_points[:, 0], model.leadsV3 if use_vision else None)

    stop_valid = (render_lead_indicator and self._experimental_mode and
                  all(sm.valid[s] and sm.alive[s] and sm.recv_frame[s] >= ui_state.started_frame
                      for s in ('carState', 'selfdriveState')))
    self._stop_bar.update(model, sm['carState'].vEgo, sm['carState'].yawRate,
                          sm.logMonoTime['carState'] * 1e-9, sm.logMonoTime['modelV2'] * 1e-9, enabled=stop_valid)
    self._stop_bar.project(self._car_space_transform, self._path_offset_z,
                           [line.raw_points for line in self._lane_lines], self._lane_line_probs,
                           [edge.raw_points for edge in self._road_edges], self._road_edge_stds)

    # Draw elements (hide when disengaged)
    if ui_state.status != UIStatus.DISENGAGED:
      self._draw_lane_lines()
      self._draw_path(sm)

      if render_lead_indicator:
        self._draw_lead_indicator()
      if self._stop_bar.points.size:
        offset = np.array([rect.x, rect.y], dtype=np.float32)
        draw_polygon(rect, self._stop_bar.points + offset, rl.Color(255, 255, 255, round(255 * 0.9)))

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

  @staticmethod
  def _use_vision_leads(sm):
    # Match the car icon's green target, before its visual color crossfade.
    plan = sm['longitudinalPlan']
    has_lead = (sm.valid['longitudinalPlan'] and sm.alive['longitudinalPlan'] and
                sm.recv_frame['longitudinalPlan'] >= ui_state.started_frame and plan.hasLead)
    fcw = (sm.valid['selfdriveState'] and sm.alive['selfdriveState'] and
           sm.recv_frame['selfdriveState'] >= ui_state.started_frame and
           sm['selfdriveState'].alertHudVisual == car.CarControl.HUDControl.VisualAlert.fcw)
    return not fcw and has_lead and plan.longitudinalPlanSource == log.LongitudinalPlan.LongitudinalPlanSource.e2e

  def _update_leads(self, radar_state, path_x_array, vision_leads=None):
    """Place road-plane footprints beneath the detected vehicles."""
    previous = self._lead_vehicles
    self._lead_vehicles = [LeadVehicle(), LeadVehicle()]
    if vision_leads is None:
      leads = (radar_state.leadOne, radar_state.leadTwo) if radar_state is not None else (VisionLeadPosition(), VisionLeadPosition())
    else:
      leads = [VisionLeadPosition(), VisionLeadPosition()]
      for i, vision in enumerate(vision_leads):
        if i >= 2:
          break
        if vision.prob > 0.5 and len(vision.x) and len(vision.y):
          # Normalize to radar distance for shared tracking; projection below
          # restores camera-relative distance for both sources.
          leads[i] = VisionLeadPosition(True, vision.x[0] - RADAR_TO_CAMERA, -vision.y[0])
    first = leads[0]
    for i, lead in enumerate(leads):
      if not lead.present:
        self._lead_bar_smoothing[i] = None
        continue
      # Radar's two solutions can describe the same vehicle.
      if i == 1 and first.present and abs(lead.dRel - first.dRel) < 3.0 and abs(lead.yRel - first.yRel) < 1.0:
        self._lead_bar_smoothing[i] = None
        continue
      identity = (vision_leads is not None, lead.radar, lead.radarTrackId)
      smoothing = self._lead_bar_smoothing[i]
      # Vision has no persistent track ID. Large position discontinuities also
      # reset the display rather than sweeping between unrelated vehicles.
      if (smoothing is None or smoothing.identity != identity or
          abs(lead.yRel - smoothing.lateral) > 3.0 or abs(lead.dRel - smoothing.distance) > 10.0):
        smoothing = LeadBarSmoothing(identity, lead.dRel, lead.yRel)
      smoothing.distance, smoothing.lateral = lead.dRel, lead.yRel
      # Radar distances are measured ahead of the camera. Restore that origin
      # before projecting; vision-only leads recover their original x here too.
      camera_distance = lead.dRel + RADAR_TO_CAMERA
      points = self._project_lead_bar(camera_distance, lead.yRel, path_x_array, smoothing)
      self._lead_bar_smoothing[i] = smoothing if points.size else None
      self._lead_vehicles[i] = LeadVehicle(points, lead.dRel, 0.8 if vision_leads is not None and i == 0 else 0.5)

    # Run once per UI frame, matching the HUD color crossfade's 0.1 s filter.
    # Retain the last visible polygon briefly when detection disappears.
    for i, current in enumerate(self._lead_vehicles):
      visible = bool(current.points.size)
      current.visibility = previous[i].visibility
      current.opacity_filter = previous[i].opacity_filter
      if visible:
        current.opacity_filter.update(current.opacity)
      alpha = current.visibility.update(float(visible))
      if not visible and round(255 * current.opacity_filter.x * alpha) > 0:
        current.points = previous[i].points
        current.distance = previous[i].distance
        current.opacity = previous[i].opacity
      elif not visible:
        current.visibility.x = 0.0
        current.opacity_filter.initialized = False

  def _project_lead_bar(self, distance, lateral, path_x_array, smoothing=None):
    """Project a camera-relative lead footprint with its rear edge at the lead."""
    empty = np.empty((0, 2), dtype=np.float32)
    if (len(path_x_array) == 0 or not np.isfinite(self._path.raw_points).all() or
        not np.isfinite([distance, lateral]).all() or
        distance <= 0.1 or distance > MAX_DRAW_DISTANCE):
      return empty

    # Near standstill, predictions can repeat or retreat slightly in x. Keep
    # forward-progressing samples in trajectory order for road interpolation.
    # Beyond that usable path, retain the last road height as before.
    forward_samples = np.r_[True, path_x_array[1:] > np.maximum.accumulate(path_x_array)[:-1]]
    path_x_array = path_x_array[forward_samples]
    path = self._path.raw_points[forward_samples]

    # Follow the local road direction, but center on the radar lead rather than the path.
    sample_x = np.clip([distance - 1.0, distance + 1.0], path_x_array[0], path_x_array[-1])
    sample_y = np.interp(sample_x, path_x_array, path[:, 1])
    heading = np.arctan2(sample_y[1] - sample_y[0], sample_x[1] - sample_x[0])
    if smoothing is not None:
      lateral = smoothing.lateral_filter.update(lateral)
      heading = smoothing.heading_filter.update(heading)
    forward = np.array([np.cos(heading), np.sin(heading)])
    sideways = np.array([-forward[1], forward[0]])
    center = np.array([distance, -lateral])  # Radar lateral is left-positive; model lateral is right-positive.
    def project_depth(depth):
      corners = np.array([
        center + forward * along + sideways * side
        for along, side in ((depth, -LEAD_BAR_WIDTH / 2),
                            (depth, LEAD_BAR_WIDTH / 2),
                            (0.0, LEAD_BAR_WIDTH / 2),
                            (0.0, -LEAD_BAR_WIDTH / 2))
      ])
      if np.any(corners[:, 0] < 0.1):
        return empty
      heights = np.interp(corners[:, 0], path_x_array, path[:, 2]) + self._path_offset_z
      projected = self._car_space_transform @ np.column_stack((corners, heights)).T
      if not np.isfinite(projected).all() or np.any(projected[2] <= 1e-3):
        return empty
      return (projected[:2] / projected[2]).T.astype(np.float32)

    points = project_depth(LEAD_BAR_DEPTH)
    if not points.size:
      return empty
    height = np.ptp(points[:, 1])
    target = max(height, LEAD_BAR_MIN_HEIGHT)
    if height == target:
      return points
    best_error = abs(height - target)
    low, high = LEAD_BAR_DEPTH, LEAD_BAR_MAX_DEPTH
    candidate = project_depth(high)
    if candidate.size:
      candidate_height = np.ptp(candidate[:, 1])
      if abs(candidate_height - target) < best_error:
        points, best_error = candidate, abs(candidate_height - target)
      # Beyond the length cap, accept a thinner footprint instead of stretching
      # it unrealistically far toward the horizon.
      if candidate_height <= target:
        return points
    for _ in range(24):
      depth = (low + high) / 2
      candidate = project_depth(depth)
      if not candidate.size:
        high = depth
        continue
      height = np.ptp(candidate[:, 1])
      error = abs(height - target)
      if error < best_error:
        points, best_error = candidate, error
      if error < 1e-5:
        break
      if height < target:
        low = depth
      else:
        high = depth
    return points

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
    if lead and lead.present:
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

    path_pts = self._path.projected_points + np.array([self._rect.x, self._rect.y], dtype=np.float32)
    max_len = min(len(path_pts) // 2, len(self._acceleration_x))

    segment_colors = []
    gradient_stops = []

    i = 0
    while i < max_len:
      # Some points (screen space) are out of frame (rect space)
      track_y = path_pts[i][1]
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
    """Draw lane lines and road edges. Two closest lines should be green (lane line or road edges)."""
    offset = np.array([self._rect.x, self._rect.y], dtype=np.float32)

    for i, lane_line in enumerate(self._lane_lines):
      if lane_line.projected_points.size == 0:
        continue

      color = self._get_ll_color(float(self._lane_line_probs[i]), i in (1, 2), i in (0, 1))
      draw_polygon(self._rect, lane_line.projected_points + offset, color)

    for i, road_edge in enumerate(self._road_edges):
      if road_edge.projected_points.size == 0:
        continue

      # if closest lane lines are not confident, make road edges green
      color = self._get_ll_color(float(1.0 - self._road_edge_stds[i]), float(self._lane_line_probs[i + 1]) < 0.25, i == 0)
      draw_polygon(self._rect, road_edge.projected_points + offset, color)

  def _draw_path(self, sm):
    """Draw path with dynamic coloring based on mode and throttle state."""
    if not self._path.projected_points.size:
      return

    allow_throttle = sm['longitudinalPlan'].allowThrottle or not self._longitudinal_control
    self._blend_filter.update(int(allow_throttle))

    path_pts = self._path.projected_points + np.array([self._rect.x, self._rect.y], dtype=np.float32)

    if self._experimental_mode:
      # Draw with acceleration coloring
      if ui_state.status == UIStatus.DISENGAGED:
        draw_polygon(self._rect, path_pts, rl.Color(0, 0, 0, 90))
      elif len(self._exp_gradient.colors) > 1:
        draw_polygon(self._rect, path_pts, gradient=self._exp_gradient)
      else:
        draw_polygon(self._rect, path_pts, rl.Color(255, 255, 255, 30))
    else:
      # Blend throttle/no throttle colors based on transition
      blend_factor = round(self._blend_filter.x * 100) / 100
      blended_colors = self._blend_colors(NO_THROTTLE_COLORS, THROTTLE_COLORS, blend_factor)
      gradient = Gradient(
        start=(0.0, 1.0),  # Bottom of path
        end=(0.0, 0.0),  # Top of path
        colors=blended_colors,
        stops=[0.0, 0.5, 1.0],
      )

      if ui_state.status == UIStatus.DISENGAGED:
        draw_polygon(self._rect, path_pts, rl.Color(0, 0, 0, 90))
      else:
        draw_polygon(self._rect, path_pts, gradient=gradient)

  def _draw_legacy_lead_indicators(self, radar_state):
    """Original mici chevrons, enabled alongside the bars for comparison."""
    path_x = self._path.raw_points[:, 0]
    for lead in (radar_state.leadOne, radar_state.leadTwo):
      if not lead.present:
        continue
      idx = self._get_path_length_idx(path_x, lead.dRel)
      z = self._path.raw_points[idx, 2]
      point = self._map_to_screen(lead.dRel, -lead.yRel, z + self._path_offset_z)
      if point is not None:
        self._draw_legacy_lead(lead.dRel, lead.vRel, point, self._rect)

  def _draw_legacy_lead(self, d_rel, v_rel, point, rect):
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

    glow = [(px + rect.x, py + rect.y) for px, py in glow]
    chevron = [(px + rect.x, py + rect.y) for px, py in chevron]
    rl.draw_triangle_fan(glow, len(glow), rl.Color(218, 202, 37, 255))
    rl.draw_triangle_fan(chevron, len(chevron), rl.Color(201, 34, 49, int(fill_alpha)))

  def _draw_lead_indicator(self):
    if self._rect.width <= 0 or self._rect.height <= 0:
      return
    offset = np.array([self._rect.x, self._rect.y], dtype=np.float32)
    # Draw farther markers first; scissoring clips offscreen corners without pinning them to an edge.
    for lead in sorted(self._lead_vehicles, key=lambda lead: lead.distance, reverse=True):
      if lead.points.size:
        draw_polygon(self._rect, lead.points + offset, rl.Color(255, 255, 255, round(255 * lead.opacity_filter.x * lead.visibility.x)))

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
