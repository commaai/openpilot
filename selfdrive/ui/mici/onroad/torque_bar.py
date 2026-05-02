import math
import time
from functools import wraps
from collections import OrderedDict

import numpy as np
import pyray as rl
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY
from openpilot.selfdrive.ui.mici.onroad import blend_colors
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.shader_polygon import draw_polygon, Gradient, ShaderState, _configure_shader_color, triangulate
from openpilot.system.ui.widgets import Widget
from openpilot.common.filter_simple import FirstOrderFilter

# TODO: arc_bar_pts doesn't consider rounded end caps part of the angle span
TORQUE_ANGLE_SPAN = 12.7

DEBUG = False

# DEBUG timing
import time as _t_time
_T = {'state': 0.0, 'arc_bg': 0.0, 'draw_bg': 0.0, 'arc_sl': 0.0, 'grad_endpts': 0.0, 'colors': 0.0, 'draw_sl': 0.0, 'count': 0, 'last_print': _t_time.monotonic()}


def _blend_rgb(a, b, f):
  fc = 0.0 if f < 0.0 else 1.0 if f > 1.0 else f
  return rl.Color(
    int(a.r + fc * (b.r - a.r)),
    int(a.g + fc * (b.g - a.g)),
    int(a.b + fc * (b.b - a.b)),
    int(a.a + fc * (b.a - a.a)),
  )


def _draw_two_polygons_batched(origin_rect, pts1, color1, pts2, gradient2):
  """Two polygons sharing one begin/end_shader_mode (skips the second batch flush)."""
  state = ShaderState.get_instance()
  state.initialize()
  pts1c = np.ascontiguousarray(pts1, dtype=np.float32)
  pts2c = np.ascontiguousarray(pts2, dtype=np.float32)
  tri1 = triangulate(pts1c)
  tri2 = triangulate(pts2c)
  rl.begin_shader_mode(state.shader)
  if len(tri1) >= 3:
    _configure_shader_color(state, color1, None, origin_rect)
    rl.draw_triangle_strip(tri1, len(tri1), rl.WHITE)
  if len(tri2) >= 3:
    _configure_shader_color(state, None, gradient2, origin_rect)
    rl.draw_triangle_strip(tri2, len(tri2), rl.WHITE)
  rl.end_shader_mode()


def quantized_lru_cache(maxsize=128):
  def decorator(func):
    cache = OrderedDict()
    @wraps(func)
    def wrapper(r_mid, thickness, a0_deg, a1_deg, **kwargs):
      # Quantize inputs: balanced for smoothness vs cache effectiveness
      key = (round(r_mid),
             round(thickness),           # 1px precision for smoother height transitions
             round(a0_deg * 10) / 10,    # 0.1° precision for smoother angle transitions
             round(a1_deg * 10) / 10,
             tuple(sorted(kwargs.items())))

      if key in cache:
        cache.move_to_end(key)
        wrapper._hits = getattr(wrapper, '_hits', 0) + 1
      else:
        if len(cache) >= maxsize:
          cache.popitem(last=False)
        result = func(r_mid, thickness, a0_deg, a1_deg, **kwargs)
        cache[key] = result
        wrapper._misses = getattr(wrapper, '_misses', 0) + 1
      total = getattr(wrapper, '_hits', 0) + getattr(wrapper, '_misses', 0)
      if total % 120 == 0:
        print(f'[arc_bar_pts cache] hits={wrapper._hits} misses={wrapper._misses} hit_rate={wrapper._hits/total*100:.1f}%', flush=True)
      return cache[key]
    return wrapper
  return decorator


@quantized_lru_cache(maxsize=256)
def arc_bar_pts(r_mid: float, thickness: float,
                a0_deg: float, a1_deg: float,
                *, max_points: int = 100, cap_segs: int = 10,
                cap_radius: float = 7, px_per_seg: float = 2.0) -> np.ndarray:
  """Return Nx2 np.float32 points for a single closed polygon (rounded thick arc), centered at origin."""

  def get_cap(left: bool, a_deg: float):
    # end cap at a1: center (a1), sweep a1→a1+180 (skip endpoints to avoid dupes)
    # quarter arc (outer corner) at a1 with fixed pixel radius cap_radius

    nx, ny = math.cos(math.radians(a_deg)), math.sin(math.radians(a_deg))  # outward normal
    tx, ty = -ny, nx  # tangent (CCW)

    mx, my = nx * r_mid, ny * r_mid  # mid-point at a1
    if DEBUG:
      rl.draw_circle(int(mx), int(my), 4, rl.PURPLE)

    ex = mx + nx * (half - cap_radius)
    ey = my + ny * (half - cap_radius)

    if DEBUG:
      rl.draw_circle(int(ex), int(ey), 2, rl.WHITE)

    # sweep 90° in the local (t,n) frame: from outer edge toward inside
    if not left:
      alpha = np.deg2rad(np.linspace(90, 0, cap_segs + 2))[1:-1]
    else:
      alpha = np.deg2rad(np.linspace(180, 90, cap_segs + 2))[1:-1]
    cap_end = np.c_[ex + np.cos(alpha) * cap_radius * tx + np.sin(alpha) * cap_radius * nx,
                    ey + np.cos(alpha) * cap_radius * ty + np.sin(alpha) * cap_radius * ny]

    # bottom quarter (inner corner) at a1
    ex2 = mx + nx * (-half + cap_radius)
    ey2 = my + ny * (-half + cap_radius)
    if DEBUG:
      rl.draw_circle(int(ex2), int(ey2), 2, rl.WHITE)

    if not left:
      alpha2 = np.deg2rad(np.linspace(0, -90, cap_segs + 1))[:-1]  # include 0 once, exclude -90
    else:
      alpha2 = np.deg2rad(np.linspace(90 - 90 - 90, 0 - 90 - 90, cap_segs + 1))[:-1]
    cap_end_bot = np.c_[ex2 + np.cos(alpha2) * cap_radius * tx + np.sin(alpha2) * cap_radius * nx,
                        ey2 + np.cos(alpha2) * cap_radius * ty + np.sin(alpha2) * cap_radius * ny]

    # append to the top quarter
    if not left:
      cap_end = np.vstack((cap_end, cap_end_bot))
    else:
      cap_end = np.vstack((cap_end_bot, cap_end))

    return cap_end

  if a1_deg < a0_deg:
    a0_deg, a1_deg = a1_deg, a0_deg
  half = thickness * 0.5

  cap_radius = min(cap_radius, half)

  span = max(1e-3, a1_deg - a0_deg)

  # pick arc segment count from arc length, clamp to shader points[] budget
  arc_len = r_mid * math.radians(span)
  arc_segs = max(6, int(arc_len / px_per_seg))
  max_arc = (max_points - (4 * cap_segs + 3)) // 2
  arc_segs = max(6, min(arc_segs, max_arc))

  # outer arc a0→a1
  ang_o = np.deg2rad(np.linspace(a0_deg, a1_deg, arc_segs + 1))
  outer = np.c_[np.cos(ang_o) * (r_mid + half),
                np.sin(ang_o) * (r_mid + half)]

  # end cap at a1
  cap_end = get_cap(False, a1_deg)

  # inner arc a1→a0
  ang_i = np.deg2rad(np.linspace(a1_deg, a0_deg, arc_segs + 1))
  inner = np.c_[np.cos(ang_i) * (r_mid - half),
                np.sin(ang_i) * (r_mid - half)]

  # start cap at a0
  cap_start = get_cap(True, a0_deg)

  pts = np.vstack((outer, cap_end, inner, cap_start, outer[:1])).astype(np.float32)

  # Rotate to start from middle of cap for proper triangulation
  pts = np.roll(pts, cap_segs, axis=0)

  if DEBUG:
    n = len(pts)
    idx = int(time.monotonic() * 12) % max(1, n)  # speed: 12 pts/sec
    for i, (x, y) in enumerate(pts):
      j = (i - idx) % n  # rotate the gradient
      t = j / n
      color = rl.Color(255, int(255 * (1 - t)), int(255 * t), 255)
      rl.draw_circle(int(x), int(y), 2, color)

  return pts


DEFAULT_MAX_LAT_ACCEL = 3.0  # m/s^2


class TorqueBar(Widget):
  def __init__(self, demo: bool = False):
    super().__init__()
    self._demo = demo
    self._torque_filter = FirstOrderFilter(0, 0.1, 1 / gui_app.target_fps)
    self._torque_line_alpha_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)

  def update_filter(self, value: float):
    """Update the torque filter value (for demo mode)."""
    self._torque_filter.update(value)

  def _update_state(self):
    if self._demo:
      return

    # torque line
    if ui_state.sm['controlsState'].lateralControlState.which() == 'angleState':
      controls_state = ui_state.sm['controlsState']
      car_state = ui_state.sm['carState']
      live_parameters = ui_state.sm['liveParameters']
      car_control = ui_state.sm['carControl']

      # Include lateral accel error in estimated torque utilization
      actual_lateral_accel = controls_state.curvature * car_state.vEgo ** 2
      desired_lateral_accel = controls_state.desiredCurvature * car_state.vEgo ** 2
      accel_diff = (desired_lateral_accel - actual_lateral_accel)

      # Include road roll in estimated torque utilization
      # Roll is less accurate near standstill, so reduce its effect at low speed
      roll_compensation = live_parameters.roll * ACCELERATION_DUE_TO_GRAVITY * np.interp(car_state.vEgo, [5, 15], [0.0, 1.0])
      lateral_acceleration = actual_lateral_accel - roll_compensation
      max_lateral_acceleration = ui_state.CP.maxLateralAccel if ui_state.CP else DEFAULT_MAX_LAT_ACCEL

      if not car_control.latActive:
        self._torque_filter.update(0.0)
      else:
        self._torque_filter.update(np.clip((lateral_acceleration + accel_diff) / max_lateral_acceleration, -1, 1))
    else:
      self._torque_filter.update(-ui_state.sm['carOutput'].actuatorsOutput.torque)

  def _render(self, rect: rl.Rectangle) -> None:
    _t0 = _t_time.perf_counter()
    # adjust y pos with torque (shared t over [0.5, 1.0])
    _t = abs(self._torque_filter.x)
    _t = 0.0 if _t <= 0.5 else 1.0 if _t >= 1.0 else (_t - 0.5) * 2.0
    torque_line_offset = 22 + _t * 4
    torque_line_height = 14 + _t * 42

    # animate alpha and angle span
    if not self._demo:
      self._torque_line_alpha_filter.update(ui_state.status != UIStatus.DISENGAGED)
    else:
      self._torque_line_alpha_filter.update(1.0)

    torque_line_bg_alpha = 0.25 + _t * 0.25
    torque_line_bg_color = rl.Color(255, 255, 255, int(255 * torque_line_bg_alpha * self._torque_line_alpha_filter.x))
    if ui_state.status != UIStatus.ENGAGED and not self._demo:
      torque_line_bg_color = rl.Color(255, 255, 255, int(255 * 0.15 * self._torque_line_alpha_filter.x))

    # draw curved line polygon torque bar
    torque_line_radius = 1200
    top_angle = -90
    torque_bg_angle_span = self._torque_line_alpha_filter.x * TORQUE_ANGLE_SPAN
    torque_start_angle = top_angle - torque_bg_angle_span / 2
    torque_end_angle = top_angle + torque_bg_angle_span / 2
    mid_r = torque_line_radius + torque_line_height / 2

    cx = rect.x + rect.width / 2 + 8
    cy = rect.y + rect.height + torque_line_radius - torque_line_offset
    offset = np.array([cx, cy], dtype=np.float32)
    _t1 = _t_time.perf_counter(); _T['state'] += _t1 - _t0

    # draw bg torque indicator line
    bg_pts = arc_bar_pts(mid_r, torque_line_height, torque_start_angle, torque_end_angle) + offset
    _t2 = _t_time.perf_counter(); _T['arc_bg'] += _t2 - _t1

    # draw torque indicator line
    a0s = top_angle
    a1s = a0s + torque_bg_angle_span / 2 * self._torque_filter.x
    sl_pts = arc_bar_pts(mid_r, torque_line_height, a0s, a1s) + offset
    _t3 = _t_time.perf_counter(); _T['arc_sl'] += _t3 - _t2

    # gradient endpoint min/max
    start_grad_pt = cx / rect.width
    if self._torque_filter.x < 0:
      end_grad_pt = (cx * (1 - 0.65) + (bg_pts[:, 0].min() * 0.65)) / rect.width
    else:
      end_grad_pt = (cx * (1 - 0.65) + (bg_pts[:, 0].max() * 0.65)) / rect.width
    _t4 = _t_time.perf_counter(); _T['grad_endpts'] += _t4 - _t3

    # blend_colors + Gradient
    start_color = _blend_rgb(
      rl.Color(255, 255, 255, int(255 * 0.9 * self._torque_line_alpha_filter.x)),
      rl.Color(255, 200, 0, int(255 * self._torque_line_alpha_filter.x)),
      max(0, abs(self._torque_filter.x) - 0.75) * 4,
    )
    end_color = _blend_rgb(
      rl.Color(255, 255, 255, int(255 * 0.9 * self._torque_line_alpha_filter.x)),
      rl.Color(255, 115, 0, int(255 * self._torque_line_alpha_filter.x)),
      max(0, abs(self._torque_filter.x) - 0.75) * 4,
    )
    if ui_state.status != UIStatus.ENGAGED and not self._demo:
      start_color = end_color = rl.Color(255, 255, 255, int(255 * 0.35 * self._torque_line_alpha_filter.x))
    gradient = Gradient(
      start=(start_grad_pt, 0),
      end=(end_grad_pt, 0),
      colors=[start_color, end_color],
      stops=[0.0, 1.0],
    )
    _t4b = _t_time.perf_counter(); _T['colors'] += _t4b - _t4; _t4 = _t4b
    draw_polygon(rect, sl_pts, gradient=gradient)
    _t4c = _t_time.perf_counter(); _T['draw_sl'] += _t4c - _t4; _t4 = _t4c
    draw_polygon(rect, bg_pts, color=torque_line_bg_color)
    _t4d = _t_time.perf_counter(); _T['draw_bg'] += _t4d - _t4c; _t4 = _t4d

    if abs(self._torque_filter.x) < 0.5:
      dot_y = self._rect.y + self._rect.height - torque_line_offset - torque_line_height / 2
      rl.draw_circle(int(cx), int(dot_y), 10 // 2,
                     rl.Color(182, 182, 182, int(255 * 0.9 * self._torque_line_alpha_filter.x)))
    _t5 = _t_time.perf_counter()
    _T['count'] += 1
    if _t5 - _T['last_print'] > 3.0:
      n = _T['count']
      total = sum(_T[k] for k in ('state','arc_bg','draw_bg','arc_sl','grad_endpts','colors','draw_sl'))
      print(f"[torque] n={n} state={_T['state']/n*1e6:.0f} arc_bg={_T['arc_bg']/n*1e6:.0f} DRAW_BG={_T['draw_bg']/n*1e6:.0f} arc_sl={_T['arc_sl']/n*1e6:.0f} grad_pts={_T['grad_endpts']/n*1e6:.0f} colors={_T['colors']/n*1e6:.0f} DRAW_SL={_T['draw_sl']/n*1e6:.0f} total={total/n*1e6:.0f}us", flush=True)
      for k in ('state','arc_bg','draw_bg','arc_sl','grad_endpts','colors','draw_sl'):
        _T[k] = 0.0
      _T['count'] = 0
      _T['last_print'] = _t5

