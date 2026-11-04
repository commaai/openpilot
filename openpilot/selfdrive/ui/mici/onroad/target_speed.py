import colorsys
import math
import raylib_nb as rl
from enum import IntEnum
from openpilot.cereal import log
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget

LongitudinalPlanSource = log.LongitudinalPlan.LongitudinalPlanSource

SET_SPEED_NA = 255

# layout
HEIGHT = 108
CORNER_RADIUS = 26
MARGIN = 12
PADDING = 18
COLUMN_GAP = 10
NUMBER_SIZE = 48
LABEL_SIZE = 23
MAX_SIZE = 17

# plan chart: speed for the next 10 s under a dashed MAX line
CHART_TOP = 60
CHART_HEIGHT = 34
CHART_MIN_WIDTH = 120
CHART_LABEL_GAP = 8
CHART_MIN_RANGE = 10.0  # m/s, so a small gap also looks small
CHART_SMOOTH = 0.2  # per frame, the plan comes at 20 Hz
CHART_REVEAL = 0.6  # s, the line draws in from the left
LINE_WIDTH = 3.0
DASH_LENGTH = 4
DASH_GAP = 4

# when to show, in display units (mph or km/h)
SHOW_GAP = 5
HIDE_GAP = 2
TARGET_STEP = 5
TARGET_HYSTERESIS = 1.0
REASON_HOLD = 0.6  # s, a new reason must stay this long before we show it

# reasons
STOP_SPEED = 2.0  # m/s
LEAD_TIME_GAP = 4.0  # s
LEAD_MIN_DISTANCE = 30.0  # m
CURVE_LAT_ACCEL = 2.0  # m/s^2, lateral accel at max speed in the curves ahead
SLOWING_MARGIN = 2.0  # m/s

ROLL_DURATION = 0.45  # s
RELEASE_HOLD = 0.9  # s, show the roll up to MAX before the capsule goes away


class Reason(IntEnum):
  CRUISING = 0
  SLOWING = 1
  CAR_AHEAD = 2
  CURVE = 3
  STOPPING = 4
  RESUMING = 5


def reason_text(reason: Reason) -> str:
  return {
    Reason.CRUISING: tr("Cruising"),
    Reason.SLOWING: tr("Slowing"),
    Reason.CAR_AHEAD: tr("Car ahead"),
    Reason.CURVE: tr("Curve ahead"),
    Reason.STOPPING: tr("Stopping"),
    Reason.RESUMING: tr("Resuming"),
  }[reason]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
  return max(low, min(high, value))


def lerp(start: float, end: float, t: float) -> float:
  return start + (end - start) * t


def ease_out_cubic(t: float) -> float:
  return 1 - (1 - t) ** 3


# the old content goes before the new content comes, so they do not overlap
def fade_out(progress: float) -> float:
  return clamp(1 - progress / 0.45)


def fade_in(progress: float) -> float:
  return clamp((progress - 0.15) / 0.45)


def white(alpha: float) -> rl.Color:
  return rl.Color(255, 255, 255, round(255 * clamp(alpha)))


def accel_color(accel: float, alpha: float) -> rl.Color:
  # same colors as the path: green to speed up, white to hold, red to slow down
  hue = clamp(60 + accel * 35, 0, 120)
  saturation = min(abs(accel * 1.5), 1)
  lightness = lerp(0.95, 0.62, saturation)
  red, green, blue = colorsys.hls_to_rgb(hue / 360, lightness, saturation)
  return rl.Color(round(red * 255), round(green * 255), round(blue * 255), round(255 * clamp(alpha)))


class Spring:
  """Damped spring, same parameters as SwiftUI's spring(response:dampingFraction:)."""

  def __init__(self, x: float, response: float, damping_fraction: float):
    self.x = x
    self.velocity = 0.0
    self._stiffness = (2 * math.pi / response) ** 2
    self._damping = 4 * math.pi * damping_fraction / response
    self._dt = 1 / gui_app.target_fps

  def update(self, target: float) -> float:
    accel = self._stiffness * (target - self.x) - self._damping * self.velocity
    self.velocity += accel * self._dt
    self.x += self.velocity * self._dt
    return self.x

  def snap(self, x: float) -> None:
    self.x = x
    self.velocity = 0.0


class DigitColumn:
  def __init__(self):
    self.char = ''
    self.prev_char = ''
    self.change_time = -ROLL_DURATION
    self.direction = 1


class RollingNumber:
  """Number where only the changed digits roll, up when the value goes up and down when it goes down."""

  def __init__(self, font: rl.Font, size: int):
    self._font = font
    self._size = size
    # same width for all digits so the number does not move when a digit changes
    self.digit_width = max(measure_text_cached(font, str(digit), size).x for digit in range(10))
    self._columns: list[DigitColumn] = []
    self._value: int | None = None
    self.width = Spring(0.0, 0.4, 0.85)

  def reset(self, digits: int) -> None:
    self._columns = []
    self._value = None
    self.width.snap(self.digit_width * digits)

  def set_value(self, value: int) -> None:
    if value == self._value:
      return

    direction = 1 if self._value is None or value > self._value else -1
    text = str(value)
    now = rl.get_time()
    while len(self._columns) < len(text):
      self._columns.append(DigitColumn())

    # columns count from the right
    for i, column in enumerate(self._columns):
      char = text[-1 - i] if i < len(text) else ''
      if char != column.char:
        column.prev_char = column.char
        column.char = char
        column.change_time = now
        column.direction = direction

    self._value = value

  def update_width(self, digits: int) -> float:
    return self.width.update(self.digit_width * digits)

  def draw(self, right: float, y: float, alpha: float) -> None:
    now = rl.get_time()
    travel = self._size * 0.3
    for i, column in enumerate(self._columns):
      center_x = right - (i + 0.5) * self.digit_width
      progress = ease_out_cubic(clamp((now - column.change_time) / ROLL_DURATION))
      # new digit comes in from below when the value goes up, old digit goes out the top
      self._draw_char(column.char, center_x, y + column.direction * (1 - progress) * travel, alpha * fade_in(progress))
      if progress < 1:
        self._draw_char(column.prev_char, center_x, y - column.direction * progress * travel, alpha * fade_out(progress))

  def _draw_char(self, char: str, center_x: float, y: float, alpha: float) -> None:
    if len(char) == 0 or alpha < 1e-2:
      return

    width = measure_text_cached(self._font, char, self._size).x
    rl.draw_text_ex(self._font, char, rl.Vector2(round(center_x - width / 2), y), self._size, 0, white(alpha))


class TargetSpeed(Widget):
  """Shows the speed openpilot drives at when it is below MAX, and why.

  MAX is only the highest speed openpilot can go. The capsule shows the speed openpilot goes for, a track
  from zero to MAX, and the reason it does not go faster.
  """

  def __init__(self):
    super().__init__()
    self._font_number = gui_app.font(FontWeight.DISPLAY)
    self._font_label = gui_app.font(FontWeight.SEMI_BOLD)
    self._font_max = gui_app.font(FontWeight.MEDIUM)

    self._target_filter = FirstOrderFilter(0.0, 1.0, 1 / gui_app.target_fps, initialized=False)
    self._started_frame = -1
    self._can_draw = True

    self._shown = False
    self._was_visible = False
    self._release_time = -RELEASE_HOLD
    self._max_speed_ms = 0.0
    self._plan_speeds: list[float] = []
    self._plan_accels: list[float] = []
    self._chart_speeds: list[float] = []
    self._chart_accels: list[float] = []
    self._reveal_time = 0.0
    self._max_speed = 0
    self._target_speed = 0
    self._reason = Reason.CRUISING
    self._prev_reason = Reason.CRUISING
    self._reason_change_time = -ROLL_DURATION
    self._candidate_reason = Reason.CRUISING
    self._candidate_time = 0.0

    self._number = RollingNumber(self._font_number, NUMBER_SIZE)
    self._presence = Spring(0.0, 0.5, 0.72)
    self._width = Spring(0.0, 0.4, 0.85)
    self._chart_floor = Spring(0.0, 0.6, 0.9)

  def set_can_draw(self, can_draw: bool) -> None:
    self._can_draw = can_draw

  def _update_state(self) -> None:
    sm = ui_state.sm
    if self._started_frame != ui_state.started_frame:
      self._target_filter.initialized = False
      self._started_frame = ui_state.started_frame

    car_state = sm['carState']
    model = sm['modelV2']
    engaged = (ui_state.started and ui_state.has_longitudinal_control and sm['selfdriveState'].enabled and
               sm.recv_frame['modelV2'] >= ui_state.started_frame and sm.alive['modelV2'] and len(model.velocity.x) > 0 and
               0 < car_state.vCruiseCluster < SET_SPEED_NA)
    if not engaged:
      self._target_filter.initialized = False
      self._shown = False
      self._release_time = -RELEASE_HOLD
      return

    now = rl.get_time()
    conversion = CV.MS_TO_KPH if ui_state.is_metric else CV.MS_TO_MPH
    max_speed_ms = car_state.vCruiseCluster * CV.KPH_TO_MS
    target_speed_ms = min(self._target_filter.update(max(0.0, model.velocity.x[-1])), max_speed_ms)

    # only the gap to MAX decides. the plan source is not good for this, it is cruise each time the cruise
    # accel limit is the lowest, also when the plan stays far below MAX
    self._max_speed = round(max_speed_ms * conversion)
    target_speed = round(target_speed_ms * conversion)
    gap = self._max_speed - target_speed
    was_shown = self._shown
    if self._shown:
      self._shown = gap >= HIDE_GAP
    else:
      self._shown = gap >= SHOW_GAP

    if was_shown and not self._shown:
      # back to MAX: roll the number up and raise the plan to the MAX line before the capsule goes away
      self._release_time = now
      self._target_speed = self._max_speed
      self._plan_speeds = [max_speed_ms] * len(self._plan_speeds)
      self._plan_accels = [0.0] * len(self._plan_accels)
      self._set_reason(Reason.RESUMING, now)

    if not self._shown:
      return

    # steps of 5 with some hysteresis, so the number only rolls when the target really changes
    if not was_shown or abs(target_speed - self._target_speed) > TARGET_STEP / 2 + TARGET_HYSTERESIS:
      self._target_speed = min(round(target_speed / TARGET_STEP) * TARGET_STEP, self._max_speed)
    self._max_speed_ms = max_speed_ms
    self._plan_speeds = [min(speed, max_speed_ms) for speed in model.velocity.x]
    self._plan_accels = list(model.acceleration.x)

    # the reason uses the plan without the filter, so it does not lag behind the chart
    plan_source = sm['longitudinalPlan'].longitudinalPlanSource
    reason = self._get_reason(min(model.velocity.x[-1], max_speed_ms), max_speed_ms, plan_source)
    if reason != self._candidate_reason:
      self._candidate_reason = reason
      self._candidate_time = now
    first_show = self._presence.x < 0.05
    if reason != self._reason and (first_show or now - self._candidate_time > REASON_HOLD):
      self._set_reason(reason, -ROLL_DURATION if first_show else now)

  def _set_reason(self, reason: Reason, change_time: float) -> None:
    self._prev_reason = self._reason
    self._reason = reason
    self._reason_change_time = change_time

  @staticmethod
  def _get_reason(target_speed: float, max_speed: float, plan_source) -> Reason:
    sm = ui_state.sm
    v_ego = sm['carState'].vEgo
    if target_speed < STOP_SPEED:
      return Reason.STOPPING

    lead = sm['radarState'].leadOne
    lead_close = lead.present and lead.dRel < max(LEAD_MIN_DISTANCE, LEAD_TIME_GAP * v_ego) and lead.vLead < max_speed - SLOWING_MARGIN
    if plan_source in (LongitudinalPlanSource.lead0, LongitudinalPlanSource.lead1) or lead_close:
      return Reason.CAR_AHEAD

    # lateral accel if we went MAX through the curves the model sees
    model = sm['modelV2']
    max_curvature = 0.0
    for yaw_rate, speed in zip(model.orientationRate.z, model.velocity.x, strict=False):
      max_curvature = max(max_curvature, abs(yaw_rate) / max(speed, 1.0))
    if max_speed ** 2 * max_curvature > CURVE_LAT_ACCEL:
      return Reason.CURVE

    if target_speed < v_ego - SLOWING_MARGIN:
      return Reason.SLOWING
    return Reason.CRUISING

  def _render(self, rect: rl.Rectangle) -> None:
    now = rl.get_time()
    releasing = now - self._release_time < RELEASE_HOLD
    visible = (self._shown or releasing) and self._can_draw

    digits = len(str(self._target_speed))
    label = reason_text(self._reason)
    max_text = f"{tr('MAX')} {self._max_speed}"
    label_width = measure_text_cached(self._font_label, label, LABEL_SIZE).x
    max_text_width = measure_text_cached(self._font_max, max_text, MAX_SIZE).x

    if visible and not self._was_visible and self._presence.x < 0.05:
      # start fresh: digits roll in and the plan draws in from the left
      self._number.reset(digits)
      self._chart_speeds = list(self._plan_speeds)
      self._chart_accels = list(self._plan_accels)
      self._chart_floor.snap(self._get_chart_floor())
      self._reveal_time = -1.0
      self._width.snap(self._content_width(self._number.width.x, label_width, max_text_width))
    self._was_visible = visible

    presence = self._presence.update(float(visible))
    if not visible and presence < 1e-2:
      self._presence.snap(0.0)
      return

    number_width = self._number.update_width(digits)
    full_width = self._width.update(self._content_width(number_width, label_width, max_text_width))

    # grow out of a small pill on the right, with a small overshoot
    grow = max(0.0, presence)
    width = max(0.0, lerp(HEIGHT * 0.5, full_width, grow))
    height = max(0.0, lerp(HEIGHT * 0.5, HEIGHT, grow))
    right = rect.x + rect.width - MARGIN - (HEIGHT - height) / 2
    top = rect.y + MARGIN + (HEIGHT - height) / 2
    capsule = rl.Rectangle(right - width, top, width, height)

    background_alpha = clamp(presence * 2)
    self._draw_capsule(capsule, background_alpha)

    # content waits until the capsule is almost full size
    content_alpha = clamp((presence - 0.8) / 0.2)
    if content_alpha < 1e-2:
      return

    # set the number here, so the digits roll in when the content shows
    self._number.set_value(self._target_speed)
    if self._reveal_time < 0:
      self._reveal_time = now

    # content stays at its full size layout, the capsule grows around it
    x = right - full_width + PADDING
    y = rect.y + MARGIN

    self._number.draw(x + number_width, y + 6, content_alpha)

    label_x = x + number_width + COLUMN_GAP
    label_y = y + 22
    progress = ease_out_cubic(clamp((now - self._reason_change_time) / ROLL_DURATION))
    if progress < 1:
      prev_label = reason_text(self._prev_reason)
      rl.draw_text_ex(self._font_label, prev_label, rl.Vector2(label_x, label_y - 6 * progress), LABEL_SIZE, 0,
                      white(content_alpha * fade_out(progress)))
    rl.draw_text_ex(self._font_label, label, rl.Vector2(label_x, label_y + 6 * (1 - progress)), LABEL_SIZE, 0,
                    white(content_alpha * fade_in(progress)))

    chart_width = full_width - PADDING * 2 - CHART_LABEL_GAP - max_text_width
    self._draw_chart(rl.Rectangle(x, y + CHART_TOP, chart_width, CHART_HEIGHT), max_text, content_alpha, now)

  @staticmethod
  def _content_width(number_width: float, label_width: float, max_text_width: float) -> float:
    top_row = number_width + COLUMN_GAP + label_width
    chart_row = CHART_MIN_WIDTH + CHART_LABEL_GAP + max_text_width
    return PADDING + max(top_row, chart_row) + PADDING

  def _get_chart_floor(self) -> float:
    # MAX is the top of the chart, show at least CHART_MIN_RANGE below it
    lowest = min(self._plan_speeds, default=self._max_speed_ms)
    return self._max_speed_ms - max(CHART_MIN_RANGE, (self._max_speed_ms - lowest) * 1.15)

  def _draw_chart(self, chart: rl.Rectangle, max_text: str, alpha: float, now: float) -> None:
    # smooth the plan, it only comes at 20 Hz
    if len(self._chart_speeds) != len(self._plan_speeds):
      self._chart_speeds = list(self._plan_speeds)
      self._chart_accels = list(self._plan_accels)
    for i in range(len(self._chart_speeds)):
      self._chart_speeds[i] += (self._plan_speeds[i] - self._chart_speeds[i]) * CHART_SMOOTH
      self._chart_accels[i] += (self._plan_accels[i] - self._chart_accels[i]) * CHART_SMOOTH

    floor = self._chart_floor.update(self._get_chart_floor())
    span = max(self._max_speed_ms - floor, 1.0)
    bottom = chart.y + chart.height

    # dashed MAX line along the top, with the MAX label at its end
    dash_x = chart.x
    while dash_x < chart.x + chart.width:
      dash_end = min(dash_x + DASH_LENGTH, chart.x + chart.width)
      rl.draw_line_ex(rl.Vector2(dash_x, chart.y), rl.Vector2(dash_end, chart.y), 1.5, white(alpha * 0.45))
      dash_x += DASH_LENGTH + DASH_GAP
    max_text_height = measure_text_cached(self._font_max, max_text, MAX_SIZE).y
    rl.draw_text_ex(self._font_max, max_text, rl.Vector2(chart.x + chart.width + CHART_LABEL_GAP, chart.y - max_text_height / 2),
                    MAX_SIZE, 0, white(alpha * 0.6))

    if len(self._chart_speeds) < 2:
      return

    # planned speed for the next 10 s, colored by the planned accel like the path
    t_end = ModelConstants.T_IDXS[-1]
    points = []
    for t, speed in zip(ModelConstants.T_IDXS, self._chart_speeds, strict=False):
      point_y = chart.y + chart.height * (1 - (speed - floor) / span)
      points.append(rl.Vector2(chart.x + chart.width * t / t_end, max(chart.y, min(bottom, point_y))))

    reveal_x = chart.x + chart.width * ease_out_cubic(clamp((now - self._reveal_time) / CHART_REVEAL))
    head = points[0]
    for i in range(len(points) - 1):
      start, end = points[i], points[i + 1]
      if start.x >= reveal_x:
        break

      if end.x > reveal_x:
        t = (reveal_x - start.x) / max(end.x - start.x, 1e-3)
        end = rl.Vector2(reveal_x, lerp(start.y, end.y, t))
      head = end

      # area under the line fades out to the bottom
      color = accel_color(self._chart_accels[i], alpha)
      area_top = round((start.y + end.y) / 2)
      rl.draw_rectangle_gradient_v(round(start.x), area_top, max(1, round(end.x) - round(start.x)), round(bottom) - area_top,
                                   accel_color(self._chart_accels[i], alpha * 0.3), accel_color(self._chart_accels[i], 0.0))
      rl.draw_line_ex(start, end, LINE_WIDTH, color)
      rl.draw_circle_v(end, LINE_WIDTH / 2, color)

    # dot at the end of the plan, that is the target speed
    rl.draw_circle_v(head, LINE_WIDTH * 1.4, white(alpha))

  @staticmethod
  def _draw_capsule(capsule: rl.Rectangle, alpha: float) -> None:
    # soft shadow, then the capsule and a faint edge
    for spread, shadow_alpha in ((8, 0.08), (4, 0.12)):
      shadow = rl.Rectangle(capsule.x - spread, capsule.y - spread + 2, capsule.width + spread * 2, capsule.height + spread * 2)
      roundness = clamp(2 * (CORNER_RADIUS + spread) / max(min(shadow.width, shadow.height), 1))
      rl.draw_rectangle_rounded(shadow, roundness, 16, rl.Color(0, 0, 0, round(255 * shadow_alpha * alpha)))
    roundness = clamp(2 * CORNER_RADIUS / max(min(capsule.width, capsule.height), 1))
    rl.draw_rectangle_rounded(capsule, roundness, 16, rl.Color(0, 0, 0, round(255 * 0.62 * alpha)))
    rl.draw_rectangle_rounded_lines_ex(capsule, roundness, 16, 1.5, white(0.1 * alpha))
