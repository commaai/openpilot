import colorsys
import math
import raylib_nb as rl
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget

SET_SPEED_NA = 255
SET_SPEED_PERSISTENCE = 2.5  # s, show MAX this long after it changes or on engage

# layout: the number is in the top right corner, the chart goes across the top to it
MARGIN_X = 18
NUMBER_SIZE = 112
NUMBER_Y = 4
LABEL_SIZE = 36
LABEL_Y = 120  # MAX under the number
LABEL_POP = 0.18  # MAX grows this much when it changes, then goes back
LABEL_POP_TIME = 0.7  # s
SHADOW_HEIGHT = 150
SHADOW_ALPHA = 0.5

# plan chart: speed for the next 10 s under a dashed MAX line
CHART_LEFT = 96  # right of the driver monitoring icon
CHART_TOP = 44  # the dashed MAX line
CHART_HEIGHT = 58
CHART_NUMBER_GAP = 24
CHART_LABEL_GAP = 4
CHART_MIN_RANGE = 10.0  # m/s, so a small gap also looks small
CHART_SMOOTH = 0.2  # per frame, the plan comes at 20 Hz
CHART_REVEAL = 0.6  # s, the line draws in from the left
LINE_WIDTH = 6.0
DOT_RADIUS = 8.0
DASH_LENGTH = 8
DASH_GAP = 7
DASH_WIDTH = 3.0

# show the target when it is this far below MAX, in display units (mph or km/h)
SHOW_GAP = 5
HIDE_GAP = 2
TARGET_STEP = 5
TARGET_HYSTERESIS = 1.0

ROLL_DURATION = 0.45  # s
RELEASE_HOLD = 1.2  # s, show the roll up to MAX before it goes away


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


class SpeedDisplay(Widget):
  """MAX and the speed openpilot drives at, in one place.

  MAX shows for a short time when it changes or on engage. When openpilot plans to stay below MAX, the big number
  is the speed it goes for (the end of the 10 s plan), with the planned speed as a line under a dashed MAX line.
  """

  def __init__(self):
    super().__init__()
    self._font_number = gui_app.font(FontWeight.DISPLAY)
    self._font_label = gui_app.font(FontWeight.SEMI_BOLD)

    self._target_filter = FirstOrderFilter(0.0, 1.0, 1 / gui_app.target_fps, initialized=False)
    self._started_frame = -1
    self._can_draw = True

    self._active = False  # engaged with a set speed
    self._set_speed = SET_SPEED_NA  # km/h, from the car
    self._max_changed_time = -SET_SPEED_PERSISTENCE
    self._max_speed = 0  # display units
    self._max_speed_ms = 0.0
    self._below_max = False
    self._release_time = -RELEASE_HOLD
    self._target_speed = 0  # display units, steps of TARGET_STEP

    self._plan_speeds: list[float] = []
    self._plan_accels: list[float] = []
    self._chart_speeds: list[float] = []
    self._chart_accels: list[float] = []
    self._reveal_time = 0.0

    self._label = ''
    self._prev_label = ''
    self._label_change_time = -ROLL_DURATION
    self._was_visible = False

    self._number = RollingNumber(self._font_number, NUMBER_SIZE)
    self._presence = Spring(0.0, 0.45, 0.8)
    self._chart_presence = Spring(0.0, 0.45, 1.0)
    self._chart_floor = Spring(0.0, 0.6, 0.9)
    self._label_alpha = FirstOrderFilter(0.0, 0.15, 1 / gui_app.target_fps)

  def set_can_draw(self, can_draw: bool) -> None:
    self._can_draw = can_draw

  def _update_state(self) -> None:
    sm = ui_state.sm
    if self._started_frame != ui_state.started_frame:
      self._target_filter.initialized = False
      self._started_frame = ui_state.started_frame

    car_state = sm['carState']
    set_speed = car_state.vCruiseCluster
    active = (ui_state.started and sm.recv_frame['carState'] >= ui_state.started_frame and sm['selfdriveState'].enabled and
              0 < set_speed < SET_SPEED_NA)
    now = rl.get_time()
    if active and (set_speed != self._set_speed or not self._active):
      self._max_changed_time = now
    self._active = active
    self._set_speed = set_speed
    if not active:
      self._below_max = False
      self._release_time = -RELEASE_HOLD
      self._target_filter.initialized = False
      return

    conversion = CV.MS_TO_KPH if ui_state.is_metric else CV.MS_TO_MPH
    self._max_speed_ms = set_speed * CV.KPH_TO_MS
    self._max_speed = round(self._max_speed_ms * conversion)

    model = sm['modelV2']
    has_plan = (ui_state.has_longitudinal_control and sm.recv_frame['modelV2'] >= ui_state.started_frame and sm.alive['modelV2'] and
                len(model.velocity.x) > 0)
    if not has_plan:
      self._below_max = False
      self._target_filter.initialized = False
      return

    # same target as before: the end of the plan, a bit smoothed
    target_speed_ms = min(self._target_filter.update(max(0.0, model.velocity.x[-1])), self._max_speed_ms)
    target_speed = round(target_speed_ms * conversion)
    gap = self._max_speed - target_speed
    was_below_max = self._below_max
    self._below_max = gap >= (HIDE_GAP if self._below_max else SHOW_GAP)

    if was_below_max and not self._below_max:
      # back to MAX: roll the number up and raise the plan to the MAX line before it goes away
      self._release_time = now
      self._plan_speeds = [self._max_speed_ms] * len(self._plan_speeds)
      self._plan_accels = [0.0] * len(self._plan_accels)

    if not self._below_max:
      return

    # steps of 5 with some hysteresis, so the number only rolls when the target really changes
    if not was_below_max or abs(target_speed - self._target_speed) > TARGET_STEP / 2 + TARGET_HYSTERESIS:
      self._target_speed = min(round(target_speed / TARGET_STEP) * TARGET_STEP, self._max_speed)
    self._plan_speeds = [min(speed, self._max_speed_ms) for speed in model.velocity.x]
    self._plan_accels = list(model.acceleration.x)

  def _render(self, rect: rl.Rectangle) -> None:
    now = rl.get_time()
    max_recent = now - self._max_changed_time < SET_SPEED_PERSISTENCE
    releasing = now - self._release_time < RELEASE_HOLD
    visible = self._active and self._can_draw and (self._below_max or max_recent or releasing)

    value = self._target_speed if self._below_max else self._max_speed
    digits = len(str(value))
    if visible and not self._was_visible and self._presence.x < 0.05:
      # start fresh: digits roll in and the plan draws in from the left
      self._number.reset(digits)
      self._chart_speeds = list(self._plan_speeds)
      self._chart_accels = list(self._plan_accels)
      self._chart_floor.snap(self._get_chart_floor())
      self._chart_presence.snap(0.0)
      self._reveal_time = now
    self._was_visible = visible

    presence = self._presence.update(float(visible))
    if not visible and presence < 1e-2:
      self._presence.snap(0.0)
      return

    alpha = clamp(presence)
    chart_presence = clamp(self._chart_presence.update(float(self._below_max and visible)))
    right = rect.x + rect.width - MARGIN_X
    # drop in from a bit above
    top = rect.y - 10 * (1 - alpha)

    # soft shade along the top so the white reads on a bright sky
    shade = rl.Color(0, 0, 0, round(255 * SHADOW_ALPHA * alpha))
    rl.draw_rectangle_gradient_v(round(rect.x), round(rect.y), round(rect.width), SHADOW_HEIGHT, shade, rl.BLANK)

    self._number.set_value(value)
    number_width = self._number.update_width(digits)
    self._number.draw(right, top + NUMBER_Y, alpha)

    # under the number it is MAX, with the chart it names the dashed line
    label = f"{tr('MAX')} {self._max_speed}" if self._below_max else tr("MAX")
    if label != self._label:
      self._prev_label = self._label
      self._label = label
      self._label_change_time = now
    # MAX is bright after it changes, and less bright as a chart label
    label_alpha = self._label_alpha.update(1.0 if max_recent or not self._below_max else 0.75) * alpha
    progress = ease_out_cubic(clamp((now - self._label_change_time) / ROLL_DURATION))
    chart_left = rect.x + CHART_LEFT
    # pop the MAX label when MAX changes
    pop = 1 + LABEL_POP * math.sin(math.pi * clamp((now - self._max_changed_time) / LABEL_POP_TIME))
    self._draw_label(self._label, right, chart_left, top, label_alpha * fade_in(progress), pop)
    if progress < 1:
      self._draw_label(self._prev_label, right, chart_left, top, label_alpha * fade_out(progress), 1.0)

    if chart_presence > 1e-2:
      chart = rl.Rectangle(chart_left, top + CHART_TOP, right - number_width - CHART_NUMBER_GAP - chart_left, CHART_HEIGHT)
      self._draw_chart(chart, alpha * chart_presence, now)

  def _draw_label(self, text: str, right: float, chart_left: float, top: float, alpha: float, scale: float) -> None:
    if len(text) == 0 or alpha < 1e-2:
      return

    font_size = LABEL_SIZE * scale
    size = measure_text_cached(self._font_label, text, round(font_size))
    if text == tr("MAX"):
      # alone, MAX goes under the number
      position = rl.Vector2(round(right - size.x), top + LABEL_Y)
    else:
      # with a value, it sits on the left end of the dashed MAX line
      position = rl.Vector2(chart_left, top + CHART_TOP - CHART_LABEL_GAP - size.y)
    rl.draw_text_ex(self._font_label, text, position, round(font_size), 0, white(alpha))

  def _get_chart_floor(self) -> float:
    # MAX is the top of the chart, show at least CHART_MIN_RANGE below it
    lowest = min(self._plan_speeds, default=self._max_speed_ms)
    return self._max_speed_ms - max(CHART_MIN_RANGE, (self._max_speed_ms - lowest) * 1.15)

  def _draw_chart(self, chart: rl.Rectangle, alpha: float, now: float) -> None:
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

    # dashed MAX line along the top
    dash_x = chart.x
    while dash_x < chart.x + chart.width:
      dash_end = min(dash_x + DASH_LENGTH, chart.x + chart.width)
      rl.draw_line_ex(rl.Vector2(dash_x, chart.y), rl.Vector2(dash_end, chart.y), DASH_WIDTH, white(alpha * 0.6))
      dash_x += DASH_LENGTH + DASH_GAP

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
                                   accel_color(self._chart_accels[i], alpha * 0.35), accel_color(self._chart_accels[i], 0.0))
      rl.draw_line_ex(start, end, LINE_WIDTH, color)
      rl.draw_circle_v(end, LINE_WIDTH / 2, color)

    # dot with a soft glow at the end of the plan, that is the big number
    rl.draw_circle_gradient(head, DOT_RADIUS * 2.5, white(alpha * 0.35), rl.BLANK)
    rl.draw_circle_v(head, DOT_RADIUS, white(alpha))
