import math
import raylib_nb as rl
from enum import IntEnum
from openpilot.cereal import log
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget

LongitudinalPlanSource = log.LongitudinalPlan.LongitudinalPlanSource

SET_SPEED_NA = 255

# layout
HEIGHT = 76
MARGIN = 12
PADDING = 20
COLUMN_GAP = 10
TRACK_HEIGHT = 4
NUMBER_SIZE = 52
LABEL_SIZE = 23
MAX_SIZE = 19

# when to show, in display units (mph or km/h)
SHOW_GAP = 5
HIDE_GAP = 2
TARGET_STEP = 5
TARGET_HYSTERESIS = 1.0
LIMITED_HOLD = 1.0  # s, the plan source can flicker to cruise for a short time
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
  return clamp((progress - 0.25) / 0.75)


def white(alpha: float) -> rl.Color:
  return rl.Color(255, 255, 255, round(255 * clamp(alpha)))


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
    self._limited_time = -LIMITED_HOLD
    self._fill_ratio = 1.0
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
    self._fill = Spring(0.0, 0.6, 0.9)

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

    plan_source = sm['longitudinalPlan'].longitudinalPlanSource
    if plan_source != LongitudinalPlanSource.cruise:
      self._limited_time = now
    limited = now - self._limited_time < LIMITED_HOLD

    self._max_speed = round(max_speed_ms * conversion)
    target_speed = round(target_speed_ms * conversion)
    gap = self._max_speed - target_speed
    was_shown = self._shown
    if self._shown:
      self._shown = limited and gap >= HIDE_GAP
    else:
      self._shown = limited and gap >= SHOW_GAP

    if was_shown and not self._shown:
      # back to MAX: roll the number up and fill the track before the capsule goes away
      self._release_time = now
      self._target_speed = self._max_speed
      self._fill_ratio = 1.0
      self._set_reason(Reason.RESUMING, now)

    if not self._shown:
      return

    # steps of 5 with some hysteresis, so the number only rolls when the target really changes
    if not was_shown or abs(target_speed - self._target_speed) > TARGET_STEP / 2 + TARGET_HYSTERESIS:
      self._target_speed = min(round(target_speed / TARGET_STEP) * TARGET_STEP, self._max_speed)
    self._fill_ratio = target_speed_ms / max(max_speed_ms, 1.0)

    reason = self._get_reason(target_speed_ms, max_speed_ms, plan_source)
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
    label_width = max(measure_text_cached(self._font_label, label, LABEL_SIZE).x,
                      measure_text_cached(self._font_max, max_text, MAX_SIZE).x)

    if visible and not self._was_visible and self._presence.x < 0.05:
      # start fresh: digits roll in and the track drains from full
      self._number.reset(digits)
      self._fill.snap(1.0)
      self._width.snap(PADDING + self._number.width.x + COLUMN_GAP + label_width + PADDING)
    self._was_visible = visible

    presence = self._presence.update(float(visible))
    if not visible and presence < 1e-2:
      self._presence.snap(0.0)
      return

    number_width = self._number.update_width(digits)
    full_width = self._width.update(PADDING + number_width + COLUMN_GAP + label_width + PADDING)

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
    fill = clamp(self._fill.update(self._fill_ratio))

    # content stays at its full size layout, the capsule grows around it
    x = right - full_width + PADDING
    y = rect.y + MARGIN

    self._number.draw(x + number_width, y + 5, content_alpha)

    label_x = x + number_width + COLUMN_GAP
    progress = ease_out_cubic(clamp((now - self._reason_change_time) / ROLL_DURATION))
    if progress < 1:
      prev_label = reason_text(self._prev_reason)
      rl.draw_text_ex(self._font_label, prev_label, rl.Vector2(label_x, y + 12 - 6 * progress), LABEL_SIZE, 0,
                      white(content_alpha * fade_out(progress)))
    rl.draw_text_ex(self._font_label, label, rl.Vector2(label_x, y + 12 + 6 * (1 - progress)), LABEL_SIZE, 0,
                    white(content_alpha * fade_in(progress)))
    rl.draw_text_ex(self._font_max, max_text, rl.Vector2(label_x, y + 36), MAX_SIZE, 0, white(content_alpha * 0.55))

    # track from zero to MAX, filled up to the target speed
    track = rl.Rectangle(x, y + HEIGHT - 16, full_width - PADDING * 2, TRACK_HEIGHT)
    rl.draw_rectangle_rounded(track, 1.0, 8, white(content_alpha * 0.2))
    fill_rect = rl.Rectangle(track.x, track.y, max(TRACK_HEIGHT, track.width * fill), TRACK_HEIGHT)
    rl.draw_rectangle_rounded(fill_rect, 1.0, 8, white(content_alpha * 0.95))

  @staticmethod
  def _draw_capsule(capsule: rl.Rectangle, alpha: float) -> None:
    # soft shadow, then the capsule and a faint edge
    for spread, shadow_alpha in ((8, 0.08), (4, 0.12)):
      shadow = rl.Rectangle(capsule.x - spread, capsule.y - spread + 2, capsule.width + spread * 2, capsule.height + spread * 2)
      rl.draw_rectangle_rounded(shadow, 1.0, 16, rl.Color(0, 0, 0, round(255 * shadow_alpha * alpha)))
    rl.draw_rectangle_rounded(capsule, 1.0, 16, rl.Color(0, 0, 0, round(255 * 0.62 * alpha)))
    rl.draw_rectangle_rounded_lines_ex(capsule, 1.0, 16, 1.5, white(0.1 * alpha))
