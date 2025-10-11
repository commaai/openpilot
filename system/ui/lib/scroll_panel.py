import math
import pyray as rl
from enum import IntEnum
from openpilot.system.ui.lib.application import gui_app, MouseEvent
from openpilot.common.filter_simple import FirstOrderFilter

# Scroll constants for smooth scrolling behavior
MOUSE_WHEEL_SCROLL_SPEED = 50
BOUNCE_RETURN_RATE = 5              # ~0.92 at 60fps
MIN_VELOCITY = 2                    # px/s, changes from auto scroll to steady state
MIN_VELOCITY_FOR_CLICKING = 2 * 60  # px/s, accepts clicks while auto scrolling below this velocity
DRAG_THRESHOLD = 12                 # pixels of movement to consider it a drag, not a click

DEBUG = False


class ScrollState(IntEnum):
  IDLE = 0              # Not dragging, content may be bouncing or scrolling with inertia
  DRAGGING_CONTENT = 1  # User is actively dragging the content


class GuiScrollPanel:
  def __init__(self):
    self._scroll_state: ScrollState = ScrollState.IDLE
    self._last_mouse_y: float = 0.0
    self._start_mouse_y: float = 0.0  # Track the initial mouse position for drag detection
    self._offset_filter_y = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)
    self._velocity_filter_y = FirstOrderFilter(0.0, 0.05, 1 / gui_app.target_fps)
    self._last_drag_time: float = 0.0

  def update(self, bounds: rl.Rectangle, content: rl.Rectangle) -> float:
    for mouse_event in gui_app.mouse_events:
      if mouse_event.slot == 0:
        self._handle_mouse_event(mouse_event, bounds, content)

    self._update_state(bounds, content)

    return float(self._offset_filter_y.x)

  def _update_state(self, bounds: rl.Rectangle, content: rl.Rectangle):
    if DEBUG:
      rl.draw_rectangle_lines(0, 0, abs(int(self._velocity_filter_y.x)), 10, rl.RED)

    # Handle mouse wheel
    self._offset_filter_y.x += rl.get_mouse_wheel_move() * MOUSE_WHEEL_SCROLL_SPEED

    max_scroll_distance = max(0, content.height - bounds.height)
    if self._scroll_state == ScrollState.IDLE:
      above_bounds, below_bounds = self._check_bounds(bounds, content)

      # Decay velocity when idle
      if abs(self._velocity_filter_y.x) > MIN_VELOCITY:
        # Faster decay if bouncing back from out of bounds
        friction = math.exp(-BOUNCE_RETURN_RATE * 1 / gui_app.target_fps)
        self._velocity_filter_y.x *= friction ** 2 if (above_bounds or below_bounds) else friction
      else:
        self._velocity_filter_y.x = 0.0

      if above_bounds or below_bounds:
        if above_bounds:
          self._offset_filter_y.update(0)
        else:
          self._offset_filter_y.update(-max_scroll_distance)

      self._offset_filter_y.x += self._velocity_filter_y.x / gui_app.target_fps

    elif self._scroll_state == ScrollState.DRAGGING_CONTENT:
      # Mouse not moving, decay velocity
      if not len(gui_app.mouse_events):
        self._velocity_filter_y.update(0.0)

    # Settle to exact bounds
    if abs(self._offset_filter_y.x) < 1e-2:
      self._offset_filter_y.x = 0.0
    elif abs(self._offset_filter_y.x + max_scroll_distance) < 1e-2:
      self._offset_filter_y.x = -max_scroll_distance

  def _handle_mouse_event(self, mouse_event: MouseEvent, bounds: rl.Rectangle, content: rl.Rectangle):
    if self._scroll_state == ScrollState.IDLE:
      if rl.check_collision_point_rec(mouse_event.pos, bounds):
        if mouse_event.left_pressed:
          self._start_mouse_y = mouse_event.pos.y
          # Interrupt scrolling with new drag
          # TODO: stop scrolling with any tap, need to fix is_touch_valid
          if abs(self._velocity_filter_y.x) > MIN_VELOCITY_FOR_CLICKING:
            self._scroll_state = ScrollState.DRAGGING_CONTENT
            # Start velocity at initial measurement for more immediate response
            self._velocity_filter_y.initialized = False

        if mouse_event.left_down:
          if abs(mouse_event.pos.y - self._start_mouse_y) > DRAG_THRESHOLD:
            self._scroll_state = ScrollState.DRAGGING_CONTENT
            # Start velocity at initial measurement for more immediate response
            self._velocity_filter_y.initialized = False

    elif self._scroll_state == ScrollState.DRAGGING_CONTENT:
      if mouse_event.left_released:
        self._scroll_state = ScrollState.IDLE
      else:
        delta_y = mouse_event.pos.y - self._last_mouse_y
        above_bounds, below_bounds = self._check_bounds(bounds, content)
        # Rubber banding effect when out of bands
        if above_bounds or below_bounds:
          delta_y /= 3

        self._offset_filter_y.x += delta_y

        # Track velocity for inertia
        dt = mouse_event.t - self._last_drag_time
        if dt > 0:
          drag_velocity = delta_y / dt
          self._velocity_filter_y.update(drag_velocity)

        # TODO: just store last mouse event!
    self._last_drag_time = mouse_event.t
    self._last_mouse_y = mouse_event.pos.y

  def _check_bounds(self, bounds: rl.Rectangle, content: rl.Rectangle) -> tuple[bool, bool]:
    max_scroll_distance = max(0, content.height - bounds.height)
    above_bounds = self._offset_filter_y.x > 0
    below_bounds = self._offset_filter_y.x < -max_scroll_distance
    return above_bounds, below_bounds

  def is_touch_valid(self):
    return self._scroll_state == ScrollState.IDLE and abs(self._velocity_filter_y.x) < MIN_VELOCITY_FOR_CLICKING

  def set_offset(self, position: float) -> None:
    self._offset_filter_y.x = position
    self._velocity_filter_y.x = 0.0
    self._scroll_state = ScrollState.IDLE
