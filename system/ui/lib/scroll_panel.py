import time
import math
import pyray as rl
from collections import deque
from enum import IntEnum
from openpilot.system.ui.lib.application import gui_app, MouseEvent, MousePos, DEFAULT_FPS
from openpilot.common.filter_simple import FirstOrderFilter

# Scroll constants for smooth scrolling behavior
MOUSE_WHEEL_SCROLL_SPEED = 30
INERTIA_FRICTION = 0.92        # The rate at which the inertia slows down
# MIN_VELOCITY = 0.5             # Minimum velocity before stopping the inertia
MIN_VELOCITY = 2               # px/s, changes from auto scroll to steady state
MIN_VELOCITY_FOR_CLICKING = 2 * 60  # px/s, accepts clicks while auto scrolling below this velocity
DRAG_THRESHOLD = 12            # Pixels of movement to consider it a drag, not a click
BOUNCE_FACTOR = 0.2            # Elastic bounce when scrolling past boundaries
BOUNCE_RETURN_SPEED = 0.15     # How quickly it returns from the bounce
MAX_BOUNCE_DISTANCE = 150      # Maximum distance for bounce effect
FLICK_MULTIPLIER = 1.8         # Multiplier for flick gestures
VELOCITY_HISTORY_SIZE = 5      # Track velocity over multiple frames for smoother motion
BOUNCE_RETURN_RATE = 5


class ScrollState(IntEnum):
  IDLE = 0
  DRAGGING_CONTENT = 1
  # DRAGGING_SCROLLBAR = 2
  # BOUNCING = 3


class GuiScrollPanel:
  def __init__(self, show_vertical_scroll_bar: bool = False):
    # TODO: remove ability to scroll x and y, we only need one and it simplifies a lot of things
    self._scroll_state: ScrollState = ScrollState.IDLE
    self._last_mouse_y: float = 0.0
    self._start_mouse_y: float = 0.0  # Track the initial mouse position for drag detection
    self._offset_filter_y = FirstOrderFilter(0.0, 0.1, 1 / DEFAULT_FPS)
    # self._offset = rl.Vector2(0, 0)
    self._view = rl.Rectangle(0, 0, 0, 0)
    self._show_vertical_scroll_bar: bool = show_vertical_scroll_bar
    self._velocity_y = 0.0  # Velocity for inertia
    self._velocity_filter_y = FirstOrderFilter(0.0, 0.05, 1 / DEFAULT_FPS)  # TODO: raise rc?
    # self._is_dragging: bool = False
    self._bounce_offset: float = 0.0
    self._velocity_history: deque[float] = deque(maxlen=VELOCITY_HISTORY_SIZE)
    self._last_drag_time: float = 0.0

  def update(self, bounds: rl.Rectangle, content: rl.Rectangle) -> float:
    # print('state', self._scroll_state)
    # TODO: HACK: this class is driven by mouse events, so we need to ensure we have at least one event to process
    print('mouse events', len(gui_app.mouse_events))
    for mouse_event in gui_app.mouse_events:  # or [MouseEvent(MousePos(0, 0), 0, False, False, False, time.monotonic())]:
      if mouse_event.slot == 0:
        self._handle_mouse_event(mouse_event, bounds, content)

    self._update_state(bounds, content)

    print('offset', self._offset_filter_y.x, 'vel', self._velocity_filter_y.x)

    return self._offset_filter_y.x

  def _update_state(self, bounds: rl.Rectangle, content: rl.Rectangle):
    rl.draw_rectangle_lines(0, 0, abs(int(self._velocity_filter_y.x)), 10, rl.RED)

    max_scroll_distance = max(0, content.height - bounds.height)
    if self._scroll_state == ScrollState.IDLE:
      above_bounds, below_bounds = self._check_bounds(bounds, content)
      print('above', above_bounds, 'below', below_bounds)

      # Decay velocity when idle
      if abs(self._velocity_filter_y.x) > MIN_VELOCITY:
        # self._offset.y += self._velocity_filter_y.x / DEFAULT_FPS
        # self._velocity_filter_y.update(0)
        # Faster decay if bouncing back from out of bounds
        friction = math.exp(-BOUNCE_RETURN_RATE * 1 / DEFAULT_FPS)
        self._velocity_filter_y.x *= friction ** 2 if (above_bounds or below_bounds) else friction
      else:
        self._velocity_filter_y.x = 0.0
      # self._offset_filter_y.x = self._offset.y

      if above_bounds or below_bounds:
        # if abs(self._velocity_filter_y.x) > MIN_VELOCITY:
        #   # self._offset_filter_y.x += self._velocity_filter_y.x / DEFAULT_FPS
        #   # self._velocity_filter_y.update(0)
        #   self._velocity_filter_y.x *= INERTIA_FRICTION
        #   self._offset_filter_y.x = self._offset_filter_y.x
        # else:
        if above_bounds:
          # self._offset_filter_y.x = self._offset_filter_y.update(0)
          self._offset_filter_y.update(0)
        else:
          # self._offset_filter_y.x = self._offset_filter_y.update(-max_scroll_distance)
          self._offset_filter_y.update(-max_scroll_distance)

      self._offset_filter_y.x += self._velocity_filter_y.x / DEFAULT_FPS

      # else:
        # if abs(self._velocity_filter_y.x) > MIN_VELOCITY:
        #   # self._offset_filter_y.x += self._velocity_filter_y.x / DEFAULT_FPS
        #   # self._velocity_filter_y.update(0)
        #   self._velocity_filter_y.x *= INERTIA_FRICTION
        # else:
        #   self._velocity_filter_y.x = 0.0
        # self._offset_filter_y.x = self._offset_filter_y.x

      # self._offset_filter_y.x += self._velocity_filter_y.x / DEFAULT_FPS

    if abs(self._offset_filter_y.x) < 1e-3:
      self._offset_filter_y.x = 0.0
    elif abs(self._offset_filter_y.x + max_scroll_distance) < 1e-3:
      self._offset_filter_y.x = -max_scroll_distance

  def _handle_mouse_event(self, mouse_event: MouseEvent, bounds: rl.Rectangle, content: rl.Rectangle):
    if self._scroll_state == ScrollState.IDLE:
      if mouse_event.left_pressed:
        self._start_mouse_y = mouse_event.pos.y

      if mouse_event.left_down:
        if abs(mouse_event.pos.y - self._start_mouse_y) > DRAG_THRESHOLD:
          self._scroll_state = ScrollState.DRAGGING_CONTENT
          # Start velocity at initial measurement for more immediate response
          self._velocity_filter_y.initialized = False
          # TODO: minimum change if y to start dragging
          # self._is_dragging = True

    elif self._scroll_state == ScrollState.DRAGGING_CONTENT:
      if mouse_event.left_released:
        self._scroll_state = ScrollState.IDLE
        # self._is_dragging = False
        # self._offset_filter_y.x = self._offset.y
      else:
        delta_y = mouse_event.pos.y - self._last_mouse_y
        above_bounds, below_bounds = self._check_bounds(bounds, content)
        # TODO: should it be anytime? match ios?
        # if above_bounds and delta_y > 0 or below_bounds and delta_y < 0:
        #   delta_y /= 2
        # Rubber banding effect when out of bands
        if above_bounds or below_bounds:
          delta_y /= 2

        # print('offset', self._offset.y, 'bounds', bounds.height, 'content', content.height)

        self._offset_filter_y.x += delta_y
        # self._offset_filter_y.x = self._offset.y
        # self._offset_filter_y.initialized = False

        # Track velocity for inertia
        dt = mouse_event.t - self._last_drag_time
        # print('dt', 'dt2', dt)
        if dt > 0:
          drag_velocity = delta_y / dt
          self._velocity_filter_y.update(drag_velocity)

        # TODO: just store last mouse event!
    self._last_drag_time = mouse_event.t
    self._last_mouse_y = mouse_event.pos.y

  def _handle_mouse_event_old(self, mouse_event: MouseEvent, bounds: rl.Rectangle, content: rl.Rectangle):
    max_scroll_y = max(content.height - bounds.height, 0)

    # Start dragging on mouse press
    if rl.check_collision_point_rec(mouse_event.pos, bounds) and mouse_event.left_pressed:
      if self._scroll_state == ScrollState.IDLE or self._scroll_state == ScrollState.BOUNCING:
        self._scroll_state = ScrollState.DRAGGING_CONTENT
        if self._show_vertical_scroll_bar:
          scrollbar_width = rl.gui_get_style(rl.GuiControl.LISTVIEW, rl.GuiListViewProperty.SCROLLBAR_WIDTH)
          scrollbar_x = bounds.x + bounds.width - scrollbar_width
          if mouse_event.pos.x >= scrollbar_x:
            self._scroll_state = ScrollState.DRAGGING_SCROLLBAR

        # TODO: hacky
        # when clicking while moving, go straight into dragging
        # self._is_dragging = abs(self._velocity_y) > MIN_VELOCITY
        self._last_mouse_y = mouse_event.pos.y
        self._start_mouse_y = mouse_event.pos.y
        self._last_drag_time = mouse_event.t
        self._velocity_history.clear()
        self._velocity_y = 0.0
        self._bounce_offset = 0.0

    # Handle active dragging
    if self._scroll_state == ScrollState.DRAGGING_CONTENT or self._scroll_state == ScrollState.DRAGGING_SCROLLBAR:
      if mouse_event.left_down:
        delta_y = mouse_event.pos.y - self._last_mouse_y

        # Track velocity for inertia
        time_since_last_drag = mouse_event.t - self._last_drag_time
        if time_since_last_drag > 0:
          # TODO: HACK: /2 since we usually get two touch events per frame
          drag_velocity = delta_y / time_since_last_drag / 60.0 / 2  # TODO: shouldn't be hardcoded
          self._velocity_history.append(drag_velocity)

        self._last_drag_time = mouse_event.t

        # Detect actual dragging
        total_drag = abs(mouse_event.pos.y - self._start_mouse_y)
        if total_drag > DRAG_THRESHOLD:
          self._is_dragging = True

        if self._scroll_state == ScrollState.DRAGGING_CONTENT:
          # Add resistance at boundaries
          if (self._offset.y > 0 and delta_y > 0) or (self._offset.y < -max_scroll_y and delta_y < 0):
            delta_y *= BOUNCE_FACTOR

          self._offset.y += delta_y
        elif self._scroll_state == ScrollState.DRAGGING_SCROLLBAR:
          scroll_ratio = content.height / bounds.height
          self._offset.y -= delta_y * scroll_ratio

        self._last_mouse_y = mouse_event.pos.y

      elif mouse_event.left_released:
        # Calculate flick velocity
        if self._velocity_history:
          total_weight = 0
          weighted_velocity = 0.0

          for i, v in enumerate(self._velocity_history):
            weight = i + 1
            weighted_velocity += v * weight
            total_weight += weight

          if total_weight > 0:
            avg_velocity = weighted_velocity / total_weight
            self._velocity_y = avg_velocity * FLICK_MULTIPLIER

        # Check bounds
        if self._offset.y > 0 or self._offset.y < -max_scroll_y:
          self._scroll_state = ScrollState.BOUNCING
        else:
          self._scroll_state = ScrollState.IDLE

    # Handle mouse wheel
    wheel_move = rl.get_mouse_wheel_move()
    if wheel_move != 0:
      self._velocity_y = 0.0

      if self._show_vertical_scroll_bar:
        self._offset.y += wheel_move * (MOUSE_WHEEL_SCROLL_SPEED - 20)
        rl.gui_scroll_panel(bounds, rl.ffi.NULL, content, self._offset, self._view)
      else:
        self._offset.y += wheel_move * MOUSE_WHEEL_SCROLL_SPEED

      if self._offset.y > 0 or self._offset.y < -max_scroll_y:
        self._scroll_state = ScrollState.BOUNCING

    # Apply inertia (continue scrolling after mouse release)
    if self._scroll_state == ScrollState.IDLE:
      if abs(self._velocity_y) > MIN_VELOCITY:
        self._offset.y += self._velocity_y
        self._velocity_y *= INERTIA_FRICTION

        if self._offset.y > 0 or self._offset.y < -max_scroll_y:
          self._scroll_state = ScrollState.BOUNCING
      else:
        self._velocity_y = 0.0

    # Handle bouncing effect
    elif self._scroll_state == ScrollState.BOUNCING:
      target_y = 0.0
      if self._offset.y < -max_scroll_y:
        target_y = -max_scroll_y

      distance = target_y - self._offset.y
      bounce_step = distance * BOUNCE_RETURN_SPEED
      self._offset.y += bounce_step
      self._velocity_y *= INERTIA_FRICTION * 0.8

      if abs(distance) < 0.5 and abs(self._velocity_y) < MIN_VELOCITY:
        self._offset.y = target_y
        self._velocity_y = 0.0
        self._scroll_state = ScrollState.IDLE

    # Limit bounce distance
    if self._scroll_state != ScrollState.DRAGGING_CONTENT:
      if self._offset.y > MAX_BOUNCE_DISTANCE:
        self._offset.y = MAX_BOUNCE_DISTANCE
      elif self._offset.y < -(max_scroll_y + MAX_BOUNCE_DISTANCE):
        self._offset.y = -(max_scroll_y + MAX_BOUNCE_DISTANCE)

  def _check_bounds(self, bounds: rl.Rectangle, content: rl.Rectangle) -> tuple[bool, bool]:
    max_scroll_distance = max(0, content.height - bounds.height)
    above_bounds = self._offset_filter_y.x > 0
    below_bounds = self._offset_filter_y.x < -max_scroll_distance
    return above_bounds, below_bounds

  def is_touch_valid(self):
    return self._scroll_state == ScrollState.IDLE and abs(self._velocity_filter_y.x) < MIN_VELOCITY_FOR_CLICKING
