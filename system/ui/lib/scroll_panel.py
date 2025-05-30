import pyray as rl
from enum import IntEnum

# Scroll constants for smooth scrolling behavior
MOUSE_WHEEL_SCROLL_SPEED = 30
INERTIA_FRICTION = 0.92        # The rate at which the inertia slows down
MIN_VELOCITY = 0.5             # Minimum velocity before stopping the inertia
DRAG_THRESHOLD = 5             # Pixels of movement to consider it a drag, not a click
BOUNCE_FACTOR = 0.2            # Elastic bounce when scrolling past boundaries
BOUNCE_RETURN_SPEED = 0.15     # How quickly it returns from the bounce
MAX_BOUNCE_DISTANCE = 150      # Maximum distance for bounce effect
FLICK_MULTIPLIER = 1.8         # Multiplier for flick gestures
VELOCITY_HISTORY_SIZE = 5      # Track velocity over multiple frames for smoother motion


class ScrollState(IntEnum):
  IDLE = 0
  DRAGGING_CONTENT = 1
  DRAGGING_SCROLLBAR = 2
  BOUNCING = 3


class GuiScrollPanel:
  def __init__(self, show_vertical_scroll_bar: bool = False):
    self._scroll_state: ScrollState = ScrollState.IDLE
    self._last_mouse_y: float = 0.0
    self._start_mouse_y: float = 0.0  # Track the initial mouse position for drag detection
    self._offset = rl.Vector2(0, 0)
    self._view = rl.Rectangle(0, 0, 0, 0)
    self._show_vertical_scroll_bar: bool = show_vertical_scroll_bar
    self._velocity_y = 0.0  # Velocity for inertia
    self._is_dragging: bool = False
    self._bounce_offset: float = 0.0
    self._last_frame_time = rl.get_time()
    self._velocity_history: list[float] = []
    self._last_drag_time: float = 0.0
    self._content_rect: rl.Rectangle | None = None
    self._bounds_rect: rl.Rectangle | None = None

  def handle_scroll(self, bounds: rl.Rectangle, content: rl.Rectangle) -> rl.Vector2:
    # Store rectangles for reference
    self._content_rect = content
    self._bounds_rect = bounds

    # Calculate time delta
    current_time = rl.get_time()
    delta_time = current_time - self._last_frame_time
    self._last_frame_time = current_time

    # Prevent large jumps
    delta_time = min(delta_time, 0.05)

    mouse_pos = rl.get_mouse_position()
    max_scroll_y = max(content.height - bounds.height, 0)

    # Start dragging on mouse press
    if rl.check_collision_point_rec(mouse_pos, bounds) and rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT):
      if self._scroll_state == ScrollState.IDLE or self._scroll_state == ScrollState.BOUNCING:
        self._scroll_state = ScrollState.DRAGGING_CONTENT
        if self._show_vertical_scroll_bar:
          scrollbar_width = rl.gui_get_style(rl.GuiControl.LISTVIEW, rl.GuiListViewProperty.SCROLLBAR_WIDTH)
          scrollbar_x = bounds.x + bounds.width - scrollbar_width
          if mouse_pos.x >= scrollbar_x:
            self._scroll_state = ScrollState.DRAGGING_SCROLLBAR

        self._last_mouse_y = mouse_pos.y
        self._start_mouse_y = mouse_pos.y
        self._last_drag_time = current_time
        self._velocity_history = []
        self._velocity_y = 0.0
        self._bounce_offset = 0.0
        self._is_dragging = False

    # Handle active dragging
    if self._scroll_state == ScrollState.DRAGGING_CONTENT or self._scroll_state == ScrollState.DRAGGING_SCROLLBAR:
      if rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT):
        delta_y = mouse_pos.y - self._last_mouse_y

        # Track velocity for inertia
        time_since_last_drag = current_time - self._last_drag_time
        if time_since_last_drag > 0:
          drag_velocity = delta_y / time_since_last_drag / 60.0
          self._velocity_history.append(drag_velocity)

          if len(self._velocity_history) > VELOCITY_HISTORY_SIZE:
            self._velocity_history.pop(0)

        self._last_drag_time = current_time

        # Detect actual dragging
        total_drag = abs(mouse_pos.y - self._start_mouse_y)
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

        self._last_mouse_y = mouse_pos.y

      elif rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT):
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

    return self._offset

  def is_click_valid(self) -> bool:
    # Check if this is a click rather than a drag
    return (
      self._scroll_state == ScrollState.IDLE
      and not self._is_dragging
      and rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT)
    )

  def get_normalized_scroll_position(self) -> float:
    """Returns the current scroll position as a value from 0.0 to 1.0"""
    if not self._content_rect or not self._bounds_rect:
      return 0.0

    max_scroll_y = max(self._content_rect.height - self._bounds_rect.height, 0)
    if max_scroll_y == 0:
      return 0.0

    normalized = -self._offset.y / max_scroll_y
    return max(0.0, min(1.0, normalized))
