import pyray as rl
from enum import IntEnum

MOUSE_WHEEL_SCROLL_SPEED = 30
INERTIA_FRICTION = 0.95  # The rate at which the inertia slows down
MIN_VELOCITY = 0.1  # Minimum velocity before stopping the inertia
DRAG_THRESHOLD = 5  # Pixels of movement to consider it a drag, not a click


class ScrollState(IntEnum):
  IDLE = 0
  DRAGGING_CONTENT = 1
  DRAGGING_SCROLLBAR = 2


class GuiScrollPanel:
  def __init__(self, show_vertical_scroll_bar: bool = False):
    self._scroll_state: ScrollState = ScrollState.IDLE
    self._last_mouse_y: float = 0.0
    self._start_mouse_y: float = 0.0  # Track the initial mouse position for drag detection
    self._offset = rl.Vector2(0, 0)
    self._view = rl.Rectangle(0, 0, 0, 0)
    self._show_vertical_scroll_bar: bool = show_vertical_scroll_bar
    self._velocity_y = 0.0  # Velocity for inertia
    self._is_dragging = False

  def handle_scroll(self, bounds: rl.Rectangle, content: rl.Rectangle) -> rl.Vector2:
    mouse_pos = rl.get_mouse_position()

    # Handle dragging logic
    if rl.check_collision_point_rec(mouse_pos, bounds) and rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT):
      if self._scroll_state == ScrollState.IDLE:
        self._scroll_state = ScrollState.DRAGGING_CONTENT
        if self._show_vertical_scroll_bar:
          scrollbar_width = rl.gui_get_style(rl.GuiControl.LISTVIEW, rl.GuiListViewProperty.SCROLLBAR_WIDTH)
          scrollbar_x = bounds.x + bounds.width - scrollbar_width
          if mouse_pos.x >= scrollbar_x:
            self._scroll_state = ScrollState.DRAGGING_SCROLLBAR

        self._last_mouse_y = mouse_pos.y
        self._start_mouse_y = mouse_pos.y  # Record starting position
        self._velocity_y = 0.0  # Reset velocity when drag starts
        self._is_dragging = False  # Reset dragging flag

    if self._scroll_state != ScrollState.IDLE:
      if rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT):
        delta_y = mouse_pos.y - self._last_mouse_y

        # Check if movement exceeds the drag threshold
        total_drag = abs(mouse_pos.y - self._start_mouse_y)
        if total_drag > DRAG_THRESHOLD:
          self._is_dragging = True

        if self._scroll_state == ScrollState.DRAGGING_CONTENT:
          self._offset.y += delta_y
        elif self._scroll_state == ScrollState.DRAGGING_SCROLLBAR:
          delta_y = -delta_y

        self._last_mouse_y = mouse_pos.y
        self._velocity_y = delta_y  # Update velocity during drag
      elif rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT):
        self._scroll_state = ScrollState.IDLE

    # Handle mouse wheel scrolling
    wheel_move = rl.get_mouse_wheel_move()
    if self._show_vertical_scroll_bar:
      self._offset.y += wheel_move * (MOUSE_WHEEL_SCROLL_SPEED - 20)
      rl.gui_scroll_panel(bounds, rl.ffi.NULL, content, self._offset, self._view)
    else:
      self._offset.y += wheel_move * MOUSE_WHEEL_SCROLL_SPEED

    # Apply inertia (continue scrolling after mouse release)
    if self._scroll_state == ScrollState.IDLE:
      self._offset.y += self._velocity_y
      self._velocity_y *= INERTIA_FRICTION  # Slow down velocity over time

      # Stop scrolling when velocity is low
      if abs(self._velocity_y) < MIN_VELOCITY:
        self._velocity_y = 0.0

    # Ensure scrolling doesn't go beyond bounds
    max_scroll_y = max(content.height - bounds.height, 0)
    self._offset.y = max(min(self._offset.y, 0), -max_scroll_y)

    return self._offset

  def is_click_valid(self) -> bool:
    return self._scroll_state == ScrollState.IDLE and not self._is_dragging and rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT)
