import pyray as rl
from cffi import FFI

MOUSE_WHEEL_SCROLL_SPEED = 30
INERTIA_FRICTION = 0.95  # The rate at which the inertia slows down
MIN_VELOCITY = 0.1  # Minimum velocity before stopping the inertia


class GuiScrollPanel:
  def __init__(self, bounds: rl.Rectangle, content: rl.Rectangle, show_vertical_scroll_bar: bool = False):
    self._dragging: bool = False
    self._last_mouse_y: float = 0.0
    self._bounds = bounds
    self._content = content
    self._scroll = rl.Vector2(0, 0)
    self._view = rl.Rectangle(0, 0, 0, 0)
    self._show_vertical_scroll_bar: bool = show_vertical_scroll_bar
    self._velocity_y = 0.0  # Velocity for inertia

  def handle_scroll(self) -> rl.Vector2:
    mouse_pos = rl.get_mouse_position()

    # Handle dragging logic
    if rl.check_collision_point_rec(mouse_pos, self._bounds) and rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT):
      if not self._dragging:
        self._dragging = True
        self._last_mouse_y = mouse_pos.y
        self._velocity_y = 0.0  # Reset velocity when drag starts

    if self._dragging:
      if rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT):
        delta_y = mouse_pos.y - self._last_mouse_y
        self._scroll.y += delta_y
        self._last_mouse_y = mouse_pos.y
        self._velocity_y = delta_y  # Update velocity during drag
      else:
        self._dragging = False

    # Handle mouse wheel scrolling
    wheel_move = rl.get_mouse_wheel_move()
    if self._show_vertical_scroll_bar:
      self._scroll.y += wheel_move * (MOUSE_WHEEL_SCROLL_SPEED - 20)
      rl.gui_scroll_panel(self._bounds, FFI().NULL, self._content, self._scroll, self._view)
    else:
      self._scroll.y += wheel_move * MOUSE_WHEEL_SCROLL_SPEED
      max_scroll_y = self._content.height - self._bounds.height
      self._scroll.y = max(min(self._scroll.y, 0), -max_scroll_y)

    # Apply inertia (continue scrolling after mouse release)
    if not self._dragging:
      self._scroll.y += self._velocity_y
      self._velocity_y *= INERTIA_FRICTION  # Slow down velocity over time

      # Stop scrolling when velocity is low
      if abs(self._velocity_y) < MIN_VELOCITY:
        self._velocity_y = 0.0

    # Ensure scrolling doesn't go beyond bounds
    max_scroll_y = max(self._content.height - self._bounds.height, 0)
    self._scroll.y = max(min(self._scroll.y, 0), -max_scroll_y)

    return self._scroll
