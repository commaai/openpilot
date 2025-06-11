import pyray as rl


class MouseState:
  def __init__(self):
    self._position: rl.Vector2 = rl.Vector2(0, 0, 0, 0)
    self._released: bool = False
    self._consumed: bool = False
    self._pressed_pos: rl.Vector2 = rl.Vector2(0, 0, 0, 0)
    self._is_down: bool = False

  def update(self):
    """Update the mouse state for the current frame."""
    self._consumed = False
    self._position = rl.get_mouse_position()
    self._pressed = rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT)
    self._released = rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT)
    self._is_down = rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT)

    if self._pressed:
      self._pressed_pos = self._position

  @property
  def position(self) -> rl.Vector2:
    return self._position

  @property
  def pressed(self) -> bool:
    return self._pressed and not self._consumed

  @property
  def released(self) -> bool:
    return self._released and not self._consumed

  @property
  def consumed(self) -> bool:
    return self._consumed

  @consumed.setter
  def consumed(self, value: bool):
    self._consumed = value

  def check_pressed(self, rect: rl.Rectangle) -> bool:
    """Check if mouse was pressed inside the given rectangle."""
    return self.pressed and rl.check_collision_point_rec(self._position, rect)

  def check_released(self, rect: rl.Rectangle) -> bool:
    """Check if mouse was released inside the given rectangle."""
    return self.released and rl.check_collision_point_rec(self._position, rect)

  def check_clicked(self, rect: rl.Rectangle) -> bool:
    """
    Check for a click inside the rectangle:
    Mouse was pressed and released inside the same rect.
    """
    return (
      self.released
      and rl.check_collision_point_rec(self._pressed_pos, rect)
      and rl.check_collision_point_rec(self._position, rect)
    )

  def check_down(self, rect: rl.Rectangle) -> bool:
    """Check if mouse is currently held down after being pressed in the rectangle."""
    return self._is_down and not self._consumed and rl.check_collision_point_rec(self._pressed_pos, rect)
