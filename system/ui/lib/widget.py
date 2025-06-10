import abc
import pyray as rl
from enum import IntEnum


class DialogResult(IntEnum):
  CANCEL = 0
  CONFIRM = 1
  NO_ACTION = -1


class Widget(abc.ABC):
  def __init__(self):
    self._is_pressed = False

  def render(self, rect: rl.Rectangle) -> bool | int | None:
    ret = self._render(rect)

    # Keep track of whether mouse down started within the widget's rectangle
    mouse_pos = rl.get_mouse_position()
    if rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT):
      if rl.check_collision_point_rec(mouse_pos, rect):
        self._is_pressed = True

    if rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT):
      if self._is_pressed and rl.check_collision_point_rec(mouse_pos, rect):
        self._handle_mouse_release(mouse_pos)
      self._is_pressed = False

    return ret

  @abc.abstractmethod
  def _render(self, rect: rl.Rectangle) -> bool | int | None:
    """Render the widget within the given rectangle."""

  def _handle_mouse_release(self, mouse_pos: rl.Vector2) -> bool:
    """Handle mouse release events, if applicable."""
    return False
