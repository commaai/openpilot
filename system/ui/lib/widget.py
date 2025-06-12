import abc
import pyray as rl
from enum import IntEnum


class DialogResult(IntEnum):
  CANCEL = 0
  CONFIRM = 1
  NO_ACTION = -1


class Widget(abc.ABC):
  def __init__(self):
    self._rect = rl.Rectangle(0, 0, 0, 0)
    self._is_pressed = False

  def set_rect(self, rect: rl.Rectangle):
    self._rect = rect

  @property
  def height2(self) -> int:
    return self._rect.height if self._rect else 0

  @property
  def width2(self) -> int:
    return self._rect.width if self._rect else 0

  @property
  def x2(self) -> int:
    return self._rect.x if self._rect else 0

  @property
  def y2(self) -> int:
    return self._rect.y if self._rect else 0

  def render(self, rect: type(rl.Rectangle) | None = None) -> bool | int | None:
    if rect is not None:
      self.set_rect(rect)

    ret = self._render(self._rect)

    # Keep track of whether mouse down started within the widget's rectangle
    mouse_pos = rl.get_mouse_position()
    if rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT):
      if rl.check_collision_point_rec(mouse_pos, self._rect):
        self._is_pressed = True

    if rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT):
      if self._is_pressed and rl.check_collision_point_rec(mouse_pos, self._rect):
        self._handle_mouse_release(mouse_pos)
      self._is_pressed = False

    return ret

  @abc.abstractmethod
  def _render(self, rect: type(rl.Rectangle) | None):  # TODO: remove this rect
    """Render the widget within the given rectangle."""

  def _handle_mouse_release(self, mouse_pos: rl.Vector2) -> bool:
    """Handle mouse release events, if applicable."""
    return False
