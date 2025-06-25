import abc
import pyray as rl
from enum import IntEnum
from collections.abc import Callable


class DialogResult(IntEnum):
  CANCEL = 0
  CONFIRM = 1
  NO_ACTION = -1


class Widget(abc.ABC):
  def __init__(self):
    self._rect: rl.Rectangle = rl.Rectangle(0, 0, 0, 0)
    self._is_pressed = False
    self._is_visible: bool | Callable[[], bool] = True
    self._click_valid_callback: Callable[[], bool] | None = None
    self._touch_valid_callback: Callable[[], bool] | None = None

  def set_click_valid_callback(self, click_callback: Callable[[], bool],
                               touch_callback: Callable[[], bool]) -> None:
    """Set a callback to determine if the widget can be clicked."""
    self._click_valid_callback = click_callback
    self._touch_valid_callback = touch_callback

  def _touch_valid(self) -> bool:
    """Check if the widget can be touched."""
    return self._touch_valid_callback() if self._touch_valid_callback else True

  @property
  def rect(self) -> rl.Rectangle:
    return self._rect

  @property
  def is_visible(self) -> bool:
    return self._is_visible() if callable(self._is_visible) else self._is_visible

  def set_visible(self, visible: bool | Callable[[], bool]) -> None:
    self._is_visible = visible

  def set_rect(self, rect: rl.Rectangle) -> None:
    changed = (self._rect.x != rect.x or self._rect.y != rect.y or
               self._rect.width != rect.width or self._rect.height != rect.height)
    self._rect = rect
    if changed:
      self._update_layout_rects()

  def set_position(self, x: float, y: float) -> None:
    """Set the position of the widget."""
    changed = (self._rect.x != x or self._rect.y != y)
    self._rect.x, self._rect.y = x, y

  def render(self, rect: rl.Rectangle = None) -> bool | int | None:
    if rect is not None:
      self.set_rect(rect)

    self._update_state()

    if not self.is_visible:
      return None

    ret = self._render(self._rect)

    # Keep track of whether mouse down started within the widget's rectangle
    mouse_pos = rl.get_mouse_position()
    if rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT) and self._touch_valid():
      if rl.check_collision_point_rec(mouse_pos, self._rect):
        self._is_pressed = True

    elif not self._touch_valid():
      self._is_pressed = False

    elif rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT):
      if self._is_pressed and rl.check_collision_point_rec(mouse_pos, self._rect):
        self._handle_mouse_release(mouse_pos)
      self._is_pressed = False

    return ret

  @abc.abstractmethod
  def _render(self, rect: rl.Rectangle) -> bool | int | None:
    """Render the widget within the given rectangle."""

  def _update_state(self):
    """Optionally update the widget's non-layout state. This is called before rendering."""

  def _update_layout_rects(self) -> None:
    """Optionally update any layout rects on Widget rect change."""

  def _handle_mouse_release(self, mouse_pos: rl.Vector2) -> bool:
    """Optionally handle mouse release events."""
    return False
