import abc
import pyray as rl
from enum import IntEnum
from collections.abc import Callable
from openpilot.system.ui.lib.application import gui_app, MousePos


class DialogResult(IntEnum):
  CANCEL = 0
  CONFIRM = 1
  NO_ACTION = -1


class Widget(abc.ABC):
  def __init__(self):
    self._rect: rl.Rectangle = rl.Rectangle(0, 0, 0, 0)
    self._parent_rect: rl.Rectangle = rl.Rectangle(0, 0, 0, 0)
    self._is_pressed = False
    self._is_visible: bool | Callable[[], bool] = True
    self._touch_valid_callback: Callable[[], bool] | None = None

  def set_touch_valid_callback(self, touch_callback: Callable[[], bool]) -> None:
    """Set a callback to determine if the widget can be clicked."""
    self._touch_valid_callback = touch_callback

  def _touch_valid(self) -> bool:
    """Check if the widget can be touched."""
    return self._touch_valid_callback() if self._touch_valid_callback else True

  @property
  def is_visible(self) -> bool:
    return self._is_visible() if callable(self._is_visible) else self._is_visible

  @property
  def rect(self) -> rl.Rectangle:
    return self._rect

  def set_visible(self, visible: bool | Callable[[], bool]) -> None:
    self._is_visible = visible

  def set_rect(self, rect: rl.Rectangle) -> None:
    changed = (self._rect.x != rect.x or self._rect.y != rect.y or
               self._rect.width != rect.width or self._rect.height != rect.height)
    self._rect = rect
    if changed:
      self._update_layout_rects()

  def set_parent_rect(self, parent_rect: rl.Rectangle) -> None:
    """Can be used like size hint in QT"""
    self._parent_rect = parent_rect

  def set_position(self, x: float, y: float) -> None:
    changed = (self._rect.x != x or self._rect.y != y)
    self._rect.x, self._rect.y = x, y
    if changed:
      self._update_layout_rects()

  def render(self, rect: rl.Rectangle = None) -> bool | int | None:
    if rect is not None:
      self.set_rect(rect)

    self._update_state()

    if not self.is_visible:
      return None

    ret = self._render(self._rect)

    # Keep track of whether mouse down started within the widget's rectangle
    for mouse_event in gui_app.mouse_events:
      if mouse_event.left_pressed and self._touch_valid():
        if rl.check_collision_point_rec(mouse_event.pos, self._rect):
          self._is_pressed = True

      elif not self._touch_valid():
        self._is_pressed = False

      elif mouse_event.left_released:
        if self._is_pressed and rl.check_collision_point_rec(mouse_event.pos, self._rect):
          self._handle_mouse_release(mouse_event.pos)
        self._is_pressed = False

    return ret

  @abc.abstractmethod
  def _render(self, rect: rl.Rectangle) -> bool | int | None:
    """Render the widget within the given rectangle."""

  def _update_state(self):
    """Optionally update the widget's non-layout state. This is called before rendering."""

  def _update_layout_rects(self) -> None:
    """Optionally update any layout rects on Widget rect change."""

  def _handle_mouse_release(self, mouse_pos: MousePos) -> bool:
    """Optionally handle mouse release events."""
    return False
