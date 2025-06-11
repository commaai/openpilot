import abc
import pyray as rl
from enum import IntEnum
from collections.abc import Callable
from openpilot.system.ui.lib.application import gui_app, MouseState


class DialogResult(IntEnum):
  CANCEL = 0
  CONFIRM = 1
  OK = 2
  NO_ACTION = -1


class Widget(abc.ABC):
  def __init__(self):
    self._rect: rl.Rectangle = rl.Rectangle(0, 0, 0, 0)
    self._is_visible: bool | Callable[[], bool] = True

  @property
  def is_visible(self) -> bool:
    return self._is_visible() if callable(self._is_visible) else self._is_visible

  def set_visible(self, visible: bool | Callable[[], bool]) -> None:
    self._is_visible = visible

  def set_rect(self, rect: rl.Rectangle) -> None:
    prev_rect = self._rect
    self._rect = rect
    if (rect.x != prev_rect.x or rect.y != prev_rect.y or
        rect.width != prev_rect.width or rect.height != prev_rect.height):
      self._update_layout_rects()

  def render(self, rect: rl.Rectangle = None) -> bool | int | None:
    if rect is not None:
      self.set_rect(rect)

    self._update_state()

    if not self.is_visible:
      return None

    ret = self._render(self._rect)

    # Check if mouse was clicked within the widget's rectangle
    if gui_app.mouse.check_clicked(rect):
      result = self._on_mouse_clicked(gui_app.mouse)
      # If the event was handled, mark the mouse input as consumed
      if result:
        gui_app.mouse.consumed = True

    return ret

  @abc.abstractmethod
  def _render(self, rect: rl.Rectangle) -> bool | int | None:
    """Render the widget within the given rectangle."""

  def _update_state(self):
    """Optionally update the widget's non-layout state. This is called before rendering."""

  def _update_layout_rects(self) -> None:
    """Optionally update any layout rects on Widget rect change."""

  def _on_mouse_clicked(self, mouse: MouseState) -> bool:
    """Optionally handle mouse release events."""
    return False
