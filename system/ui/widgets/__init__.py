from __future__ import annotations

import abc
import pyray as rl
from enum import IntEnum
from collections.abc import Callable
from openpilot.system.ui.lib.application import gui_app, MousePos, MAX_TOUCH_SLOTS, MouseEvent

try:
  from openpilot.selfdrive.ui.ui_state import device
except ImportError:
  class Device:
    awake = True
  device = Device()


class DialogResult(IntEnum):
  CANCEL = 0
  CONFIRM = 1
  NO_ACTION = -1


class Widget(abc.ABC):
  def __init__(self):
    self._rect: rl.Rectangle = rl.Rectangle(0, 0, 0, 0)
    self._parent_rect: rl.Rectangle | None = None
    self.__is_pressed = [False] * MAX_TOUCH_SLOTS
    # if current mouse/touch down started within the widget's rectangle
    self.__tracking_is_pressed = [False] * MAX_TOUCH_SLOTS
    self._enabled: bool | Callable[[], bool] = True
    self._is_visible: bool | Callable[[], bool] = True
    self._touch_valid_callback: Callable[[], bool] | None = None
    self._click_callback: Callable[[], None] | None = None
    self._multi_touch = False
    self.__was_awake = True

  @property
  def rect(self) -> rl.Rectangle:
    return self._rect

  def set_rect(self, rect: rl.Rectangle) -> None:
    changed = (self._rect.x != rect.x or self._rect.y != rect.y or
               self._rect.width != rect.width or self._rect.height != rect.height)
    self._rect = rect
    if changed:
      self._update_layout_rects()

  def set_parent_rect(self, parent_rect: rl.Rectangle) -> None:
    """Can be used like size hint in QT"""
    self._parent_rect = parent_rect

  @property
  def is_pressed(self) -> bool:
    return any(self.__is_pressed)

  @property
  def enabled(self) -> bool:
    return self._enabled() if callable(self._enabled) else self._enabled

  def set_enabled(self, enabled: bool | Callable[[], bool]) -> None:
    self._enabled = enabled

  @property
  def is_visible(self) -> bool:
    return self._is_visible() if callable(self._is_visible) else self._is_visible

  def set_visible(self, visible: bool | Callable[[], bool]) -> None:
    self._is_visible = visible

  def set_click_callback(self, click_callback: Callable[[], None] | None) -> None:
    """Set a callback to be called when the widget is clicked."""
    self._click_callback = click_callback

  def set_touch_valid_callback(self, touch_callback: Callable[[], bool]) -> None:
    """Set a callback to determine if the widget can be clicked."""
    self._touch_valid_callback = touch_callback

  def _touch_valid(self) -> bool:
    """Check if the widget can be touched."""
    return self._touch_valid_callback() if self._touch_valid_callback else True

  def set_position(self, x: float, y: float) -> None:
    changed = (self._rect.x != x or self._rect.y != y)
    self._rect = rl.Rectangle(x, y, self._rect.width, self._rect.height)
    if changed:
      self._update_layout_rects()

  @property
  def _hit_rect(self) -> rl.Rectangle:
    # restrict touches to within parent rect if set, useful inside Scroller
    if self._parent_rect is None:
      return self._rect
    return rl.get_collision_rec(self._rect, self._parent_rect)

  def render(self, rect: rl.Rectangle | None = None) -> bool | int | None:
    if rect is not None:
      self.set_rect(rect)

    self._update_state()

    if not self.is_visible:
      return None

    self._layout()
    ret = self._render(self._rect)

    if gui_app.show_touches:
      self._draw_debug_rect()

    # Keep track of whether mouse down started within the widget's rectangle
    if self.enabled and self.__was_awake:
      self._process_mouse_events()
    else:
      # TODO: ideally we emit release events when going disabled
      self.__is_pressed = [False] * MAX_TOUCH_SLOTS
      self.__tracking_is_pressed = [False] * MAX_TOUCH_SLOTS

    self.__was_awake = device.awake

    return ret

  def _draw_debug_rect(self) -> None:
    rl.draw_rectangle_lines(int(self._rect.x), int(self._rect.y),
                            max(int(self._rect.width), 1), max(int(self._rect.height), 1), rl.RED)

  def _process_mouse_events(self) -> None:
    hit_rect = self._hit_rect
    touch_valid = self._touch_valid()

    for mouse_event in gui_app.mouse_events:
      if not self._multi_touch and mouse_event.slot != 0:
        continue

      mouse_in_rect = rl.check_collision_point_rec(mouse_event.pos, hit_rect)
      # Ignores touches/presses that start outside our rect
      # Allows touch to leave the rect and come back in focus if mouse did not release
      if mouse_event.left_pressed and touch_valid:
        if mouse_in_rect:
          self._handle_mouse_press(mouse_event.pos)
          self.__is_pressed[mouse_event.slot] = True
          self.__tracking_is_pressed[mouse_event.slot] = True
          self._handle_mouse_event(mouse_event)

      # Callback such as scroll panel signifies user is scrolling
      elif not touch_valid:
        self.__is_pressed[mouse_event.slot] = False
        self.__tracking_is_pressed[mouse_event.slot] = False

      elif mouse_event.left_released:
        self._handle_mouse_event(mouse_event)
        if self.__is_pressed[mouse_event.slot] and mouse_in_rect:
          self._handle_mouse_release(mouse_event.pos)
        self.__is_pressed[mouse_event.slot] = False
        self.__tracking_is_pressed[mouse_event.slot] = False

      # Mouse/touch is still within our rect
      elif mouse_in_rect:
        if self.__tracking_is_pressed[mouse_event.slot]:
          self.__is_pressed[mouse_event.slot] = True
          self._handle_mouse_event(mouse_event)

      # Mouse/touch left our rect but may come back into focus later
      elif not mouse_in_rect:
        self.__is_pressed[mouse_event.slot] = False
        self._handle_mouse_event(mouse_event)

  def _layout(self) -> None:
    """Optionally lay out child widgets separately. This is called before rendering."""

  def _update_state(self):
    """Optionally update the widget's non-layout state. This is called before rendering."""

  @abc.abstractmethod
  def _render(self, rect: rl.Rectangle) -> bool | int | None:
    """Render the widget within the given rectangle."""

  def _update_layout_rects(self) -> None:
    """Optionally update any layout rects on Widget rect change."""

  def _handle_mouse_press(self, mouse_pos: MousePos) -> None:
    """Optionally handle mouse press events."""

  def _handle_mouse_release(self, mouse_pos: MousePos) -> None:
    """Optionally handle mouse release events."""
    if self._click_callback:
      self._click_callback()

  def _handle_mouse_event(self, mouse_event: MouseEvent) -> None:
    """Optionally handle mouse events. This is called before rendering."""
    # Default implementation does nothing, can be overridden by subclasses

  def show_event(self):
    """Optionally handle show event. Parent must manually call this"""
    # TODO: iterate through all child objects, check for subclassing from Widget/Layout (Scroller)

  def hide_event(self):
    """Optionally handle hide event. Parent must manually call this"""
