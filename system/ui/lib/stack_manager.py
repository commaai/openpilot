from typing import Optional
from collections.abc import Callable
from dataclasses import dataclass

import pyray as rl

# TODO: actually use Widget, but theres a circular dependency
# TODO: wrap all widgets with NavWidget here


@dataclass
class StackEntry:
  widget: object
  callback: Callable | None = None


class StackManager:
  def __init__(self, screen_rect: rl.Rectangle):
    self._stack: list[StackEntry] = []
    self._screen_rect = screen_rect
    self._top_shown = False

  def push(self, widget: object, callback: Callable | None = None) -> None:
    self._stack.append(StackEntry(widget=widget, callback=callback))
    self._top_shown = False

  def pop(self) -> Optional[object]:
    if not self._stack:
      return None

    entry = self._stack.pop()
    widget = entry.widget
    self._top_shown = False

    if hasattr(widget, 'hide_event'):
      widget.hide_event()

    return widget

  def depth(self) -> int:
    return len(self._stack)

  def render(self) -> None:
    if not self._stack:
      return

    print(f"Rendering stack: {self._stack}")
    entry = self._stack[-1]
    widget = entry.widget

    if not self._top_shown:
      if hasattr(widget, 'show_event'):
        widget.show_event()
      self._top_shown = True

    result = None
    if hasattr(widget, 'render'):
      result = widget.render(self._screen_rect)
    elif callable(widget):
      result = widget()
    else:
      raise TypeError(f"Widget must have a render() method or be callable, got {type(widget)}")

    if result is not None and result >= 0:
      entry = self._stack.pop()
      self._top_shown = False

      if hasattr(entry.widget, 'hide_event'):
        entry.widget.hide_event()

      if entry.callback is not None:
        entry.callback(result)
