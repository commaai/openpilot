from typing import Optional

import pyray as rl

# TODO: actually use Widget, but theres a circular dependency
# TODO: wrap all widgets with NavWidget here?

class StackManager:
  def __init__(self, screen_rect: rl.Rectangle):
    self._stack: list[object] = []
    self._screen_rect = screen_rect
    self._top_shown = False

  def push(self, widget: object) -> None:
    self._stack.append(widget)
    self._top_shown = False

  def pop(self) -> Optional[object]:
    if not self._stack:
      return None

    widget = self._stack.pop()
    self._top_shown = False

    if hasattr(widget, 'hide_event'):
      widget.hide_event()

    return widget

  def render(self) -> None:
    if not self._stack:
      return

    top_widget = self._stack[-1]

    if not self._top_shown:
      top_widget.show_event()
      self._top_shown = True

    if hasattr(top_widget, 'render'):
      result = top_widget.render(self._screen_rect)
    else:
      result = None

    if result is not None and result != -1:
      self.pop()
