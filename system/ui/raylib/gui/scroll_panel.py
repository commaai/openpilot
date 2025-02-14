import pyray as rl
from cffi import FFI

class GuiScrollPanel:
  def __init__(self, bounds: rl.Rectangle, content: rl.Rectangle, show_vertical_scroll_bar: bool = False):
    self._dragging: bool = False
    self._last_mouse_y: float = 0.0
    self._bounds = bounds
    self._content = content
    self._scroll = rl.Vector2()
    self._view = rl.Rectangle()
    self._show_vertical_scroll_bar: bool = show_vertical_scroll_bar

  def handle_scroll(self)-> rl.Vector2:
    mouse_point = rl.get_mouse_position()
    if rl.check_collision_point_rec(mouse_point, self._bounds) and rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT):
      if not self._dragging:
        self._dragging = True
        self._last_mouse_y = rl.get_mouse_y()

    if self._dragging and rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT):
      mouse_y = rl.get_mouse_y()
      self._scroll.y += (mouse_y - self._last_mouse_y)
      self._last_mouse_y = mouse_y

    if rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT):
      self._dragging = False

    if self._show_vertical_scroll_bar:
      rl.gui_scroll_panel(self._bounds, FFI().NULL, self._content, self._scroll, self._view)

    return self._scroll
