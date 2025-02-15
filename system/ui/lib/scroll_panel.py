import pyray as rl
from cffi import FFI

MOUSE_WHEEL_SCROLL_SPEED = 30

class GuiScrollPanel:
  def __init__(self, bounds: rl.Rectangle, content: rl.Rectangle, show_vertical_scroll_bar: bool = False):
    self._dragging: bool = False
    self._last_mouse_y: float = 0.0
    self._bounds = bounds
    self._content = content
    self._scroll = rl.Vector2(0, 0)
    self._view = rl.Rectangle(0, 0, 0, 0)
    self._show_vertical_scroll_bar: bool = show_vertical_scroll_bar

  def handle_scroll(self)-> rl.Vector2:
    mouse_pos = rl.get_mouse_position()
    if rl.check_collision_point_rec(mouse_pos, self._bounds) and rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT):
      if not self._dragging:
        self._dragging = True
        self._last_mouse_y = mouse_pos.y

    if self._dragging:
      if rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT):
        delta_y = mouse_pos.y - self._last_mouse_y
        self._scroll.y += delta_y
        self._last_mouse_y = mouse_pos.y
      else:
        self._dragging = False

    wheel_move = rl.get_mouse_wheel_move()
    if self._show_vertical_scroll_bar:
      self._scroll.y += wheel_move * (MOUSE_WHEEL_SCROLL_SPEED - 20)
      rl.gui_scroll_panel(self._bounds, FFI().NULL, self._content, self._scroll, self._view)
    else:
      self._scroll.y += wheel_move * MOUSE_WHEEL_SCROLL_SPEED
      max_scroll_y = self._content.height - self._bounds.height
      self._scroll.y = max(min(self._scroll.y, 0), -max_scroll_y)

    return self._scroll
