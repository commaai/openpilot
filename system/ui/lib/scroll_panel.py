import pyray as rl
from cffi import FFI

MOUSE_WHEEL_SCROLL_SPEED = 30


class GuiScrollPanel:
  def __init__(self):
    self._dragging: bool = False
    self._last_mouse_y: float = 0.0
    self._scroll: rl.Vector2 = rl.Vector2(0, 0)
    self._view = rl.Rectangle(0, 0, 0, 0)

  def handle_scroll(self, bounds: rl.Rectangle, content: rl.Rectangle, show_vertical_scroll_bar: bool = False) -> rl.Vector2:
    mouse_pos = rl.get_mouse_position()
    if rl.check_collision_point_rec(mouse_pos, bounds) and rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT):
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
    if show_vertical_scroll_bar:
      self._scroll.y += wheel_move * (MOUSE_WHEEL_SCROLL_SPEED - 20)
      rl.gui_scroll_panel(bounds, FFI().NULL, content, self._scroll, self._view)
    else:
      self._scroll.y += wheel_move * MOUSE_WHEEL_SCROLL_SPEED
      max_scroll_y = max(content.height - bounds.height, 0)
      self._scroll.y = max(min(self._scroll.y, 0), -max_scroll_y)

    return self._scroll
