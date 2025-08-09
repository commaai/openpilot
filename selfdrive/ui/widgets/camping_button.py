import pyray as rl
from openpilot.common.params import Params
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.widget import Widget


class CampingModeButton(Widget):
  def __init__(self):
    super().__init__()

    self.img_width = 80
    self.horizontal_padding = 50
    self.button_height = 110

    self.params = Params()
    self.camping_mode = self.params.get_bool("CampingMode")
    self.is_pressed = False

  def _get_gradient_colors(self):
    alpha = 0xCC if self.is_pressed else 0xFF
    if self.camping_mode:
      return rl.Color(20, 255, 171, alpha), rl.Color(35, 149, 255, alpha)
    else:
      return rl.Color(180, 180, 180, alpha), rl.Color(120, 120, 120, alpha)

  def _draw_gradient_background(self, rect):
    start_color, end_color = self._get_gradient_colors()
    rl.draw_rectangle_gradient_h(int(rect.x), int(rect.y), int(rect.width), int(rect.height),
                                 start_color, end_color)

  def _handle_interaction(self, rect):
    mouse_pos = rl.get_mouse_position()
    mouse_in_rect = rl.check_collision_point_rec(mouse_pos, rect)

    self.is_pressed = mouse_in_rect and rl.is_mouse_button_down(rl.MOUSE_BUTTON_LEFT)
    return mouse_in_rect and rl.is_mouse_button_released(rl.MOUSE_BUTTON_LEFT)

  def _render(self, rect):
    if self._handle_interaction(rect):
      self.camping_mode = not self.camping_mode
      self.params.put_bool("CampingMode", self.camping_mode)

    rl.draw_rectangle_rounded(rect, 0.08, 20, rl.Color(255, 255, 255, 255))

    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))
    self._draw_gradient_background(rect)
    rl.end_scissor_mode()

    # Draw text label
    text = "CAMPING MODE ON" if self.camping_mode else "CAMPING MODE OFF"
    text_x = rect.x + self.horizontal_padding
    text_y = rect.y + rect.height / 2 - 42 // 2

    rl.draw_text_ex(gui_app.font(FontWeight.NORMAL), text, rl.Vector2(int(text_x), int(text_y)), 42, 0, rl.Color(0, 0, 0, 255))
