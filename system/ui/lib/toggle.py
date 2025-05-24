import pyray as rl

ON_COLOR = rl.GREEN
OFF_COLOR = rl.Color(0x39, 0x39, 0x39, 255)
KNOB_COLOR = rl.WHITE
BG_HEIGHT = 60
KNOB_HEIGHT = 80
WIDTH = 160


class Toggle:
  def __init__(self, x, y, initial_state=False):
    self._state = initial_state
    self._rect = rl.Rectangle(x, y, WIDTH, KNOB_HEIGHT)

  def handle_input(self):
    if rl.is_mouse_button_pressed(rl.MOUSE_LEFT_BUTTON):
      mouse_pos = rl.get_mouse_position()
      if rl.check_collision_point_rec(mouse_pos, self._rect):
        self._state = not self._state

  def get_state(self):
    return self._state

  def render(self):
    self._draw_background()
    self._draw_knob()

  def _draw_background(self):
    bg_rect = rl.Rectangle(
      self._rect.x + 5,
      self._rect.y + (KNOB_HEIGHT - BG_HEIGHT) / 2,
      self._rect.width - 10,
      BG_HEIGHT,
    )
    rl.draw_rectangle_rounded(bg_rect, 1.0, 10, ON_COLOR if self._state else OFF_COLOR)

  def _draw_knob(self):
    knob_radius = KNOB_HEIGHT / 2
    knob_x = self._rect.x + knob_radius if not self._state else self._rect.x + self._rect.width - knob_radius
    knob_y = self._rect.y + knob_radius
    rl.draw_circle(int(knob_x), int(knob_y), knob_radius, KNOB_COLOR)


if __name__ == "__main__":
  from openpilot.system.ui.lib.application import gui_app

  gui_app.init_window("Text toggle example")
  toggle = Toggle(100, 100)
  for _ in gui_app.render():
    toggle.handle_input()
    toggle.render()

