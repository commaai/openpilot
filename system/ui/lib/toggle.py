import pyray as rl

ON_COLOR = rl.Color(51, 171, 76, 255)
OFF_COLOR = rl.Color(0x39, 0x39, 0x39, 255)
KNOB_COLOR = rl.WHITE
WIDTH, HEIGHT = 160, 80
BG_HEIGHT = 60
ANIMATION_SPEED = 8.0


class Toggle:
  def __init__(self, initial_state=False):
    self._state = initial_state
    self._rect = rl.Rectangle(0, 0, WIDTH, HEIGHT)
    self._progress = 1.0 if initial_state else 0.0
    self._target = self._progress

  def handle_input(self):
    if rl.is_mouse_button_pressed(rl.MOUSE_LEFT_BUTTON):
      if rl.check_collision_point_rec(rl.get_mouse_position(), self._rect):
        self._state = not self._state
        self._target = 1.0 if self._state else 0.0
        return 1
    return 0

  def get_state(self):
    return self._state

  def set_state(self, state: bool):
    self._state = state

  def update(self):
    if abs(self._progress - self._target) > 0.01:
      delta = rl.get_frame_time() * ANIMATION_SPEED
      self._progress += delta if self._progress < self._target else -delta
      self._progress = max(0.0, min(1.0, self._progress))

  def render(self, rect: rl.Rectangle):
    self._rect.x, self._rect.y = rect.x, rect.y
    self. update()
    # Draw background
    bg_rect = rl.Rectangle(self._rect.x + 5, self._rect.y + 10, WIDTH - 10, BG_HEIGHT)
    bg_color = self._blend_color(OFF_COLOR, ON_COLOR, self._progress)
    rl.draw_rectangle_rounded(bg_rect, 1.0, 10, bg_color)

    # Draw knob
    knob_x = self._rect.x + HEIGHT / 2 + (WIDTH - HEIGHT) * self._progress
    knob_y = self._rect.y + HEIGHT / 2
    rl.draw_circle(int(knob_x), int(knob_y), HEIGHT / 2, KNOB_COLOR)

    return self.handle_input()

  def _blend_color(self, c1, c2, t):
    return rl.Color(int(c1.r + (c2.r - c1.r) * t), int(c1.g + (c2.g - c1.g) * t), int(c1.b + (c2.b - c1.b) * t), 255)
