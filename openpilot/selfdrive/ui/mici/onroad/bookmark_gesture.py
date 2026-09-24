import pyray as rl

from openpilot.system.ui.lib.application import MouseEvent


PARENT_SCROLL_BLOCK_THRESHOLD = 1.0  # px leftward movement before blocking parent scroll
PRIMARY_TOUCH_SLOT = 0


class BookmarkParentScrollBlocker:
  def __init__(self) -> None:
    self._start_x: float | None = None
    self._blocking = False

  def update(self, mouse_events: list[MouseEvent], hit_rect: rl.Rectangle) -> bool:
    for mouse_event in mouse_events:
      if mouse_event.slot != PRIMARY_TOUCH_SLOT:
        continue

      if mouse_event.left_pressed:
        if rl.check_collision_point_rec(mouse_event.pos, hit_rect):
          self._start_x = mouse_event.pos.x
          self._blocking = False

      elif mouse_event.left_down and self._start_x is not None:
        if self._start_x - mouse_event.pos.x > PARENT_SCROLL_BLOCK_THRESHOLD:
          self._blocking = True

      elif mouse_event.left_released:
        self._start_x = None
        self._blocking = False

    return self._blocking
