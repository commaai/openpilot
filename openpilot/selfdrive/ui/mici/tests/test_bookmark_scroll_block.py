import pyray as rl
import unittest

from openpilot.system.ui.lib.application import MouseEvent, MousePos
from openpilot.selfdrive.ui.mici.onroad.bookmark_gesture import BookmarkParentScrollBlocker


BOOKMARK_RECT = rl.Rectangle(0, 0, 100, 100)
PRESS_X = 80
PRESS_Y = 50
LEFT_DRAG_X = 60
RIGHT_DRAG_X = 90
TOUCH_TIME = 1.0
TOUCH_SLOT = 0


def mouse_event(x: float, *, left_pressed=False, left_down=False, left_released=False) -> MouseEvent:
  return MouseEvent(MousePos(x, PRESS_Y), TOUCH_SLOT, left_pressed, left_released, left_down, TOUCH_TIME)


class TestBookmarkScrollBlock(unittest.TestCase):
  def setUp(self):
    self.blocker = BookmarkParentScrollBlocker()

  def test_blocks_parent_scroll_before_bookmark_handler_sees_left_drag(self):
    assert not self.blocker.update([mouse_event(PRESS_X, left_pressed=True)], BOOKMARK_RECT)

    assert self.blocker.update([mouse_event(LEFT_DRAG_X, left_down=True)], BOOKMARK_RECT)

  def test_right_drag_does_not_block_parent_scroll(self):
    assert not self.blocker.update([mouse_event(PRESS_X, left_pressed=True)], BOOKMARK_RECT)

    assert not self.blocker.update([mouse_event(RIGHT_DRAG_X, left_down=True)], BOOKMARK_RECT)

  def test_release_clears_parent_scroll_block(self):
    self.blocker.update([
      mouse_event(PRESS_X, left_pressed=True),
      mouse_event(LEFT_DRAG_X, left_down=True),
    ], BOOKMARK_RECT)

    assert not self.blocker.update([mouse_event(LEFT_DRAG_X, left_released=True)], BOOKMARK_RECT)
