import math
import raylib_nb as rl

from openpilot.common.parameterized import parameterized
from openpilot.common.test import OpenpilotTestCase
from openpilot.system.ui.lib.application import gui_app, MouseEvent, MousePos
from openpilot.system.ui.lib.scroll_panel2 import GuiScrollPanel2

BOUNDS = rl.Rectangle(0, 0, 536, 240)
PAGES_SIZE = 3 * 536
LIST_SIZE = 1000


class TestNestedScrollPanels(OpenpilotTestCase):
  """A vertical list inside a horizontal pager, like the mici alerts page"""
  def setUp(self):
    self.pager = GuiScrollPanel2(horizontal=True)
    self.list = GuiScrollPanel2(horizontal=False)
    self.t = 0.0

  def _frame(self, events: list[MouseEvent]):
    gui_app._mouse_events = events
    # outer panel first, like Scroller lays out its items
    self.pager.update(BOUNDS, PAGES_SIZE)
    self.list.update(BOUNDS, LIST_SIZE)
    gui_app._frame += 1

  def _ev(self, x: float, y: float, pressed: bool = False, released: bool = False) -> MouseEvent:
    self.t += 1 / 140
    return MouseEvent(MousePos(x, y), 0, pressed, released, not released, self.t)

  def _drag(self, points: list[tuple[float, float]]):
    # two touch samples per frame, like the 140 Hz touch controller
    events = [self._ev(*points[0], pressed=True)] + [self._ev(*p) for p in points[1:]]
    for i in range(0, len(events), 2):
      self._frame(events[i:i + 2])
    self._frame([self._ev(*points[-1], released=True)])

  def _swipe(self, deg: float, length: float = 150, start=(268, 200)):
    # deg is the angle above horizontal, the swipe goes up and to the left
    dx, dy = -math.cos(math.radians(deg)), -math.sin(math.radians(deg))
    self._drag([(start[0] + dx * d, start[1] + dy * d) for d in range(0, int(length) + 1, 6)])

  @parameterized.expand([(0, 'pager'), (20, 'pager'), (35, 'list'), (55, 'list'), (90, 'list')])
  def test_dominant_axis_with_list_bias(self, deg, owner):
    self._swipe(deg)
    self.assertEqual(self.pager.get_offset() < 0, owner == 'pager')
    self.assertEqual(self.list.get_offset() < 0, owner == 'list')

  def test_list_that_cant_scroll_doesnt_bias(self):
    self.list.set_offset(-(LIST_SIZE - BOUNDS.height))  # at the bottom, can't scroll further down
    self._swipe(35)
    self.assertLess(self.pager.get_offset(), 0)
    self.assertEqual(self.list.get_offset(), -(LIST_SIZE - BOUNDS.height))

  def test_claim_is_sticky(self):
    self._drag([(268, 200 - d) for d in range(0, 60, 6)] + [(268 - d, 140) for d in range(0, 200, 6)])
    self.assertEqual(self.pager.get_offset(), 0)
    self.assertLess(self.list.get_offset(), 0)

  def test_tap(self):
    events = [self._ev(268, 120, pressed=True), self._ev(272, 126)]
    self._frame(events)
    self.assertTrue(self.pager.is_touch_valid() and self.list.is_touch_valid())
    self._frame([self._ev(272, 126, released=True)])
    self.assertTrue(self.pager.is_touch_valid() and self.list.is_touch_valid())
    self.assertEqual((self.pager.get_offset(), self.list.get_offset()), (0, 0))

  def test_drag_is_not_a_tap(self):
    self._frame([self._ev(268, 200, pressed=True), self._ev(268, 180)])
    self.assertFalse(self.list.is_touch_valid())

