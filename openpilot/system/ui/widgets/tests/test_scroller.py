from types import SimpleNamespace
from unittest.mock import Mock

import itertools
import unittest

from openpilot.system.ui.widgets.scroller import _Scroller


class TestScroller(unittest.TestCase):
  def test_scroll_to_bounds(self):
    cases = [
      (0, 300, [360, 402], -182),  # Last item cannot be centered.
      (-100, -300, [360, 402], 0),  # First item cannot be centered.
      (-50, 50, [360, 402], -100),  # In-bounds relative movement.
      (0, 300, [200], 0),  # Content fits in the viewport.
      (-182, 100, [200], 0),  # Content shrank before the next layout.
      (0, 300, [], 0),
    ]
    for horizontal, smooth, (current, distance, sizes, expected) in itertools.product((True, False), (True, False), cases):
      with self.subTest(horizontal=horizontal, smooth=smooth, current=current, distance=distance, sizes=sizes):
        scroller = _Scroller.__new__(_Scroller)
        scroller._horizontal = horizontal
        scroller._rect = SimpleNamespace(width=640 if horizontal else 180, height=180 if horizontal else 640)
        scroller._spacing = scroller._pad = 20
        scroller._items = [SimpleNamespace(is_visible=True, rect=SimpleNamespace(width=size, height=size)) for size in sizes]
        scroller._items.append(SimpleNamespace(is_visible=False, rect=SimpleNamespace(width=1000, height=1000)))
        scroller.scroll_panel = Mock()
        scroller.scroll_panel.get_offset.return_value = current
        scroller._scrolling_to_filter = SimpleNamespace(x=0)

        scroller.scroll_to(distance, smooth=smooth, block_interrupt=smooth, block_widget_interaction=smooth)

        if smooth:
          assert scroller._scrolling_to == (expected, True, True)
          assert scroller._scrolling_to_filter.x == current
          scroller.scroll_panel.set_offset.assert_not_called()
        else:
          scroller.scroll_panel.set_offset.assert_called_once_with(expected)
