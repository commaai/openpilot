import pyray as rl
import numpy as np
from collections.abc import Callable

from openpilot.common.filter_simple import FirstOrderFilter, BounceFilter
from openpilot.common.swaglog import cloudlog
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.scroll_panel2 import GuiScrollPanel2, ScrollState
from openpilot.system.ui.widgets import Widget

ITEM_SPACING = 20
LINE_COLOR = rl.GRAY
LINE_PADDING = 40
ANIMATION_SCALE = 0.6

MOVE_LIFT = 20
MOVE_OVERLAY_ALPHA = 0.65
SCROLL_RC = 0.15

EDGE_SHADOW_WIDTH = 20

MIN_ZOOM_ANIMATION_TIME = 0.075  # seconds
DO_ZOOM = False
DO_JELLO = False


class ScrollIndicator(Widget):
  HORIZONTAL_MARGIN = 4

  def __init__(self):
    super().__init__()
    self._txt_scroll_indicator = gui_app.texture("icons_mici/settings/horizontal_scroll_indicator.png", 96, 48)
    self._scroll_offset: float = 0.0
    self._content_size: float = 0.0
    self._viewport: rl.Rectangle = rl.Rectangle(0, 0, 0, 0)

  def update(self, scroll_offset: float, content_size: float, viewport: rl.Rectangle) -> None:
    self._scroll_offset = scroll_offset
    self._content_size = content_size
    self._viewport = viewport

  def _render(self, _):
    # scale indicator width based on content size
    indicator_w = float(np.interp(self._content_size, [1000, 3000], [300, 100]))

    # position based on scroll ratio
    slide_range = self._viewport.width - indicator_w
    max_scroll = self._content_size - self._viewport.width
    scroll_ratio = -self._scroll_offset / max_scroll
    x = self._viewport.x + scroll_ratio * slide_range
    # don't bounce up when NavWidget shows
    y = max(self._viewport.y, 0) + self._viewport.height - self._txt_scroll_indicator.height / 2

    # squeeze when overscrolling past edges
    dest_left = max(x, self._viewport.x)
    dest_right = min(x + indicator_w, self._viewport.x + self._viewport.width)
    dest_w = max(indicator_w / 2, dest_right - dest_left)

    # keep within viewport after applying minimum width
    dest_left = min(dest_left, self._viewport.x + self._viewport.width - dest_w)
    dest_left = max(dest_left, self._viewport.x)

    src_rec = rl.Rectangle(0, 0, self._txt_scroll_indicator.width, self._txt_scroll_indicator.height)
    dest_rec = rl.Rectangle(dest_left, y, dest_w, self._txt_scroll_indicator.height)
    rl.draw_texture_pro(self._txt_scroll_indicator, src_rec, dest_rec, rl.Vector2(0, 0), 0.0,
                        rl.Color(255, 255, 255, int(255 * 0.45)))


class Scroller(Widget):
  def __init__(self, items: list[Widget], horizontal: bool = True, snap_items: bool = False, spacing: int = ITEM_SPACING,
               pad: int = ITEM_SPACING, scroll_indicator: bool = True, edge_shadows: bool = True):
    super().__init__()
    self._items: list[Widget] = []
    self._horizontal = horizontal
    self._snap_items = snap_items
    self._spacing = spacing
    self._pad = pad

    self._reset_scroll_at_show = True

    self._scrolling_to: tuple[float | None, bool] = (None, False)  # target offset, block user scrolling
    self._scrolling_to_filter = FirstOrderFilter(0.0, SCROLL_RC, 1 / gui_app.target_fps)
    self._zoom_filter = FirstOrderFilter(1.0, 0.2, 1 / gui_app.target_fps)
    self._zoom_out_t: float = 0.0

    # layout state
    self._visible_items: list[Widget] = []
    self._content_size: float = 0.0
    self._scroll_offset: float = 0.0

    self._item_pos_filter = BounceFilter(0.0, 0.05, 1 / gui_app.target_fps)

    # when not pressed, snap to closest item to be center
    self._scroll_snap_filter = FirstOrderFilter(0.0, 0.05, 1 / gui_app.target_fps)

    self.scroll_panel = GuiScrollPanel2(self._horizontal, handle_out_of_bounds=not self._snap_items)
    self._scroll_enabled: bool | Callable[[], bool] = True

    self._show_scroll_indicator = scroll_indicator and self._horizontal
    self._scroll_indicator = ScrollIndicator()
    self._edge_shadows = edge_shadows and self._horizontal

    # move animation state
    # on move; lift src widget -> wait -> move all -> wait -> drop src widget
    self._overlay_filter = FirstOrderFilter(0.0, 0.05, 1 / gui_app.target_fps)
    self._move_animations: dict[Widget, FirstOrderFilter] = {}
    self._move_lift: dict[Widget, FirstOrderFilter] = {}
    # these are used to wait before moving/dropping, also to move onto next part of the animation earlier for timing
    self._pending_lift: set[Widget] = set()
    self._pending_move: set[Widget] = set()

    for item in items:
      self.add_widget(item)

  def set_reset_scroll_at_show(self, scroll: bool):
    self._reset_scroll_at_show = scroll

  def scroll_to(self, pos: float, smooth: bool = False, block: bool = False):
    assert not block or smooth, "Instant scroll cannot be blocking"

    # already there
    if abs(pos) < 1:
      return

    # FIXME: the padding correction doesn't seem correct
    scroll_offset = self.scroll_panel.get_offset() - pos
    if smooth:
      self._scrolling_to_filter.x = self.scroll_panel.get_offset()
      self._scrolling_to = scroll_offset, block
    else:
      self.scroll_panel.set_offset(scroll_offset)

  @property
  def is_auto_scrolling(self) -> bool:
    return self._scrolling_to[0] is not None

  @property
  def items(self) -> list[Widget]:
    return self._items

  @property
  def content_size(self) -> float:
    return self._content_size

  def add_widget(self, item: Widget) -> None:
    self._items.append(item)

    # preserve original touch valid callback
    original_touch_valid_callback = item._touch_valid_callback
    item.set_touch_valid_callback(lambda: self.scroll_panel.is_touch_valid() and self.enabled and self._scrolling_to[0] is None
                                          and not self.moving_items and (original_touch_valid_callback() if
                                                                         original_touch_valid_callback else True))

  def set_scrolling_enabled(self, enabled: bool | Callable[[], bool]) -> None:
    """Set whether scrolling is enabled (does not affect widget enabled state)."""
    self._scroll_enabled = enabled

  def _update_state(self):
    if DO_ZOOM:
      if self._scrolling_to[0] is not None or self.scroll_panel.state != ScrollState.STEADY:
        self._zoom_out_t = rl.get_time() + MIN_ZOOM_ANIMATION_TIME
        self._zoom_filter.update(0.85)
      else:
        if self._zoom_out_t is not None:
          if rl.get_time() > self._zoom_out_t:
            self._zoom_filter.update(1.0)
          else:
            self._zoom_filter.update(0.85)

    # Cancel auto-scroll if user starts manually scrolling (unless blocking)
    if (self.scroll_panel.state in (ScrollState.PRESSED, ScrollState.MANUAL_SCROLL) and
        self._scrolling_to[0] is not None and not self._scrolling_to[1]):
      self._scrolling_to = None, False

    if self._scrolling_to[0] is not None and len(self._pending_lift) == 0:
      self._scrolling_to_filter.update(self._scrolling_to[0])
      self.scroll_panel.set_offset(self._scrolling_to_filter.x)

      if abs(self._scrolling_to_filter.x - self._scrolling_to[0]) < 1:
        self.scroll_panel.set_offset(self._scrolling_to[0])
        self._scrolling_to = None, False

  def _get_scroll(self, visible_items: list[Widget], content_size: float) -> float:
    scroll_enabled = self._scroll_enabled() if callable(self._scroll_enabled) else self._scroll_enabled
    self.scroll_panel.set_enabled(scroll_enabled and self.enabled and not self._scrolling_to[1])
    self.scroll_panel.update(self._rect, content_size)
    if not self._snap_items:
      return round(self.scroll_panel.get_offset())

    # Snap closest item to center
    center_pos = self._rect.x + self._rect.width / 2 if self._horizontal else self._rect.y + self._rect.height / 2
    closest_delta_pos = float('inf')
    scroll_snap_idx: int | None = None
    for idx, item in enumerate(visible_items):
      if self._horizontal:
        delta_pos = (item.rect.x + item.rect.width / 2) - center_pos
      else:
        delta_pos = (item.rect.y + item.rect.height / 2) - center_pos
      if abs(delta_pos) < abs(closest_delta_pos):
        closest_delta_pos = delta_pos
        scroll_snap_idx = idx

    if scroll_snap_idx is not None:
      snap_item = visible_items[scroll_snap_idx]
      if self.is_pressed:
        # no snapping until released
        self._scroll_snap_filter.x = 0
      else:
        # TODO: this doesn't handle two small buttons at the edges well
        if self._horizontal:
          snap_delta_pos = (center_pos - (snap_item.rect.x + snap_item.rect.width / 2)) / 10
          snap_delta_pos = min(snap_delta_pos, -self.scroll_panel.get_offset() / 10)
          snap_delta_pos = max(snap_delta_pos, (self._rect.width - self.scroll_panel.get_offset() - content_size) / 10)
        else:
          snap_delta_pos = (center_pos - (snap_item.rect.y + snap_item.rect.height / 2)) / 10
          snap_delta_pos = min(snap_delta_pos, -self.scroll_panel.get_offset() / 10)
          snap_delta_pos = max(snap_delta_pos, (self._rect.height - self.scroll_panel.get_offset() - content_size) / 10)
        self._scroll_snap_filter.update(snap_delta_pos)

      self.scroll_panel.set_offset(self.scroll_panel.get_offset() + self._scroll_snap_filter.x)

    return self.scroll_panel.get_offset()

  @property
  def moving_items(self) -> bool:
    return len(self._move_animations) > 0 or len(self._move_lift) > 0

  def move_item(self, from_idx: int, to_idx: int):
    assert self._horizontal
    if from_idx == to_idx:
      return

    if self.moving_items:
      cloudlog.warning(f"Already moving items, cannot move from {from_idx} to {to_idx}")
      return

    item = self._items.pop(from_idx)
    self._items.insert(to_idx, item)

    # store original position in content space of all affected widgets to animate from
    for idx in range(min(from_idx, to_idx), max(from_idx, to_idx) + 1):
      affected_item = self._items[idx]
      self._move_animations[affected_item] = FirstOrderFilter(affected_item.rect.x - self._scroll_offset, SCROLL_RC, 1 / gui_app.target_fps)
      self._pending_move.add(affected_item)

    # lift only src widget to make it more clear which one is moving
    self._move_lift[item] = FirstOrderFilter(0.0, SCROLL_RC, 1 / gui_app.target_fps)
    self._pending_lift.add(item)

  def _do_move_animation(self, item: Widget, target_x: float, target_y: float) -> tuple[float, float]:
    # wait a frame before moving so we match potential pending scroll animation
    can_start_move = len(self._pending_lift) == 0

    if item in self._move_lift:
      lift_filter = self._move_lift[item]

      # Animate lift
      if len(self._pending_move) > 0:
        lift_filter.update(MOVE_LIFT)
        # start moving when almost lifted
        if abs(lift_filter.x - MOVE_LIFT) < 2:
          self._pending_lift.discard(item)
      else:
        # if done moving, animate down
        lift_filter.update(0)
        if abs(lift_filter.x) < 1:
          del self._move_lift[item]
      target_y -= lift_filter.x

    # Animate move
    if item in self._move_animations:
      move_filter = self._move_animations[item]

      # compare/update in content space to match filter
      content_x = target_x - self._scroll_offset
      if can_start_move:
        move_filter.update(content_x)

        # drop when close to target
        if abs(move_filter.x - content_x) < 10:
          self._pending_move.discard(item)

        # finished moving
        if abs(move_filter.x - content_x) < 1:
          del self._move_animations[item]
      target_x = move_filter.x + self._scroll_offset

    return target_x, target_y

  def _layout(self):
    self._visible_items = [item for item in self._items if item.is_visible]

    self._content_size = sum(item.rect.width if self._horizontal else item.rect.height for item in self._visible_items)
    self._content_size += self._spacing * (len(self._visible_items) - 1)
    self._content_size += self._pad * 2

    self._scroll_offset = self._get_scroll(self._visible_items, self._content_size)

    self._item_pos_filter.update(self._scroll_offset)

    cur_pos = 0
    for idx, item in enumerate(self._visible_items):
      spacing = self._spacing if (idx > 0) else self._pad
      # Nicely lay out items horizontally/vertically
      if self._horizontal:
        x = self._rect.x + cur_pos + spacing
        y = self._rect.y + (self._rect.height - item.rect.height) / 2
        cur_pos += item.rect.width + spacing
      else:
        x = self._rect.x + (self._rect.width - item.rect.width) / 2
        y = self._rect.y + cur_pos + spacing
        cur_pos += item.rect.height + spacing

      # Consider scroll
      if self._horizontal:
        x += self._scroll_offset
      else:
        y += self._scroll_offset

      # Add some jello effect when scrolling
      if DO_JELLO:
        if self._horizontal:
          cx = self._rect.x + self._rect.width / 2
          jello_offset = self._scroll_offset - np.interp(x + item.rect.width / 2,
                                                         [self._rect.x, cx, self._rect.x + self._rect.width],
                                                         [self._item_pos_filter.x, self._scroll_offset, self._item_pos_filter.x])
          x -= np.clip(jello_offset, -20, 20)
        else:
          cy = self._rect.y + self._rect.height / 2
          jello_offset = self._scroll_offset - np.interp(y + item.rect.height / 2,
                                                         [self._rect.y, cy, self._rect.y + self._rect.height],
                                                         [self._item_pos_filter.x, self._scroll_offset, self._item_pos_filter.x])
          y -= np.clip(jello_offset, -20, 20)

      # Animate moves if needed
      x, y = self._do_move_animation(item, x, y)

      # Update item state
      item.set_position(round(x), round(y))  # round to prevent jumping when settling
      item.set_parent_rect(self._rect)

  def _render_item(self, item: Widget):
    # Skip rendering if not in viewport
    if not rl.check_collision_recs(item.rect, self._rect):
      return

    # Scale each element around its own origin when scrolling
    scale = self._zoom_filter.x
    if scale != 1.0:
      rl.rl_push_matrix()
      rl.rl_scalef(scale, scale, 1.0)
      rl.rl_translatef((1 - scale) * (item.rect.x + item.rect.width / 2) / scale,
                       (1 - scale) * (item.rect.y + item.rect.height / 2) / scale, 0)
      item.render()
      rl.rl_pop_matrix()
    else:
      item.render()

  def _render(self, _):
    rl.begin_scissor_mode(int(self._rect.x), int(self._rect.y),
                          int(self._rect.width), int(self._rect.height))

    for item in reversed(self._visible_items):
      if item in self._move_lift:
        continue
      self._render_item(item)

    # Dim background if moving items, lifted items are above
    self._overlay_filter.update(MOVE_OVERLAY_ALPHA if len(self._pending_move) else 0.0)
    if self._overlay_filter.x > 0.01:
      rl.draw_rectangle_rec(self._rect, rl.Color(0, 0, 0, int(255 * self._overlay_filter.x)))

    for item in self._move_lift:
      self._render_item(item)

    rl.end_scissor_mode()

    # Draw edge shadows on top of scroller content
    if self._edge_shadows:
      rl.draw_rectangle_gradient_h(int(self._rect.x), int(self._rect.y),
                                   EDGE_SHADOW_WIDTH, int(self._rect.height),
                                   rl.Color(0, 0, 0, 204), rl.BLANK)

      right_x = int(self._rect.x + self._rect.width - EDGE_SHADOW_WIDTH)
      rl.draw_rectangle_gradient_h(right_x, int(self._rect.y),
                                   EDGE_SHADOW_WIDTH, int(self._rect.height),
                                   rl.BLANK, rl.Color(0, 0, 0, 204))

    # Draw scroll indicator on top of edge shadows
    if self._show_scroll_indicator and len(self._visible_items) > 0:
      self._scroll_indicator.update(self._scroll_offset, self._content_size, self._rect)
      self._scroll_indicator.render()

  def show_event(self):
    super().show_event()
    for item in self._items:
      item.show_event()

    if self._reset_scroll_at_show:
      self.scroll_panel.set_offset(0.0)

    self._overlay_filter.x = 0.0
    self._move_animations.clear()
    self._move_lift.clear()
    self._pending_lift.clear()
    self._pending_move.clear()
    self._scrolling_to = None, False
    self._scrolling_to_filter.x = 0.0

  def hide_event(self):
    super().hide_event()
    for item in self._items:
      item.hide_event()
