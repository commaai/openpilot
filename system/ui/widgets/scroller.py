import pyray as rl
import numpy as np
from collections.abc import Callable

from openpilot.common.filter_simple import FirstOrderFilter, BounceFilter
from openpilot.system.ui.lib.application import gui_app, MouseEvent
from openpilot.system.ui.lib.scroll_panel2 import GuiScrollPanel2, ScrollState
from openpilot.system.ui.widgets import Widget

ITEM_SPACING = 20
LINE_COLOR = rl.GRAY
LINE_PADDING = 40
ANIMATION_SCALE = 0.6

MIN_ZOOM_ANIMATION_TIME = 0.075  # seconds
DO_ZOOM = False
DO_JELLO = False
SCROLL_BAR = False

# Space reserved at bottom for horizontal scroll bar
HORIZONTAL_SCROLL_BAR_MARGIN = 10

# Scroll indicator dot dimensions
SCROLL_DOT_WIDTH = 20
SCROLL_DOT_HEIGHT = 14
SCROLL_DOT_SPACING = 8

# Exclusive touch zone height for scroll indicator (from bottom of screen)
SCROLL_INDICATOR_TOUCH_ZONE = 30

# Position indicator pill dimensions
POSITION_PILL_WIDTH = 36
POSITION_PILL_HEIGHT = 16
POSITION_PILL_MARGIN = 8  # 8px margin from left/right edges of scroll indicator


class LineSeparator(Widget):
  def __init__(self, height: int = 1):
    super().__init__()
    self._rect = rl.Rectangle(0, 0, 0, height)

  def set_parent_rect(self, parent_rect: rl.Rectangle) -> None:
    super().set_parent_rect(parent_rect)
    self._rect.width = parent_rect.width

  def _render(self, _):
    rl.draw_line(int(self._rect.x) + LINE_PADDING, int(self._rect.y),
                 int(self._rect.x + self._rect.width) - LINE_PADDING, int(self._rect.y),
                 LINE_COLOR)


class ScrollIndicatorDot:
  """A single dot in the scroll indicator."""

  def __init__(self, index: int):
    self.index = index
    self.x = 0.0
    self.y = 0.0
    self.base_y = 0.0  # Original y position before animation
    self.width = SCROLL_DOT_WIDTH
    self.height = SCROLL_DOT_HEIGHT
    # Animation filter for smooth vertical movement
    self._y_offset_filter = FirstOrderFilter(0.0, 0.08, 1 / gui_app.target_fps)
    # Toggle state for coloring
    self.is_toggled_on = False
    # Whether this dot should be circular (for BigCircleToggle/BigCircleButton)
    self.is_circular = False
    # Whether this dot is for a red button (power off)
    self.is_red = False

  def set_position(self, x: float, y: float):
    self.x = x
    self.base_y = y
    self.y = y

  def set_y_offset(self, offset: float):
    """Set target y offset for animation (negative = move up)."""
    self._y_offset_filter.update(offset)

  def get_animated_y(self) -> float:
    """Get the current animated y position."""
    return self.base_y + self._y_offset_filter.x

  def get_center(self) -> tuple[float, float]:
    return self.x + self.width / 2, self.base_y + self.height / 2


class ScrollIndicator(Widget):
  """Interactive scroll indicator with dots for each item in a scroller."""

  DOT_DRAG_HYSTERESIS = 5  # px

  def __init__(self, on_dot_clicked: Callable[[int], None], on_dot_drag: Callable[[int], None] | None = None):
    super().__init__()
    self._on_dot_clicked = on_dot_clicked
    self._on_dot_drag = on_dot_drag
    self._dots: list[ScrollIndicatorDot] = []
    self._closest_dot: tuple[ScrollIndicatorDot | None, float] = (None, float('inf'))
    self._dragging_on_indicator = False
    self._current_indices: set[int] = {0}  # Track which items are currently active/centered
    self._parent_bottom = 0.0  # Bottom edge of parent scroller for touch zone
    self._last_drag_index: int | None = None  # Track last dragged index to avoid redundant calls
    self._scroll_progress = 0.0  # 0.0 to 1.0, tracks scroll position

  def set_item_count(self, count: int):
    """Update the number of dots to match the scroller's item count."""
    if len(self._dots) != count:
      self._dots = [ScrollIndicatorDot(i) for i in range(count)]

  def set_current_indices(self, indices: set[int]):
    """Set which dots should be highlighted as active."""
    self._current_indices = indices if indices else {0}

  def set_toggle_state(self, index: int, is_toggled_on: bool):
    """Set the toggle state for a specific dot (for toggle buttons)."""
    if 0 <= index < len(self._dots):
      self._dots[index].is_toggled_on = is_toggled_on

  def set_circular(self, index: int, is_circular: bool):
    """Set whether a dot should be circular (for BigCircleToggle/BigCircleButton)."""
    if 0 <= index < len(self._dots):
      self._dots[index].is_circular = is_circular

  def set_red(self, index: int, is_red: bool):
    """Set whether a dot is for a red button (power off)."""
    if 0 <= index < len(self._dots):
      self._dots[index].is_red = is_red

  def set_scroll_progress(self, progress: float):
    """Set scroll progress (0.0 = start, 1.0 = end) for position indicator."""
    self._scroll_progress = max(0.0, min(1.0, progress))

  def _get_total_width(self) -> float:
    """Calculate total width of all dots with spacing."""
    if not self._dots:
      return 0
    return len(self._dots) * SCROLL_DOT_WIDTH + (len(self._dots) - 1) * SCROLL_DOT_SPACING

  def _is_in_indicator_area(self, x: float, y: float) -> bool:
    """Check if position is within the indicator's exclusive touch zone."""
    if not self._dots:
      return False
    # Touch zone height: 30px from the bottom of the parent scroller
    zone_top = self._parent_bottom - SCROLL_INDICATOR_TOUCH_ZONE

    # Touch zone width: from left edge to 40px past the rightmost dot
    # Get the rightmost dot's position
    last_dot = self._dots[-1]
    zone_right = last_dot.x + last_dot.width + 40  # 40px past the rightmost dot

    return (0 <= x <= zone_right and
            zone_top <= y <= self._parent_bottom)

  def set_parent_bottom(self, bottom: float):
    """Set the bottom edge of the parent scroller for touch zone calculation."""
    self._parent_bottom = bottom

  def is_touch_in_zone(self) -> bool:
    """Check if current touch/press is in the indicator's exclusive zone."""
    for mouse_event in gui_app.mouse_events:
      if mouse_event.left_pressed or mouse_event.left_down:
        if self._is_in_indicator_area(mouse_event.pos.x, mouse_event.pos.y):
          return True
    return self._dragging_on_indicator

  def get_closest_dot_index(self) -> int | None:
    """Get the index of the currently closest dot, or None if not touching."""
    if self._closest_dot[0] is not None:
      return self._closest_dot[0].index
    return None

  def _get_closest_dot(self, mouse_x: float, mouse_y: float) -> tuple[ScrollIndicatorDot | None, float]:
    """Find the dot closest to the mouse position."""
    closest: tuple[ScrollIndicatorDot | None, float] = (None, float('inf'))
    for dot in self._dots:
      cx, cy = dot.get_center()
      dist = abs(cx - mouse_x) + abs(cy - mouse_y)
      if dist < closest[1]:
        # Apply hysteresis to prevent jitter when dragging
        if (self._closest_dot[0] is None or
            dot is self._closest_dot[0] or
            dist < self._closest_dot[1] - self.DOT_DRAG_HYSTERESIS):
          closest = (dot, dist)
    return closest

  def _process_indicator_input(self):
    """Process mouse/touch input for the indicator - similar to keyboard input handling."""
    for mouse_event in gui_app.mouse_events:
      if mouse_event.left_pressed:
        # Start dragging if pressed within indicator area
        if self._is_in_indicator_area(mouse_event.pos.x, mouse_event.pos.y):
          self._dragging_on_indicator = True
          self._last_drag_index = None  # Reset on new drag

      if mouse_event.left_released:
        # On release, jump to the closest dot if we were dragging
        if self._dragging_on_indicator and self._closest_dot[0] is not None:
          self._on_dot_clicked(self._closest_dot[0].index)
        self._dragging_on_indicator = False
        self._closest_dot = (None, float('inf'))
        self._last_drag_index = None

      if mouse_event.left_down and self._dragging_on_indicator:
        # Update closest dot while dragging
        self._closest_dot = self._get_closest_dot(mouse_event.pos.x, mouse_event.pos.y)
        # Call drag callback if closest dot changed
        if self._closest_dot[0] is not None and self._on_dot_drag is not None:
          current_idx = self._closest_dot[0].index
          if current_idx != self._last_drag_index:
            self._last_drag_index = current_idx
            self._on_dot_drag(current_idx)

  def _layout(self):
    """Position all dots left-justified within the rect."""
    if not self._dots:
      return

    # Left-justify: start at the left edge of the rect
    cur_x = self._rect.x

    for i, dot in enumerate(self._dots):
      # Circular dots are 14px wide, pills are 20px wide
      dot_visual_width = SCROLL_DOT_HEIGHT if dot.is_circular else SCROLL_DOT_WIDTH
      dot_y = self._rect.y + (self._rect.height - SCROLL_DOT_HEIGHT) / 2
      dot.set_position(cur_x, dot_y)
      # Store the visual width for rendering
      dot.width = dot_visual_width
      cur_x += dot_visual_width + SCROLL_DOT_SPACING

  def _render(self, _):
    self._layout()

    # Process input each frame
    self._process_indicator_input()

    # Animate dots based on closest dot
    for dot in self._dots:
      if self._closest_dot[0] is not None:
        # Calculate offset based on distance from closest dot
        closest_idx = self._closest_dot[0].index
        distance = abs(dot.index - closest_idx)

        if distance == 0:
          # Closest dot: move up 40px
          dot.set_y_offset(-40)
        elif distance == 1:
          # Adjacent dots: move up 20px
          dot.set_y_offset(-20)
        else:
          # Other dots: no offset
          dot.set_y_offset(0)
      else:
        # No touch: reset all to base position
        dot.set_y_offset(0)

    for dot in self._dots:
      # Determine dot color based on state
      is_active = self._closest_dot[0] is dot or dot.index in self._current_indices

      if dot.is_toggled_on:
        # Toggled-on dots: always green (0, 255, 38) with full opacity, even when active
        color = rl.Color(0, 255, 38, 255)
      elif dot.is_red and is_active:
        # Red button (power off) when selected: red (255, 0, 21)
        color = rl.Color(255, 0, 21, 255)
      elif is_active:
        # Active dot (being touched or current item): full opacity white
        color = rl.Color(255, 255, 255, 255)
      else:
        # Inactive dots: 25% opacity white
        color = rl.Color(255, 255, 255, 64)

      if dot.is_circular:
        # Draw as a circle (14x14) for BigCircleToggle/BigCircleButton
        circle_size = SCROLL_DOT_HEIGHT  # 14px
        # Center the circle horizontally within the dot's width
        center_x = dot.x + dot.width / 2
        center_y = dot.get_animated_y() + dot.height / 2
        rl.draw_circle(int(center_x), int(center_y), circle_size / 2, color)
      else:
        # Draw the dot as a pill-shaped rounded rectangle using animated y position
        dot_rect = rl.Rectangle(dot.x, dot.get_animated_y(), dot.width, dot.height)
        rl.draw_rectangle_rounded(dot_rect, 1.0, 8, color)

    # Draw position indicator pill (half cut off at the bottom of the screen)
    if self._dots:
      first_dot = self._dots[0]
      last_dot = self._dots[-1]

      # Track range: 8px left of first dot to 8px right of last dot, minus pill width
      track_left = first_dot.x - POSITION_PILL_MARGIN
      track_right = last_dot.x + last_dot.width + POSITION_PILL_MARGIN - POSITION_PILL_WIDTH

      # Position pill based on scroll progress
      pill_x = track_left + self._scroll_progress * (track_right - track_left)
      # Half cut off at the bottom of the screen (only top 8px visible)
      pill_y = self._parent_bottom - POSITION_PILL_HEIGHT / 2

      pill_rect = rl.Rectangle(pill_x, pill_y, POSITION_PILL_WIDTH, POSITION_PILL_HEIGHT)
      rl.draw_rectangle_rounded(pill_rect, 1.0, 8, rl.Color(255, 255, 255, 255))


class Scroller(Widget):
  def __init__(self, items: list[Widget], horizontal: bool = True, snap_items: bool = True, spacing: int = ITEM_SPACING,
               line_separator: bool = False, pad_start: int = ITEM_SPACING, pad_end: int = ITEM_SPACING,
               scroll_bar_margin: bool = False):
    super().__init__()
    self._items: list[Widget] = []
    self._horizontal = horizontal
    self._snap_items = snap_items
    self._spacing = spacing
    self._line_separator = LineSeparator() if line_separator else None
    self._pad_start = pad_start
    self._pad_end = pad_end
    self._scroll_bar_margin = scroll_bar_margin  # Reserve space at bottom for scroll bar

    self._reset_scroll_at_show = True

    self._scrolling_to: float | None = None
    self._scroll_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)
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

    self._txt_scroll_indicator = gui_app.texture("icons_mici/settings/vertical_scroll_indicator.png", 40, 80)

    # Interactive scroll indicator for horizontal scrollers with margin
    self._scroll_indicator: ScrollIndicator | None = None
    self._indicator_highlighted_index: int | None = None  # Track which item is highlighted by indicator
    self._item_y_offsets: dict[int, FirstOrderFilter] = {}  # Animation filters for item y-offsets
    if self._horizontal and self._scroll_bar_margin:
      self._scroll_indicator = ScrollIndicator(
        on_dot_clicked=self._on_indicator_dot_clicked,
        on_dot_drag=self._on_indicator_dot_drag
      )

    for item in items:
      self.add_widget(item)

  def set_reset_scroll_at_show(self, scroll: bool):
    self._reset_scroll_at_show = scroll

  def scroll_to(self, pos: float, smooth: bool = False):
    # already there
    if abs(pos) < 1:
      return

    # FIXME: the padding correction doesn't seem correct
    scroll_offset = self.scroll_panel.get_offset() - pos
    if smooth:
      self._scrolling_to = scroll_offset
    else:
      self.scroll_panel.set_offset(scroll_offset)

  @property
  def is_auto_scrolling(self) -> bool:
    return self._scrolling_to is not None

  def _is_touch_in_bottom_zone(self) -> bool:
    """Check if current touch is in the bottom scroll bar zone (entire 30px area)."""
    if not self._scroll_bar_margin:
      return False
    zone_top = self._rect.y + self._rect.height - SCROLL_INDICATOR_TOUCH_ZONE
    for mouse_event in gui_app.mouse_events:
      if mouse_event.left_pressed or mouse_event.left_down:
        if mouse_event.pos.y >= zone_top:
          return True
    return False

  def _calculate_clamped_scroll_offset(self, index: int) -> float | None:
    """Calculate scroll offset to show item at index, clamped to scroll bounds with margin."""
    if not self._visible_items or index >= len(self._visible_items):
      return None

    content_rect = self._get_content_rect()
    center_pos = content_rect.x + content_rect.width / 2
    bounds_size = content_rect.width

    # Calculate cumulative position of the item
    item_pos = self._pad_start
    for i, item in enumerate(self._visible_items):
      if i == index:
        # Found the item - calculate scroll offset to center it
        item_center = item_pos + item.rect.width / 2
        target_offset = center_pos - content_rect.x - item_center

        # Calculate scroll bounds (same as scroll panel)
        # max_offset = 0 means content starts at left edge
        # min_offset = bounds_size - content_size means content ends at right edge
        max_offset = 0.0
        min_offset = min(0.0, bounds_size - self._content_size)

        # Clamp to bounds - items at edges won't be centered if that would
        # push beyond the natural scroll limits
        target_offset = max(min_offset, min(max_offset, target_offset))
        return target_offset
      item_pos += item.rect.width + self._spacing

    return None

  def _on_indicator_dot_clicked(self, index: int):
    """Handle click on scroll indicator dot - scroll to center that item."""
    target_offset = self._calculate_clamped_scroll_offset(index)
    if target_offset is not None:
      self._scrolling_to = target_offset
      self._scroll_filter.x = self.scroll_panel.get_offset()

  def _on_indicator_dot_drag(self, index: int):
    """Handle drag over scroll indicator dot - scroll to center that item in real-time."""
    target_offset = self._calculate_clamped_scroll_offset(index)
    if target_offset is not None:
      self._scrolling_to = target_offset
      self._scroll_filter.x = self.scroll_panel.get_offset()

  def add_widget(self, item: Widget) -> None:
    self._items.append(item)
    item.set_touch_valid_callback(lambda: self.scroll_panel.is_touch_valid() and self.enabled)

  def set_scrolling_enabled(self, enabled: bool | Callable[[], bool]) -> None:
    """Set whether scrolling is enabled (does not affect widget enabled state)."""
    self._scroll_enabled = enabled

  def _update_state(self):
    if DO_ZOOM:
      if self._scrolling_to is not None or self.scroll_panel.state != ScrollState.STEADY:
        self._zoom_out_t = rl.get_time() + MIN_ZOOM_ANIMATION_TIME
        self._zoom_filter.update(0.85)
      else:
        if self._zoom_out_t is not None:
          if rl.get_time() > self._zoom_out_t:
            self._zoom_filter.update(1.0)
          else:
            self._zoom_filter.update(0.85)

    # Cancel auto-scroll if user starts manually scrolling
    if self._scrolling_to is not None and (self.scroll_panel.state == ScrollState.PRESSED or self.scroll_panel.state == ScrollState.MANUAL_SCROLL):
      self._scrolling_to = None

    if self._scrolling_to is not None:
      self._scroll_filter.update(self._scrolling_to)
      self.scroll_panel.set_offset(self._scroll_filter.x)

      if abs(self._scroll_filter.x - self._scrolling_to) < 1:
        self.scroll_panel.set_offset(self._scrolling_to)
        self._scrolling_to = None
    else:
      # keep current scroll position up to date
      self._scroll_filter.x = self.scroll_panel.get_offset()

  def _get_scroll(self, visible_items: list[Widget], content_size: float) -> float:
    content_rect = self._get_content_rect()

    scroll_enabled = self._scroll_enabled() if callable(self._scroll_enabled) else self._scroll_enabled
    # Disable scroll panel if touch is in the bottom scroll bar zone (entire 30px area)
    touch_in_bottom_zone = self._is_touch_in_bottom_zone()
    self.scroll_panel.set_enabled(scroll_enabled and self.enabled and not touch_in_bottom_zone)
    self.scroll_panel.update(content_rect, content_size)
    if not self._snap_items:
      return round(self.scroll_panel.get_offset())

    # Snap closest item to center
    center_pos = content_rect.x + content_rect.width / 2 if self._horizontal else content_rect.y + content_rect.height / 2
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
          snap_delta_pos = max(snap_delta_pos, (content_rect.width - self.scroll_panel.get_offset() - content_size) / 10)
        else:
          snap_delta_pos = (center_pos - (snap_item.rect.y + snap_item.rect.height / 2)) / 10
          snap_delta_pos = min(snap_delta_pos, -self.scroll_panel.get_offset() / 10)
          snap_delta_pos = max(snap_delta_pos, (content_rect.height - self.scroll_panel.get_offset() - content_size) / 10)
        self._scroll_snap_filter.update(snap_delta_pos)

      self.scroll_panel.set_offset(self.scroll_panel.get_offset() + self._scroll_snap_filter.x)

    return self.scroll_panel.get_offset()

  def _get_content_rect(self) -> rl.Rectangle:
    """Get the effective content area rect, accounting for scroll bar margin if enabled."""
    if self._horizontal and self._scroll_bar_margin:
      return rl.Rectangle(
        self._rect.x,
        self._rect.y,
        self._rect.width,
        self._rect.height - HORIZONTAL_SCROLL_BAR_MARGIN
      )
    return self._rect

  def _layout(self):
    self._visible_items = [item for item in self._items if item.is_visible]

    # Add line separator between items
    if self._line_separator is not None:
      l = len(self._visible_items)
      for i in range(1, len(self._visible_items)):
        self._visible_items.insert(l - i, self._line_separator)

    self._content_size = sum(item.rect.width if self._horizontal else item.rect.height for item in self._visible_items)
    self._content_size += self._spacing * (len(self._visible_items) - 1)
    self._content_size += self._pad_start + self._pad_end

    self._scroll_offset = self._get_scroll(self._visible_items, self._content_size)

    # Use effective content rect (smaller for horizontal to make room for scroll bar)
    content_rect = self._get_content_rect()

    rl.begin_scissor_mode(int(content_rect.x), int(content_rect.y),
                          int(content_rect.width), int(content_rect.height))

    self._item_pos_filter.update(self._scroll_offset)

    # Get highlighted item index from scroll indicator (if active)
    highlighted_idx = None
    if self._scroll_indicator is not None:
      highlighted_idx = self._scroll_indicator.get_closest_dot_index()

    # Count actual items (not line separators) for index mapping
    actual_item_indices = []
    for idx, item in enumerate(self._visible_items):
      if not isinstance(item, LineSeparator):
        actual_item_indices.append(idx)

    cur_pos = 0
    actual_idx = 0  # Track index among actual items (not separators)
    for idx, item in enumerate(self._visible_items):
      spacing = self._spacing if (idx > 0) else self._pad_start
      # Nicely lay out items horizontally/vertically
      if self._horizontal:
        x = content_rect.x + cur_pos + spacing
        # Center in rect space, shift up if scroll bar margin is enabled
        y = self._rect.y + (self._rect.height - item.rect.height) / 2
        if self._scroll_bar_margin:
          y -= HORIZONTAL_SCROLL_BAR_MARGIN
        cur_pos += item.rect.width + spacing
      else:
        x = content_rect.x + (content_rect.width - item.rect.width) / 2
        y = content_rect.y + cur_pos + spacing
        cur_pos += item.rect.height + spacing

      # Consider scroll
      if self._horizontal:
        x += self._scroll_offset
      else:
        y += self._scroll_offset

      # Add some jello effect when scrolling
      if DO_JELLO:
        if self._horizontal:
          cx = content_rect.x + content_rect.width / 2
          jello_offset = self._scroll_offset - np.interp(x + item.rect.width / 2,
                                                         [content_rect.x, cx, content_rect.x + content_rect.width],
                                                         [self._item_pos_filter.x, self._scroll_offset, self._item_pos_filter.x])
          x -= np.clip(jello_offset, -20, 20)
        else:
          cy = content_rect.y + content_rect.height / 2
          jello_offset = self._scroll_offset - np.interp(y + item.rect.height / 2,
                                                         [content_rect.y, cy, content_rect.y + content_rect.height],
                                                         [self._item_pos_filter.x, self._scroll_offset, self._item_pos_filter.x])
          y -= np.clip(jello_offset, -20, 20)

      # Apply y-offset animation for items highlighted by scroll indicator
      if self._horizontal and not isinstance(item, LineSeparator):
        # Create filter if needed
        if actual_idx not in self._item_y_offsets:
          self._item_y_offsets[actual_idx] = FirstOrderFilter(0.0, 0.08, 1 / gui_app.target_fps)

        # Set target offset: -15px if highlighted, 0 otherwise
        target_offset = -15 if (highlighted_idx is not None and actual_idx == highlighted_idx) else 0
        self._item_y_offsets[actual_idx].update(target_offset)

        # Apply animated offset
        y += self._item_y_offsets[actual_idx].x
        actual_idx += 1

      # Update item state
      item.set_position(round(x), round(y))  # round to prevent jumping when settling
      item.set_parent_rect(content_rect)

    # Update scroll indicator if enabled
    if self._scroll_indicator is not None:
      # Count only actual items (not line separators)
      actual_items = [item for item in self._visible_items if not isinstance(item, LineSeparator)]
      self._scroll_indicator.set_item_count(len(actual_items))

      # Update toggle states and shape for each item
      for idx, item in enumerate(actual_items):
        # Check if the item has a _checked attribute (toggle state)
        is_toggled = getattr(item, '_checked', False)
        self._scroll_indicator.set_toggle_state(idx, is_toggled)

        # Check if item is a circular button (BigCircleToggle, BigCircleButton)
        # These have a 180x180 size (circle buttons are smaller than rectangle buttons)
        is_circular = item.rect.width == 180 and item.rect.height == 180
        self._scroll_indicator.set_circular(idx, is_circular)

        # Check if item is a red button (power off)
        is_red = getattr(item, '_red', False)
        self._scroll_indicator.set_red(idx, is_red)

      # Find which items should be indicated as current
      # An item is "selected" if the screen center falls within its horizontal bounds
      center_pos = content_rect.x + content_rect.width / 2
      selected_indices: set[int] = set()

      for idx, item in enumerate(actual_items):
        item_left = item.rect.x
        item_right = item.rect.x + item.rect.width
        if item_left <= center_pos <= item_right:
          selected_indices.add(idx)

      # Edge cases: if first/last item is fully visible, include it
      edge_margin = 20
      first_item = actual_items[0] if actual_items else None
      last_item = actual_items[-1] if actual_items else None

      if first_item and (first_item.rect.x >= content_rect.x - edge_margin):
        selected_indices.add(0)
      if last_item and (last_item.rect.x + last_item.rect.width <= content_rect.x + content_rect.width + edge_margin):
        selected_indices.add(len(actual_items) - 1)

      # Fallback: if nothing selected, pick the closest to center
      if not selected_indices:
        closest_idx = 0
        closest_dist = float('inf')
        for idx, item in enumerate(actual_items):
          item_center = item.rect.x + item.rect.width / 2
          dist = abs(item_center - center_pos)
          if dist < closest_dist:
            closest_dist = dist
            closest_idx = idx
        selected_indices.add(closest_idx)

      self._scroll_indicator.set_current_indices(selected_indices)

      # Position the indicator at the bottom of the screen
      # 10px from bottom edge, 40px from left edge, left-justified
      indicator_rect = rl.Rectangle(
        self._rect.x + 40,  # 40px from left edge
        self._rect.y + self._rect.height - 10 - SCROLL_DOT_HEIGHT,  # 10px from bottom edge
        self._rect.width - 40,  # Remaining width
        SCROLL_DOT_HEIGHT
      )
      self._scroll_indicator.set_rect(indicator_rect)
      self._scroll_indicator.set_parent_bottom(self._rect.y + self._rect.height)

      # Calculate and pass scroll progress (0.0 = start, 1.0 = end)
      bounds_size = content_rect.width
      min_offset = min(0.0, bounds_size - self._content_size)
      if min_offset < 0:
        progress = self._scroll_offset / min_offset  # offset is 0 at start, min_offset at end
        progress = max(0.0, min(1.0, progress))
      else:
        progress = 0.0
      self._scroll_indicator.set_scroll_progress(progress)

  def _render(self, _):
    content_rect = self._get_content_rect()

    for item in self._visible_items:
      # Skip rendering if not in viewport
      if not rl.check_collision_recs(item.rect, content_rect):
        continue

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

    # Draw vertical scroll indicator (legacy)
    if SCROLL_BAR and not self._horizontal and len(self._visible_items) > 0:
      _real_content_size = self._content_size - self._rect.height + self._txt_scroll_indicator.height
      scroll_bar_y = -self._scroll_offset / _real_content_size * self._rect.height
      scroll_bar_y = min(max(scroll_bar_y, self._rect.y), self._rect.y + self._rect.height - self._txt_scroll_indicator.height)
      rl.draw_texture_ex(self._txt_scroll_indicator, rl.Vector2(self._rect.x, scroll_bar_y), 0, 1.0, rl.WHITE)

    rl.end_scissor_mode()

    # Draw interactive scroll indicator for horizontal scrollers
    if self._scroll_indicator is not None:
      self._scroll_indicator.render()

  def show_event(self):
    super().show_event()
    if self._reset_scroll_at_show:
      self.scroll_panel.set_offset(0.0)

    for item in self._items:
      item.show_event()

  def hide_event(self):
    super().hide_event()
    for item in self._items:
      item.hide_event()
