from __future__ import annotations

import abc
import pyray as rl
from enum import IntEnum
from collections.abc import Callable
from openpilot.common.filter_simple import BounceFilter, FirstOrderFilter
from openpilot.system.ui.lib.application import gui_app, MousePos, MAX_TOUCH_SLOTS, MouseEvent

try:
  from openpilot.selfdrive.ui.ui_state import device
except ImportError:
  class Device:
    awake = True
  device = Device()


class DialogResult(IntEnum):
  CANCEL = 0
  CONFIRM = 1
  NO_ACTION = -1


class Widget(abc.ABC):
  def __init__(self):
    self._rect: rl.Rectangle = rl.Rectangle(0, 0, 0, 0)
    self._parent_rect: rl.Rectangle | None = None
    self.__is_pressed = [False] * MAX_TOUCH_SLOTS
    # if current mouse/touch down started within the widget's rectangle
    self.__tracking_is_pressed = [False] * MAX_TOUCH_SLOTS
    self._enabled: bool | Callable[[], bool] = True
    self._is_visible: bool | Callable[[], bool] = True
    self._touch_valid_callback: Callable[[], bool] | None = None
    self._click_callback: Callable[[], None] | None = None
    self._multi_touch = False
    self.__was_awake = True

  @property
  def rect(self) -> rl.Rectangle:
    return self._rect

  def set_rect(self, rect: rl.Rectangle) -> None:
    changed = (self._rect.x != rect.x or self._rect.y != rect.y or
               self._rect.width != rect.width or self._rect.height != rect.height)
    self._rect = rect
    if changed:
      self._update_layout_rects()

  def set_parent_rect(self, parent_rect: rl.Rectangle) -> None:
    """Can be used like size hint in QT"""
    self._parent_rect = parent_rect

  @property
  def is_pressed(self) -> bool:
    return any(self.__is_pressed)

  @property
  def enabled(self) -> bool:
    return self._enabled() if callable(self._enabled) else self._enabled

  def set_enabled(self, enabled: bool | Callable[[], bool]) -> None:
    self._enabled = enabled

  @property
  def is_visible(self) -> bool:
    return self._is_visible() if callable(self._is_visible) else self._is_visible

  def set_visible(self, visible: bool | Callable[[], bool]) -> None:
    self._is_visible = visible

  def set_click_callback(self, click_callback: Callable[[], None] | None) -> None:
    """Set a callback to be called when the widget is clicked."""
    self._click_callback = click_callback

  def set_touch_valid_callback(self, touch_callback: Callable[[], bool]) -> None:
    """Set a callback to determine if the widget can be clicked."""
    self._touch_valid_callback = touch_callback

  def _touch_valid(self) -> bool:
    """Check if the widget can be touched."""
    return self._touch_valid_callback() if self._touch_valid_callback else True

  def set_position(self, x: float, y: float) -> None:
    changed = (self._rect.x != x or self._rect.y != y)
    self._rect = rl.Rectangle(x, y, self._rect.width, self._rect.height)
    if changed:
      self._update_layout_rects()

  @property
  def _hit_rect(self) -> rl.Rectangle:
    # restrict touches to within parent rect if set, useful inside Scroller
    if self._parent_rect is None:
      return self._rect
    return rl.get_collision_rec(self._rect, self._parent_rect)

  def render(self, rect: rl.Rectangle | None = None) -> bool | int | None:
    if rect is not None:
      self.set_rect(rect)

    self._update_state()

    if not self.is_visible:
      return None

    self._layout()
    ret = self._render(self._rect)

    # Keep track of whether mouse down started within the widget's rectangle
    if self.enabled and self.__was_awake:
      self._process_mouse_events()

    self.__was_awake = device.awake

    return ret

  def _process_mouse_events(self) -> None:
    hit_rect = self._hit_rect
    touch_valid = self._touch_valid()

    for mouse_event in gui_app.mouse_events:
      if not self._multi_touch and mouse_event.slot != 0:
        continue

      mouse_in_rect = rl.check_collision_point_rec(mouse_event.pos, hit_rect)
      # Ignores touches/presses that start outside our rect
      # Allows touch to leave the rect and come back in focus if mouse did not release
      if mouse_event.left_pressed and touch_valid:
        if mouse_in_rect:
          self._handle_mouse_press(mouse_event.pos)
          self.__is_pressed[mouse_event.slot] = True
          self.__tracking_is_pressed[mouse_event.slot] = True
          self._handle_mouse_event(mouse_event)

      # Callback such as scroll panel signifies user is scrolling
      elif not touch_valid:
        self.__is_pressed[mouse_event.slot] = False
        self.__tracking_is_pressed[mouse_event.slot] = False

      elif mouse_event.left_released:
        self._handle_mouse_event(mouse_event)
        if self.__is_pressed[mouse_event.slot] and mouse_in_rect:
          self._handle_mouse_release(mouse_event.pos)
        self.__is_pressed[mouse_event.slot] = False
        self.__tracking_is_pressed[mouse_event.slot] = False

      # Mouse/touch is still within our rect
      elif mouse_in_rect:
        if self.__tracking_is_pressed[mouse_event.slot]:
          self.__is_pressed[mouse_event.slot] = True
          self._handle_mouse_event(mouse_event)

      # Mouse/touch left our rect but may come back into focus later
      elif not mouse_in_rect:
        self.__is_pressed[mouse_event.slot] = False
        self._handle_mouse_event(mouse_event)

  def _layout(self) -> None:
    """Optionally lay out child widgets separately. This is called before rendering."""

  def _update_state(self):
    """Optionally update the widget's non-layout state. This is called before rendering."""

  @abc.abstractmethod
  def _render(self, rect: rl.Rectangle) -> bool | int | None:
    """Render the widget within the given rectangle."""

  def _update_layout_rects(self) -> None:
    """Optionally update any layout rects on Widget rect change."""

  def _handle_mouse_press(self, mouse_pos: MousePos) -> None:
    """Optionally handle mouse press events."""

  def _handle_mouse_release(self, mouse_pos: MousePos) -> None:
    """Optionally handle mouse release events."""
    if self._click_callback:
      self._click_callback()

  def _handle_mouse_event(self, mouse_event: MouseEvent) -> None:
    """Optionally handle mouse events. This is called before rendering."""
    # Default implementation does nothing, can be overridden by subclasses

  def show_event(self):
    """Optionally handle show event. Parent must manually call this"""

  def hide_event(self):
    """Optionally handle hide event. Parent must manually call this"""


SWIPE_AWAY_THRESHOLD = 80  # px to dismiss after releasing
START_DISMISSING_THRESHOLD = 40  # px to start dismissing while dragging
BLOCK_SWIPE_AWAY_THRESHOLD = 60  # px horizontal movement to block swipe away

NAV_BAR_MARGIN = 6
NAV_BAR_WIDTH = 205
NAV_BAR_HEIGHT = 8

DISMISS_PUSH_OFFSET = 50 + NAV_BAR_MARGIN + NAV_BAR_HEIGHT  # px extra to push down when dismissing
DISMISS_TIME_SECONDS = 2.0


class NavBar(Widget):
  def __init__(self):
    super().__init__()
    self.set_rect(rl.Rectangle(0, 0, NAV_BAR_WIDTH, NAV_BAR_HEIGHT))
    self._alpha = 1.0
    self._alpha_filter = FirstOrderFilter(1.0, 0.1, 1 / gui_app.target_fps)
    self._fade_time = 0.0

  def set_alpha(self, alpha: float) -> None:
    self._alpha = alpha
    self._fade_time = rl.get_time()

  def show_event(self):
    super().show_event()
    self._alpha = 1.0
    self._alpha_filter.x = 1.0
    self._fade_time = rl.get_time()

  def _render(self, _):
    if rl.get_time() - self._fade_time > DISMISS_TIME_SECONDS:
      self._alpha = 0.0
    alpha = self._alpha_filter.update(self._alpha)

    # white bar with black border
    rl.draw_rectangle_rounded(self._rect, 1.0, 6, rl.Color(255, 255, 255, int(255 * 0.9 * alpha)))
    rl.draw_rectangle_rounded_lines_ex(self._rect, 1.0, 6, 2, rl.Color(0, 0, 0, int(255 * 0.3 * alpha)))


class NavWidget(Widget, abc.ABC):
  """
  A full screen widget that supports back navigation by swiping down from the top.
  """
  BACK_TOUCH_AREA_PERCENTAGE = 0.65

  def __init__(self):
    super().__init__()
    self._back_callback: Callable[[], None] | None = None
    self._back_button_start_pos: MousePos | None = None
    self._swiping_away = False  # currently swiping away
    self._can_swipe_away = True  # swipe away is blocked after certain horizontal movement

    self._pos_filter = BounceFilter(0.0, 0.1, 1 / gui_app.target_fps, bounce=1)
    self._playing_dismiss_animation = False
    self._trigger_animate_in = False
    self._nav_bar_show_time = 0.0
    self._back_enabled: bool | Callable[[], bool] = True
    self._nav_bar = NavBar()

    self._nav_bar_y_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)

    self._set_up = False

  @property
  def back_enabled(self) -> bool:
    return self._back_enabled() if callable(self._back_enabled) else self._back_enabled

  def set_back_enabled(self, enabled: bool | Callable[[], bool]) -> None:
    self._back_enabled = enabled

  def set_back_callback(self, callback: Callable[[], None]) -> None:
    self._back_callback = callback

  def _handle_mouse_event(self, mouse_event: MouseEvent) -> None:
    super()._handle_mouse_event(mouse_event)

    if not self.back_enabled:
      self._back_button_start_pos = None
      self._swiping_away = False
      self._can_swipe_away = True
      return

    if mouse_event.left_pressed:
      # user is able to swipe away if starting near top of screen, or anywhere if scroller is at top
      self._pos_filter.update_alpha(0.04)
      in_dismiss_area = mouse_event.pos.y < self._rect.height * self.BACK_TOUCH_AREA_PERCENTAGE

      scroller_at_top = False
      vertical_scroller = False
      # TODO: -20? snapping in WiFi dialog can make offset not be positive at the top
      if hasattr(self, '_scroller'):
        scroller_at_top = self._scroller.scroll_panel.get_offset() >= -20 and not self._scroller._horizontal
        vertical_scroller = not self._scroller._horizontal
      elif hasattr(self, '_scroll_panel'):
        scroller_at_top = self._scroll_panel.get_offset() >= -20 and not self._scroll_panel._horizontal
        vertical_scroller = not self._scroll_panel._horizontal

      # Vertical scrollers need to be at the top to swipe away to prevent erroneous swipes
      if (not vertical_scroller and in_dismiss_area) or scroller_at_top:
        self._can_swipe_away = True
        self._back_button_start_pos = mouse_event.pos

    elif mouse_event.left_down:
      if self._back_button_start_pos is not None:
        # block swiping away if too much horizontal or upward movement
        horizontal_movement = abs(mouse_event.pos.x - self._back_button_start_pos.x) > BLOCK_SWIPE_AWAY_THRESHOLD
        upward_movement = mouse_event.pos.y - self._back_button_start_pos.y < -BLOCK_SWIPE_AWAY_THRESHOLD
        if not self._swiping_away and (horizontal_movement or upward_movement):
          self._can_swipe_away = False
          self._back_button_start_pos = None

        # block horizontal swiping if now swiping away
        if self._can_swipe_away:
          if mouse_event.pos.y - self._back_button_start_pos.y > START_DISMISSING_THRESHOLD:
            self._swiping_away = True

    elif mouse_event.left_released:
      self._pos_filter.update_alpha(0.1)
      # if far enough, trigger back navigation callback
      if self._back_button_start_pos is not None:
        if mouse_event.pos.y - self._back_button_start_pos.y > SWIPE_AWAY_THRESHOLD:
          self._playing_dismiss_animation = True

      self._back_button_start_pos = None
      self._swiping_away = False

  def _update_state(self):
    super()._update_state()

    # Disable self's scroller while swiping away
    if not self._set_up:
      self._set_up = True
      if hasattr(self, '_scroller'):
        original_enabled = self._scroller._enabled
        self._scroller.set_enabled(lambda: not self._swiping_away and (original_enabled() if callable(original_enabled) else
                                                                       original_enabled))
      elif hasattr(self, '_scroll_panel'):
        original_enabled = self._scroll_panel.enabled
        self._scroll_panel.set_enabled(lambda: not self._swiping_away and (original_enabled() if callable(original_enabled) else
                                                                          original_enabled))

    if self._trigger_animate_in:
      self._pos_filter.x = self._rect.height
      self._nav_bar_y_filter.x = -NAV_BAR_MARGIN - NAV_BAR_HEIGHT
      self._nav_bar_show_time = rl.get_time()
      self._trigger_animate_in = False

    new_y = 0.0

    if self._back_button_start_pos is not None:
      last_mouse_event = gui_app.last_mouse_event
      # push entire widget as user drags it away
      new_y = max(last_mouse_event.pos.y - self._back_button_start_pos.y, 0)
      if new_y < SWIPE_AWAY_THRESHOLD:
        new_y /= 2  # resistance until mouse release would dismiss widget

    if self._swiping_away:
      self._nav_bar.set_alpha(1.0)

    if self._playing_dismiss_animation:
      new_y = self._rect.height + DISMISS_PUSH_OFFSET

    new_y = round(self._pos_filter.update(new_y))
    if abs(new_y) < 1 and self._pos_filter.velocity.x == 0.0:
      new_y = self._pos_filter.x = 0.0

    if new_y > self._rect.height + DISMISS_PUSH_OFFSET - 10:
      if self._back_callback is not None:
        self._back_callback()

      self._playing_dismiss_animation = False
      self._back_button_start_pos = None
      self._swiping_away = False

    self.set_position(self._rect.x, new_y)

  def render(self, rect: rl.Rectangle | None = None) -> bool | int | None:
    ret = super().render(rect)

    if self.back_enabled:
      bar_x = self._rect.x + (self._rect.width - self._nav_bar.rect.width) / 2
      nav_bar_delayed = rl.get_time() - self._nav_bar_show_time < 0.4
      # User dragging or dismissing, nav bar follows NavWidget
      if self._back_button_start_pos is not None or self._playing_dismiss_animation:
        self._nav_bar_y_filter.x = NAV_BAR_MARGIN + self._pos_filter.x
      # Waiting to show
      elif nav_bar_delayed:
        self._nav_bar_y_filter.x = -NAV_BAR_MARGIN - NAV_BAR_HEIGHT
      # Animate back to top
      else:
        self._nav_bar_y_filter.update(NAV_BAR_MARGIN)

      # draw black above widget when dismissing
      if self._rect.y > 0:
        rl.draw_rectangle(int(self._rect.x), 0, int(self._rect.width), int(self._rect.y), rl.BLACK)

      self._nav_bar.set_position(bar_x, round(self._nav_bar_y_filter.x))
      self._nav_bar.render()

    return ret

  def show_event(self):
    super().show_event()
    # FIXME: we don't know the height of the rect at first show_event since it's before the first render :(
    #  so we need this hacky bool for now
    self._trigger_animate_in = True
    self._nav_bar.show_event()
