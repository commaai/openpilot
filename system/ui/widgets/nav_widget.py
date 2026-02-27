from __future__ import annotations

import abc
import pyray as rl
from collections.abc import Callable
from openpilot.system.ui.widgets import Widget
from openpilot.common.filter_simple import BounceFilter, FirstOrderFilter
from openpilot.system.ui.lib.application import gui_app, MousePos, MouseEvent

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

  @property
  def back_enabled(self) -> bool:
    return self._back_enabled() if callable(self._back_enabled) else self._back_enabled

  def set_back_enabled(self, enabled: bool | Callable[[], bool]) -> None:
    self._back_enabled = enabled

  def set_back_callback(self, callback: Callable[[], None]) -> None:
    self._back_callback = callback

  def _handle_mouse_event(self, mouse_event: MouseEvent) -> None:
    # FIXME: disabling this widget on new push_widget still causes this widget to track mouse events without mouse down
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

      # TODO: remove vertical scrolling and then this hacky logic to check if scroller is at top
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

    if self._trigger_animate_in:
      self._pos_filter.x = self._rect.height
      self._nav_bar_y_filter.x = -NAV_BAR_MARGIN - NAV_BAR_HEIGHT
      self._nav_bar_show_time = rl.get_time()
      self._trigger_animate_in = False

    new_y = 0.0

    if not self.enabled:
      self._back_button_start_pos = None

    # TODO: why is this not in handle_mouse_event? have to hack above
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

  def _layout(self):
    # Dim whatever is behind this widget, fading with position (runs after _update_state so position is correct)
    overlay_alpha = int(200 * max(0.0, min(1.0, 1.0 - self._rect.y / self._rect.height))) if self._rect.height > 0 else 0
    rl.draw_rectangle(0, 0, int(self._rect.width), int(self._rect.height), rl.Color(0, 0, 0, overlay_alpha))

    bounce_height = 20
    rl.draw_rectangle(int(self._rect.x), int(self._rect.y), int(self._rect.width), int(self._rect.height + bounce_height), rl.BLACK)

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

      self._nav_bar.set_position(bar_x, round(self._nav_bar_y_filter.x))
      self._nav_bar.render()

    return ret

  def show_event(self):
    super().show_event()
    # FIXME: we don't know the height of the rect at first show_event since it's before the first render :(
    #  so we need this hacky bool for now
    self._trigger_animate_in = True
    self._nav_bar.show_event()

    # Reset state
    self._pos_filter.update_alpha(0.1)
    self._back_button_start_pos = None
    self._swiping_away = False
