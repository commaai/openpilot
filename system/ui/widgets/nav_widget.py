from __future__ import annotations

import abc
import pyray as rl
from openpilot.system.ui.widgets import Widget
from openpilot.common.filter_simple import BounceFilter, FirstOrderFilter
from openpilot.system.ui.lib.application import gui_app, MousePos, MouseEvent

SWIPE_AWAY_THRESHOLD = 80  # px to dismiss after releasing
START_DISMISSING_THRESHOLD = 40  # px to start dismissing while dragging
BLOCK_SWIPE_AWAY_THRESHOLD = 60  # px horizontal movement to block swipe away

NAV_BAR_MARGIN = 6
NAV_BAR_WIDTH = 205
NAV_BAR_HEIGHT = 8

DISMISS_PUSH_OFFSET = NAV_BAR_MARGIN + NAV_BAR_HEIGHT + 50  # px extra to push down when dismissing


class NavBar(Widget):
  FADE_AFTER_SECONDS = 2.0

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
    if rl.get_time() - self._fade_time > self.FADE_AFTER_SECONDS:
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
    # State
    self._drag_start_pos: MousePos | None = None  # cleared after certain amount of horizontal movement
    self._dragging_down = False  # swiped down enough to trigger dismissing on release
    self._playing_dismiss_animation = False  # released and animating away
    self._y_pos_filter = BounceFilter(0.0, 0.1, 1 / gui_app.target_fps, bounce=1)

    # TODO: move this state into NavBar
    self._nav_bar = NavBar()
    self._nav_bar_show_time = 0.0
    self._nav_bar_y_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)

  def _back_enabled(self) -> bool:
    # Children can override this to block swipe away, like when not at
    # the top of a vertical scroll panel to prevent erroneous swipes
    return True

  def _handle_mouse_event(self, mouse_event: MouseEvent) -> None:
    super()._handle_mouse_event(mouse_event)

    if mouse_event.left_pressed:
      # user is able to swipe away if starting near top of screen
      self._y_pos_filter.update_alpha(0.04)
      in_dismiss_area = mouse_event.pos.y < self._rect.height * self.BACK_TOUCH_AREA_PERCENTAGE

      if in_dismiss_area and self._back_enabled():
        self._drag_start_pos = mouse_event.pos

    elif mouse_event.left_down:
      if self._drag_start_pos is not None:
        # block swiping away if too much horizontal or upward movement
        # block (lock-in) threshold is higher than start dismissing
        horizontal_movement = abs(mouse_event.pos.x - self._drag_start_pos.x) > BLOCK_SWIPE_AWAY_THRESHOLD
        upward_movement = mouse_event.pos.y - self._drag_start_pos.y < -BLOCK_SWIPE_AWAY_THRESHOLD

        if not (horizontal_movement or upward_movement):
          # no blocking movement, check if we should start dismissing
          if mouse_event.pos.y - self._drag_start_pos.y > START_DISMISSING_THRESHOLD:
            self._dragging_down = True
        else:
          if not self._dragging_down:
            self._drag_start_pos = None

    elif mouse_event.left_released:
      # reset rc for either slide up or down animation
      self._y_pos_filter.update_alpha(0.1)

      # if far enough, trigger back navigation callback
      if self._drag_start_pos is not None:
        if mouse_event.pos.y - self._drag_start_pos.y > SWIPE_AWAY_THRESHOLD:
          self._playing_dismiss_animation = True

      self._drag_start_pos = None
      self._dragging_down = False

  def _update_state(self):
    super()._update_state()

    new_y = 0.0

    if self._dragging_down:
      self._nav_bar.set_alpha(1.0)

    # FIXME: disabling this widget on new push_widget still causes this widget to track mouse events without mouse down
    if not self.enabled:
      self._drag_start_pos = None

    if self._drag_start_pos is not None:
      last_mouse_event = gui_app.last_mouse_event
      # push entire widget as user drags it away
      new_y = max(last_mouse_event.pos.y - self._drag_start_pos.y, 0)
      if new_y < SWIPE_AWAY_THRESHOLD:
        new_y /= 2  # resistance until mouse release would dismiss widget

    if self._playing_dismiss_animation:
      new_y = self._rect.height + DISMISS_PUSH_OFFSET

    new_y = round(self._y_pos_filter.update(new_y))
    if abs(new_y) < 1 and self._y_pos_filter.velocity.x == 0.0:
      new_y = self._y_pos_filter.x = 0.0

    if new_y > self._rect.height + DISMISS_PUSH_OFFSET - 10:
      gui_app.pop_widget()

      self._playing_dismiss_animation = False
      self._drag_start_pos = None
      self._dragging_down = False

    self.set_position(self._rect.x, new_y)

  def _layout(self):
    # Dim whatever is behind this widget, fading with position (runs after _update_state so position is correct)
    overlay_alpha = int(200 * max(0.0, min(1.0, 1.0 - self._rect.y / self._rect.height))) if self._rect.height > 0 else 0
    rl.draw_rectangle(0, 0, int(self._rect.width), int(self._rect.height), rl.Color(0, 0, 0, overlay_alpha))

    bounce_height = 20
    rl.draw_rectangle(int(self._rect.x), int(self._rect.y), int(self._rect.width), int(self._rect.height + bounce_height), rl.BLACK)

  def render(self, rect: rl.Rectangle | None = None) -> bool | int | None:
    ret = super().render(rect)

    bar_x = self._rect.x + (self._rect.width - self._nav_bar.rect.width) / 2
    nav_bar_delayed = rl.get_time() - self._nav_bar_show_time < 0.4
    # User dragging or dismissing, nav bar follows NavWidget
    if self._drag_start_pos is not None or self._playing_dismiss_animation:
      self._nav_bar_y_filter.x = NAV_BAR_MARGIN + self._y_pos_filter.x
    # Waiting to show
    elif nav_bar_delayed:
      self._nav_bar_y_filter.x = -NAV_BAR_MARGIN - NAV_BAR_HEIGHT
    # Animate back to top
    else:
      self._nav_bar_y_filter.update(NAV_BAR_MARGIN)

    self._nav_bar.set_position(bar_x, round(self._nav_bar_y_filter.x))
    self._nav_bar.render()

    return ret

  @property
  def is_dismissing(self) -> bool:
    return self._dragging_down or self._playing_dismiss_animation

  def show_event(self):
    super().show_event()
    self._nav_bar.show_event()

    # Reset state
    self._drag_start_pos = None
    self._dragging_down = False
    self._playing_dismiss_animation = False
    # Start NavWidget off-screen, no matter how tall it is
    self._y_pos_filter.update_alpha(0.1)
    self._y_pos_filter.x = gui_app.height

    self._nav_bar_y_filter.x = -NAV_BAR_MARGIN - NAV_BAR_HEIGHT
    self._nav_bar_show_time = rl.get_time()
