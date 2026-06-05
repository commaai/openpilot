import os
import math
import pyray as rl
from collections.abc import Callable
from enum import Enum
from typing import cast
from openpilot.system.ui.lib.application import gui_app, MouseEvent
from openpilot.system.hardware import TICI
from collections import deque

MIN_VELOCITY = 10  # px/s, changes from auto scroll to steady state
MIN_VELOCITY_FOR_CLICKING = 2 * 60  # px/s, accepts clicks while auto scrolling below this velocity
MIN_DRAG_PIXELS = 12
AUTO_SCROLL_TC_SNAP = 0.025
AUTO_SCROLL_TC = 0.18
BOUNCE_RETURN_RATE = 10.0
SNAP_RATE = 6.3  # matches previous Scroller snapping. exp rate of approach to snap target, 1/s
REJECT_DECELERATION_FACTOR = 3
MAX_SPEED = 10000.0  # px/s

DEBUG = os.getenv("DEBUG_SCROLL", "0") == "1"


# Weights older (steadier) velocity samples more heavily on release.
# Finger-lift samples are noisy; trusting earlier samples gives consistent fling velocity.
# Reverse-engineered from iOS UIScrollView (tuned at 120Hz touch) by Flutter team:
# https://github.com/flutter/flutter/pull/60501
# 3 samples ≈ 25ms at 120Hz (iOS) / ~21ms at 140Hz (comma). Scale if touch rate changes.
def weighted_velocity(buffer: deque) -> float:
  if len(buffer) >= 3:
    return buffer[-3] * 0.6 + buffer[-2] * 0.35 + buffer[-1] * 0.05
  elif len(buffer) == 2:
    return buffer[-2] * 0.7 + buffer[-1] * 0.3
  elif len(buffer) == 1:
    return buffer[-1]
  return 0.0


# from https://ariya.io/2011/10/flick-list-with-its-momentum-scrolling-and-deceleration
class ScrollState(Enum):
  STEADY = 0
  PRESSED = 1
  MANUAL_SCROLL = 2
  AUTO_SCROLL = 3


class GuiScrollPanel2:
  def __init__(self, horizontal: bool = True) -> None:
    self._horizontal = horizontal
    self._state = ScrollState.STEADY
    self._offset: rl.Vector2 = rl.Vector2(0, 0)
    self._initial_click_event: MouseEvent | None = None
    self._previous_mouse_event: MouseEvent | None = None
    self._velocity = 0.0  # pixels per second
    self._velocity_buffer: deque[float] = deque(maxlen=12 if TICI else 6)
    self._enabled: bool | Callable[[], bool] = True

  def set_enabled(self, enabled: bool | Callable[[], bool]) -> None:
    self._enabled = enabled

  @property
  def enabled(self) -> bool:
    return self._enabled() if callable(self._enabled) else self._enabled

  def update(self, bounds: rl.Rectangle, content_size: float, snap_target: float | None = None) -> float:
    if DEBUG:
      print('Old state:', self._state)

    bounds_size = bounds.width if self._horizontal else bounds.height

    for mouse_event in gui_app.mouse_events:
      self._handle_mouse_event(mouse_event, bounds, bounds_size, content_size)
      self._previous_mouse_event = mouse_event

    self._update_state(bounds_size, content_size, snap_target)

    if DEBUG:
      print('Velocity:', self._velocity)
      print('Offset X:', self._offset.x, 'Y:', self._offset.y)
      print('New state:', self._state)
      print()
    return self.get_offset()

  def _get_offset_bounds(self, bounds_size: float, content_size: float) -> tuple[float, float]:
    """Returns (max_offset, min_offset) for the given bounds and content size."""
    return 0.0, min(0.0, bounds_size - content_size)

  def _update_state(self, bounds_size: float, content_size: float, snap_target: float | None) -> None:
    """Runs per render frame, independent of mouse events. Updates auto-scrolling state and velocity."""
    max_offset, min_offset = self._get_offset_bounds(bounds_size, content_size)

    if self._state == ScrollState.STEADY:
      # if we find ourselves out of bounds, scroll back in (from external layout dimension changes, etc.)
      if self.get_offset() > max_offset or self.get_offset() < min_offset:
        self._state = ScrollState.AUTO_SCROLL

    elif self._state == ScrollState.AUTO_SCROLL:
      # simple exponential return if out of bounds
      # out of bounds is handled by snapping, so skip if set
      out_of_bounds = self.get_offset() > max_offset or self.get_offset() < min_offset
      if out_of_bounds and snap_target is None:
        target = max_offset if self.get_offset() > max_offset else min_offset

        dt = rl.get_frame_time() or 1e-6
        factor = 1.0 - math.exp(-BOUNCE_RETURN_RATE * dt)

        dist = target - self.get_offset()
        self.set_offset(self.get_offset() + dist * factor)  # ease toward the edge
        self._velocity *= (1.0 - factor)  # damp any leftover fling

        # Steady once we are close enough to the target
        if abs(dist) < 1 and abs(self._velocity) < MIN_VELOCITY:
          self.set_offset(target)
          self._velocity = 0.0
          self._state = ScrollState.STEADY

      elif abs(self._velocity) < MIN_VELOCITY:
        self._velocity = 0.0
        self._state = ScrollState.STEADY

      # Update the offset based on the current velocity
      dt = rl.get_frame_time()
      self.set_offset(self.get_offset() + self._velocity * dt)  # Adjust the offset based on velocity
      # fast decay in snap mode so velocity yields to the snap pull instead of fighting it
      auto_scroll_tc = AUTO_SCROLL_TC_SNAP if snap_target is not None else AUTO_SCROLL_TC
      alpha = 1 - (dt / (auto_scroll_tc + dt))
      self._velocity *= alpha

    # Ease toward snap target when not in user control. Composes with velocity coast above:
    # high velocity dominates initially, snap dominates as velocity decays.
    if snap_target is not None and self._state not in (ScrollState.PRESSED, ScrollState.MANUAL_SCROLL):
      snap_target = max(min_offset, min(max_offset, snap_target))
      dist = snap_target - self.get_offset()
      if abs(dist) < 1:  # finished snap
        self.set_offset(snap_target)
      else:
        dt = rl.get_frame_time() or 1e-6
        factor = 1.0 - math.exp(-SNAP_RATE * dt)
        self.set_offset(self.get_offset() + dist * factor)

  def _handle_mouse_event(self, mouse_event: MouseEvent, bounds: rl.Rectangle, bounds_size: float,
                          content_size: float) -> None:
    max_offset, min_offset = self._get_offset_bounds(bounds_size, content_size)
    # simple exponential return if out of bounds
    out_of_bounds = self.get_offset() > max_offset or self.get_offset() < min_offset
    if DEBUG:
      print('Mouse event:', mouse_event)

    mouse_pos = self._get_mouse_pos(mouse_event)

    if not self.enabled:
      # Reset state if not enabled
      self._state = ScrollState.STEADY
      self._velocity = 0.0
      self._velocity_buffer.clear()

    elif self._state == ScrollState.STEADY:
      if rl.check_collision_point_rec(mouse_event.pos, bounds):
        if mouse_event.left_pressed:
          self._state = ScrollState.PRESSED
          self._initial_click_event = mouse_event

    elif self._state == ScrollState.PRESSED:
      initial_click_pos = self._get_mouse_pos(cast(MouseEvent, self._initial_click_event))
      diff = abs(mouse_pos - initial_click_pos)
      if mouse_event.left_released:
        # Special handling for down and up clicks across two frames
        # TODO: not sure what that means or if it's accurate anymore
        if out_of_bounds:
          self._state = ScrollState.AUTO_SCROLL
        elif diff <= MIN_DRAG_PIXELS:
          self._state = ScrollState.STEADY
        else:
          self._state = ScrollState.MANUAL_SCROLL
      elif diff > MIN_DRAG_PIXELS:
        self._state = ScrollState.MANUAL_SCROLL

    elif self._state == ScrollState.MANUAL_SCROLL:
      if mouse_event.left_released:
        # Touch rejection: when releasing finger after swiping and stopping, panel
        # reports a few erroneous touch events with high velocity, try to ignore.

        # If velocity decelerates very quickly, assume user doesn't intend to auto scroll.
        # Catches two cases: 1) swipe, stop finger, then lift (stale high velocity in buffer)
        # 2) dirty finger lift where finger rotates/slides producing spurious velocity spike.
        # TODO: this heuristic false-positives on fast swipes because 140Hz touch polling
        #  jitter causes velocity to oscillate (not real deceleration). Better approaches:
        #  - Use evdev kernel timestamps to eliminate velocity oscillation at the source
        #  - Replace with a time-since-last-event check (40ms timeout) for swipe-stop-lift
        high_decel = False
        if len(self._velocity_buffer) > 2:
          # We limit max to first half since final few velocities can surpass first few
          abs_velocity_buffer = [(abs(v), i) for i, v in enumerate(self._velocity_buffer)]
          max_idx = max(abs_velocity_buffer[:len(abs_velocity_buffer) // 2])[1]
          min_idx = min(abs_velocity_buffer)[1]
          if DEBUG:
            print('min_idx:', min_idx, 'max_idx:', max_idx, 'velocity buffer:', self._velocity_buffer)
          if (abs(self._velocity_buffer[min_idx]) * REJECT_DECELERATION_FACTOR < abs(self._velocity_buffer[max_idx]) and
              max_idx < min_idx):
            if DEBUG:
              print('deceleration too high, going to STEADY')
            high_decel = True

        self._velocity = weighted_velocity(self._velocity_buffer)

        # If final velocity is below some threshold, switch to steady state too
        low_speed = abs(self._velocity) <= MIN_VELOCITY_FOR_CLICKING * 1.5  # plus some margin

        if out_of_bounds or not (high_decel or low_speed):
          self._state = ScrollState.AUTO_SCROLL
        else:
          # TODO: we should just set velocity and let autoscroll go back to steady. delays one frame but who cares
          self._velocity = 0.0
          self._state = ScrollState.STEADY
        self._velocity_buffer.clear()
      else:
        # Update velocity for when we release the mouse button.
        # Do not update velocity on the same frame the mouse was released
        previous_mouse_pos = self._get_mouse_pos(cast(MouseEvent, self._previous_mouse_event))
        delta_x = mouse_pos - previous_mouse_pos
        delta_t = max((mouse_event.t - cast(MouseEvent, self._previous_mouse_event).t), 1e-6)
        self._velocity = delta_x / delta_t
        self._velocity = max(-MAX_SPEED, min(MAX_SPEED, self._velocity))
        self._velocity_buffer.append(self._velocity)

        # rubber-banding: reduce dragging when out of bounds
        # TODO: this drifts when dragging quickly
        if out_of_bounds:
          delta_x *= 0.25

        # Update the offset based on the mouse movement
        # Use internal _offset directly to preserve precision (don't round via get_offset())
        # TODO: make get_offset return float
        current_offset = self._offset.x if self._horizontal else self._offset.y
        self.set_offset(current_offset + delta_x)

    elif self._state == ScrollState.AUTO_SCROLL:
      if mouse_event.left_pressed:
        # Decide whether to click or scroll (block click if moving too fast)
        if abs(self._velocity) <= MIN_VELOCITY_FOR_CLICKING:
          # Traveling slow enough, click
          self._state = ScrollState.PRESSED
          self._initial_click_event = mouse_event
        else:
          # Go straight into manual scrolling to block erroneous input
          self._state = ScrollState.MANUAL_SCROLL
          # Reset velocity for touch down and up events that happen in back-to-back frames
          self._velocity = 0.0

  def _get_mouse_pos(self, mouse_event: MouseEvent) -> float:
    return mouse_event.pos.x if self._horizontal else mouse_event.pos.y

  def get_offset(self) -> float:
    return self._offset.x if self._horizontal else self._offset.y

  def set_offset(self, value: float) -> None:
    if self._horizontal:
      self._offset.x = value
    else:
      self._offset.y = value

  @property
  def state(self) -> ScrollState:
    return self._state

  def is_touch_valid(self) -> bool:
    # MIN_VELOCITY_FOR_CLICKING is checked in auto-scroll state
    return bool(self._state != ScrollState.MANUAL_SCROLL)
