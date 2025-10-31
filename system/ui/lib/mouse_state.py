import pyray as rl
import time
import threading
from collections import deque
from typing import NamedTuple
from openpilot.common.realtime import Ratekeeper


MAX_TOUCH_SLOTS = 2
MOUSE_THREAD_RATE = 140  # touch controller runs at 140Hz


class MousePos(NamedTuple):
  x: float
  y: float


class MousePosWithTime(NamedTuple):
  x: float
  y: float
  t: float


class MouseEvent(NamedTuple):
  pos: MousePos
  slot: int
  left_pressed: bool
  left_released: bool
  left_down: bool
  t: float


class MouseState:
  def __init__(self, scale: float = 1.0):
    self._scale = scale
    self._events: deque[MouseEvent] = deque(maxlen=MOUSE_THREAD_RATE)  # bound event list
    self._prev_mouse_event: list[MouseEvent | None] = [None] * MAX_TOUCH_SLOTS

    self._rk = Ratekeeper(MOUSE_THREAD_RATE, print_delay_threshold=None)
    self._lock = threading.Lock()
    self._exit_event = threading.Event()
    self._thread = None

  def get_events(self) -> list[MouseEvent]:
    with self._lock:
      events = list(self._events)
      self._events.clear()
    return events

  def start(self):
    self._exit_event.clear()
    if self._thread is None or not self._thread.is_alive():
      self._thread = threading.Thread(target=self._run_thread, daemon=True)
      self._thread.start()

  def stop(self):
    self._exit_event.set()
    if self._thread is not None and self._thread.is_alive():
      self._thread.join()

  def _run_thread(self):
    while not self._exit_event.is_set():
      rl.poll_input_events()
      self._handle_mouse_event()
      self._rk.keep_time()

  def _handle_mouse_event(self):
    for slot in range(MAX_TOUCH_SLOTS):
      mouse_pos = rl.get_touch_position(slot)
      x = mouse_pos.x / self._scale if self._scale != 1.0 else mouse_pos.x
      y = mouse_pos.y / self._scale if self._scale != 1.0 else mouse_pos.y
      ev = MouseEvent(
        MousePos(x, y),
        slot,
        rl.is_mouse_button_pressed(slot),  # noqa: TID251
        rl.is_mouse_button_released(slot),  # noqa: TID251
        rl.is_mouse_button_down(slot),
        time.monotonic(),
      )
      # Only add changes
      if self._prev_mouse_event[slot] is None or ev[:-1] != self._prev_mouse_event[slot][:-1]:
        with self._lock:
          self._events.append(ev)
        self._prev_mouse_event[slot] = ev
