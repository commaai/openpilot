"""Utilities for reading real time clocks and keeping soft real time constraints."""
import gc
import os
import time
from collections import deque
from typing import Optional, List, Union

from setproctitle import getproctitle  # pylint: disable=no-name-in-module

from common.clock import sec_since_boot  # pylint: disable=no-name-in-module, import-error
from system.hardware import PC


# time step for each process
DT_CTRL = 0.01  # controlsd
DT_MDL = 0.05  # model
DT_TRML = 0.5  # thermald and manager
DT_DMON = 0.05  # driver monitoring


class Priority:
  # CORE 2
  # - modeld = 55
  # - camerad = 54
  CTRL_LOW = 51 # plannerd & radard

  # CORE 3
  # - boardd = 55
  CTRL_HIGH = 53


def set_realtime_priority(level: int) -> None:
  if not PC:
    os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(level))  # pylint: disable=no-member


def set_core_affinity(cores: List[int]) -> None:
  if not PC:
    os.sched_setaffinity(0, cores)  # pylint: disable=no-member


def config_realtime_process(cores: Union[int, List[int]], priority: int) -> None:
  gc.disable()
  set_realtime_priority(priority)
  c = cores if isinstance(cores, list) else [cores, ]
  set_core_affinity(c)


class Ratekeeper:
  def __init__(self, rate: float, print_delay_threshold: Optional[float] = 0.0) -> None:
    """Rate in Hz for ratekeeping. print_delay_threshold must be nonnegative."""
    self._interval = 1. / rate
    self._next_frame_time = sec_since_boot() + self._interval
    self._print_delay_threshold = print_delay_threshold
    self._frame = 0
    self._remaining = 0.0
    self._process_name = getproctitle()
    self._dts = deque([self._interval], maxlen=100)
    self._last_monitor_time = sec_since_boot()

  @property
  def frame(self) -> int:
    return self._frame

  @property
  def remaining(self) -> float:
    return self._remaining

  @property
  def lagging(self) -> bool:
    avg_dt = sum(self._dts) / len(self._dts)
    expected_dt = self._interval * (1 / 0.9)
    return avg_dt > expected_dt

  # Maintain loop rate by calling this at the end of each loop
  def keep_time(self) -> bool:
    lagged = self.monitor_time()
    if self._remaining > 0:
      time.sleep(self._remaining)
    return lagged

  # this only monitor the cumulative lag, but does not enforce a rate
  def monitor_time(self) -> bool:
    prev = self._last_monitor_time
    self._last_monitor_time = sec_since_boot()
    self._dts.append(self._last_monitor_time - prev)

    lagged = False
    remaining = self._next_frame_time - sec_since_boot()
    self._next_frame_time += self._interval
    if self._print_delay_threshold is not None and remaining < -self._print_delay_threshold:
      print(f"{self._process_name} lagging by {-remaining * 1000:.2f} ms")
      lagged = True
    self._frame += 1
    self._remaining = remaining
    return lagged


class DurationTimer:
  def __init__(self, duration=0, step=DT_CTRL) -> None:
    self.step = step
    self.duration = duration
    self.was_reset = False
    self.timer = 0
    self.min = float("-inf") # type: float
    self.max = float("inf") # type: float

  def tick_obj(self) -> None:
    self.timer += self.step
    # reset on overflow
    self.reset() if (self.timer == (self.max or self.min)) else None

  def reset(self) -> None:
    """Resets this objects timer"""
    self.timer = 0
    self.was_reset = True

  def active(self) -> bool:
    """Returns true if time since last reset is less than duration"""
    return bool(round(self.timer,2) < self.duration)

  def adjust(self, duration) -> None:
    """Adjusts the duration of the timer"""
    self.duration = duration

  def once_after_reset(self) -> bool: 
    """Returns true only one time after calling reset()"""
    ret = self.was_reset
    self.was_reset = False
    return ret

  @staticmethod
  def interval_obj(rate, frame) -> bool:
    if frame % rate == 0:
      return True
    return False

class ModelTimer(DurationTimer):
  frame = -1 # type: int
  objects = [] # type: List[DurationTimer]
  def __init__(self, duration=0) -> None:
    self.step = DT_MDL
    super().__init__(duration, self.step)
    self.__class__.objects.append(self)

  @classmethod
  def tick(cls) -> None:
    cls.frame += 1
    for obj in cls.objects:
      ModelTimer.tick_obj(obj)

  @classmethod
  def reset_all(cls) -> None:
    for obj in cls.objects:
      obj.reset()

  @classmethod
  def interval(cls, rate) -> bool:
    return ModelTimer.interval_obj(rate, cls.frame)

class ControlsTimer(DurationTimer):
  frame = -1
  objects = [] # type: List[DurationTimer]
  def __init__(self, duration=0) -> None:
    self.step = DT_CTRL
    super().__init__(duration=duration, step=self.step)
    self.__class__.objects.append(self)

  @classmethod
  def tick(cls) -> None:
    cls.frame += 1
    for obj in cls.objects:
      ControlsTimer.tick_obj(obj)

  @classmethod
  def reset_all(cls) -> None:
    for obj in cls.objects:
      obj.reset()

  @classmethod
  def interval(cls, rate) -> bool:
    return ControlsTimer.interval_obj(rate, cls.frame)
  
import unittest
class TestDurationTimer(unittest.TestCase):

  def test_timer_initialization(self):
    timer = DurationTimer(duration=5)
    self.assertEqual(timer.duration, 5)
    self.assertEqual(timer.timer, 0)

  def test_tick_obj(self):
    timer = DurationTimer(duration=5)
    timer.tick_obj()
    self.assertEqual(timer.timer, DT_CTRL)

  def test_reset(self):
    timer = DurationTimer(duration=5)
    timer.tick_obj()
    timer.reset()
    self.assertEqual(timer.timer, 0)
    self.assertTrue(timer.was_reset)

  def test_active(self):
    timer = DurationTimer(duration=5)
    self.assertTrue(timer.active())
    timer.timer = 5
    self.assertFalse(timer.active())

  def test_adjust(self):
    timer = DurationTimer(duration=5)
    timer.adjust(10)
    self.assertEqual(timer.duration, 10)

  def test_once_after_reset(self):
    timer = DurationTimer(duration=5)
    timer.reset()
    self.assertTrue(timer.once_after_reset())
    self.assertFalse(timer.once_after_reset())

  def test_interval_obj(self):
    self.assertTrue(DurationTimer.interval_obj(2, 4))
    self.assertFalse(DurationTimer.interval_obj(2, 5))

class TestModelTimer(unittest.TestCase):

  def test_model_timer_initialization(self):
    timer = ModelTimer(duration=5)
    self.assertEqual(timer.duration, 5)
    self.assertEqual(timer.step, DT_MDL)

  def test_tick(self):
    timer = ModelTimer(duration=5)
    ModelTimer.tick()
    self.assertEqual(timer.timer, DT_MDL)

  def test_reset_all(self):
    timer1 = ModelTimer(duration=5)
    timer2 = ModelTimer(duration=10)
    timer1.tick_obj()
    timer2.tick_obj()
    ModelTimer.reset_all()
    self.assertEqual(timer1.timer, 0)
    self.assertEqual(timer2.timer, 0)

  def test_interval(self):
    ModelTimer.frame = 4
    self.assertTrue(ModelTimer.interval(2))
    ModelTimer.frame = 5
    self.assertFalse(ModelTimer.interval(2))

class TestControlsTimer(unittest.TestCase):

  def test_controls_timer_initialization(self):
    timer = ControlsTimer(duration=5)
    self.assertEqual(timer.duration, 5)
    self.assertEqual(timer.step, DT_CTRL)

  def test_tick(self):
    timer = ControlsTimer(duration=5)
    ControlsTimer.tick()
    self.assertEqual(timer.timer, DT_CTRL)

  def test_reset_all(self):
    timer1 = ControlsTimer(duration=5)
    timer2 = ControlsTimer(duration=10)
    timer1.tick_obj()
    timer2.tick_obj()
    ControlsTimer.reset_all()
    self.assertEqual(timer1.timer, 0)
    self.assertEqual(timer2.timer, 0)

  def test_interval(self):
    ControlsTimer.frame = 4
    self.assertTrue(ControlsTimer.interval(2))
    ControlsTimer.frame = 5
    self.assertFalse(ControlsTimer.interval(2))
        
class TestTimers(unittest.TestCase):

  def test_increment_and_interval(self):
    # Assume previous test cases passed so cls frame is not 0
    self.assertFalse(ControlsTimer.frame == 0)
    self.assertFalse(ModelTimer.frame == 0)
    # reset frame for class methods
    ControlsTimer.frame = 0
    ModelTimer.frame = 0
    self.assertTrue(ControlsTimer.frame == 0)
    self.assertTrue(ModelTimer.frame == 0)
    # Create control timers
    control_timer1 = ControlsTimer(1)
    control_timer2 = ControlsTimer(5)
    # Create model timers
    model_timer1 = ModelTimer(1)
    model_timer2 = ModelTimer(5)
    self.assertTrue(control_timer1.active())
    self.assertTrue(control_timer2.active())
    self.assertTrue(model_timer1.active())
    self.assertTrue(model_timer2.active())
    for i in range(1, 2002):
      if i % 1 == 0:
        # Increment control timers  to simulate 100Hz
        ControlsTimer.tick()
      if i % 5 == 0: 
        # Increment model timers to simulate 20Hz
        ModelTimer.tick()
          
      # Test interval method for control timers
      if i % 200 == 0:  # 1-second interval (100Hz)
        self.assertTrue(ControlsTimer.interval(200))
      else:
        self.assertFalse(ControlsTimer.interval(200))
      if i % 1000 == 0:  # 5-second interval (100Hz)
        self.assertTrue(ControlsTimer.interval(1000))
      else:
        self.assertFalse(ControlsTimer.interval(1000))
        
      # Test interval method for model timers
      if i % 50 == 0:  # 1-second interval (20Hz)
        self.assertTrue(ModelTimer.interval(5))
      elif i % 10 == 0:
        self.assertFalse(ModelTimer.interval(5))
      if i % 1000 == 0:  # 5-second interval (20Hz)
        self.assertTrue(ModelTimer.interval(200))
      elif i % 10 == 0:
        self.assertFalse(ModelTimer.interval(200))
        
      # Test active method for control timers
      if i < 100:  # 1-second duration
        self.assertTrue(control_timer1.active())
      else:
        self.assertFalse(control_timer1.active())
      if i < 500:  # 5-second duration
        self.assertTrue(control_timer2.active())
      else:
        self.assertFalse(control_timer2.active())
        
      # Test active method for model timers
      if i < 100:  # 1-second duration
        self.assertTrue(model_timer1.active())
      else:
        self.assertFalse(model_timer1.active())
      if i < 500:  # 5-second duration
        self.assertTrue(model_timer2.active())
      else:
        self.assertFalse(model_timer2.active())
            
if __name__ == '__main__':
  unittest.main()