import unittest, time
from unittest.case import skipIf

from extra.bench_log import BenchEvent, InstantBenchEvent, WallTimeEvent, KernelTimeEvent, log_event_instant, _events, clear_events
from tinygrad.helpers import Context, CI
from tinygrad.tensor import Tensor
from tinygrad.device import Device

class TestBenchLog(unittest.TestCase):
  def setUp(self):
    clear_events()

  def test_log_single_wall_time(self):
    for event in BenchEvent:
      with WallTimeEvent(event):
        time.sleep(0.1)

    # check event list
    for event in BenchEvent:
      self.assertEqual(len(_events[event]["wall"]), 1)
      self.assertGreater(_events[event]["wall"][0], 0)

  def test_log_double_wall_time(self):
    for event in BenchEvent:
      with WallTimeEvent(event):
        time.sleep(0.1)

    for event in reversed(BenchEvent):
      with WallTimeEvent(event):
        time.sleep(0.2)

    # check event list
    for event in BenchEvent:
      self.assertEqual(len(_events[event]["wall"]), 2)
      self.assertGreater(_events[event]["wall"][0], 0)
      self.assertGreater(_events[event]["wall"][1], 0)

  @skipIf(CI, "ci timing is not accurate")
  def test_log_single_kernel_time(self):
    wall_times = []

    with Context(DEBUG=2):
      for event in BenchEvent:
        with KernelTimeEvent(event):
          st = time.perf_counter()
          Tensor.rand(32, 32).sum().realize().item()
          wall_times.append(time.perf_counter() - st)

    # check event list
    for event in BenchEvent:
      self.assertEqual(len(_events[event]["kernel"]), 1)
      self.assertLess(_events[event]["kernel"][0], wall_times[0])
      self.assertGreater(_events[event]["kernel"][0], 0)

  @skipIf(CI and Device.DEFAULT == "CUDA", "ci cuda timing is not accurate")
  def test_interleaved_wall_kernel_time(self):
    wall_times = []
    with Context(DEBUG=2):
      for event in BenchEvent:
        with KernelTimeEvent(event):
          st = time.perf_counter()
          Tensor.rand(32, 32).sum().realize().item()
          wall_times.append(time.perf_counter() - st)

        with WallTimeEvent(event):
          st = time.perf_counter()
          Tensor.rand(32, 32).sum().realize().item()
          wall_times.append(time.perf_counter() - st)

    # check event list
    for event in BenchEvent:
      self.assertEqual(len(_events[event]["wall"]), 1)
      self.assertEqual(len(_events[event]["kernel"]), 1)
      self.assertLess(_events[event]["kernel"][0], wall_times[0])
      self.assertGreater(_events[event]["kernel"][0], 0)

  @skipIf(CI and Device.DEFAULT == "CUDA", "ci cuda timing is not accurate")
  def test_stacked_wall_kernel_time(self):
    with Context(DEBUG=2):
      for event in BenchEvent:
        with KernelTimeEvent(event):
          with WallTimeEvent(event):
            Tensor.rand(32, 32).sum().realize().item()

      for event in BenchEvent:
        with WallTimeEvent(event):
          with KernelTimeEvent(event):
            Tensor.rand(32, 32).sum().realize().item()

    for event in BenchEvent:
      self.assertEqual(len(_events[event]["wall"]), 2)
      self.assertEqual(len(_events[event]["kernel"]), 2)
      self.assertLess(_events[event]["kernel"][0], _events[event]["wall"][0])
      self.assertGreater(_events[event]["kernel"][0], 0)
      self.assertLess(_events[event]["kernel"][1], _events[event]["wall"][1])
      self.assertGreater(_events[event]["kernel"][1], 0)

  def test_log_instant_event(self):
    for event in InstantBenchEvent:
      log_event_instant(event, 1000)

    # check event list
    for event in InstantBenchEvent:
      self.assertEqual(len(_events[event]), 1)
      self.assertEqual(_events[event][0], 1000)

if __name__ == '__main__':
  unittest.main()
