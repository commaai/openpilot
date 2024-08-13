import pytest
import time
from openpilot.common.realtime import Ratekeeper

class TestRatekeeper:
  def setup_method(self):
    self.rate = 100
    self.tolerance = 0.01
    self.rk = Ratekeeper(self.rate)

  def test_stability(self):
    start_time = time.perf_counter()
    iterations = 100
    lagging_count = 0

    for _ in range(iterations):
      lagging = self.rk.keep_time()
      if lagging:
        lagging_count += 1

      # Measure elapsed time
      elapsed_time = time.perf_counter() - start_time
      expected_time = (self.rk.frame) * self.rk._interval

      assert abs(elapsed_time - expected_time) <= self.tolerance * self.rk._interval, \
        f"Timing error exceeded tolerance: {elapsed_time - expected_time:.6f} seconds"

    assert lagging_count < iterations, "The loop lagged too many times"

  def test_lagging_detection(self):
    for _ in range(10):
      time.sleep(1.5 * self.rk._interval)
      lagging = self.rk.keep_time()

      assert lagging, "Expected the loop to lag"

  def test_no_lag_detection(self):
    for _ in range(10):
      lagging = self.rk.keep_time()

      assert not lagging, "Unexpected lag detected"

  def test_frame_count_increment(self):
    initial_frame = self.rk.frame
    iterations = 50

    for _ in range(iterations):
      self.rk.keep_time()

    assert self.rk.frame == initial_frame + iterations, \
      f"Frame count mismatch: expected {initial_frame + iterations}, got {self.rk.frame}"

  @pytest.mark.skip(reason="Temporarily disabled")
  def test_drift_compensation(self):
    for _ in range(5):
      time.sleep(1.5 * self.rk._interval)
      self.rk.keep_time()

    start_time = time.perf_counter()

    for _ in range(5):
      self.rk.keep_time()

    elapsed_time = time.perf_counter() - start_time
    expected_time = 5 * self.rk._interval

    assert abs(elapsed_time - expected_time) <= self.rk._interval, \
      f"Drift compensation failed: elapsed time {elapsed_time:.6f} vs expected {expected_time:.6f}"
