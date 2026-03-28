import time
from collections import deque

# Threshold for the magnitude of longitudinal acceleration spike (m/s²).
# A bump causes a sudden jolt; typical driving accelerations are well below this.
# Note: ideally this would use IMU z-axis (vertical) acceleration, but carState.aEgo
# (longitudinal) is used here as it's already available in ui_state.sm.
BUMP_ACCEL_THRESHOLD = 4.0  # m/s²

# Threshold for how quickly the acceleration must change (m/s³) to count as a spike.
BUMP_JERK_THRESHOLD = 10.0  # m/s³

# Minimum time between consecutive bump detections (seconds).
BUMP_COOLDOWN = 30.0

# How far back in time to look when measuring the jerk (rate of acceleration change).
HISTORY_WINDOW = 0.3  # seconds


class BumpDetector:
  """
  Detects sudden acceleration spikes in carState.aEgo that may indicate a road bump.

  Call update() each frame with the current longitudinal acceleration value.
  Returns True on the frame a bump is first detected.
  """

  def __init__(self):
    # Circular buffer of (timestamp, aEgo) samples
    self._history: deque[tuple[float, float]] = deque()
    self._last_trigger: float = -BUMP_COOLDOWN  # allow immediate trigger on startup

  def update(self, a_ego: float) -> bool:
    now = time.monotonic()
    self._history.append((now, a_ego))

    # Prune samples older than the look-back window
    while self._history and now - self._history[0][0] > HISTORY_WINDOW:
      self._history.popleft()

    # Respect cooldown
    if now - self._last_trigger < BUMP_COOLDOWN:
      return False

    # Need at least two samples to estimate jerk
    if len(self._history) < 2:
      return False

    oldest_t, oldest_a = self._history[0]
    newest_t, newest_a = self._history[-1]

    dt = newest_t - oldest_t
    if dt < 1e-6:
      return False

    jerk = abs(newest_a - oldest_a) / dt

    if abs(newest_a) > BUMP_ACCEL_THRESHOLD and jerk > BUMP_JERK_THRESHOLD:
      self._last_trigger = now
      return True

    return False

  def reset_cooldown(self):
    """Force the cooldown to expire so the next spike will trigger immediately."""
    self._last_trigger = -BUMP_COOLDOWN
