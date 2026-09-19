"""Experimental UI-only held stopping marker; never used by vehicle controls."""
from dataclasses import dataclass, field

import numpy as np


ROLLING_SPEED = 2.0
MIN_SLOWDOWN = 2.0
MOVE_OFF_CONFIRMATION = 0.5
STOP_BAR_MIN_HEIGHT = 6.0
STOP_BAR_MAX_HEIGHT = 12.0
STOP_BAR_WIDTH = 3.6


def motion_prediction(model):
  times, speeds = np.asarray(model.velocity.t), np.asarray(model.velocity.x)
  pt = np.asarray(model.position.t)
  coordinates = [np.asarray(getattr(model.position, axis)) for axis in ('x', 'y', 'z')]
  if (len(times) < 2 or len(speeds) != len(times) or len(pt) < 2 or
      any(len(axis) != len(pt) for axis in coordinates) or
      not all(np.isfinite(a).all() for a in (times, speeds, pt, *coordinates)) or
      np.any(np.diff(times) <= 0) or np.any(np.diff(pt) <= 0)):
    return None
  return times, speeds, pt, coordinates


def stop_candidate(prediction, speed):
  times, speeds, pt, coordinates = prediction
  future = (times >= 0.5) & (times >= pt[0]) & (times <= pt[-1])
  if not future.any():
    return None
  slowest = max(0.0, float(speeds[future].min()))
  if slowest > ROLLING_SPEED or max(speed, speeds[0]) - slowest < MIN_SLOWDOWN:
    return None
  # Use the beginning of the slowest portion, allowing a rolling stop and
  # avoiding arbitrary placement at the very end of a creeping trajectory.
  indices = np.flatnonzero(future & (speeds <= min(ROLLING_SPEED, slowest + 0.3)))
  if not len(indices):
    return None
  time = times[indices[0]]
  point = np.array([np.interp(time, pt, axis) for axis in coordinates])
  if not 1.0 < point[0] <= 100.0:
    return None
  before, after = np.clip([time - 0.25, time + 0.25], pt[0], pt[-1])
  dx = np.interp(after, pt, coordinates[0]) - np.interp(before, pt, coordinates[0])
  dy = np.interp(after, pt, coordinates[1]) - np.interp(before, pt, coordinates[1])
  return point, float(np.arctan2(dy, max(dx, 0.0)))


@dataclass
class HeldStopBar:
  point: np.ndarray | None = None
  heading: float = 0.0
  last_time: float | None = None
  last_model_time: float | None = None
  last_speed: float = 0.0
  last_yaw_rate: float = 0.0
  move_off_since: float | None = None
  points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))

  def reset(self):
    self.point = None
    self.last_time = self.last_model_time = self.move_off_since = None
    self.points = np.empty((0, 2), dtype=np.float32)

  def update(self, model, speed, yaw_rate, now, model_time, enabled=True):
    if not enabled or model is None or not np.isfinite([speed, yaw_rate, now, model_time]).all():
      self.reset()
      return
    if self.last_time is not None and (now < self.last_time or now - self.last_time > 1.0):
      self.reset()  # Replay seek or missing odometry: do not retain a stale anchor.
    if self.last_model_time is not None and model_time < self.last_model_time:
      self.reset()
    dt = 0.0 if self.last_time is None else now - self.last_time
    if self.point is not None and dt > 0:
      velocity = max(0.0, (self.last_speed + speed) / 2)
      # carState yaw is left-positive; model/camera lateral is right-positive.
      omega = -(self.last_yaw_rate + yaw_rate) / 2
      angle = omega * dt
      c, s = np.cos(angle), np.sin(angle)
      travel = np.array([velocity * dt, 0.0]) if abs(omega) < 1e-6 else velocity / omega * np.array([s, 1 - c])
      self.point[:2] = np.array([[c, s], [-s, c]]) @ (self.point[:2] - travel)
      self.heading -= angle
      if self.point[0] <= 0.3:
        self.point = None
        self.move_off_since = None
        # Consume this frame so the just-passed target cannot immediately relatch.
        self.last_model_time = model_time
    self.last_time, self.last_speed, self.last_yaw_rate = now, speed, yaw_rate
    if model_time == self.last_model_time:
      return
    self.last_model_time = model_time
    prediction = motion_prediction(model)
    if prediction is None:
      self.reset()
      return
    candidate = stop_candidate(prediction, speed)
    if self.point is None:
      if candidate is not None and model.action.desiredAcceleration <= 0.1:
        self.point, self.heading = candidate
      return
    times, speeds, _, _ = prediction
    moving_off = (candidate is None and np.isfinite(model.action.desiredAcceleration) and
                  model.action.desiredAcceleration > 0.2 and times[0] <= 2.0 <= times[-1] and
                  np.interp(2.0, times, speeds) > max(ROLLING_SPEED + 0.5, speed + 1.0))
    if moving_off:
      if self.move_off_since is None:
        self.move_off_since = model_time
      elif model_time - self.move_off_since >= MOVE_OFF_CONFIRMATION:
        self.point = None
        self.move_off_since = None
    else:
      self.move_off_since = None

  def project(self, transform, camera_height, lane_lines, probabilities, road_edges, edge_stds):
    self.points = np.empty((0, 2), dtype=np.float32)
    if self.point is None:
      return
    boundaries = []
    for side in range(2):
      line = lane_lines[side + 1] if probabilities[side + 1] >= 0.25 else road_edges[side]
      confident = probabilities[side + 1] >= 0.25 or 1.0 - edge_stds[side] >= 0.25
      if (not confident or len(line) < 2 or not np.isfinite(line).all() or np.any(np.diff(line[:, 0]) <= 0) or
          not line[0, 0] <= self.point[0] <= line[-1, 0]):
        boundaries = []
        break
      boundaries.append(line)
    if boundaries and np.interp(self.point[0], boundaries[0][:, 0], boundaries[0][:, 1]) >= np.interp(
        self.point[0], boundaries[1][:, 0], boundaries[1][:, 1]):
      boundaries = []
    forward = np.array([np.cos(self.heading), np.sin(self.heading)])
    sideways = np.array([-forward[1], forward[0]])

    def project_depth(depth):
      corners = np.array([self.point[:2] - forward * behind + sideways * side
                          for behind, side in ((0, -STOP_BAR_WIDTH / 2), (0, STOP_BAR_WIDTH / 2),
                                               (depth, STOP_BAR_WIDTH / 2), (depth, -STOP_BAR_WIDTH / 2))])
      heights = np.full(4, self.point[2] + camera_height)
      if boundaries:
        sides = (boundaries[0], boundaries[1], boundaries[1], boundaries[0])
        if all(line[0, 0] <= corner[0] <= line[-1, 0] for corner, line in zip(corners, sides, strict=True)):
          for i, line in enumerate(sides):
            corners[i, 1] = np.interp(corners[i, 0], line[:, 0], line[:, 1])
            heights[i] = np.interp(corners[i, 0], line[:, 0], line[:, 2])
      if np.any(corners[:, 0] <= 0.1):
        return None
      projected = transform @ np.column_stack((corners, heights)).T
      if not np.isfinite(projected).all() or np.any(projected[2] <= 1e-3):
        return None
      return (projected[:2] / projected[2]).T.astype(np.float32)

    high = self.point[0] * 0.9
    depth = min(6.0, high)
    points = project_depth(depth)
    for _ in range(20):
      if points is not None:
        break
      high, depth = depth, depth / 2
      points = project_depth(depth)
    if points is None:
      return
    height = np.ptp(points[:, 1])
    target = np.clip(height, STOP_BAR_MIN_HEIGHT, STOP_BAR_MAX_HEIGHT)
    self.points = points
    if height == target:
      return
    low, high = (0.0, depth) if height > target else (depth, high)
    error = abs(height - target)
    for _ in range(20):
      depth = (low + high) / 2
      candidate = project_depth(depth)
      if candidate is None:
        high = depth
        continue
      height = np.ptp(candidate[:, 1])
      if abs(height - target) < error:
        self.points, error = candidate, abs(height - target)
      if error < 1e-5:
        break
      if height < target:
        low = depth
      else:
        high = depth
