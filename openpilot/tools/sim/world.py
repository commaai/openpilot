import math
import threading
import time
import numpy as np
from collections import deque

from openpilot.common.realtime import DT_MDL
from openpilot.common.transformations.camera import view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.tools.sim.route import CameraFiles, FrameStream, RouteData
from openpilot.tools.sim.vehicle import Vehicle
from openpilot.tools.sim.warp import Warp, warp_params

MAX_LATERAL_ERROR = 2.0  # m
MAX_HEADING_ERROR = math.radians(8)


class Camera:
  def __init__(self, route: RouteData, files: CameraFiles, start_frame: int, K: np.ndarray, view_from_calib: np.ndarray, warp: Warp):
    self.route, self.files, self.start_frame = route, files, start_frame
    self.K = K
    self.view_from_calib = view_from_calib
    self.warp = warp
    self.stream: FrameStream | None = None
    # a stream at the start frame stays ready, so a restart doesn't have to wait on the first frames to decode
    self.next_stream = FrameStream(route, files, start_frame)

  def restart(self):
    if self.stream is not None:
      self.stream.close()
    self.stream, self.next_stream = self.next_stream, FrameStream(self.route, self.files, self.start_frame)

  def close(self):
    for s in (self.stream, self.next_stream):
      if s is not None:
        s.close()


class LogWorld:
  """Replays a logged segment until openpilot engages, then drives a simulated car along it.
  Each camera frame is the logged frame closest in distance travelled, warped to the simulated pose."""
  def __init__(self, route: RouteData, start_frame: int = 0):
    self.route = route
    self.start_frame = start_frame
    self.vehicle = Vehicle()
    Warp.init()

    # the latest (road, wide) NV12 frames in VisionIPC buffer layout, None until the first render
    self.images: tuple[np.ndarray, np.ndarray] | None = None
    self.new_images = threading.Event()

    device_from_calib = rot_from_euler(route.rpy_calib)
    cam = route.camera
    wide_from_device = rot_from_euler(route.wide_from_device_euler)
    # the start segment's files download before the episode starts, later ones while it plays
    for files in (route.road_files, route.wide_files):
      files.get(route.segment(start_frame)[0])
    self.cameras = [Camera(route, route.road_files, start_frame, cam.narrow_road.intrinsics, view_frame_from_device_frame @ device_from_calib,
                           Warp(*cam.narrow_road.size)),
                    Camera(route, route.wide_files, start_frame, cam.wide_road.intrinsics,
                           view_frame_from_device_frame @ wide_from_device @ device_from_calib, Warp(*cam.wide_road.size))]
    # the first real warp compiles the shader, which takes a while on llvmpipe
    for c in self.cameras:
      c.warp(np.zeros(c.warp.w * c.warp.h * 3 // 2, dtype=np.uint8), warp_params(c.K, c.view_from_calib, route.height, 0., 0.1, 0.))
    self.reset()

  def reset(self):
    """Start the episode over, replaying from the start frame until openpilot engages again"""
    for cam in self.cameras:
      cam.restart()

    self.t_idx = self.start_frame  # logged frame at the current sim time
    self.frame_idx = self.start_frame  # logged frame closest to the simulated car
    self.rel_pose: np.ndarray | None = None  # simulated pose relative to frame_idx, None while replaying
    self.lead_gap = math.nan
    self.closed_loop = False
    self.engaged = False
    self.ended: str | None = None
    self.frame_times: deque[tuple[float, float]] = deque(maxlen=20)  # (waiting on decode, warping) of the last second of frames
    self.stats = {"engaged_time": 0., "max_lateral_error": 0., "min_lead_gap": math.inf}
    route = self.route
    self.vehicle.set_state(route.pose[self.start_frame], route.v[self.start_frame], route.yaw_rate[self.start_frame])

  def recenter(self):
    """Puts the car back where the logged car was at the current frame, with its speed, and continues the episode from there"""
    if self.ended == "route_end":
      return
    route, i = self.route, self.frame_idx
    self.t_idx = i
    self.vehicle.set_state(route.pose[i], route.v[i], route.yaw_rate[i])
    if self.rel_pose is not None:
      self.rel_pose = np.zeros(3)
    self.ended = None

  def apply_controls(self, steer_angle, throttle_out, brake_out):
    if self.closed_loop and self.ended is None:
      self.vehicle.step(steer_angle, throttle_out * 1.6 - brake_out * 4.0)

  def tick(self):
    if self.ended is not None:
      return

    route = self.route
    self.t_idx += 1
    if not self.closed_loop:
      self.frame_idx = min(self.t_idx, len(route) - 1)
      self.vehicle.set_state(route.pose[self.frame_idx], route.v[self.frame_idx], route.yaw_rate[self.frame_idx])
      self.lead_gap = route.lead_gap[self.frame_idx]
    else:
      rel = route.relative_pose(self.frame_idx, self.vehicle.pose)
      while self.frame_idx + 1 < len(route):
        nxt = route.relative_pose(self.frame_idx + 1, self.vehicle.pose)
        if abs(nxt[0]) >= abs(rel[0]):
          break
        self.frame_idx += 1
        rel = nxt
      self.rel_pose = rel

      if abs(rel[1]) > MAX_LATERAL_ERROR or abs(rel[2]) > MAX_HEADING_ERROR:
        self.ended = "out_of_envelope"
        return
      if self.engaged:
        self.stats["engaged_time"] += DT_MDL
        self.stats["max_lateral_error"] = max(self.stats["max_lateral_error"], abs(rel[1]))
      # the logged lead is frozen in the frames, so compare against where it really was at this time
      self.lead_gap = math.nan
      if self.t_idx < len(route) and not math.isnan(route.lead_gap[self.t_idx]):
        self.lead_gap = route.s[self.t_idx] + route.lead_gap[self.t_idx] - (route.s[self.frame_idx] + rel[0])
        self.stats["min_lead_gap"] = min(self.stats["min_lead_gap"], self.lead_gap)
        if self.lead_gap < 0:
          self.ended = "collision"
          return

    if self.frame_idx == len(route) - 1:
      self.ended = "route_end"

  def status(self) -> dict:
    rel = self.rel_pose if self.rel_pose is not None else (0., 0., 0.)
    return {"t": (self.frame_idx - self.start_frame) * DT_MDL, "duration": (len(self.route) - 1 - self.start_frame) * DT_MDL,
            "closed_loop": self.closed_loop, "offset": float(rel[1]), "heading": math.degrees(rel[2]), "lead_gap": float(self.lead_gap),
            "speed": self.vehicle.v, "steer": self.vehicle.steer.x, "ended": self.ended,
            "decode_wait": max((w for w, _ in self.frame_times), default=0.), "warp": max((w for _, w in self.frame_times), default=0.), **self.stats}

  def set_engaged(self, engaged: bool):
    self.engaged = engaged
    self.closed_loop |= engaged

  def render(self):
    images = []
    wait = warp = 0.
    for cam in self.cameras:
      assert cam.stream is not None
      params = None if self.rel_pose is None else warp_params(cam.K, cam.view_from_calib, self.route.height, *self.rel_pose)
      t0 = time.monotonic()
      frame = cam.stream.get(self.frame_idx)
      t1 = time.monotonic()
      # copied out, since the camera thread may still be sending the previous frame from the warp's buffer
      images.append(cam.warp(frame, params).copy())
      wait, warp = wait + t1 - t0, warp + time.monotonic() - t1
    self.images = (images[0], images[1])
    self.new_images.set()
    self.frame_times.append((wait, warp))

  def close(self):
    for cam in self.cameras:
      cam.close()
