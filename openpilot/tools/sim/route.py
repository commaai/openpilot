import multiprocessing
import os
import queue
import threading
import numpy as np
from concurrent.futures import Future, ThreadPoolExecutor

from openpilot.common.hardware.hw import Paths
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.tools.lib.file_sources import openpilotci_source
from openpilot.tools.lib.filereader import FileReader
from openpilot.tools.lib.framereader import FrameIterator, get_video_index
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.route import FileName, Route, SegmentRange


def _parse_segment(segment: str) -> dict:
  """The parts of a segment's log the sim needs, as plain arrays so they can come back from a worker process"""
  cs, dm, radar, road, wide = [], [], [], {}, set()
  seg: dict = {"calib": None}
  for m in LogReader(segment):
    w = m.which()
    t = m.logMonoTime * 1e-9
    if w == 'carState':
      cs.append((t, m.carState.vEgo))
    elif w == 'deviceMotion':
      dm.append((t, m.deviceMotion.angularVelocityDevice.z))
    elif w == 'radarState':
      radar.append((t, m.radarState.leadOne.dRel if m.radarState.leadOne.present else np.nan))
    elif w == 'narrowRoadEncodeIdx':
      road[m.narrowRoadEncodeIdx.segmentId] = m.narrowRoadEncodeIdx.timestampEof * 1e-9
    elif w == 'wideRoadEncodeIdx':
      wide.add(m.wideRoadEncodeIdx.segmentId)
    elif w == 'extrinsicsCalibration':
      c = m.extrinsicsCalibration
      seg["calib"] = (m.as_builder().to_bytes(), list(c.rpyCalib), list(c.wideFromDeviceEuler), c.height[0])
    elif w == 'initData':
      seg["device_type"] = str(m.initData.deviceType)
    elif w == 'narrowRoadCameraState':
      seg["sensor"] = str(m.narrowRoadCameraState.sensor)

  # frames in both cameras' files, from the start of the segment
  n = next(i for i in range(len(road) + 1) if i not in road or i not in wide)
  seg |= {"carState": np.array(cs).reshape(-1, 2), "deviceMotion": np.array(dm).reshape(-1, 2), "radarState": np.array(radar).reshape(-1, 2),
          "frame_t": np.array([road[i] for i in range(n)])}
  return seg


class CameraFiles:
  """One camera's files for each segment, downloaded to a local cache the first time they're needed"""
  def __init__(self, sr: SegmentRange, segs: list[int], fn: tuple[str, ...]):
    urls = openpilotci_source(sr, segs, fn)
    if len(urls) < len(segs):
      route = Route(sr.route_name)
      paths = route.camera_paths() if fn == FileName.FCAMERA else route.ecamera_paths()
      urls = {seg: urls.get(seg) or paths[seg] for seg in segs}
    self.urls = [urls[seg] for seg in segs]
    name = sr.route_name.replace('|', '_')
    self.paths = [os.path.join(Paths.download_cache_root(), "sim", f"{name}--{seg}--{fn[0]}") for seg in segs]
    self.pool = ThreadPoolExecutor(2)
    self.lock = threading.Lock()
    self.files: dict[int, Future] = {}

  def get(self, i: int) -> tuple[str, dict]:
    """Path and index of segment i's file, waiting for it to download"""
    return self.fetch(i).result()

  def fetch(self, i: int) -> Future:
    with self.lock:
      if i not in self.files:
        self.files[i] = self.pool.submit(self._download, i)
      return self.files[i]

  def _download(self, i: int) -> tuple[str, dict]:
    path = self.paths[i]
    if not os.path.exists(path):
      assert self.urls[i] is not None, f"no camera file for segment {i}"
      os.makedirs(os.path.dirname(path), exist_ok=True)
      with FileReader(self.urls[i]) as f:
        dat = f.read()
      with open(path + ".tmp", "wb") as out:
        out.write(dat)
      os.replace(path + ".tmp", path)
    return path, get_video_index(path)


class RouteData:
  """Logged segments of a route: per-frame path (x fwd, y right, heading right-positive), speed, lead and calibration, and the camera files"""
  def __init__(self, route: str):
    sr = SegmentRange(route)
    segs = sr.seg_idxs
    # spawned, since forking a process with threads can deadlock the workers
    with multiprocessing.get_context("spawn").Pool(min(len(segs), os.cpu_count() or 1)) as pool:
      parsed = pool.map(_parse_segment, [f"{sr.route_name}/{seg}" for seg in segs])

    calib_msg, rpy_calib, wide_from_device_euler, self.height = next(p["calib"] for p in reversed(parsed) if p["calib"] is not None)
    self.calib_msg = calib_msg
    self.rpy_calib = np.array(rpy_calib)
    self.wide_from_device_euler = np.array(wide_from_device_euler)
    self.camera = DEVICE_CAMERAS[(parsed[0]["device_type"], parsed[0]["sensor"])]
    assert self.camera.narrow_road.size == (1928, 1208), f"unsupported camera {parsed[0]['device_type']} {parsed[0]['sensor']}"

    self.road_files = CameraFiles(sr, segs, FileName.FCAMERA)
    self.wide_files = CameraFiles(sr, segs, FileName.ECAMERA)
    self.seg_frames = [len(p["frame_t"]) for p in parsed]
    self.seg_start = np.cumsum([0, *self.seg_frames[:-1]])
    self.t = np.concatenate([p["frame_t"] for p in parsed])

    # carState and deviceMotion are logged with logMonoTime, encodeIdx with the frame's EOF, both on the monotonic clock
    cs = np.concatenate([p["carState"] for p in parsed])
    dm = np.concatenate([p["deviceMotion"] for p in parsed])
    radar = np.concatenate([p["radarState"] for p in parsed])
    cs_t, v = cs[:, 0], cs[:, 1]
    yaw_rate = np.interp(cs_t, dm[:, 0], dm[:, 1])
    dt = np.diff(cs_t, prepend=cs_t[0])
    heading = np.cumsum(yaw_rate * dt)
    x = np.cumsum(v * np.cos(heading) * dt)
    y = np.cumsum(v * np.sin(heading) * dt)
    s = np.cumsum(v * dt)

    self.v = np.interp(self.t, cs_t, v)
    self.yaw_rate = np.interp(self.t, cs_t, yaw_rate)
    self.s = np.interp(self.t, cs_t, s)
    self.pose = np.stack([np.interp(self.t, cs_t, x), np.interp(self.t, cs_t, y), np.interp(self.t, cs_t, heading)], axis=1)
    self.lead_gap = radar[np.clip(np.searchsorted(radar[:, 0], self.t), 0, len(radar) - 1), 1]

  def __len__(self) -> int:
    return len(self.t)

  def segment(self, frame: int) -> tuple[int, int]:
    """Segment of a frame, and the frame's index in that segment's files"""
    i = int(np.searchsorted(self.seg_start, frame, side='right')) - 1
    return i, frame - int(self.seg_start[i])

  def relative_pose(self, i: int, pose: np.ndarray) -> np.ndarray:
    """(ds, e, psi) of pose in the frame of logged frame i"""
    dx, dy = pose[:2] - self.pose[i, :2]
    c, s = np.cos(self.pose[i, 2]), np.sin(self.pose[i, 2])
    return np.array([c * dx + s * dy, -s * dx + c * dy, pose[2] - self.pose[i, 2]])


class FrameStream:
  """Decodes a camera's files forward from a frame in a background thread, prefetching the next segment's file.
  Frames are requested with non-decreasing indices."""
  TIMEOUT = 30.  # s

  def __init__(self, route: RouteData, files: CameraFiles, start: int, buffer: int = 20):
    self.q: queue.Queue = queue.Queue(buffer)
    self.idx = start - 1
    self.frame: np.ndarray | None = None
    self.closed = threading.Event()
    threading.Thread(target=self._decode, args=(route, files, start), daemon=True).start()

  def _decode(self, route: RouteData, files: CameraFiles, start: int) -> None:
    try:
      seg, local = route.segment(start)
      for i in range(seg, len(route.seg_frames)):
        path, index = files.get(i)
        if i + 1 < len(route.seg_frames):
          files.fetch(i + 1)
        for frame in FrameIterator(path, index, pix_fmt='nv12', start_fidx=local, end_fidx=route.seg_frames[i]):
          if not self._put(frame):
            return
        local = 0
      self._put(None)
    except Exception as e:
      self._put(e)

  def _put(self, item) -> bool:
    while not self.closed.is_set():
      try:
        self.q.put(item, timeout=0.1)
        return True
      except queue.Full:
        pass
    return False

  def get(self, idx: int) -> np.ndarray:
    while self.idx < idx:
      try:
        frame = self.q.get(timeout=self.TIMEOUT)
      except queue.Empty:
        raise TimeoutError(f"no frame from the camera files in {self.TIMEOUT:.0f} s") from None
      if isinstance(frame, Exception):
        raise frame
      self.frame = frame
      self.idx += 1
    assert self.frame is not None, f"camera files end before frame {idx}"
    return self.frame

  def close(self) -> None:
    self.closed.set()
