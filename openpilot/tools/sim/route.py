import itertools
import queue
import threading
import numpy as np
from collections.abc import Iterator

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.tools.lib.file_sources import openpilotci_source
from openpilot.tools.lib.framereader import FrameIterator, get_video_index
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.route import FileName, Route, SegmentRange


def camera_url(sr: SegmentRange, seg: int, fn: tuple[str, ...]) -> str:
  if url := openpilotci_source(sr, [seg], fn).get(seg):
    return url
  route = Route(sr.route_name)
  return (route.camera_paths() if fn == FileName.FCAMERA else route.ecamera_paths())[seg]


class RouteData:
  """One logged segment: per-frame path (x fwd, y right, heading right-positive), speed, lead and calibration."""
  def __init__(self, segment: str):
    sr = SegmentRange(segment)
    seg = sr.seg_idxs[0]
    self.road_url = camera_url(sr, seg, FileName.FCAMERA)
    self.wide_url = camera_url(sr, seg, FileName.ECAMERA)
    # indexing reads the whole file, so it's done once here instead of every time a FrameStream starts
    self.road_index = get_video_index(self.road_url)
    self.wide_index = get_video_index(self.wide_url)

    msgs: dict[str, list] = {k: [] for k in ('carState', 'deviceMotion', 'radarState', 'narrowRoadEncodeIdx', 'wideRoadEncodeIdx')}
    for m in LogReader(f"{sr.route_name}/{seg}"):
      w = m.which()
      if w in msgs:
        msgs[w].append((m.logMonoTime * 1e-9, getattr(m, w)))
      elif w == 'extrinsicsCalibration':
        self.calib_msg = m.as_builder().to_bytes()
        calib = m.extrinsicsCalibration
      elif w == 'initData':
        device_type = str(m.initData.deviceType)
      elif w == 'narrowRoadCameraState':
        sensor = str(m.narrowRoadCameraState.sensor)

    self.camera = DEVICE_CAMERAS[(device_type, sensor)]
    assert self.camera.narrow_road.size == (1928, 1208), f"unsupported camera {device_type} {sensor}"
    self.rpy_calib = np.array(calib.rpyCalib)
    self.wide_from_device_euler = np.array(calib.wideFromDeviceEuler)
    self.height = calib.height[0]

    road_idx = {e.segmentId: e.timestampEof * 1e-9 for _, e in msgs['narrowRoadEncodeIdx']}
    wide_idx = {e.segmentId for _, e in msgs['wideRoadEncodeIdx']}
    n = next(i for i in range(len(road_idx) + 1) if i not in road_idx or i not in wide_idx)
    self.t = np.array([road_idx[i] for i in range(n)])

    # carState and deviceMotion are logged with logMonoTime, encodeIdx with the frame's EOF, both on the monotonic clock
    cs_t = np.array([t for t, _ in msgs['carState']])
    v = np.array([cs.vEgo for _, cs in msgs['carState']])
    dm_t = np.array([t for t, _ in msgs['deviceMotion']])
    yaw_rate = np.interp(cs_t, dm_t, [dm.angularVelocityDevice.z for _, dm in msgs['deviceMotion']])
    dt = np.diff(cs_t, prepend=cs_t[0])
    heading = np.cumsum(yaw_rate * dt)
    x = np.cumsum(v * np.cos(heading) * dt)
    y = np.cumsum(v * np.sin(heading) * dt)
    s = np.cumsum(v * dt)

    self.v = np.interp(self.t, cs_t, v)
    self.yaw_rate = np.interp(self.t, cs_t, yaw_rate)
    self.s = np.interp(self.t, cs_t, s)
    self.pose = np.stack([np.interp(self.t, cs_t, x), np.interp(self.t, cs_t, y), np.interp(self.t, cs_t, heading)], axis=1)

    lead_t = np.array([t for t, _ in msgs['radarState']])
    lead_d = np.array([rs.leadOne.dRel if rs.leadOne.present else np.nan for _, rs in msgs['radarState']])
    self.lead_gap = lead_d[np.clip(np.searchsorted(lead_t, self.t), 0, len(lead_t) - 1)]

  def __len__(self) -> int:
    return len(self.t)

  def relative_pose(self, i: int, pose: np.ndarray) -> np.ndarray:
    """(ds, e, psi) of pose in the frame of logged frame i"""
    dx, dy = pose[:2] - self.pose[i, :2]
    c, s = np.cos(self.pose[i, 2]), np.sin(self.pose[i, 2])
    return np.array([c * dx + s * dy, -s * dx + c * dy, pose[2] - self.pose[i, 2]])


class FrameStream:
  """Decodes a camera file forward in a background thread. Frames are requested with non-decreasing indices."""
  TIMEOUT = 10.  # s

  def __init__(self, url: str, index: dict, start: int, buffer: int = 20):
    self.q: queue.Queue = queue.Queue(buffer)
    self.idx = start - 1
    self.frame: np.ndarray | None = None
    self.closed = threading.Event()
    threading.Thread(target=self._decode, args=(url, index, start), daemon=True).start()

  def _decode(self, url: str, index: dict, start: int) -> None:
    try:
      frames: Iterator[np.ndarray | Exception | None] = itertools.chain(FrameIterator(url, index, pix_fmt='nv12', start_fidx=start), [None])
      for frame in frames:
        if not self._put(frame):
          return
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
        raise TimeoutError(f"no frame from the camera file in {self.TIMEOUT:.0f} s") from None
      if isinstance(frame, Exception):
        raise frame
      self.frame = frame
      self.idx += 1
    assert self.frame is not None, f"camera file ends before frame {idx}"
    return self.frame

  def close(self) -> None:
    self.closed.set()
