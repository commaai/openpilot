from collections import namedtuple
from cereal.visionipc import VisionStreamType
from common.realtime import DT_MDL, DT_DMON
from common.transformations.camera import tici_f_frame_size, tici_d_frame_size, tici_e_frame_size, eon_f_frame_size, eon_d_frame_size

class CameraFrameSizes:
  def __init__(self, eon_size, tici_size):
    self.eon = eon_size
    self.tici = tici_size
  
  def __getitem__(self, key):
    if key in ["tici", "tizi"]:
      return self.tici
    else:
      return self.eon

VideoStreamMeta = namedtuple("VideoStreamMeta", ["camera_state", "stream", "dt", "frame_sizes"])
VIPC_STREAM_METADATA = [
  # metadata: (state_msg_type, stream_type, dt)
  ("roadCameraState", VisionStreamType.VISION_STREAM_ROAD, DT_MDL, CameraFrameSizes(eon_f_frame_size, tici_f_frame_size)),
  ("wideRoadCameraState", VisionStreamType.VISION_STREAM_WIDE_ROAD, DT_MDL, CameraFrameSizes(None, tici_e_frame_size)),
  ("driverCameraState", VisionStreamType.VISION_STREAM_DRIVER, DT_DMON, CameraFrameSizes(eon_d_frame_size, tici_d_frame_size)),
]


def meta_from_camera_state(state):
  meta = next((VideoStreamMeta(*meta) for meta in VIPC_STREAM_METADATA if meta[0] == state), None)
  return meta 


def meta_from_stream_type(stream_type):
  meta = next((VideoStreamMeta(*meta) for meta in VIPC_STREAM_METADATA if meta[1] == stream_type), None)
  return meta


def available_streams(lr=None):
  if lr is None:
    return [VideoStreamMeta(*meta) for meta in VIPC_STREAM_METADATA]

  result = []
  for meta in VIPC_STREAM_METADATA:
    has_cam_state = next((True for m in lr if m.which() == meta[0]), False)
    if has_cam_state:
      result.append(VideoStreamMeta(*meta))

  return result
