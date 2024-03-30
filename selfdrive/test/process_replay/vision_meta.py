from collections import namedtuple
from cereal.visionipc import VisionStreamType
from openpilot.common.realtime import DT_MDL, DT_DMON
from openpilot.common.transformations.camera import DEVICE_CAMERAS

VideoStreamMeta = namedtuple("VideoStreamMeta", ["camera_state", "encode_index", "stream", "dt", "frame_sizes"])
ROAD_CAMERA_FRAME_SIZES = {k: (v.dcam.width, v.dcam.height) for k, v in DEVICE_CAMERAS.items()}
WIDE_ROAD_CAMERA_FRAME_SIZES = {k: (v.ecam.width, v.ecam.height) for k, v in DEVICE_CAMERAS.items() if v.ecam is not None}
DRIVER_CAMERA_FRAME_SIZES = {k: (v.dcam.width, v.dcam.height) for k, v in DEVICE_CAMERAS.items()}
VIPC_STREAM_METADATA = [
  # metadata: (state_msg_type, encode_msg_type, stream_type, dt, frame_sizes)
  ("roadCameraState", "roadEncodeIdx", VisionStreamType.VISION_STREAM_ROAD, DT_MDL, ROAD_CAMERA_FRAME_SIZES),
  ("wideRoadCameraState", "wideRoadEncodeIdx", VisionStreamType.VISION_STREAM_WIDE_ROAD, DT_MDL, WIDE_ROAD_CAMERA_FRAME_SIZES),
  ("driverCameraState", "driverEncodeIdx", VisionStreamType.VISION_STREAM_DRIVER, DT_DMON, DRIVER_CAMERA_FRAME_SIZES),
]


def meta_from_camera_state(state):
  meta = next((VideoStreamMeta(*meta) for meta in VIPC_STREAM_METADATA if meta[0] == state), None)
  return meta


def meta_from_encode_index(encode_index):
  meta = next((VideoStreamMeta(*meta) for meta in VIPC_STREAM_METADATA if meta[1] == encode_index), None)
  return meta


def meta_from_stream_type(stream_type):
  meta = next((VideoStreamMeta(*meta) for meta in VIPC_STREAM_METADATA if meta[2] == stream_type), None)
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
