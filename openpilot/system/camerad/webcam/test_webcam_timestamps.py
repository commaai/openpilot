import numpy as np
from msgq.visionipc import VisionStreamType

from openpilot.system.camerad.webcam import camera as camera_module
from openpilot.system.camerad.webcam.camera import Camera
from openpilot.system.camerad.webcam.camerad import Camerad, camera_device


class FakeVisionIpcServer:
  def __init__(self):
    self.sent = []

  def send(self, *args):
    self.sent.append(args)


class FakePubMaster:
  def __init__(self):
    self.sent = []

  def send(self, service, message):
    self.sent.append((service, message.as_reader()))


class FakeCapture:
  def __init__(self):
    self.settings = {}

  def set(self, prop, value):
    self.settings[prop] = value
    return True

  def get(self, prop):
    return {
      camera_module.cv.CAP_PROP_FRAME_WIDTH: 1280,
      camera_module.cv.CAP_PROP_FRAME_HEIGHT: 720,
    }.get(prop, 0)

  def isOpened(self):
    return True


def test_camera_capture_matches_encoder_cadence(monkeypatch):
  capture = FakeCapture()
  monkeypatch.setattr(camera_module.cv, "VideoCapture", lambda _camera_id: capture)

  Camera("roadCameraState", VisionStreamType.VISION_STREAM_ROAD, "0")

  assert capture.settings[camera_module.cv.CAP_PROP_FPS] == 20.0
  assert capture.settings[camera_module.cv.CAP_PROP_BUFFERSIZE] == 1.0


def test_linux_camera_device_preserves_absolute_paths():
  assert camera_device("3", system="Linux") == "/dev/video3"
  assert camera_device("/dev/video0", system="Linux") == "/dev/video0"
  assert camera_device("/dev/v4l/by-id/road-camera", system="Linux") == "/dev/v4l/by-id/road-camera"


def test_bgr_to_nv12_has_expected_size():
  width, height = 64, 48
  nv12 = Camera.bgr2nv12(np.zeros((height, width, 3), dtype=np.uint8))

  assert nv12.nbytes == width * height * 3 // 2


def test_send_yuv_preserves_monotonic_capture_timestamps():
  camerad = Camerad.__new__(Camerad)
  camerad.vipc_server = FakeVisionIpcServer()
  camerad.pm = FakePubMaster()
  timestamp_sof = 12_345_000_000
  timestamp_eof = 12_346_000_000

  camerad._send_yuv(
    b"nv12",
    7,
    "roadCameraState",
    VisionStreamType.VISION_STREAM_ROAD,
    timestamp_sof,
    timestamp_eof,
  )

  assert camerad.vipc_server.sent == [
    (VisionStreamType.VISION_STREAM_ROAD, b"nv12", 7, timestamp_sof, timestamp_eof),
  ]
  service, message = camerad.pm.sent[0]
  assert service == "roadCameraState"
  assert message.roadCameraState.frameId == 7
  assert message.roadCameraState.timestampSof == timestamp_sof
  assert message.roadCameraState.timestampEof == timestamp_eof
