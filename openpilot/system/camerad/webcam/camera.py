import av
import cv2 as cv
import time


class Camera:
  def __init__(self, cam_type_state, stream_type, camera_id):
    try:
      camera_id = int(camera_id)
    except ValueError:  # allow strings, ex: /dev/video0
      pass
    self.cam_type_state = cam_type_state
    self.stream_type = stream_type
    self.cur_frame_id = 0

    print(f"Opening {cam_type_state} at {camera_id}")

    self.cap = cv.VideoCapture(camera_id)

    self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280.0)
    self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720.0)
    # Match VisionIPC/encoderd's fixed 20 Hz cadence. Capturing faster than the
    # consumer can drain V4L2 risks buffering stale frames and corrupting the
    # apparent camera-to-CAN timing.
    self.cap.set(cv.CAP_PROP_FPS, 20.0)
    self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1.0)

    if not self.cap.isOpened():
      raise RuntimeError(f"failed to open camera {camera_id}")

    self.W = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
    self.H = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    if self.W <= 0 or self.H <= 0:
      raise RuntimeError(f"camera {camera_id} returned an invalid resolution: {self.W}x{self.H}")

  @classmethod
  def bgr2nv12(self, bgr):
    frame = av.VideoFrame.from_ndarray(bgr, format='bgr24')
    return frame.reformat(format='nv12').to_ndarray()

  def read_frames(self):
    while True:
      timestamp_sof = time.monotonic_ns()
      ret, frame = self.cap.read()
      timestamp_eof = time.monotonic_ns()
      if not ret:
        break
      # Rotate the frame 180 degrees (flip both axes)
      frame = cv.flip(frame, -1)
      yuv = Camera.bgr2nv12(frame)
      yield yuv.data.tobytes(), timestamp_sof, timestamp_eof
    self.cap.release()
