import av
import cv2 as cv

class Camera:
  def __init__(self, cam_type_state, stream_type, camera_id):
    use_gst = False
    # 数値に変換できない場合は文字列パイプラインとして扱う
    try:
      camera_id = int(camera_id)
    except ValueError:  # allow strings, ex: /dev/video0 or gst pipeline
      use_gst = True

    self.cam_type_state = cam_type_state
    self.stream_type = stream_type
    self.cur_frame_id = 0

    print(f"Opening {cam_type_state} at {camera_id}")

    if use_gst:
      # GStreamerパイプライン入力を許可（CSIカメラ向け）
      self.cap = cv.VideoCapture(camera_id, cv.CAP_GSTREAMER)
    else:
      self.cap = cv.VideoCapture(camera_id)

    # GStreamerパイプラインの場合は caps で解像度/フレームレートを指定済みのため
    # set() は呼ばない。V4L2 デバイスの場合のみ設定する。
    if not use_gst:
      self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280.0)
      self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720.0)
      self.cap.set(cv.CAP_PROP_FPS, 25.0)

    if not self.cap.isOpened():
      raise RuntimeError(f"Failed to open camera {camera_id}")

    self.W = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
    self.H = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)

  @classmethod
  def bgr2nv12(self, bgr):
    frame = av.VideoFrame.from_ndarray(bgr, format='bgr24')
    return frame.reformat(format='nv12').to_ndarray()

  def read_frames(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      # # Rotate the frame 180 degrees (flip both axes)
      # frame = cv.flip(frame, -1)
      # 反転なし
      yuv = Camera.bgr2nv12(frame)
      yield yuv.data.tobytes()
    self.cap.release()
