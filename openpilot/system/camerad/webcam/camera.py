import cv2 as cv
import numpy as np

_Y_R = np.uint32(16829)
_Y_G = np.uint32(33039)
_Y_B = np.uint32(6416)
_U_R = np.int32(-9714)
_U_G = np.int32(-19071)
_U_B = np.int32(28784)
_V_R = np.int32(28784)
_V_G = np.int32(-24103)
_V_B = np.int32(-4681)

class Camera:
  def __init__(self, cam_type_state, stream_type, camera_id):
    try:
      camera_id = int(camera_id)
    except ValueError: # allow strings, ex: /dev/video0
      pass
    self.cam_type_state = cam_type_state
    self.stream_type = stream_type
    self.cur_frame_id = 0

    print(f"Opening {cam_type_state} at {camera_id}")

    self.cap = cv.VideoCapture(camera_id)

    self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280.0)
    self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720.0)
    self.cap.set(cv.CAP_PROP_FPS, 25.0)

    self.W = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
    self.H = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    self._bgr2nv12_shape = None
    self._bgr2nv12_buffers = None

  @staticmethod
  def _check_bgr_frame(bgr):
    if not isinstance(bgr, np.ndarray):
      raise ValueError(f"expected BGR frame as np.ndarray, got {type(bgr).__name__}")

    if bgr.dtype != np.uint8:
      raise ValueError(f"expected uint8 BGR frame, got {bgr.dtype}")

    if bgr.ndim != 3 or bgr.shape[2] != 3:
      raise ValueError(f"expected BGR frame with shape (height, width, 3), got {bgr.shape}")

    h, w = bgr.shape[:2]
    if h % 2 or w % 2:
      raise ValueError(f"NV12 conversion requires even frame dimensions, got {w}x{h}")

    return h, w

  @staticmethod
  def _make_bgr2nv12_buffers(h, w):
    return (
      np.empty((h * 3 // 2, w), dtype=np.uint8),
      np.empty((h, w), dtype=np.uint32),
      np.empty((h, w), dtype=np.uint32),
      np.empty((h // 2, w // 2), dtype=np.int32),
      np.empty((h // 2, w // 2), dtype=np.int32),
      np.empty((h // 2, w // 2), dtype=np.int32),
      np.empty((h // 2, w // 2), dtype=np.int32),
      np.empty((h // 2, w // 2), dtype=np.int32),
    )

  @staticmethod
  def _bgr2nv12_into(bgr, nv12, y, y_tmp, b, g, r, uv_tmp, chroma_tmp):
    h, w = bgr.shape[:2]

    # BT.601 limited-range conversion, with coefficients scaled by 2^16.
    np.multiply(bgr[:, :, 2], _Y_R, out=y, casting="unsafe")
    np.multiply(bgr[:, :, 1], _Y_G, out=y_tmp, casting="unsafe")
    y += y_tmp
    np.multiply(bgr[:, :, 0], _Y_B, out=y_tmp, casting="unsafe")
    y += y_tmp
    y += 32768
    y >>= 16
    y += 16
    nv12[:h] = y

    b[:] = bgr[0::2, 0::2, 0]
    b += bgr[0::2, 1::2, 0]
    b += bgr[1::2, 0::2, 0]
    b += bgr[1::2, 1::2, 0]
    b += 2
    b >>= 2

    g[:] = bgr[0::2, 0::2, 1]
    g += bgr[0::2, 1::2, 1]
    g += bgr[1::2, 0::2, 1]
    g += bgr[1::2, 1::2, 1]
    g += 2
    g >>= 2

    r[:] = bgr[0::2, 0::2, 2]
    r += bgr[0::2, 1::2, 2]
    r += bgr[1::2, 0::2, 2]
    r += bgr[1::2, 1::2, 2]
    r += 2
    r >>= 2

    uv = nv12[h:].reshape(h // 2, w // 2, 2)
    np.multiply(b, _U_B, out=uv_tmp, casting="unsafe")
    np.multiply(g, _U_G, out=chroma_tmp, casting="unsafe")
    uv_tmp += chroma_tmp
    np.multiply(r, _U_R, out=chroma_tmp, casting="unsafe")
    uv_tmp += chroma_tmp
    uv_tmp += 32768
    uv_tmp >>= 16
    uv_tmp += 128
    uv[:, :, 0] = uv_tmp

    np.multiply(b, _V_B, out=uv_tmp, casting="unsafe")
    np.multiply(g, _V_G, out=chroma_tmp, casting="unsafe")
    uv_tmp += chroma_tmp
    np.multiply(r, _V_R, out=chroma_tmp, casting="unsafe")
    uv_tmp += chroma_tmp
    uv_tmp += 32768
    uv_tmp >>= 16
    uv_tmp += 128
    uv[:, :, 1] = uv_tmp

    return nv12

  @staticmethod
  def bgr2nv12(bgr):
    h, w = Camera._check_bgr_frame(bgr)
    return Camera._bgr2nv12_into(bgr, *Camera._make_bgr2nv12_buffers(h, w))

  def _reused_bgr2nv12(self, bgr):
    h, w = Camera._check_bgr_frame(bgr)
    if self._bgr2nv12_shape != (h, w):
      self._bgr2nv12_shape = (h, w)
      self._bgr2nv12_buffers = Camera._make_bgr2nv12_buffers(h, w)

    return Camera._bgr2nv12_into(bgr, *self._bgr2nv12_buffers)

  def read_frames(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      # Rotate the frame 180 degrees (flip both axes)
      frame = frame[::-1, ::-1]
      yuv = self._reused_bgr2nv12(frame)
      yield yuv.data.tobytes()
    self.cap.release()
