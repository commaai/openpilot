import base64
import os
import pickle

import numpy as np

from openpilot.system.camerad.cameras.nv12_info import get_nv12_info


class OpenCvCpuWarp:
  """CPU NV12 perspective warp matching compile_modeld's tinygrad preprocessing."""

  def __init__(self, cam_w: int, cam_h: int):
    import cv2

    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    self.cv2 = cv2
    self.cam_w = cam_w
    self.cam_h = cam_h
    self.stride, self.y_height, self.uv_height, self.buffer_size = get_nv12_info(cam_w, cam_h)
    self.model_w = 512
    self.model_h = 256

  def prepare(self, frame, transform: np.ndarray) -> np.ndarray:
    flat = np.frombuffer(frame.data, dtype=np.uint8)
    stride = int(getattr(frame, 'stride', self.stride))
    uv_offset = int(getattr(frame, 'uv_offset', self.stride * self.y_height))
    uv_rows = self.cam_h // 2
    required_size = uv_offset + stride * uv_rows
    if flat.size < required_size:
      raise ValueError(f"NV12 buffer is too small: got {flat.size}, need {required_size}")
    y = flat[:uv_offset].reshape(-1, stride)[:self.cam_h, :self.cam_w]
    uv = flat[uv_offset:required_size].reshape(uv_rows, stride)[:, :self.cam_w]
    u, v = uv[:, 0::2], uv[:, 1::2]

    flags = self.cv2.INTER_NEAREST | self.cv2.WARP_INVERSE_MAP
    y_warped = self.cv2.warpPerspective(y, transform, (self.model_w, self.model_h),
                                        flags=flags, borderMode=self.cv2.BORDER_REPLICATE)
    uv_transform = transform * np.array([[1.0, 1.0, 0.5],
                                          [1.0, 1.0, 0.5],
                                          [2.0, 2.0, 1.0]], dtype=np.float32)
    uv_size = (self.model_w // 2, self.model_h // 2)
    u_warped = self.cv2.warpPerspective(u, uv_transform, uv_size, flags=flags,
                                        borderMode=self.cv2.BORDER_REPLICATE)
    v_warped = self.cv2.warpPerspective(v, uv_transform, uv_size, flags=flags,
                                        borderMode=self.cv2.BORDER_REPLICATE)

    return np.stack((y_warped[0::2, 0::2], y_warped[1::2, 0::2],
                     y_warped[0::2, 1::2], y_warped[1::2, 1::2],
                     u_warped, v_warped))

  def run(self, frames: dict, transforms: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack((self.prepare(frames['img'], transforms['img']),
                     self.prepare(frames['big_img'], transforms['big_img'])))


class OnnxCpuPolicy:
  """NumPy queue management and ONNX Runtime inference for CPU-only CI."""

  def __init__(self, model_path: str, frame_skip: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = int(os.environ.get('ONNX_CPU_THREADS', '2'))
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    self.session = ort.InferenceSession(model_path, sess_options=options, providers=['CPUExecutionProvider'])
    self.input_shapes = {item.name: tuple(int(dim) for dim in item.shape) for item in self.session.get_inputs()}
    metadata = self.session.get_modelmeta().custom_metadata_map
    if 'output_slices' not in metadata:
      raise RuntimeError(f"output_slices metadata is missing from {model_path}")
    self.output_slices = pickle.loads(base64.b64decode(metadata['output_slices']))
    self.frame_skip = frame_skip
    self.reset()

  def reset(self) -> None:
    img_shape = self.input_shapes['img']
    frames = img_shape[1] // 6
    queue_frames = self.frame_skip * (frames - 1) + 1
    self.img_q = np.zeros((queue_frames, 6, *img_shape[2:]), dtype=np.uint8)
    self.big_img_q = np.zeros_like(self.img_q)

    desire_shape = self.input_shapes['desire_pulse']
    feature_shape = self.input_shapes['features_buffer']
    self.desire_q = np.zeros((self.frame_skip * desire_shape[1], desire_shape[0], desire_shape[2]),
                             dtype=np.float32)
    self.feature_q = np.zeros((self.frame_skip * feature_shape[1], feature_shape[0], feature_shape[2]),
                              dtype=np.float32)

  @staticmethod
  def _shift(queue: np.ndarray, value: np.ndarray) -> None:
    queue[:-1] = queue[1:]
    queue[-1:] = value

  def run(self, warped: np.ndarray, desire: np.ndarray, traffic_convention: np.ndarray,
          action_t: np.ndarray, previous_feature: np.ndarray) -> np.ndarray:
    self._shift(self.img_q, warped[0:1])
    self._shift(self.big_img_q, warped[1:2])
    self._shift(self.desire_q, desire.reshape(1, 1, -1))
    self._shift(self.feature_q, previous_feature.reshape(1, 1, -1))

    inputs = {
      'img': self.img_q[::self.frame_skip].reshape(self.input_shapes['img']),
      'big_img': self.big_img_q[::self.frame_skip].reshape(self.input_shapes['big_img']),
      'features_buffer': self.feature_q[::self.frame_skip].reshape(self.input_shapes['features_buffer']),
      'desire_pulse': self.desire_q.reshape(-1, self.frame_skip, 1, desire.shape[-1]).max(axis=1).reshape(
        self.input_shapes['desire_pulse']),
      'traffic_convention': traffic_convention.astype(np.float32, copy=False),
      'action_t': action_t.astype(np.float32, copy=False),
    }
    return self.session.run(None, inputs)[0].astype(np.float32, copy=False)
