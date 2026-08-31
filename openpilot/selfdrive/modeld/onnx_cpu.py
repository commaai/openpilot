import os

import numpy as np


class OnnxCpuPolicy:
  """NumPy queue management and ONNX Runtime inference for CPU-only CI."""

  def __init__(self, model_path: str, input_shapes: dict[str, tuple[int, ...]], frame_skip: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = int(os.environ.get('ONNX_CPU_THREADS', '2'))
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    self.session = ort.InferenceSession(model_path, sess_options=options, providers=['CPUExecutionProvider'])
    self.input_shapes = input_shapes
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
