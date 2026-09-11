import importlib.util
import unittest

import numpy as np

from openpilot.selfdrive.modeld.onnx_cpu import OnnxCpuPolicy, OpenCvCpuWarp


CV2_AVAILABLE = importlib.util.find_spec("cv2") is not None


class FakeSession:
  def __init__(self):
    self.inputs = None

  def run(self, _outputs, inputs):
    self.inputs = inputs
    return [np.zeros((1, 2576), dtype=np.float32)]


class TestOnnxCpuPolicy(unittest.TestCase):
  shapes = {
    'img': (1, 12, 2, 3),
    'big_img': (1, 12, 2, 3),
    'features_buffer': (1, 2, 4),
    'desire_pulse': (1, 2, 3),
  }

  def make_policy(self):
    policy = OnnxCpuPolicy.__new__(OnnxCpuPolicy)
    policy.input_shapes = self.shapes
    policy.frame_skip = 2
    policy.session = FakeSession()
    policy.reset()
    return policy

  def test_queues(self):
    policy = self.make_policy()
    for value in range(4):
      warped = np.full((2, 6, 2, 3), value, dtype=np.uint8)
      desire = np.array([value, 0, 0], dtype=np.float32)
      feature = np.full((1, 4), value, dtype=np.float32)
      output = policy.run(warped, desire, np.array([[1, 0]], dtype=np.float32),
                          np.array([[.1, .2]], dtype=np.float32), feature)

    self.assertEqual(output.shape, (1, 2576))
    self.assertEqual(policy.session.inputs['img'].shape, self.shapes['img'])
    np.testing.assert_array_equal(policy.session.inputs['img'][0, :, 0, 0], [1] * 6 + [3] * 6)
    np.testing.assert_array_equal(policy.session.inputs['desire_pulse'][0, :, 0], [1, 3])
    np.testing.assert_array_equal(policy.session.inputs['features_buffer'][0, :, 0], [0, 2])

  def test_reset(self):
    policy = self.make_policy()
    policy.img_q[:] = 1
    policy.feature_q[:] = 1
    policy.reset()
    self.assertFalse(policy.img_q.any())
    self.assertFalse(policy.feature_q.any())

  @unittest.skipUnless(CV2_AVAILABLE, "OpenCV is not installed")
  def test_opencv_warp_identity(self):
    warp = OpenCvCpuWarp(512, 256)
    frame = np.zeros(warp.buffer_size, dtype=np.uint8)
    y = frame[:warp.stride * warp.y_height].reshape(warp.y_height, warp.stride)
    uv = frame[warp.stride * warp.y_height:warp.stride * (warp.y_height + warp.uv_height)]
    uv = uv.reshape(warp.uv_height, warp.stride)
    y[:256, :512] = np.arange(512, dtype=np.uint8)
    uv[:128, :512:2] = 80
    uv[:128, 1:512:2] = 160

    result = warp.run({'img': frame, 'big_img': frame},
                      {'img': np.eye(3, dtype=np.float32), 'big_img': np.eye(3, dtype=np.float32)})
    self.assertEqual(result.shape, (2, 6, 128, 256))
    np.testing.assert_array_equal(result[0, 0], y[:256:2, :512:2])
    np.testing.assert_array_equal(result[0, 1], y[1:256:2, :512:2])
    np.testing.assert_array_equal(result[0, 2], y[:256:2, 1:512:2])
    np.testing.assert_array_equal(result[0, 3], y[1:256:2, 1:512:2])
    self.assertTrue(np.all(result[:, 4] == 80))
    self.assertTrue(np.all(result[:, 5] == 160))

  @unittest.skipUnless(CV2_AVAILABLE, "OpenCV is not installed")
  def test_opencv_warp_contiguous_sim_buffer(self):
    warp = OpenCvCpuWarp(512, 256)
    data = np.zeros(512 * 256 * 3 // 2, dtype=np.uint8)
    y = data[:512 * 256].reshape(256, 512)
    uv = data[512 * 256:].reshape(128, 512)
    y[:] = np.arange(512, dtype=np.uint8)
    uv[:, 0::2] = 80
    uv[:, 1::2] = 160

    class SimVisionBuf:
      stride = 512
      uv_offset = 512 * 256

      def __init__(self, buf):
        self.data = buf.data

    result = warp.prepare(SimVisionBuf(data), np.eye(3, dtype=np.float32))
    np.testing.assert_array_equal(result[0], y[0::2, 0::2])
    np.testing.assert_array_equal(result[3], y[1::2, 1::2])
    self.assertTrue(np.all(result[4] == 80))
    self.assertTrue(np.all(result[5] == 160))

  @unittest.skipUnless(CV2_AVAILABLE, "OpenCV is not installed")
  def test_opencv_warp_matches_reference(self):
    warp = OpenCvCpuWarp(512, 256)
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, warp.buffer_size, dtype=np.uint8)
    transform = np.array([[1.01, 0.02, 3.2], [-0.01, 0.99, 2.7],
                          [0.00002, -0.00001, 1.0]], dtype=np.float32)

    flat = frame[:warp.stride * warp.y_height].reshape(warp.y_height, warp.stride)
    y = flat[:warp.cam_h, :warp.cam_w]
    uv = frame[warp.stride * warp.y_height:warp.stride * (warp.y_height + warp.uv_height)]
    uv = uv.reshape(warp.uv_height, warp.stride)[:warp.cam_h // 2, :warp.cam_w]

    def sample(src, matrix, width, height):
      dst_y, dst_x = np.indices((height, width), dtype=np.float32)
      src_w = matrix[2, 0] * dst_x + matrix[2, 1] * dst_y + matrix[2, 2]
      src_x = np.rint((matrix[0, 0] * dst_x + matrix[0, 1] * dst_y + matrix[0, 2]) / src_w).astype(np.int32)
      src_y = np.rint((matrix[1, 0] * dst_x + matrix[1, 1] * dst_y + matrix[1, 2]) / src_w).astype(np.int32)
      return src[np.clip(src_y, 0, src.shape[0] - 1), np.clip(src_x, 0, src.shape[1] - 1)]

    uv_transform = transform * np.array([[1.0, 1.0, 0.5], [1.0, 1.0, 0.5],
                                          [2.0, 2.0, 1.0]], dtype=np.float32)
    y_warped = sample(y, transform, 512, 256)
    expected = np.stack((y_warped[0::2, 0::2], y_warped[1::2, 0::2],
                         y_warped[0::2, 1::2], y_warped[1::2, 1::2],
                         sample(uv[:, 0::2], uv_transform, 256, 128),
                         sample(uv[:, 1::2], uv_transform, 256, 128)))
    # OpenCV uses a fixed-point transform table, so coordinates exactly on a
    # nearest-neighbor boundary can differ from the float32 reference.
    actual = warp.prepare(frame, transform)
    self.assertLess(np.count_nonzero(actual != expected) / expected.size, 1e-4)

