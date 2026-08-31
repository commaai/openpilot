import unittest

import numpy as np

from openpilot.selfdrive.modeld.onnx_cpu import OnnxCpuPolicy


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
