import unittest

import numpy as np

from openpilot.common.hardware import COMMA_HARDWARE


@unittest.skipUnless(COMMA_HARDWARE, 'requires compiled QCOM model artifacts')
class TestModelState(unittest.TestCase):
  def test_recurrent_histories(self):
    from openpilot.selfdrive.modeld.modeld import ModelState

    model = ModelState(1344, 760, False)
    model.warmup()
    expected = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in model.input_shapes.items() if name in model.state_pairs}
    for name, value in expected.items():
      np.testing.assert_array_equal(model.input_queues[name].numpy(), value)
    transforms = dict.fromkeys(('img', 'big_img'), np.eye(3, dtype=np.float32))
    previous_desire = np.zeros(8, dtype=np.float32)

    for frame in range(128):
      image = np.full(model.frame_copy_size, frame % 251, dtype=np.uint8)
      desire = np.zeros(8, dtype=np.float32)
      if frame in (40, 41, 70):
        desire[1] = 1
      pulse = np.where(desire - previous_desire > .99, desire, 0)
      previous_desire[:] = desire
      model.run(dict.fromkeys(('img', 'big_img'), image), transforms,
                {'desire_pulse': desire, 'traffic_convention': np.array([1, 0], dtype=np.float32),
                 'action_t': np.array([0.2, 0.2], dtype=np.float32)})
      raw = model.outputs['outputs'].numpy()
      self.assertTrue(np.isfinite(raw).all())
      hidden = raw[:, model.output_slices['hidden_state']]
      expected['state_feat_q'] = np.concatenate((expected['state_feat_q'][1:], hidden[None]), axis=0)
      expected['state_desire_q'] = np.concatenate((expected['state_desire_q'][1:], pulse[None, None]), axis=0)
      new_img = np.full(model.input_queues['new_img'].shape, frame % 251, dtype=np.uint8)
      expected['state_img_q'] = np.concatenate((expected['state_img_q'][:, 1:], new_img[:, None]), axis=1)
      for name, value in expected.items():
        with self.subTest(frame=frame, history=name):
          np.testing.assert_array_equal(model.input_queues[name].numpy(), value)


if __name__ == '__main__':
  unittest.main()
