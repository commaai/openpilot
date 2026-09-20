from collections import defaultdict
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from openpilot.selfdrive.modeld import modeld


class TestModeldFailover(unittest.TestCase):
  def run_modeld(self, *, chestnut=True, fail_frame=None, failure=RuntimeError):
    published, runs, models, values = defaultdict(list), [], {}, {}
    camera = SimpleNamespace(width=2, height=2, buffer_len=6, frame_id=0, timestamp_sof=0, timestamp_eof=0)

    def recv():
      if camera.frame_id == 4:
        raise KeyboardInterrupt
      camera.frame_id += 1
      camera.timestamp_sof = camera.frame_id * 50_000_000
      camera.timestamp_eof = camera.timestamp_sof + 10_000_000
      return SimpleNamespace(frame_id=camera.frame_id)

    camera.recv, camera.connect = recv, lambda blocking: True
    vipc = Mock(return_value=camera)
    vipc.available_streams.return_value = [modeld.VisionStreamType.VISION_STREAM_NARROW_ROAD]
    sm = MagicMock()
    sm.updated = {'extrinsicsCalibration': False}
    sm.__getitem__.side_effect = {
      'driverMonitoringState': SimpleNamespace(isRHD=False), 'narrowRoadCameraState': SimpleNamespace(frameId=0),
      'carState': SimpleNamespace(vEgo=10., leftBlinker=False, rightBlinker=False),
      'carControl': SimpleNamespace(latActive=False), 'lateralDelay': SimpleNamespace(lateralDelay=0.1),
    }.__getitem__
    params = Mock()
    params.put_bool.side_effect = values.__setitem__
    params.remove.side_effect = lambda key: values.pop(key, None)
    params.get_bool.side_effect = lambda key: values.get(key, False)
    pm = Mock()
    pm.send.side_effect = lambda name, msg: published[name].append(
      modeld.messaging.log_from_bytes(msg) if isinstance(msg, bytes) else msg.as_reader())

    class ModelBase:
      chestnut = False
      vision_input_names = ('img', 'big_img')

      def __init__(self, *args):
        self.warmup = Mock()
        models['big' if self.chestnut else 'small'] = self

      def infer(self, bufs, transforms, inputs, callback=None):
        frame = bufs['img'].frame_id
        runs.append((self.chestnut, frame))
        if self.chestnut and frame == fail_frame:
          raise failure('injected primary failure')
        if callback is not None:
          callback()
        output = {'frame': frame, 'action': np.array([[0., float(frame)]])}
        for name, width in [('pose', 6), ('wide_from_device_euler', 3), ('road_transform', 6)]:
          output[name] = np.full((1, width), float(frame))
          output[name + '_stds'] = np.ones((1, width))
        return output

    class SmallModel(ModelBase):
      def run(self, bufs, transforms, inputs, callback=None):
        return self.infer(bufs, transforms, inputs, callback)

    class BigModel(ModelBase):
      chestnut = True

      def __init__(self, width, height, cleanup=None):
        super().__init__(width, height)
        self.worker = Mock()
        if cleanup is not None:
          cleanup.callback(self.worker.close)

      def run(self, bufs, transforms, inputs, send_state, deadline):
        output = self.infer(bufs, transforms, inputs)
        messages = [modeld.messaging.new_message('chestnutGpuState').to_bytes()] if send_state else []
        return output, messages

    def fill_model(msg, output, action, state, frame, extra_frame, camera_frame, drops, eof, execution_time, valid):
      self.assertEqual(output['frame'], frame)
      msg.valid = valid
      m = msg.modelV2
      m.frameId, m.frameIdExtra, m.timestampEof = frame, extra_frame, eof
      m.frameDropPerc, m.modelExecutionTime, m.action = drops * 100, execution_time, action
      m.meta.desireState = [0.] * modeld.ModelConstants.DESIRE_LEN
      for line in m.init('laneLines', 4):
        line.y = [0.]
      m.laneLineProbs = [0.] * 4
      for axis in ('x', 'y', 'z'):
        setattr(m.position, axis, [0.] * modeld.ModelConstants.IDX_N)

    replacements: dict[str, Any] = {
      'chestnut_present': Mock(return_value=chestnut), 'chestnut_compiled': Mock(return_value=True),
      'VisionIpcClient': vipc, 'ModelState': SmallModel, 'BigModelProcess': BigModel,
      'Params': Mock(return_value=params), 'PubMaster': Mock(return_value=pm), 'SubMaster': Mock(return_value=sm),
      'get_demo_car_params': Mock(return_value=SimpleNamespace(brand='mock', longitudinalActuatorDelay=0.2)),
      'fill_model_msg': fill_model, 'config_realtime_process': Mock(), 'cloudlog': Mock(),
      'get_action_from_model': Mock(wraps=modeld.get_action_from_model), 'DesireHelper': Mock(wraps=modeld.DesireHelper),
    }
    with patch.multiple(modeld, **replacements), ExitStack() as cleanup, self.assertRaises(KeyboardInterrupt):
      modeld.modeld_thread(True, cleanup)

    expected = [frame for frame in range(1, 5) if frame != fail_frame]
    for service in ('modelV2', 'drivingModelData', 'cameraOdometry'):
      self.assertEqual([getattr(msg, service).frameId for msg in published[service]], expected)
    for index, frame in enumerate(expected):
      model, driving, pose = (getattr(published[name][index], name) for name in ('modelV2', 'drivingModelData', 'cameraOdometry'))
      self.assertEqual(model.frameIdExtra, driving.frameIdExtra)
      self.assertEqual(model.timestampEof, pose.timestampEof)
      self.assertEqual(model.timestampEof, frame * 50_000_000 + 10_000_000)
      self.assertEqual(pose.trans[0], frame)
      self.assertEqual(model.big, chestnut and (fail_frame is None or frame < fail_frame))
      self.assertEqual(driving.big, model.big)
      self.assertEqual(published['modelV2'][index].valid, published['drivingModelData'][index].valid)
      self.assertAlmostEqual(driving.action.desiredAcceleration, model.action.desiredAcceleration)
      previous = replacements['get_action_from_model'].call_args_list[index].args[1].desiredAcceleration
      self.assertAlmostEqual(previous, published['modelV2'][index - 1].modelV2.action.desiredAcceleration if index else 0.)
    replacements['DesireHelper'].assert_called_once_with()
    replacements['config_realtime_process'].assert_called_once_with(7, 54)
    self.assertFalse(values['ChestnutLoading'])
    if chestnut:
      self.assertTrue(published['chestnutGpuState'])
      models['small'].warmup.assert_called_once_with()
      models['big'].worker.close.assert_called_once_with()
      self.assertEqual(models['big'].worker.stop.call_count, int(fail_frame is not None))
      self.assertEqual(values['ChestnutActive'], fail_frame is None)
    else:
      self.assertNotIn('chestnutGpuState', published)
      self.assertNotIn('big', models)
      models['small'].warmup.assert_not_called()
    self.assertEqual(runs, [(chestnut and (fail_frame is None or frame <= fail_frame), frame) for frame in range(1, 5)])

  def test_normal_primary_keeps_all_production_outputs(self):
    self.run_modeld()

  def test_primary_failure_uses_new_frame_and_preserves_action(self):
    for failure in (RuntimeError, TimeoutError):
      for frame in (1, 3):
        with self.subTest(failure=failure, frame=frame):
          self.run_modeld(fail_frame=frame, failure=failure)

  def test_no_chestnut_uses_only_synchronous_small_model(self):
    self.run_modeld(chestnut=False)


if __name__ == '__main__':
  unittest.main()
