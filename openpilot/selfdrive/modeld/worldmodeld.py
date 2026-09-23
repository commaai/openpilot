#!/usr/bin/env python3
import os
os.environ['GMMU'] = '0'
os.environ.setdefault('AM_POWER_LIMIT', '100')
import time
import platform
from functools import lru_cache
from pathlib import Path

import numpy as np
from tinygrad import Context, Device, Tensor

from openpilot.cereal import messaging
from openpilot.cereal.services import SERVICE_LIST
from openpilot.cereal.visionipc import VisionStreamType
from msgq.visionipc import VisionIpcClient
from opendbc.car.structs import car
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, set_core_affinity
from openpilot.common.swaglog import cloudlog
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.selfdrive.modeld.worldmodel_pkl import load_worldmodel
from openpilot.selfdrive.modeld.helpers import WORLDMODEL_DIR
from openpilot.selfdrive.modeld.constants import LAT_SMOOTH_SECONDS, LONG_SMOOTH_SECONDS

WORLD_MODEL_FREQ = SERVICE_LIST['worldModelPlan'].frequency
CAMERA_FRAME_STRIDE = int(SERVICE_LIST['narrowRoadCameraState'].frequency / WORLD_MODEL_FREQ)


class WorldModelRunner:
  def __init__(self, directory: Path):
    device = Device[Device.DEFAULT]
    if device.is_usb:
      device.iface.dev_impl.smu.set_clocks(level=None)
    artifact = load_worldmodel(directory / 'model.pkl')
    self.live_frames = artifact['live_frames']
    self.input_names = artifact.get('inputs', ('images',))
    programs = artifact['programs'][platform.machine().lower()]
    self.jit, self.reset_jit = programs['run'], programs['reset']
    self.run(np.zeros((1, 6, 128, 256), dtype=np.uint8), np.full((1, 2), .5, dtype=np.float32))
    self.reset()
    self.execution_time = 1 / WORLD_MODEL_FREQ

  def reset(self):
    self.reset_jit()

  def run(self, images, action_t):
    start = time.monotonic()
    inputs = {'images': images, 'action_t': action_t}
    outputs = self.jit(*(Tensor(inputs[name], device='NPY') for name in self.input_names))
    if isinstance(outputs, Tensor):
      outputs = {'plan': outputs}
    outputs = {name: value.numpy() for name, value in outputs.items()}
    if not all(np.isfinite(value).all() for value in outputs.values()):
      raise RuntimeError('Worldmodel output is not finite')
    self.execution_time = time.monotonic() - start
    return outputs

@lru_cache(maxsize=8)
def warp_indices(width, height, transform):
  yy, xx = np.indices((128, 256), dtype=np.float64)
  warp = np.asarray(transform).reshape(3, 3) @ np.diag([2., 2., 1.])
  denominator = warp[2, 0] * xx + warp[2, 1] * yy + warp[2, 2]
  sx = np.rint((warp[0, 0] * xx + warp[0, 1] * yy + warp[0, 2]) / denominator).astype(np.int32).clip(0, width - 1)
  sy = np.rint((warp[1, 0] * xx + warp[1, 1] * yy + warp[1, 2]) / denominator).astype(np.int32).clip(0, height - 1)
  return sx, sy


def prepare_image(buffer, width, height, transform):
  stride, y_height, uv_height, _ = get_nv12_info(width, height)
  data = np.frombuffer(buffer.data, dtype=np.uint8)
  y = data[:stride * y_height].reshape(y_height, stride)
  uv = data[stride * y_height:stride * (y_height + uv_height)].reshape(uv_height, stride)
  sx, sy = warp_indices(width, height, tuple(transform.ravel()))
  # BT.601 limited-range NV12 conversion, after gathering the nearest source pixels.
  luma = np.maximum(y[sy, sx].astype(np.int32) - 16, 0) * 1220542
  u = uv[sy // 2, (sx // 2) * 2].astype(np.int32) - 128
  v = uv[sy // 2, (sx // 2) * 2 + 1].astype(np.int32) - 128
  rgb = np.stack((luma + 1673527 * v, luma - 409993 * u - 852492 * v, luma + 2116026 * u))
  return ((rgb + (1 << 19)) >> 20).clip(0, 255).astype(np.uint8)


def main():
  config_realtime_process([4, 5], 5)
  directory = Path(WORLDMODEL_DIR)
  with Context(DEV='USB+AMD:LLVM', TC_OPT=2, TC_MIN_GLOBALS=32, JIT_BATCH_SIZE=0):
    runner = WorldModelRunner(directory)
    set_core_affinity([6])
    pm = messaging.PubMaster(['worldModelPlan'])
    sm = messaging.SubMaster(['deviceState', 'narrowRoadCameraState', 'extrinsicsCalibration', 'lateralDelay'])
    CP = messaging.log_from_bytes(Params().get('CarParams', block=True), car.CarParams)
    long_delay = CP.longitudinalActuatorDelay + LONG_SMOOTH_SECONDS
    cameras = [VisionIpcClient('camerad', stream, True) for stream in
               (VisionStreamType.VISION_STREAM_NARROW_ROAD, VisionStreamType.VISION_STREAM_WIDE_ROAD)]
    for camera in cameras:
      while not camera.connect(False):
        time.sleep(.1)
    last_timestamp, last_frame_id, history_frames = 0, None, 0
    cloudlog.info('Worldmodel planner ready; target %d FPS', WORLD_MODEL_FREQ)
    while True:
      narrow = cameras[0].recv()
      if narrow is None:
        continue
      # Count camera frames so timestamp jitter cannot skip a complete planner period.
      if last_frame_id is not None and 0 <= cameras[0].frame_id - last_frame_id < CAMERA_FRAME_STRIDE:
        continue
      wide = cameras[1].recv()
      while narrow is not None and wide is not None and abs(cameras[0].timestamp_sof - cameras[1].timestamp_sof) > 10_000_000:
        if cameras[0].timestamp_sof < cameras[1].timestamp_sof:
          narrow = cameras[0].recv()
        else:
          wide = cameras[1].recv()
      if narrow is None or wide is None:
        continue
      timestamp = cameras[0].timestamp_eof
      frame_id = cameras[0].frame_id
      sm.update(0)
      if not all(sm.seen.values()):
        continue
      if (last_frame_id is not None and frame_id < last_frame_id) or timestamp <= last_timestamp or timestamp - last_timestamp > 2e9 / WORLD_MODEL_FREQ:
        runner.reset()
        history_frames = 0
      last_timestamp = timestamp
      last_frame_id = frame_id
      rpy = np.array(sm['extrinsicsCalibration'].rpyCalib, dtype=np.float32)
      dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['narrowRoadCameraState'].sensor))]
      transforms = (get_warp_matrix(rpy, dc.narrow_road.intrinsics), get_warp_matrix(rpy, dc.wide_road.intrinsics, True))
      start = time.monotonic()
      images = np.concatenate([prepare_image(buf, camera.width, camera.height, tfm)
                               for buf, camera, tfm in zip((narrow, wide), cameras, transforms, strict=True)], axis=0)[None]
      delay = time.monotonic() - timestamp / 1e9 + runner.execution_time + .5 / WORLD_MODEL_FREQ
      action_t = np.array([[sm['lateralDelay'].lateralDelay + LAT_SMOOTH_SECONDS + delay, long_delay + delay]], dtype=np.float32)
      outputs = runner.run(images, action_t)
      history_frames += 1
      msg = messaging.new_message('worldModelPlan')
      msg.valid = history_frames >= runner.live_frames
      msg.worldModelPlan.frameId = cameras[0].frame_id
      msg.worldModelPlan.timestampEof = timestamp
      msg.worldModelPlan.modelExecutionTime = time.monotonic() - start
      msg.worldModelPlan.plan = outputs['plan'].ravel().tolist()
      msg.worldModelPlan.action = outputs.get('action', np.empty(0)).ravel().tolist()
      msg.worldModelPlan.actionT = action_t.ravel().tolist()
      pm.send('worldModelPlan', msg)


if __name__ == '__main__':
  main()
