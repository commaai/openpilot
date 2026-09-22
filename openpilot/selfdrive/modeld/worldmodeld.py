#!/usr/bin/env python3
import gc
import json
import os
os.environ['GMMU'] = '0'
import time
from functools import partial
from pathlib import Path

import numpy as np
from tinygrad import Context, Device, Tensor, TinyJit, dtypes
from tinygrad.nn.onnx import OnnxRunner

from openpilot.cereal import messaging
from openpilot.cereal.visionipc import VisionStreamType
from msgq.visionipc import VisionIpcClient
from openpilot.common.swaglog import cloudlog
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.selfdrive.modeld.worldmodel import WorldModel, load_weights

WORLD_MODEL_FREQ = 5
LIVE_FRAMES = 5


class WorldModelRunner:
  def __init__(self, directory: Path, live_frames=LIVE_FRAMES):
    assert 1 <= live_frames <= 10
    self.live_frames = live_frames
    device = Device[Device.DEFAULT]
    if device.arch not in ('gfx1200', 'gfx1201'):
      raise RuntimeError(f'Worldmodel requires an RDNA4 GPU, got {device.arch}')
    if device.is_usb:
      device.iface.dev_impl.smu.set_clocks(level=None)
    config = json.loads((directory / 'hparams.json').read_text())['model']
    self.model = WorldModel(config, load_weights(directory / 'weights.fp8.safetensors', None, True))
    self.encoder = OnnxRunner(directory / 'encoder' / 'encoder.onnx')
    weights = {n.inputs[1] for n in self.encoder.graph_nodes if n.op == 'MatMul' and n.inputs[1] in self.encoder.const_names}
    for name in weights:
      self.encoder.graph_values[name] = self.encoder.graph_values[name].cast(dtypes.float16)
    self.encoder.onnx_ops = dict(self.encoder.onnx_ops)

    def matmul(a, b):
      return a.cast(dtypes.float16).contiguous().realize().matmul(
        b.cast(dtypes.float16).contiguous().realize(), dtype=dtypes.float32).realize()

    self.encoder.onnx_ops['MatMul'] = matmul
    for value in self.encoder.graph_values.values():
      if isinstance(value, Tensor):
        value.realize()
    sample = Tensor.zeros(1, 6, 128, 256).contiguous().realize()
    self.encoder({'imgs': sample})['latents'].realize()
    del sample

    self.prefix_frames = self.model.frames - live_frames
    self.model.setup_cache(1, self.prefix_frames)
    fidx = np.array([[10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int64)

    def conditions(start, end, timestep):
      return {
        't': Tensor.full((1, end - start), timestep, dtype=dtypes.bfloat16).contiguous().realize(),
        'augments_pos_ref_augment': Tensor.zeros(1, end - start, 3, dtype=dtypes.bfloat16).contiguous().realize(),
        'ref_augment_from_augments_euler': Tensor.zeros(1, end - start, 3, dtype=dtypes.bfloat16).contiguous().realize(),
        'pose_mask': Tensor.ones(1, end - start, dtype=dtypes.int64).contiguous().realize(),
        'fidx': Tensor(fidx[:, start:end].copy()).realize(),
      }

    # Unobserved future and older context slots have fixed noise at t=1.
    rng = np.random.default_rng(0)
    prefix = Tensor(rng.standard_normal((1, self.prefix_frames, 32, 16, 32), dtype=np.float32)).cast(dtypes.bfloat16).realize()
    prefill = TinyJit(partial(self.model, **conditions(0, self.prefix_frames, 1), return_plan=False), prune=True)
    prefill.cnt = 1
    prefill(prefix)
    device.synchronize()
    del prefill, prefix
    gc.collect()
    device.allocator.free_cache()
    self.forward = partial(self.model, start_frame=self.prefix_frames, **conditions(self.prefix_frames, self.model.frames, 0))
    self.history = Tensor.zeros(1, live_frames, 32, 16, 32, dtype=dtypes.bfloat16).contiguous().realize()
    self.jit = TinyJit(self._run, prune=True)
    self.jit.cnt = 1
    self.run(np.zeros((1, 6, 128, 256), dtype=np.uint8))
    self.reset()

  def _run(self, images):
    latent = self.encoder({'imgs': images.float() / 127.5 - 1.0})['latents']
    latent = ((latent - self.model.config['compressor_mean']) / self.model.config['compressor_std']).cast(dtypes.bfloat16)
    self.history.assign(self.history[:, 1:].cat(latent.unsqueeze(1), dim=1)).realize()
    return self.forward(self.history)['plan'].float().realize()

  def reset(self):
    self.history.assign(0).realize()

  def run(self, images):
    plan = self.jit(Tensor(images).realize()).numpy()
    if not np.isfinite(plan).all():
      raise RuntimeError('Worldmodel plan is not finite')
    return plan


def prepare_image(buffer, width, height, transform):
  stride, y_height, uv_height, _ = get_nv12_info(width, height)
  data = np.frombuffer(buffer.data, dtype=np.uint8)
  y = data[:stride * y_height].reshape(y_height, stride)
  uv = data[stride * y_height:stride * (y_height + uv_height)].reshape(uv_height, stride)
  yy, xx = np.indices((128, 256), dtype=np.float64)
  warp = transform @ np.diag([2., 2., 1.])
  denominator = warp[2, 0] * xx + warp[2, 1] * yy + warp[2, 2]
  sx = np.rint((warp[0, 0] * xx + warp[0, 1] * yy + warp[0, 2]) / denominator).astype(np.int32).clip(0, width - 1)
  sy = np.rint((warp[1, 0] * xx + warp[1, 1] * yy + warp[1, 2]) / denominator).astype(np.int32).clip(0, height - 1)
  # BT.601 limited-range NV12 conversion, after gathering the nearest source pixels.
  luma = np.maximum(y[sy, sx].astype(np.int32) - 16, 0) * 1220542
  u = uv[sy // 2, (sx // 2) * 2].astype(np.int32) - 128
  v = uv[sy // 2, (sx // 2) * 2 + 1].astype(np.int32) - 128
  rgb = np.stack((luma + 1673527 * v, luma - 409993 * u - 852492 * v, luma + 2116026 * u))
  return ((rgb + (1 << 19)) >> 20).clip(0, 255).astype(np.uint8)


def main():
  directory = Path(os.environ['WORLDMODEL_DIR'])
  with Context(DEV='USB+AMD:LLVM'):
    runner = WorldModelRunner(directory)
    pm = messaging.PubMaster(['worldModelPlan'])
    sm = messaging.SubMaster(['deviceState', 'narrowRoadCameraState', 'extrinsicsCalibration'])
    cameras = [VisionIpcClient('camerad', stream, True) for stream in
               (VisionStreamType.VISION_STREAM_NARROW_ROAD, VisionStreamType.VISION_STREAM_WIDE_ROAD)]
    for camera in cameras:
      while not camera.connect(False):
        time.sleep(.1)
    last_timestamp, history_frames = 0, 0
    cloudlog.info('Worldmodel planner ready; target %d FPS', WORLD_MODEL_FREQ)
    while True:
      narrow = cameras[0].recv()
      if narrow is None:
        continue
      timestamp = cameras[0].timestamp_eof
      if timestamp - last_timestamp < 1e9 / WORLD_MODEL_FREQ:
        continue
      wide = cameras[1].recv()
      while wide is not None and cameras[1].timestamp_sof + 10_000_000 < cameras[0].timestamp_sof:
        wide = cameras[1].recv()
      if wide is None or abs(cameras[0].timestamp_sof - cameras[1].timestamp_sof) > 10_000_000:
        continue
      sm.update(0)
      if not all(sm.seen.values()):
        continue
      if timestamp - last_timestamp > 2e9 / WORLD_MODEL_FREQ:
        runner.reset()
        history_frames = 0
      last_timestamp = timestamp
      rpy = np.array(sm['extrinsicsCalibration'].rpyCalib, dtype=np.float32)
      dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['narrowRoadCameraState'].sensor))]
      transforms = (get_warp_matrix(rpy, dc.narrow_road.intrinsics), get_warp_matrix(rpy, dc.wide_road.intrinsics, True))
      start = time.monotonic()
      images = np.concatenate([prepare_image(buf, camera.width, camera.height, tfm)
                               for buf, camera, tfm in zip((narrow, wide), cameras, transforms, strict=True)], axis=0)[None]
      plan = runner.run(images)
      history_frames += 1
      msg = messaging.new_message('worldModelPlan')
      msg.valid = history_frames >= LIVE_FRAMES
      msg.worldModelPlan.frameId = cameras[0].frame_id
      msg.worldModelPlan.timestampEof = timestamp
      msg.worldModelPlan.modelExecutionTime = time.monotonic() - start
      msg.worldModelPlan.plan = plan.ravel().tolist()
      pm.send('worldModelPlan', msg)


if __name__ == '__main__':
  main()
