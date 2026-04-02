#!/usr/bin/env python3
import time
import pickle
import numpy as np
from pathlib import Path
from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.onnx import OnnxRunner

from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.common.transformations.model import MEDMODEL_INPUT_SIZE
from openpilot.common.transformations.camera import _ar_ox_fisheye, _os_fisheye
from openpilot.selfdrive.modeld.compile_warp import make_frame_prepare
from openpilot.selfdrive.modeld.constants import ModelConstants

MODELS_DIR = Path(__file__).parent / 'models'

CAMERA_CONFIGS = [
  (_ar_ox_fisheye.width, _ar_ox_fisheye.height),  # tici: 1928x1208
  (_os_fisheye.width, _os_fisheye.height),         # mici: 1344x760
]

IMG_QUEUE_SHAPE = (6*(ModelConstants.MODEL_RUN_FREQ//ModelConstants.MODEL_CONTEXT_FREQ + 1), 128, 256)


def modeld_pkl_path(w, h):
  return MODELS_DIR / f'modeld_{w}x{h}_tinygrad.pkl'


def _make_run_modeld(vision_runner, on_policy_runner, off_policy_runner,
                     cam_w, cam_h, vision_features_slice, frame_skip):
  model_w, model_h = MEDMODEL_INPUT_SIZE
  frame_warp = make_frame_prepare(cam_w, cam_h, model_w, model_h)

  def update_bufs(frame, img_q, tfm):
    new = frame_warp(frame, tfm.to(Device.DEFAULT))
    img_q.assign(img_q[6:].cat(new, dim=0).contiguous())
    return Tensor.cat(img_q[:6], img_q[-6:], dim=0).reshape(1, 12, model_h//2, model_w//2).contiguous()

  def run_modeld(img_q, big_img_q, frame, big_frame, tfm, big_tfm,
                 feat_q, des_buf, traf):
    # warp
    img = update_bufs(frame, img_q, tfm)
    big_img = update_bufs(big_frame, big_img_q, big_tfm)

    # vision
    vision_out = next(iter(vision_runner({'img': img, 'big_img': big_img}).values())).cast('float32')

    # features queue
    des_buf, traf = des_buf.to(Device.DEFAULT), traf.to(Device.DEFAULT)
    feat_q.assign(feat_q[:, 1:].cat(vision_out[:, vision_features_slice].reshape(1, 1, -1), dim=1).contiguous())
    feat_buf = feat_q[:, frame_skip - 1::frame_skip]

    policy_inputs = {'features_buffer': feat_buf, 'desire_pulse': des_buf, 'traffic_convention': traf}

    # on-policy + off-policy
    on_out = next(iter(on_policy_runner(policy_inputs).values())).cast('float32')
    off_out = next(iter(off_policy_runner(policy_inputs).values())).cast('float32')

    return vision_out[:, :vision_features_slice.start].contiguous(), on_out, off_out
  return run_modeld


class CompileState:
  """Holds tensors needed during compilation and at runtime for the combined JIT."""
  def __init__(self):
    with open(MODELS_DIR / 'driving_vision_metadata.pkl', 'rb') as f:
      vision_metadata = pickle.load(f)
      self.vision_output_slices = vision_metadata['output_slices']
      self.vision_features_slice = self.vision_output_slices.pop('hidden_state')

    with open(MODELS_DIR / 'driving_on_policy_metadata.pkl', 'rb') as f:
      on_policy_metadata = pickle.load(f)
      self.on_policy_input_shapes = on_policy_metadata['input_shapes']
      self.on_policy_output_slices = on_policy_metadata['output_slices']

    with open(MODELS_DIR / 'driving_off_policy_metadata.pkl', 'rb') as f:
      off_policy_metadata = pickle.load(f)
      self.off_policy_output_slices = off_policy_metadata['output_slices']

    self.frame_skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ
    fb = self.on_policy_input_shapes['features_buffer']
    self.features_queue = Tensor.zeros(fb[0], fb[1] * self.frame_skip, fb[2]).contiguous().realize()

    self.img_queues = {
      'img': Tensor.zeros(IMG_QUEUE_SHAPE, dtype='uint8').contiguous().realize(),
      'big_img': Tensor.zeros(IMG_QUEUE_SHAPE, dtype='uint8').contiguous().realize(),
    }
    self.transforms_np = {k: np.zeros((3, 3), dtype=np.float32) for k in self.img_queues}
    self.transforms = {k: Tensor(v, device='NPY').realize() for k, v in self.transforms_np.items()}

    self.desire_np = np.zeros(self.on_policy_input_shapes['desire_pulse'], dtype=np.float32)
    self.desire_tensor = Tensor(self.desire_np, device='NPY').realize()
    self.traffic_np = np.zeros(self.on_policy_input_shapes['traffic_convention'], dtype=np.float32)
    self.traffic_tensor = Tensor(self.traffic_np, device='NPY').realize()


def compile_modeld(cam_w, cam_h):
  print(f"Compiling combined modeld JIT for {cam_w}x{cam_h}...")

  vision_runner = OnnxRunner(MODELS_DIR / 'driving_vision.onnx')
  on_policy_runner = OnnxRunner(MODELS_DIR / 'driving_on_policy.onnx')
  off_policy_runner = OnnxRunner(MODELS_DIR / 'driving_off_policy.onnx')

  state = CompileState()
  _run = _make_run_modeld(vision_runner, on_policy_runner, off_policy_runner,
                          cam_w, cam_h, state.vision_features_slice, state.frame_skip)
  run_modeld = TinyJit(_run, prune=True)

  _, _, _, yuv_size = get_nv12_info(cam_w, cam_h)

  for i in range(10):
    frame = Tensor(np.random.randint(0, 255, yuv_size, dtype=np.uint8)).realize()
    big_frame = Tensor(np.random.randint(0, 255, yuv_size, dtype=np.uint8)).realize()

    st = time.perf_counter()
    outs = run_modeld(
      state.img_queues['img'], state.img_queues['big_img'],
      frame, big_frame,
      state.transforms['img'], state.transforms['big_img'],
      state.features_queue,
      state.desire_tensor, state.traffic_tensor,
    )
    t_enqueue = time.perf_counter()
    Device.default.synchronize()
    t_sync = time.perf_counter()
    vision_out = outs[0].uop.base.buffer.numpy().flatten()
    on_policy_out = outs[1].uop.base.buffer.numpy().flatten()
    off_policy_out = outs[2].uop.base.buffer.numpy().flatten()
    t_copy = time.perf_counter()
    print(f"  [{i+1}/10] enqueue {(t_enqueue-st)*1e3:.1f} ms  sync {(t_sync-st)*1e3:.1f} ms  copy out {(t_copy-t_sync)*1e3:.1f} ms  total {(t_copy-st)*1e3:.1f} ms")

  pkl_path = modeld_pkl_path(cam_w, cam_h)
  with open(pkl_path, "wb") as f:
    pickle.dump(run_modeld, f)
  print(f"  Saved to {pkl_path}")

  # validate pickle roundtrip
  with open(pkl_path, "rb") as f:
    loaded = pickle.load(f)
  loaded(
    state.img_queues['img'], state.img_queues['big_img'],
    frame, big_frame,
    state.transforms['img'], state.transforms['big_img'],
    state.features_queue,
    state.desire_tensor, state.traffic_tensor,
  )
  Device.default.synchronize()
  print("  Pickle roundtrip validated")

  return run_modeld


if __name__ == "__main__":
  for cam_w, cam_h in CAMERA_CONFIGS:
    compile_modeld(cam_w, cam_h)
