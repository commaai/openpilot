#!/usr/bin/env python3
import time
import pickle
import numpy as np
from pathlib import Path
from tinygrad.tensor import Tensor
from tinygrad.helpers import Context
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit

from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.common.transformations.model import MEDMODEL_INPUT_SIZE, DM_INPUT_SIZE
from openpilot.common.transformations.camera import _ar_ox_fisheye, _os_fisheye

MODELS_DIR = Path(__file__).parent / 'models'

CAMERA_CONFIGS = [
  (_ar_ox_fisheye.width, _ar_ox_fisheye.height),  # tici: 1928x1208
  (_os_fisheye.width, _os_fisheye.height),        # mici: 1344x760
]

UV_SCALE_MATRIX = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]], dtype=np.float32)
UV_SCALE_MATRIX_INV = np.linalg.inv(UV_SCALE_MATRIX)

IMG_BUFFER_SHAPE = (30, MEDMODEL_INPUT_SIZE[1] // 2, MEDMODEL_INPUT_SIZE[0] // 2)


def warp_pkl_path(w, h):
  return MODELS_DIR / f'warp_{w}x{h}_tinygrad.pkl'


def dm_warp_pkl_path(w, h):
  return MODELS_DIR / f'dm_warp_{w}x{h}_tinygrad.pkl'


def warp_perspective_tinygrad(src_flat, M_inv, dst_shape, src_shape, stride_pad, ratio):
  w_dst, h_dst = dst_shape
  h_src, w_src = src_shape

  x = Tensor.arange(w_dst).reshape(1, w_dst).expand(h_dst, w_dst)
  y = Tensor.arange(h_dst).reshape(h_dst, 1).expand(h_dst, w_dst)
  ones = Tensor.ones_like(x)
  dst_coords = x.reshape(1, -1).cat(y.reshape(1, -1)).cat(ones.reshape(1, -1))

  src_coords = M_inv @ dst_coords
  src_coords = src_coords / src_coords[2:3, :]

  x_nn_clipped = Tensor.round(src_coords[0]).clip(0, w_src - 1).cast('int')
  y_nn_clipped = Tensor.round(src_coords[1]).clip(0, h_src - 1).cast('int')
  idx = y_nn_clipped * w_src + (y_nn_clipped * ratio).cast('int') * stride_pad + x_nn_clipped

  sampled = src_flat[idx]
  return sampled


def frames_to_tensor(frames, model_w, model_h):
  H = (frames.shape[0] * 2) // 3
  W = frames.shape[1]
  in_img1 = Tensor.cat(frames[0:H:2, 0::2],
                       frames[1:H:2, 0::2],
                       frames[0:H:2, 1::2],
                       frames[1:H:2, 1::2],
                       frames[H:H+H//4].reshape((H//2, W//2)),
                       frames[H+H//4:H+H//2].reshape((H//2, W//2)), dim=0).reshape((6, H//2, W//2))
  return in_img1


def make_frame_prepare(cam_w, cam_h, model_w, model_h):
  stride, y_height, _, _ = get_nv12_info(cam_w, cam_h)
  uv_offset = stride * y_height
  stride_pad = stride - cam_w

  def frame_prepare_tinygrad(input_frame, M_inv):
    tg_scale = Tensor(UV_SCALE_MATRIX)
    M_inv_uv = tg_scale @ M_inv @ Tensor(UV_SCALE_MATRIX_INV)
    with Context(SPLIT_REDUCEOP=0):
      y = warp_perspective_tinygrad(input_frame[:cam_h*stride],
                                    M_inv, (model_w, model_h),
                                    (cam_h, cam_w), stride_pad, 1).realize()
      u = warp_perspective_tinygrad(input_frame[uv_offset:uv_offset + (cam_h//4)*stride],
                                    M_inv_uv, (model_w//2, model_h//2),
                                    (cam_h//2, cam_w//2), stride_pad, 0.5).realize()
      v = warp_perspective_tinygrad(input_frame[uv_offset + (cam_h//4)*stride:uv_offset + (cam_h//2)*stride],
                                    M_inv_uv, (model_w//2, model_h//2),
                                    (cam_h//2, cam_w//2), stride_pad, 0.5).realize()
    yuv = y.cat(u).cat(v).reshape((model_h * 3 // 2, model_w))
    tensor = frames_to_tensor(yuv, model_w, model_h)
    return tensor
  return frame_prepare_tinygrad


def make_update_img_input(frame_prepare, model_w, model_h):
  def update_img_input_tinygrad(tensor, frame, M_inv):
    M_inv = M_inv.to(Device.DEFAULT)
    new_img = frame_prepare(frame, M_inv)
    full_buffer = tensor[6:].cat(new_img, dim=0).contiguous()
    return full_buffer, Tensor.cat(full_buffer[:6], full_buffer[-6:], dim=0).contiguous().reshape(1, 12, model_h//2, model_w//2)
  return update_img_input_tinygrad


def make_update_both_imgs(frame_prepare, model_w, model_h):
  update_img = make_update_img_input(frame_prepare, model_w, model_h)

  def update_both_imgs_tinygrad(calib_img_buffer, new_img, M_inv,
                                calib_big_img_buffer, new_big_img, M_inv_big):
    calib_img_buffer, calib_img_pair = update_img(calib_img_buffer, new_img, M_inv)
    calib_big_img_buffer, calib_big_img_pair = update_img(calib_big_img_buffer, new_big_img, M_inv_big)
    return calib_img_buffer, calib_img_pair, calib_big_img_buffer, calib_big_img_pair
  return update_both_imgs_tinygrad


def make_warp_dm(cam_w, cam_h, dm_w, dm_h):
  stride, y_height, _, _ = get_nv12_info(cam_w, cam_h)
  stride_pad = stride - cam_w

  def warp_dm(input_frame, M_inv):
    M_inv = M_inv.to(Device.DEFAULT)
    with Context(SPLIT_REDUCEOP=0):
      result = warp_perspective_tinygrad(input_frame[:cam_h*stride], M_inv, (dm_w, dm_h), (cam_h, cam_w), stride_pad, 1).reshape(-1, dm_h * dm_w)
    return result
  return warp_dm


# ---- numpy reference implementations ----

def warp_perspective_numpy(src, M_inv, dst_shape, src_shape, stride_pad, ratio):
  w_dst, h_dst = dst_shape
  h_src, w_src = src_shape
  xs, ys = np.meshgrid(np.arange(w_dst), np.arange(h_dst))

  ones = np.ones_like(xs)
  dst_hom = np.stack([xs, ys, ones], axis=0).reshape(3, -1)

  src_hom = M_inv @ dst_hom
  src_hom /= src_hom[2:3, :]

  src_x = np.clip(np.round(src_hom[0, :]).astype(int), 0, w_src - 1)
  src_y = np.clip(np.round(src_hom[1, :]).astype(int), 0, h_src - 1)
  idx = src_y * w_src + (src_y * ratio).astype(np.int32) * stride_pad + src_x
  return src[idx]


def frames_to_tensor_numpy(frames):
  H = (frames.shape[0] * 2) // 3
  W = frames.shape[1]
  p1 = frames[0:H:2, 0::2]
  p2 = frames[1:H:2, 0::2]
  p3 = frames[0:H:2, 1::2]
  p4 = frames[1:H:2, 1::2]
  p5 = frames[H:H+H//4].reshape((H//2, W//2))
  p6 = frames[H+H//4:H+H//2].reshape((H//2, W//2))
  return np.concatenate([p1, p2, p3, p4, p5, p6], axis=0).reshape((6, H//2, W//2))


def make_frame_prepare_numpy(cam_w, cam_h, model_w, model_h):
  stride, y_height, _, _ = get_nv12_info(cam_w, cam_h)
  uv_offset = stride * y_height
  stride_pad = stride - cam_w

  def frame_prepare_numpy(input_frame, M_inv):
    M_inv_uv = UV_SCALE_MATRIX @ M_inv @ UV_SCALE_MATRIX_INV
    y = warp_perspective_numpy(input_frame[:cam_h*stride],
                               M_inv, (model_w, model_h), (cam_h, cam_w), stride_pad, 1)
    u = warp_perspective_numpy(input_frame[uv_offset:uv_offset + (cam_h//4)*stride],
                               M_inv_uv, (model_w//2, model_h//2), (cam_h//2, cam_w//2), stride_pad, 0.5)
    v = warp_perspective_numpy(input_frame[uv_offset + (cam_h//4)*stride:uv_offset + (cam_h//2)*stride],
                               M_inv_uv, (model_w//2, model_h//2), (cam_h//2, cam_w//2), stride_pad, 0.5)
    yuv = np.concatenate([y, u, v]).reshape(model_h * 3 // 2, model_w)
    return frames_to_tensor_numpy(yuv)
  return frame_prepare_numpy


def make_update_img_input_numpy(frame_prepare, model_w, model_h):
  def update_img_input_numpy(tensor, frame, M_inv):
    tensor[:-6] = tensor[6:]
    tensor[-6:] = frame_prepare(frame, M_inv)
    return tensor, np.concatenate([tensor[:6], tensor[-6:]], axis=0).reshape((1, 12, model_h//2, model_w//2))
  return update_img_input_numpy


def make_update_both_imgs_numpy(frame_prepare, model_w, model_h):
  update_img = make_update_img_input_numpy(frame_prepare, model_w, model_h)

  def update_both_imgs_numpy(calib_img_buffer, new_img, M_inv,
                             calib_big_img_buffer, new_big_img, M_inv_big):
    calib_img_buffer, calib_img_pair = update_img(calib_img_buffer, new_img, M_inv)
    calib_big_img_buffer, calib_big_img_pair = update_img(calib_big_img_buffer, new_big_img, M_inv_big)
    return calib_img_buffer, calib_img_pair, calib_big_img_buffer, calib_big_img_pair
  return update_both_imgs_numpy


def compile_modeld_warp(cam_w, cam_h):
  model_w, model_h = MEDMODEL_INPUT_SIZE
  _, _, _, yuv_size = get_nv12_info(cam_w, cam_h)

  print(f"Compiling modeld warp for {cam_w}x{cam_h}...")

  frame_prepare = make_frame_prepare(cam_w, cam_h, model_w, model_h)
  update_both_imgs = make_update_both_imgs(frame_prepare, model_w, model_h)
  update_img_jit = TinyJit(update_both_imgs, prune=True)

  frame_prepare_np = make_frame_prepare_numpy(cam_w, cam_h, model_w, model_h)
  update_both_imgs_np = make_update_both_imgs_numpy(frame_prepare_np, model_w, model_h)

  full_buffer = Tensor.zeros(IMG_BUFFER_SHAPE, dtype='uint8').contiguous().realize()
  big_full_buffer = Tensor.zeros(IMG_BUFFER_SHAPE, dtype='uint8').contiguous().realize()
  full_buffer_np = np.zeros(IMG_BUFFER_SHAPE, dtype=np.uint8)
  big_full_buffer_np = np.zeros(IMG_BUFFER_SHAPE, dtype=np.uint8)

  for i in range(10):
    new_frame_np = (32 * np.random.randn(yuv_size).astype(np.float32) + 128).clip(0, 255).astype(np.uint8)
    img_inputs = [full_buffer,
                  Tensor.from_blob(new_frame_np.ctypes.data, (yuv_size,), dtype='uint8').realize(),
                  Tensor(Tensor.randn(3, 3).mul(8).realize().numpy(), device='NPY')]
    new_big_frame_np = (32 * np.random.randn(yuv_size).astype(np.float32) + 128).clip(0, 255).astype(np.uint8)
    big_img_inputs = [big_full_buffer,
                      Tensor.from_blob(new_big_frame_np.ctypes.data, (yuv_size,), dtype='uint8').realize(),
                      Tensor(Tensor.randn(3, 3).mul(8).realize().numpy(), device='NPY')]
    inputs = img_inputs + big_img_inputs
    Device.default.synchronize()

    inputs_np = [x.numpy() for x in inputs]
    inputs_np[0] = full_buffer_np
    inputs_np[3] = big_full_buffer_np

    st = time.perf_counter()
    out = update_img_jit(*inputs)
    full_buffer = out[0].contiguous().realize().clone()
    big_full_buffer = out[2].contiguous().realize().clone()
    mt = time.perf_counter()
    Device.default.synchronize()
    et = time.perf_counter()
    print(f"  [{i+1}/10] enqueue {(mt-st)*1e3:6.2f} ms -- total {(et-st)*1e3:6.2f} ms")

    out_np = update_both_imgs_np(*inputs_np)
    full_buffer_np = out_np[0]
    big_full_buffer_np = out_np[2]

    for a, b in zip(out_np, (x.numpy() for x in out), strict=True):
      mismatch = np.abs(a - b) > 0
      mismatch_percent = sum(mismatch.flatten()) / len(mismatch.flatten()) * 100
      assert mismatch_percent < 1e-2, f"mismatch {mismatch_percent:.4f}% exceeds tolerance"

  pkl_path = warp_pkl_path(cam_w, cam_h)
  with open(pkl_path, "wb") as f:
    pickle.dump(update_img_jit, f)
  print(f"  Saved to {pkl_path}")

  jit = pickle.load(open(pkl_path, "rb"))
  jit(*inputs)


def compile_dm_warp(cam_w, cam_h):
  dm_w, dm_h = DM_INPUT_SIZE
  _, _, _, yuv_size = get_nv12_info(cam_w, cam_h)

  print(f"Compiling DM warp for {cam_w}x{cam_h}...")

  warp_dm = make_warp_dm(cam_w, cam_h, dm_w, dm_h)
  warp_dm_jit = TinyJit(warp_dm, prune=True)

  for i in range(10):
    inputs = [Tensor.from_blob((32 * Tensor.randn(yuv_size,) + 128).cast(dtype='uint8').realize().numpy().ctypes.data, (yuv_size,), dtype='uint8'),
              Tensor(Tensor.randn(3, 3).mul(8).realize().numpy(), device='NPY')]
    Device.default.synchronize()
    st = time.perf_counter()
    out = warp_dm_jit(*inputs)
    mt = time.perf_counter()
    Device.default.synchronize()
    et = time.perf_counter()
    print(f"  [{i+1}/10] enqueue {(mt-st)*1e3:6.2f} ms -- total {(et-st)*1e3:6.2f} ms")

  pkl_path = dm_warp_pkl_path(cam_w, cam_h)
  with open(pkl_path, "wb") as f:
    pickle.dump(warp_dm_jit, f)
  print(f"  Saved to {pkl_path}")


def run_and_save_pickle():
  for cam_w, cam_h in CAMERA_CONFIGS:
    compile_modeld_warp(cam_w, cam_h)
    compile_dm_warp(cam_w, cam_h)


if __name__ == "__main__":
  run_and_save_pickle()
