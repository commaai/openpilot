#!/usr/bin/env python3
import time
import pickle
import numpy as np
from pathlib import Path
from tinygrad.tensor import Tensor
from tinygrad.helpers import Context
from tinygrad.device import Device
from common.transformations.camera import get_nv12_info


WARP_PKL_PATH = Path(__file__).parent / 'models/warp_tinygrad.pkl'
DM_WARP_PKL_PATH = Path(__file__).parent / 'models/dm_warp_tinygrad.pkl'

MODEL_WIDTH = 512
MODEL_HEIGHT = 256
MODEL_FRAME_SIZE = MODEL_WIDTH * MODEL_HEIGHT * 3 // 2
IMG_BUFFER_SHAPE = (30, 128, 256)
W, H = 1928, 1208

YUV_SIZE, STRIDE, UV_OFFSET = get_nv12_info(W, H)

UV_SCALE_MATRIX = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]], dtype=np.float32)
UV_SCALE_MATRIX_INV = np.linalg.inv(UV_SCALE_MATRIX)


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

def frames_to_tensor(frames):
  H = (frames.shape[0]*2)//3
  W = frames.shape[1]
  in_img1 = Tensor.cat(frames[0:H:2, 0::2],
                        frames[1:H:2, 0::2],
                        frames[0:H:2, 1::2],
                        frames[1:H:2, 1::2],
                        frames[H:H+H//4].reshape((H//2,W//2)),
                        frames[H+H//4:H+H//2].reshape((H//2,W//2)), dim=0).reshape((6, H//2, W//2))
  return in_img1

def frame_prepare_tinygrad(input_frame, M_inv):
  tg_scale = Tensor(UV_SCALE_MATRIX)
  M_inv_uv = tg_scale @ M_inv @ Tensor(UV_SCALE_MATRIX_INV)
  with Context(SPLIT_REDUCEOP=0):
    y = warp_perspective_tinygrad(input_frame[:H*STRIDE], M_inv, (MODEL_WIDTH, MODEL_HEIGHT), (H, W), STRIDE - W, 1).realize()
    u = warp_perspective_tinygrad(input_frame[UV_OFFSET:UV_OFFSET + (H//4)*STRIDE], M_inv_uv, (MODEL_WIDTH//2, MODEL_HEIGHT//2), (H//2, W//2), STRIDE - W, 0.5).realize()
    v = warp_perspective_tinygrad(input_frame[UV_OFFSET + (H//4)*STRIDE:UV_OFFSET + (H//2)*STRIDE], M_inv_uv, (MODEL_WIDTH//2, MODEL_HEIGHT//2), (H//2, W//2), STRIDE - W, 0.5).realize()
  yuv = y.cat(u).cat(v).reshape((MODEL_HEIGHT*3//2,MODEL_WIDTH))
  tensor = frames_to_tensor(yuv)
  return tensor

def update_img_input_tinygrad(tensor, frame, M_inv):
  frame = frame.flatten().to(Device.DEFAULT)
  M_inv = M_inv.to(Device.DEFAULT)
  new_img = frame_prepare_tinygrad(frame, M_inv)
  full_buffer = tensor[6:].cat(new_img, dim=0).contiguous()
  return full_buffer, Tensor.cat(full_buffer[:6], full_buffer[-6:], dim=0).contiguous().reshape(1,12,MODEL_HEIGHT//2,MODEL_WIDTH//2)

def update_both_imgs_tinygrad(calib_img_buffer, new_img, M_inv,
                              calib_big_img_buffer, new_big_img, M_inv_big):
  calib_img_buffer, calib_img_pair = update_img_input_tinygrad(calib_img_buffer, new_img, M_inv)
  calib_big_img_buffer, calib_big_img_pair = update_img_input_tinygrad(calib_big_img_buffer, new_big_img, M_inv_big)
  return calib_img_buffer, calib_img_pair, calib_big_img_buffer, calib_big_img_pair

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


def frames_to_tensor_np(frames):
  H = (frames.shape[0]*2)//3
  W = frames.shape[1]
  p1 = frames[0:H:2, 0::2]
  p2 = frames[1:H:2, 0::2]
  p3 = frames[0:H:2, 1::2]
  p4 = frames[1:H:2, 1::2]
  p5 = frames[H:H+H//4].reshape((H//2, W//2))
  p6 = frames[H+H//4:H+H//2].reshape((H//2, W//2))
  return np.concatenate([p1, p2, p3, p4, p5, p6], axis=0)\
           .reshape((6, H//2, W//2))

def frame_prepare_np(input_frame, M_inv):
  M_inv_uv = UV_SCALE_MATRIX @ M_inv @ UV_SCALE_MATRIX_INV
  y  = warp_perspective_numpy(input_frame[:H*STRIDE],
                                 M_inv, (MODEL_WIDTH, MODEL_HEIGHT), (H, W), STRIDE - W, 1)
  u  = warp_perspective_numpy(input_frame[UV_OFFSET:UV_OFFSET + (H//4)*STRIDE],
                                 M_inv_uv, (MODEL_WIDTH//2, MODEL_HEIGHT//2), (H//2, W//2), STRIDE - W, 0.5)
  v  = warp_perspective_numpy(input_frame[UV_OFFSET + (H//4)*STRIDE:UV_OFFSET + (H//2)*STRIDE],
                                 M_inv_uv, (MODEL_WIDTH//2, MODEL_HEIGHT//2), (H//2, W//2), STRIDE - W, 0.5)
  yuv = np.concatenate([y, u, v]).reshape( MODEL_HEIGHT*3//2, MODEL_WIDTH)
  return frames_to_tensor_np(yuv)

def update_img_input_np(tensor, frame, M_inv):
  tensor[:-6]  = tensor[6:]
  tensor[-6:] = frame_prepare_np(frame, M_inv)
  return tensor, np.concatenate([tensor[:6], tensor[-6:]], axis=0).reshape((1,12,MODEL_HEIGHT//2, MODEL_WIDTH//2))

def update_both_imgs_np(calib_img_buffer, new_img, M_inv,
                        calib_big_img_buffer, new_big_img, M_inv_big):
  calib_img_buffer, calib_img_pair = update_img_input_np(calib_img_buffer, new_img, M_inv)
  calib_big_img_buffer, calib_big_img_pair = update_img_input_np(calib_big_img_buffer, new_big_img, M_inv_big)
  return calib_img_buffer, calib_img_pair, calib_big_img_buffer, calib_big_img_pair

def run_and_save_pickle():
  from tinygrad.engine.jit import TinyJit
  from tinygrad.device import Device
  update_img_jit = TinyJit(update_both_imgs_tinygrad, prune=True)

  full_buffer = Tensor.zeros(IMG_BUFFER_SHAPE, dtype='uint8').contiguous().realize()
  big_full_buffer = Tensor.zeros(IMG_BUFFER_SHAPE, dtype='uint8').contiguous().realize()
  full_buffer_np = np.zeros(IMG_BUFFER_SHAPE, dtype=np.uint8)
  big_full_buffer_np = np.zeros(IMG_BUFFER_SHAPE, dtype=np.uint8)

  step_times = []
  for _ in range(10):
    new_frame_np = (32*np.random.randn(YUV_SIZE).astype(np.float32) + 128).clip(0,255).astype(np.uint8)
    new_frame = Tensor.from_blob(new_frame_np.ctypes.data, (YUV_SIZE,), dtype='uint8').realize()
    img_inputs = [full_buffer,
                  Tensor.from_blob(new_frame_np.ctypes.data, (YUV_SIZE,), dtype='uint8').realize(),
                  Tensor(Tensor.randn(3,3).mul(8).realize().numpy(), device='NPY')]
    new_big_frame_np = (32*np.random.randn(YUV_SIZE).astype(np.float32) + 128).clip(0,255).astype(np.uint8)
    big_img_inputs = [big_full_buffer,
                      Tensor.from_blob(new_big_frame_np.ctypes.data, (YUV_SIZE,), dtype='uint8').realize(),
                      Tensor(Tensor.randn(3,3).mul(8).realize().numpy(), device='NPY')]
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
    step_times.append((et-st)*1e3)
    print(f"enqueue {(mt-st)*1e3:6.2f} ms -- total run {step_times[-1]:6.2f} ms")
    out_np = update_both_imgs_np(*inputs_np)
    full_buffer_np = out_np[0]
    big_full_buffer_np = out_np[2]

    for a, b in zip(out_np, (x.numpy() for x in out), strict=True):
      mismatch = np.abs(a - b) > 0
      mismatch_percent = sum(mismatch.flatten()) / len(mismatch.flatten()) * 100
      mismatch_percent_tol = 1e-2
      assert mismatch_percent < mismatch_percent_tol, f"input mismatch percent {mismatch_percent} exceeds tolerance {mismatch_percent_tol}"

  with open(WARP_PKL_PATH, "wb") as f:
    pickle.dump(update_img_jit, f)

  jit = pickle.load(open(WARP_PKL_PATH, "rb"))
  # test function after loading
  jit(*inputs)


  def warp_dm(input_frame, M_inv):
    input_frame = input_frame.to(Device.DEFAULT)
    M_inv = M_inv.to(Device.DEFAULT)
    return warp_perspective_tinygrad(input_frame[:H*STRIDE], M_inv, (1440, 960), (H, W), STRIDE - W, 1).reshape(-1,960*1440)
  warp_dm_jit = TinyJit(warp_dm, prune=True)
  step_times = []
  for _ in range(10):
    inputs = [Tensor.from_blob((32*Tensor.randn(YUV_SIZE,) + 128).cast(dtype='uint8').realize().numpy().ctypes.data, (YUV_SIZE,), dtype='uint8'),
                  Tensor(Tensor.randn(3,3).mul(8).realize().numpy(), device='NPY')]

    Device.default.synchronize()
    st = time.perf_counter()
    out = warp_dm_jit(*inputs)
    mt = time.perf_counter()
    Device.default.synchronize()
    et = time.perf_counter()
    step_times.append((et-st)*1e3)
    print(f"enqueue {(mt-st)*1e3:6.2f} ms -- total run {step_times[-1]:6.2f} ms")

  with open(DM_WARP_PKL_PATH, "wb") as f:
    pickle.dump(warp_dm_jit, f)

if __name__ == "__main__":
    run_and_save_pickle()
