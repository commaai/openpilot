import time
from tinygrad.tensor import Tensor, Device

MODEL_WIDTH = 512
MODEL_HEIGHT = 256
MODEL_FRAME_SIZE = MODEL_WIDTH * MODEL_HEIGHT * 3 // 2
IMG_INPUT_SHAPE = (1, 12, 128, 256)

def tensor_arange(end): return Tensor([float(i) for i in range(end)])
def tensor_round(tensor:Tensor): return (tensor + 0.5).floor()

h_src, w_src = 1208, 1928
h_dst, w_dst = MODEL_HEIGHT, MODEL_WIDTH
x = tensor_arange(w_dst).reshape(1, w_dst).expand(h_dst, w_dst)
y = tensor_arange(h_dst).reshape(h_dst, 1).expand(h_dst, w_dst)
ones = Tensor.ones_like(x)
dst_coords = x.reshape((1,-1)).cat(y.reshape((1,-1))).cat(ones.reshape((1,-1)))

def warp_perspective_tinygrad(src:Tensor, M_inv:Tensor) -> Tensor:
  src_coords = M_inv @ dst_coords
  src_coords = src_coords / src_coords[2:3, :]

  x_src = src_coords[0].reshape(h_dst, w_dst)
  y_src = src_coords[1].reshape(h_dst, w_dst)

  x_nearest = tensor_round(x_src).clip(0, w_src - 1).cast('int')
  y_nearest = tensor_round(y_src).clip(0, h_src - 1).cast('int')

  # TODO: make 2d indexing fast
  idx = y_nearest*src.shape[1] + x_nearest
  dst = src.flatten()[idx]
  return dst.reshape(h_dst, w_dst)

if __name__ == "__main__":
  from tinygrad.engine.jit import TinyJit
  update_img_jit = TinyJit(warp_perspective_tinygrad, prune=True)

  step_times = []
  for _ in range(10):
    # regenerate inputs
    inputs = [Tensor.randn(1928,1208), Tensor.randn(3,3)]
    Tensor.realize(*inputs)
    Device.default.synchronize()

    # do the warp
    st = time.perf_counter()
    out = update_img_jit(*inputs)
    mt = time.perf_counter()
    val = out.contiguous().realize()
    Device.default.synchronize()
    et = time.perf_counter()

    # measure the time
    step_times.append((et-st)*1e3)
    print(f"enqueue {(mt-st)*1e3:6.2f} ms -- total run {step_times[-1]:6.2f} ms")
