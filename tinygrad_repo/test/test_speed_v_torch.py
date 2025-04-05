import os
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import unittest
import torch
torch.set_num_threads(1)
import time
import numpy as np
np.set_printoptions(linewidth=160)
from tinygrad import Tensor, Device, GlobalCounters, TinyJit
from tinygrad.nn import Conv2d
from tinygrad.helpers import colorize_float, getenv, CI

IN_CHANS = [int(x) for x in getenv("IN_CHANS", "4,16,64").split(",")]

torch_dt = torch.float16 if getenv("HALF", 0) else torch.float32
torch_device = torch.device('mps' if getenv("MPS", 0) else ('cuda' if getenv("TORCHCUDA", 0) else 'cpu'))
if str(torch_device) == "mps":
  import torch.mps
  def sync(): torch.mps.synchronize()
elif str(torch_device) == "cuda":
  import torch.cuda
  def sync(): torch.cuda.synchronize()
else:
  def sync(): pass

save_ops, save_mem = 0, 0
CNT = getenv("CNT", 8)
def helper_test_speed(f1, *args):
  global save_ops, save_mem
  ets = []
  ret = None
  cache_defeat = np.zeros((2048,2048))
  for i in range(CNT):
    del ret

    # operation cache defeats
    args = [(x+1).realize() if isinstance(x, Tensor) else (None if x is None else (x+1)) for x in args]
    args = [(x-1).realize() if isinstance(x, Tensor) else (None if x is None else (x-1)) for x in args]

    # force syncing
    [x.numpy() if isinstance(x, Tensor) or str(torch_device) == "cpu" else x.cpu().numpy() for x in args if x is not None]

    # clear 32MB global memory cache (CPU and global memory only)
    cache_defeat += 1

    # manual pre sync
    if isinstance(args[0], Tensor): Device[args[0].device].synchronize()
    else: sync()

    GlobalCounters.global_ops = 0
    GlobalCounters.global_mem = 0
    st = time.perf_counter()
    ret = f1(*args)
    if isinstance(ret, Tensor): Device[ret.device].synchronize()
    else: sync()
    et = (time.perf_counter() - st) * 1000
    if i >= 1: ets.append(et)
    if GlobalCounters.global_ops:
      save_ops, save_mem = GlobalCounters.global_ops, GlobalCounters.global_mem
  return ret.numpy() if isinstance(ret, Tensor) else ret.cpu().numpy(), np.min(ets)

def helper_test_generic_square(name, N, f1, f2, onearg=False):
  torch.manual_seed(0)
  torch_a = (torch.rand(N, N, dtype=torch_dt) - 0.5).to(torch_device)
  torch_b = (torch.rand(N, N, dtype=torch_dt) - 0.5).to(torch_device) if not onearg else None

  tiny_a = Tensor(torch_a.cpu().numpy())
  tiny_b = Tensor(torch_b.cpu().numpy()) if not onearg else None

  helper_test_generic(f"{name:30s} {N:5d}x{N:5d}", f1, (torch_a, torch_b), TinyJit(f2), (tiny_a, tiny_b))

def helper_test_matvec(name, N, M):
  torch.manual_seed(0)
  torch_a = (torch.rand(N, dtype=torch_dt) - 0.5).to(torch_device)
  torch_b = (torch.rand(N, M, dtype=torch_dt) - 0.5).to(torch_device)

  tiny_a = Tensor(torch_a.cpu().numpy())
  tiny_b = Tensor(torch_b.cpu().numpy())

  helper_test_generic(f"{name:30s} {N:5d}x{M:5d}", lambda a,b: a@b, (torch_a, torch_b), TinyJit(lambda a,b:a@b), (tiny_a, tiny_b))

prefix = None
def helper_test_generic(name, f1, f1_args, f2, f2_args):
  global prefix
  with torch.no_grad():
    val_torch, et_torch = helper_test_speed(f1, *f1_args)
  val_tinygrad, et_tinygrad = helper_test_speed(f2, *f2_args)

  desc = "faster" if et_torch > et_tinygrad else "slower"
  flops = save_ops*1e-6
  mem = save_mem*1e-6
  print(("\r" if not CI else "")+f"{name:42s} {et_torch:7.2f} ms ({flops/et_torch:9.2f} GFLOPS {mem/et_torch:7.2f} GB/s) in torch, {et_tinygrad:7.2f} ms ({flops/et_tinygrad:9.2f} GFLOPS {mem/et_tinygrad:7.2f} GB/s) in tinygrad, {colorize_float(et_tinygrad/et_torch)} {desc} {flops:10.2f} MOPS {mem:8.2f} MB")  # noqa: E501
  atol, rtol = (1e-2, 1e-2) if torch_dt == torch.float16 else (1e-3, 1e-3)
  np.testing.assert_allclose(val_tinygrad, val_torch, atol=atol, rtol=rtol)

def helper_test_conv(bs, in_chans, out_chans, kernel_size, img_size_y, img_size_x):
  torch.manual_seed(0)
  torch_dat = torch.rand(bs, in_chans, img_size_y, img_size_x, dtype=torch_dt).to(torch_device)
  torch_conv = torch.nn.Conv2d(in_chans, out_chans, kernel_size, bias=None, dtype=torch_dt).to(torch_device)

  tiny_dat = Tensor(torch_dat.cpu().numpy())
  tiny_conv = Conv2d(in_chans, out_chans, kernel_size, bias=None)
  tiny_conv.weight = Tensor(torch_conv.weight.detach().cpu().numpy())

  def f1(torch_dat): return torch_conv(torch_dat)
  def f2(tiny_dat): return tiny_conv(tiny_dat).realize()
  helper_test_generic(f"conv bs:{bs:3d} chans:{in_chans:3d} -> {out_chans:3d} k:{kernel_size}", f1, (torch_dat,), TinyJit(f2), (tiny_dat,))

@unittest.skipIf(getenv("BIG") == 0, "no big tests")
@unittest.skipIf(getenv("MOCKGPU"), "no MOCKGPUs")
class TestBigSpeed(unittest.TestCase):
  def test_add(self):
    def f(a, b): return a+b
    helper_test_generic_square('add', 8192, f, f)
  def test_exp(self):
    def f(a, b): return a.exp()
    helper_test_generic_square('exp', 8192, f, f, onearg=True)
  def test_gemm_2048(self):
    def f(a, b): return a @ b
    helper_test_generic_square('gemm', 2048, f, f)
  def test_gemm_4096(self):
    def f(a, b): return a @ b
    helper_test_generic_square('gemm', 4096, f, f)
  def test_large_conv_1x1(self): helper_test_conv(bs=32, in_chans=128, out_chans=128, kernel_size=1, img_size_y=128, img_size_x=128)
  def test_large_conv_3x3(self): helper_test_conv(bs=4, in_chans=128, out_chans=128, kernel_size=3, img_size_y=130, img_size_x=130)
  def test_large_conv_5x5(self): helper_test_conv(bs=4, in_chans=128, out_chans=128, kernel_size=5, img_size_y=132, img_size_x=132)
  def test_matvec_4096_16384(self): helper_test_matvec('matvec_4096_16384', 4096, 16384)
  def test_matvec_16384_4096(self): helper_test_matvec('matvec_16384_4096', 16384, 4096)

@unittest.skipIf(getenv("BIG") == 1, "only big tests")
@unittest.skipIf(getenv("MOCKGPU"), "no MOCKGPUs")
class TestSpeed(unittest.TestCase):
  def test_sub(self):
    def f(a, b): return a-b
    helper_test_generic_square('sub', 4096, f, f)

  def test_pow(self):
    def f(a, b): return a.pow(b)
    helper_test_generic_square('pow', 2048, f, f)

  def test_sum(self):
    def f(a, b): return a.sum()
    helper_test_generic_square('sum', 2048, f, f, onearg=True)
    helper_test_generic_square('sum', 4096, f, f, onearg=True)

  def test_partial_sum(self):
    R = 256
    def f(a, b): return a.reshape(int(4096//R), int(4096*R)).sum(axis=1)
    helper_test_generic_square('partial_sum', 4096, f, f, onearg=True)

  @unittest.skip("not really used in models")
  def test_cumsum(self):
    def f0(a, b): return a.cumsum(axis=0)
    def f1(a, b): return a.cumsum(axis=1)
    helper_test_generic_square('cumsum_0', 256, f0, f0, onearg=True)
    helper_test_generic_square('cumsum_1', 256, f1, f1, onearg=True)

  def test_cat(self):
    helper_test_generic_square('cat_0', 256, lambda x,y: torch.cat((x,y),dim=0), lambda x,y: x.cat(y,dim=0))
    helper_test_generic_square('cat_1', 256, lambda x,y: torch.cat((x,y),dim=1), lambda x,y: x.cat(y,dim=1))

  def test_array_packing(self):
    N = 2048
    def f(a, b): return a.reshape(N, N // 32, 32).permute(1,0,2).contiguous()
    helper_test_generic_square('array_packing', N, f, f, onearg=True)

  def test_permute(self):
    for N in [1024, 4096]:
      # this is a 64MB tensor, M1 L1 cache is 128kB
      # to fit easily in L1, rotations should be 128x128 chunks. 128x128 is also the AMX size
      def f(a, b): return a.permute(1,0).contiguous()
      helper_test_generic_square('permute', N, f, f, onearg=True)

  def test_double_permute(self):
    N = 64
    torch.manual_seed(0)
    torch_a = (torch.rand(N, N, N, N, dtype=torch_dt) - 0.5).to(torch_device)
    tiny_a = Tensor(torch_a.cpu().numpy())
    def f(a): return a.permute(1,0,3,2).contiguous()
    helper_test_generic(f"double_permute {tiny_a.shape}", f, (torch_a,), TinyJit(lambda a: f(a).realize()), (tiny_a,))

  def test_neg(self):
    def f(a, b): return -a
    helper_test_generic_square('neg', 4096, f, f, onearg=True)

  def test_exp(self):
    def f(a, b): return a.exp()
    helper_test_generic_square('exp', 2048, f, f, onearg=True)

  def test_sqrt(self):
    def f(a, b): return a.sqrt()
    helper_test_generic_square('sqrt', 2048, f, f, onearg=True)

  def test_relu(self):
    def f(a, b): return a.relu()
    helper_test_generic_square('relu', 4096, f, f, onearg=True)

  def test_max(self):
    def f(a, b): return a.max()
    helper_test_generic_square('max', 4096, f, f, onearg=True)

  def test_mul_sum(self):
    def f(a, b): return (a*b).sum()
    helper_test_generic_square('mul_sum', 4096, f, f)

  def test_add_a(self):
    def f(a, b): return a + b
    helper_test_generic_square('add', 1, f, f)

  def test_add_big(self):
    for N in [1024, 4096]:
      def f(a, b): return a + b
      helper_test_generic_square('add', N, f, f)

  def test_add_constant(self):
    def f(a, b): return a+2.0
    helper_test_generic_square('add_constant', 4096, f, f, onearg=True)

  def test_add_sq(self):
    def f(a, b): return a*a + b*b
    helper_test_generic_square('add_sq', 4096, f, f)

  def test_gemm(self):
    def f(a, b): return a @ b
    helper_test_generic_square('gemm', 1024, f, f)

  def test_gemm_small(self):
    def f(a, b): return a @ b
    helper_test_generic_square('gemm', 256, f, f)

  def test_gemm_unrolled(self):
    N = 512
    def f1(a, b): return a@b.T
    def f2(a, b): return (a.reshape(N, 1, N).expand(N, N, N) * b.reshape(1, N, N).expand(N, N, N)).sum(axis=2)
    helper_test_generic_square('gemm_unrolled', N, f1, f2)

  def test_gemm_unrolled_permute_l(self):
    N = 512
    def f1(a, b): return a.T@b.T
    def f2(a, b): return (a.permute(1,0).reshape(N, 1, N).expand(N, N, N) * b.reshape(1, N, N).expand(N, N, N)).sum(axis=2)
    helper_test_generic_square('gemm_unrolled_permute_l', N, f1, f2)

  def test_gemm_unrolled_permute_r(self):
    N = 512
    def f1(a, b): return a@b
    def f2(a, b): return (a.reshape(N, 1, N).expand(N, N, N) * b.permute(1,0).reshape(1, N, N).expand(N, N, N)).sum(axis=2)
    helper_test_generic_square('gemm_unrolled_permute_r', N, f1, f2)

  def test_gemm_unrolled_permute_lr(self):
    N = 512
    def f1(a, b): return a.T@b
    def f2(a, b): return (a.permute(1,0).reshape(N, 1, N).expand(N, N, N) * b.permute(1,0).reshape(1, N, N).expand(N, N, N)).sum(axis=2)
    helper_test_generic_square('gemm_unrolled_permute_lr', N, f1, f2)

  def test_matvec_1024_1024(self): helper_test_matvec('matvec_1024_1024', 1024, 1024)
  def test_matvec_1024_4096(self): helper_test_matvec('matvec_1024_4096', 1024, 4096)
  def test_matvec_4096_1024(self): helper_test_matvec('matvec_4096_1024', 4096, 1024)
  def test_matvec_4096_4096(self): helper_test_matvec('matvec_4096_4096', 4096, 4096)

  def test_openpilot_conv2d(self):
    bs, in_chans, out_chans = 1,12,32
    torch.manual_seed(0)
    torch_dat = torch.rand(bs, 64, 128, 12, dtype=torch_dt).to(torch_device)
    torch_conv = torch.nn.Conv2d(in_chans, out_chans, 3, bias=None, padding=1, dtype=torch_dt).to(torch_device)

    tiny_dat = Tensor(torch_dat.cpu().numpy())
    tiny_conv = Conv2d(in_chans, out_chans, 3, bias=None, padding=1)
    tiny_conv.weight = Tensor(torch_conv.weight.detach().cpu().numpy())

    def f1(torch_dat): return torch_conv(torch_dat.permute(0,3,1,2))
    def f2(tiny_dat): return tiny_conv(tiny_dat.permute(0,3,1,2)).realize()
    helper_test_generic(f"conv bs:{bs:3d} chans:{in_chans:3d} -> {out_chans:3d} k:3", f1, (torch_dat,), TinyJit(f2), (tiny_dat,))

  def test_conv2d(self):
    for bs in [32]:
      for in_chans in IN_CHANS:
        for out_chans in [32]:
          helper_test_conv(bs, in_chans, out_chans, 3, 34, 34)

if __name__ == '__main__':
  unittest.main()
