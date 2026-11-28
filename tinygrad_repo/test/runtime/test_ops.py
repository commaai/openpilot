import time, math, unittest, functools, warnings, sys
import numpy as np
import torch
from tinygrad.helpers import getenv, DEBUG, DEV, IMAGE, Context, to_tuple
from tinygrad import Tensor, Device, dtypes
from tinygrad.renderer.nir import NIRRenderer
from test.helpers import TensorTestCase, prepare_test_op

TINY_BACKEND = getenv("TINY_BACKEND")
if TINY_BACKEND:
  import tinygrad.nn.torch # noqa: F401 # pylint: disable=unused-import
  torch.set_default_device("tiny")

warnings.filterwarnings("ignore", message="Non-empty compiler output encountered")

FORWARD_ONLY = getenv("FORWARD_ONLY", 0)
PRINT_TENSORS = getenv("PRINT_TENSORS", 0)
COMPILE_ONLY = Device.DEFAULT == "NULL"
# WGSL lets the compiler assume no inf/nan, mesa NIR does too without float controls: x-x and x*0 fold to 0
NAN_UNSAFE = Device.DEFAULT == "WEBGPU" or isinstance(Device[Device.DEFAULT].renderer, NIRRenderer)

def slow_test(test_func):
  return unittest.skipIf(getenv("SKIP_SLOW_TEST"), "Skipping slow test")(test_func)

def helper_test_op(shps, torch_fxn, tinygrad_fxn=None, atol=1e-6, rtol=1e-3, grad_atol=1e-4, grad_rtol=1e-3,
                   forward_only=False, vals=None, low=-2, high=2):
  if tinygrad_fxn is None: tinygrad_fxn = torch_fxn
  ts, tst = prepare_test_op(low, high, shps, vals, forward_only)

  st = time.monotonic()
  out = torch_fxn(*ts)
  torch_fp = time.monotonic() - st

  # move inputs to a different device, test the device of intermediate tensors are correct
  if mt:=getenv("MOVE_TENSOR", ""):
    for t in tst: t.to_(mt)

  st = time.monotonic()
  ret = to_tuple(tinygrad_fxn(*tst))
  Tensor.realize(*ret)
  tinygrad_fp = time.monotonic() - st

  def compare(s, tinygrad_output, torch_output, atol, rtol):
    if COMPILE_ONLY: return
    if PRINT_TENSORS or DEBUG >= 6: print(s, tinygrad_output, torch_output)
    try:
      assert tinygrad_output.shape == torch_output.shape, f"shape mismatch: tinygrad={tinygrad_output.shape} | torch={torch_output.shape}"
      assert tinygrad_output.dtype == torch_output.dtype, f"dtype mismatch: tinygrad={tinygrad_output.dtype} | torch={torch_output.dtype}"
      if np.issubdtype(tinygrad_output.dtype, np.floating):
        np.testing.assert_allclose(tinygrad_output, torch_output, atol=atol, rtol=rtol)
      else:
        np.testing.assert_equal(tinygrad_output, torch_output)
    except Exception as e:
      raise Exception(f"{s} failed shape {tinygrad_output.shape}: {e}")

  if DEBUG >= 6: np.set_printoptions(linewidth=200, suppress=True)
  for i, (t, torch_out) in enumerate(zip(ret, to_tuple(out), strict=True)):
    compare(f"forward pass tensor {i}", t.numpy(), torch_out.detach().cpu().numpy(), atol=atol, rtol=rtol)

  torch_fbp, tinygrad_fbp = np.nan, np.nan
  if not forward_only and not FORWARD_ONLY and ts and tst:
    st = time.monotonic()
    torch_grads = torch.autograd.grad(out.sum(), ts)
    torch_fbp = time.monotonic() - st

    st = time.monotonic()
    # NOTE: we now have to recompute the forward pass since we realized it
    tiny_grads = tinygrad_fxn(*tst).sum().gradient(*tst)
    Tensor.realize(*tiny_grads)
    tinygrad_fbp = time.monotonic() - st

    for i, (t, torch_grad) in enumerate(zip(tiny_grads, torch_grads)):
      compare(f"backward pass tensor {i}", t.numpy(), torch_grad.detach().cpu().numpy(), atol=grad_atol, rtol=grad_rtol)

  if sys.stdout.isatty():
    print("\ntesting %40r   torch/tinygrad fp: %.2f / %.2f ms  bp: %.2f / %.2f ms " % \
          (shps, torch_fp*1000, tinygrad_fp*1000, torch_fbp*1000, tinygrad_fbp*1000), end="")

class TestOps(TensorTestCase):
  def test_nested_shrink(self):
    a = Tensor.arange(32).reshape(4, 8).contiguous().realize()
    out = a.shrink(((1, 4), (1, 7))).shrink(((1, 3), (2, 5)))
    self.assertEqual(out.shape, (2, 3))
    self.assertEqual(out.tolist(), [[19, 20, 21], [27, 28, 29]])

  def test_full_like(self):
    a = Tensor([[1,2,3],[4,5,6]], dtype=dtypes.float32)
    b = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32)
    helper_test_op([], lambda: torch.full_like(b, 4), lambda: Tensor.full_like(a, 4), forward_only=True)

    a = Tensor([[1,2,3],[4,5,6]], dtype=dtypes.int32)
    b = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.int32)
    helper_test_op([], lambda: torch.full_like(b, 4), lambda: Tensor.full_like(a, 4), forward_only=True)

  def test_full(self):
    helper_test_op([], lambda: torch.full((45,65), 4, dtype=torch.int32), lambda: Tensor.full((45,65), 4), forward_only=True)

  def test_zeros(self):
    helper_test_op([], lambda: torch.zeros(45,65), lambda: Tensor.zeros(45,65), forward_only=True)
    helper_test_op([], lambda: torch.zeros([45,65]), lambda: Tensor.zeros([45,65]), forward_only=True)
    helper_test_op([], lambda: torch.zeros([]), lambda: Tensor.zeros([]), forward_only=True)

  def test_zeros_like(self):
    a = Tensor([[1,2,3],[4,5,6]], dtype=dtypes.float32)
    b = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32)
    helper_test_op([], lambda: torch.zeros_like(b), lambda: Tensor.zeros_like(a), forward_only=True)

    a = Tensor([[1,2,3],[4,5,6]], dtype=dtypes.int32)
    b = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.int32)
    helper_test_op([], lambda: torch.zeros_like(b), lambda: Tensor.zeros_like(a), forward_only=True)

  def test_empty_0(self):
    helper_test_op([], lambda: torch.empty(45,65)*0/0, lambda: Tensor.empty(45,65)*0/0, forward_only=True)

  def test_ones(self):
    helper_test_op([], lambda: torch.ones(45,65), lambda: Tensor.ones(45,65), forward_only=True)
    helper_test_op([], lambda: torch.ones([45,65]), lambda: Tensor.ones([45,65]), forward_only=True)
    helper_test_op([], lambda: torch.ones([]), lambda: Tensor.ones([]), forward_only=True)

  def test_ones_like(self):
    a = Tensor([[1,2,3],[4,5,6]], dtype=dtypes.float32)
    b = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32)
    helper_test_op([], lambda: torch.ones_like(b), lambda: Tensor.ones_like(a), forward_only=True)

    a = Tensor([[1,2,3],[4,5,6]], dtype=dtypes.int32)
    b = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.int32)
    helper_test_op([], lambda: torch.ones_like(b), lambda: Tensor.ones_like(a), forward_only=True)

  def test_eye(self):
    helper_test_op([], lambda: torch.eye(10), lambda: Tensor.eye(10), forward_only=True)
    helper_test_op([], lambda: torch.eye(3, 5), lambda: Tensor.eye(3, 5), forward_only=True)
    helper_test_op([], lambda: torch.eye(5, 3), lambda: Tensor.eye(5, 3), forward_only=True)
    helper_test_op([], lambda: torch.eye(1), lambda: Tensor.eye(1), forward_only=True)
    helper_test_op([], lambda: torch.eye(0), lambda: Tensor.eye(0), forward_only=True)

  def test_split(self):
    def tensor(s): return torch.arange(math.prod(s), dtype=torch.int32).reshape(s), Tensor.arange(math.prod(s)).reshape(s)
    test_cases = [
      (tensor((10,)),       5, {}),
      (tensor((10,)), [1,4,5], {}),
      (tensor((10,)),       3, {}),
      (tensor((3,4,)),      1, {}),
      (tensor((3,4,)),      1, {'dim':1}),
      (tensor((4,4,)),  [2,2], {}),
      (tensor((4,4,)),  [2,2], {'dim':1}),
      (tensor((10000,)), 2500, {}),
    ]

    for (tor, ten), sizes, args in test_cases:
      tor_splits, ten_splits = tor.split(sizes, **args), ten.split(sizes, **args)
      assert len(tor_splits) == len(ten_splits)
      for tor_chunk, ten_chunk in zip(tor_splits, ten_splits):
        helper_test_op([], lambda: tor_chunk, lambda: ten_chunk, forward_only=True)

  def test_chunk(self):
    tor = torch.arange(13, dtype=torch.int32).repeat(8, 1).chunk(6, 1)
    ten = Tensor.arange(13).repeat((8, 1)).chunk(6, 1)
    assert len(tor) == len(ten)
    for i in range(len(tor)):
      helper_test_op([], lambda: tor[i], lambda: ten[i], forward_only=True)

    tor = torch.arange(13, dtype=torch.int32).repeat(8, 1).chunk(6, 0)
    ten = Tensor.arange(13).repeat((8, 1)).chunk(6, 0)
    assert len(tor) == len(ten)
    for i in range(len(tor)):
      helper_test_op([], lambda: tor[i], lambda: ten[i], forward_only=True)

    tor = torch.arange(13, dtype=torch.int32).repeat(8, 1).chunk(3, -1)
    ten = Tensor.arange(13).repeat((8, 1)).chunk(3, -1)
    assert len(tor) == len(ten)
    for i in range(len(tor)):
      helper_test_op([], lambda: tor[i], lambda: ten[i], forward_only=True)

    tor = torch.arange(13, dtype=torch.int32).repeat(8, 3, 3).chunk(3, -2)
    ten = Tensor.arange(13).repeat((8, 3, 3)).chunk(3, -2)
    assert len(tor) == len(ten)
    for i in range(len(tor)):
      helper_test_op([], lambda: tor[i], lambda: ten[i], forward_only=True)

  def test_unfold(self):
    helper_test_op([(8,)], lambda x: x.unfold(0, 2, 1))
    helper_test_op([(8,)], lambda x: x.unfold(0, 2, 2))
    helper_test_op([(8,)], lambda x: x.unfold(0, 7, 3))
    helper_test_op([(3,3,3)], lambda x: x.unfold(2, 2, 8))
    helper_test_op([(3,3,3)], lambda x: x.unfold(1, 0, 8))
    helper_test_op([(3,3,3,3,3)], lambda x: x.unfold(-1, 2, 2))
    helper_test_op([(0,)], lambda x: x.unfold(0, 0, 1))
    helper_test_op([(2,0,3)], lambda x: x.unfold(1, 0, 5))

    self.helper_test_exception([(8,)], lambda x: x.unfold(0, 9, 3), expected=RuntimeError)
    self.helper_test_exception([(8,)], lambda x: x.unfold(1, 8, 3), expected=IndexError)
    self.helper_test_exception([(8,)], lambda x: x.unfold(0, 1, -1), expected=RuntimeError)

  def test_meshgrid(self):
    x, xt = torch.tensor([0.,1.,2.], requires_grad=True), Tensor([0.,1.,2.])
    y, yt = torch.tensor([3.,4.,5.,6.], requires_grad=True), Tensor([3.,4.,5.,6.])
    z, zt = torch.tensor([7.,8.,9.], requires_grad=True), Tensor([7.,8.,9.])
    for indexing in ("ij", "xy"):
      tor = torch.meshgrid(x, indexing=indexing)
      ten = xt.meshgrid(indexing=indexing)
      self.assertEqual(len(tor), len(ten))
      for tor_i, ten_i in zip(tor, ten):
        helper_test_op([], lambda: tor_i, lambda: ten_i)
      tor = torch.meshgrid(x, y, indexing=indexing)
      ten = xt.meshgrid(yt, indexing=indexing)
      self.assertEqual(len(tor), len(ten))
      for tor_i, ten_i in zip(tor, ten):
        helper_test_op([], lambda: tor_i, lambda: ten_i)
      tor = torch.meshgrid(x, torch.tensor(10., requires_grad=True), y, z, indexing=indexing)
      ten = xt.meshgrid(Tensor(10.), yt, zt, indexing=indexing)
      self.assertEqual(len(tor), len(ten))
      for tor_i, ten_i in zip(tor, ten):
        helper_test_op([], lambda: tor_i, lambda: ten_i)

    self.helper_test_exception([], lambda: torch.meshgrid(x, indexing="bad"), lambda: xt.meshgrid(indexing="bad"), expected=RuntimeError)

  def test_arange(self):
    helper_test_op([], lambda: torch.arange(10, dtype=torch.int32), lambda: Tensor.arange(10), forward_only=True)
    helper_test_op([], lambda: torch.arange(36, dtype=torch.int32), lambda: Tensor.arange(36), forward_only=True)
    helper_test_op([], lambda: torch.arange(5, 10, 3, dtype=torch.int32), lambda: Tensor.arange(5, 10, 3), forward_only=True)
    helper_test_op([], lambda: torch.arange(10, 5, -3, dtype=torch.int32), lambda: Tensor.arange(10, 5, -3), forward_only=True)
    helper_test_op([], lambda: torch.arange(11, 5, -3, dtype=torch.int32), lambda: Tensor.arange(11, 5, -3), forward_only=True)
    helper_test_op([], lambda: torch.arange(1, 78, 2, dtype=torch.int32), lambda: Tensor.arange(1, 78, 2), forward_only=True)
    helper_test_op([], lambda: torch.arange(5.5, 175.5, 2.5), lambda: Tensor.arange(5.5, 175.5, 2.5), forward_only=True)
    helper_test_op([], lambda: torch.arange(-30.2, -0.3, 0.75), lambda: Tensor.arange(-30.2, -0.3, 0.75), forward_only=True)
    helper_test_op([], lambda: torch.arange(-50.3, -380.2, -2.25), lambda: Tensor.arange(-50.3, -380.2, -2.25), forward_only=True)
    helper_test_op([], lambda: torch.arange(0, 3, 0.6), lambda: Tensor.arange(0, 3, 0.6), forward_only=True)
    # boundary values that fit exactly in int8 (min=-128, max=127)
    helper_test_op([], lambda: torch.arange(128, dtype=torch.int8), lambda: Tensor.arange(128, dtype=dtypes.int8), forward_only=True)
    helper_test_op([], lambda: torch.arange(-128, 128, dtype=torch.int8), lambda: Tensor.arange(-128, 128, dtype=dtypes.int8), forward_only=True)
    helper_test_op([], lambda: torch.arange(127, -129, -1, dtype=torch.int8),
                   lambda: Tensor.arange(127, -129, -1, dtype=dtypes.int8), forward_only=True)
    # an int range too large for default_int picks int64
    self.assertEqual(Tensor.arange(2**31, 2**31+3).dtype, dtypes.int64)
    # overflow: tinygrad raises (torch silently wraps)
    with self.assertRaises(OverflowError): Tensor.arange(2**33, dtype=dtypes.int)
    with self.assertRaises(OverflowError): Tensor.arange(129, dtype=dtypes.int8)            # last=128 overflows
    with self.assertRaises(OverflowError): Tensor.arange(-129, 128, dtype=dtypes.int8)      # start=-129 overflows
    with self.assertRaises(OverflowError): Tensor.arange(128, 0, -1, dtype=dtypes.int8)     # start=128 overflows
    with self.assertRaises(OverflowError): Tensor.arange(127, -130, -1, dtype=dtypes.int8)  # last=-129 overflows

  def test_arange_low_precision_dtype(self):
    for dt in (*dtypes.fp8s, dtypes.half, dtypes.bfloat16):
      for args in ((0, 10, 3), (0, 200, 1), (5.5, 175.5, 2.5)):
        self.assertEqual(Tensor.arange(*args, dtype=dt).float().tolist(), Tensor.arange(*args).cast(dt).float().tolist())

  def test_arange_big(self):
    helper_test_op([], lambda: torch.arange(256, dtype=torch.int32), lambda: Tensor.arange(256), forward_only=True)

  def test_arange_4096(self):
    helper_test_op([], lambda: torch.arange(4096, dtype=torch.int32), lambda: Tensor.arange(4096), forward_only=True)

  def test_linspace(self):
    helper_test_op([], lambda: torch.linspace(5, 10, 3), lambda: Tensor.linspace(5, 10, 3), forward_only=True)
    helper_test_op([], lambda: torch.linspace(5, 10, 1), lambda: Tensor.linspace(5, 10, 1), forward_only=True)
    helper_test_op([], lambda: torch.linspace(5, 10, 0), lambda: Tensor.linspace(5, 10, 0), forward_only=True)
    helper_test_op([], lambda: torch.linspace(5, 10, 30), lambda: Tensor.linspace(5, 10, 30), forward_only=True)
    helper_test_op([], lambda: torch.linspace(-5.5, 5.5, 10), lambda: Tensor.linspace(-5.5, 5.5, 10), forward_only=True)
    helper_test_op([], lambda: torch.linspace(5.5, -5.5, 10), lambda: Tensor.linspace(5.5, -5.5, 10), forward_only=True)
    helper_test_op([], lambda: torch.linspace(5, 10, 3, dtype=torch.int32), lambda: Tensor.linspace(5, 10, 3, dtype="int32"), forward_only=True)
    helper_test_op([], lambda: torch.linspace(5, 10, 20, dtype=torch.int32), lambda: Tensor.linspace(5, 10, 20, dtype="int32"), forward_only=True)
    helper_test_op([], lambda: torch.linspace(5, -5, 20, dtype=torch.int32), lambda: Tensor.linspace(5, -5, 20, dtype="int32"), forward_only=True)
    self.helper_test_exception([], lambda: torch.linspace(5, 10, 3, dtype=torch.bool), lambda: Tensor.linspace(5, 10, 3, dtype="bool"),
                               expected=(RuntimeError, ValueError))
    self.helper_test_exception([], lambda: torch.linspace(1, 2, -1), lambda: Tensor.linspace(1, 2, -1), expected=(RuntimeError, ValueError))

  def test_sum_fake(self):
    helper_test_op([(256, 1)], lambda x: x.sum(axis=1))

  def test_sum_collapse(self):
    helper_test_op([], lambda: torch.ones(256,256).sum(axis=1), lambda: Tensor.ones(256,256).sum(axis=1), forward_only=True)

  def test_sum_collapse_neg(self):
    helper_test_op([], lambda: (-torch.ones(3,3)).sum(axis=1), lambda: (-Tensor.ones(3,3)).sum(axis=1), forward_only=True)

  def test_sum_pad_collapse(self):
    helper_test_op([], lambda: torch.nn.functional.pad(torch.ones(256,256), pad=(0,64,0,0)).sum(axis=1),
                       lambda: Tensor.ones(256,256).pad(((0,0), (0,64))).sum(axis=1), forward_only=True)

  def test_sum_twice(self):
    helper_test_op([(4, 4, 4)], lambda x: x.sum((0, 1)).sum())
    helper_test_op([(4, 4, 4)], lambda x: x.sum((0, 2)).sum())
    helper_test_op([(4, 4, 4)], lambda x: x.sum((1, 2)).sum())

  # this is more complex and won't fold for a while
  def test_sum_cat_collapse(self):
    helper_test_op([], lambda: torch.cat([torch.ones(256,256), torch.zeros(256,64)], dim=1).sum(axis=1),
                       lambda: Tensor.cat(Tensor.ones(256,256), Tensor.zeros(256,64), dim=1).sum(axis=1), forward_only=True)

  def test_max_dont_collapse(self):
    helper_test_op([], lambda: torch.ones(256,256).max(1)[0], lambda: Tensor.ones(256,256).max(1), forward_only=True)

  def test_where(self):
    helper_test_op([], lambda: torch.where(torch.tensor([True, False]), 1, 3).type(torch.int32),
                   lambda: Tensor([True, False]).where(1, 3).clone(), forward_only=True)
    helper_test_op([], lambda: torch.where(torch.tensor([True, False]), 3., 0.1).eq(0.1),
                   lambda: Tensor([True, False]).where(3., 0.1).eq(0.1), forward_only=True)
    helper_test_op([], lambda: torch.where(torch.tensor([True, False]), torch.tensor(0.).half(), 0.3).float() > 0.30001,
                   lambda: Tensor([True, False]).where(Tensor(0., dtype=dtypes.half), 0.3).float() > 0.30001, forward_only=True)
    helper_test_op(
      [(100,)],
      lambda x: torch.where(x > 0.5, 4, 2).type(torch.int32),
      lambda x: (x > 0.5).where(4, 2).clone(), forward_only=True)

    for shps in [[(8,),(1,),(1,)], [(10,10),(10,),(10,)], [(100,)]*3, [(10,10)]*3]:
      helper_test_op(
        shps,
        lambda x, a, b: torch.where(x > 0.5, a, b),
        lambda x, a, b: (x > 0.5).where(a, b), forward_only=True)

  def test_where_permute(self):
    helper_test_op(
      [(5, 5)],
      lambda x: torch.where(x > 0.5, 4, 2).type(torch.int32).permute((1, 0)),
      lambda x: (x > 0.5).where(4, 2).clone().permute((1, 0)), forward_only=True)

  def test_where_nan_cond(self):
    # a NaN compares false against everything except !=.
    for fxn in (lambda x: x<1, lambda x: x>1, lambda x: x!=1, lambda x: x==1):
      helper_test_op(None, lambda x,a,b: torch.where(fxn(x), a, b), lambda x,a,b: fxn(x).where(a, b), forward_only=True,
                     vals=[[math.nan, 1.0, 2.0, -1.0], [10, 20, 30, 40], [-1, -2, -3, -4]])

  def _test_cmp(self, fxn, reverse=True):
    # test different dtypes
    helper_test_op(None, fxn, fxn, forward_only=True, vals=[[0.,1,2], [2.,1,0]])
    helper_test_op(None, fxn, fxn, forward_only=True, vals=[[0,1,2], [2,1,0]])
    helper_test_op(None, fxn, fxn, forward_only=True, vals=[[True, True, False], [False,True,False]])
    # test broadcasting
    for shps in [[(3, 4, 5), (3, 4, 5)], [(3, 4, 5), (5,)], [(5,), (3, 4, 5)]]:
      helper_test_op(shps, fxn, fxn, forward_only=True)
    # test cmp with const
    helper_test_op(None, lambda x,y: fxn(x,2), lambda x,y: fxn(x,2), forward_only=True, vals=[[0.,1,2], [2.,1,0]])
    if reverse: helper_test_op(None, lambda x,y: fxn(2,y), lambda x,y: fxn(2,y), forward_only=True, vals=[[0.,1,2], [2.,1,0]])
    # test special floats  # TODO: fix nan
    specials = [0.0, 1.0, -1.0, math.inf, -math.inf]#, math.nan]
    for s0 in specials:
      for s1 in specials:
        helper_test_op(None, fxn, fxn, forward_only=True, vals=[[s0], [s1]])

  def test_cmp_eq(self): self._test_cmp(lambda x,y: x==y, reverse=False)
  def test_cmp_ne(self): self._test_cmp(lambda x,y: x!=y)

  def test_cmp_eq_ne_specials(self):
    vals = [[math.nan, math.inf, -math.inf, 0.0, -0.0, 1.0], [math.nan, math.inf, math.inf, -0.0, 0.0, 2.0]]
    for fxn in (lambda x,y: x==y, lambda x,y: x!=y):
      helper_test_op(None, fxn, forward_only=True, vals=vals)
      helper_test_op(None, lambda x: fxn(x, x), forward_only=True, vals=vals[:1])

  def test_cmp_eq_padding(self):
    # Padding a comparison fills False; comparing two zero-padded inputs fills True.
    for fxn in (lambda x,y: x==y, lambda x,y: x!=y):
      helper_test_op(None, lambda x,y: torch.nn.functional.pad(fxn(x,y), (1,2)).float().sum(),
                     lambda x,y: fxn(x,y).pad((1,2)).float().sum(), vals=[[1,2,3], [1,0,3]], forward_only=True)
      helper_test_op(None, lambda x,y: fxn(torch.nn.functional.pad(x, (1,2)), torch.nn.functional.pad(y, (1,2))).float().sum(),
                     lambda x,y: fxn(x.pad((1,2)), y.pad((1,2))).float().sum(), vals=[[1,2,3], [1,0,3]], forward_only=True)

  def test_cmp_gt(self): self._test_cmp(lambda x,y: x>y)
  def test_cmp_ge(self): self._test_cmp(lambda x,y: x>=y)
  def test_cmp_lt(self): self._test_cmp(lambda x,y: x<y)
  def test_cmp_le(self): self._test_cmp(lambda x,y: x<=y)

  def test_cmp_ne_backwards(self):
    # new grad zeroes these out
    """
    t1 = torch.ones(4, requires_grad=True)
    t2 = torch.ones(4, requires_grad=True)
    self.assertRaises(RuntimeError, (t1 != t2).sum().backward)
    tt1 = Tensor.ones(4)
    tt2 = Tensor.ones(4)
    self.assertRaises(RuntimeError, (tt1 != tt2).sum().backward)
    """
    tt = Tensor.randn(4)
    (tt*(tt != 0)).sum().backward()
    t = torch.tensor(tt.numpy(), requires_grad=True)
    (t*(t != 0)).sum().backward()
    np.testing.assert_allclose(t.grad.cpu().numpy(), tt.grad.numpy(), rtol=1e-5)

  # TODO: fix backward of these functions
  def test_trunc(self):
    helper_test_op([()], lambda x: x.trunc(), forward_only=True)
    helper_test_op([(45,35)], lambda x: x.trunc(), forward_only=True)
    helper_test_op(None, lambda x: x.trunc(), vals=[[1.499, 1.5, 1.501, 1.0, 2.1, 0.0, -5.0, -2.499, -2.5, -2.501, 1e12, -1e12]], forward_only=True)
  def test_floor(self):
    helper_test_op([()], lambda x: x.floor(), forward_only=True)
    helper_test_op([(45,35)], lambda x: x.floor(), forward_only=True)
    helper_test_op(None, lambda x: x.floor(), vals=[[1.499, 1.5, 1.501, 1.0, 2.1, 0.0, -5.0, -2.499, -2.5, -2.501]], forward_only=True)
  def test_ceil(self):
    helper_test_op([()], lambda x: x.ceil(), forward_only=True)
    helper_test_op([(45,35)], lambda x: x.ceil(), forward_only=True)
    helper_test_op(None, lambda x: x.ceil(), vals=[[1.499, 1.5, 1.501, 1.0, 2.1, 0.0, -5.0, -2.499, -2.5, -2.501]], forward_only=True)
  def test_round(self):
    helper_test_op([()], lambda x: x.round(), forward_only=True)
    helper_test_op([(45,35)], lambda x: x.round(), forward_only=True)
    helper_test_op(None, lambda x: x.round(), vals=[[1.499, 1.5, 1.501, 1.0, 2.1, 0.0, -5.0, -2.499, -2.5, -2.501]], forward_only=True)
    helper_test_op(None, lambda x: x.round(), vals=[[2.5, -1.5]], forward_only=True)
    helper_test_op(None, lambda x: x.round(), vals=[[16777217, -3, 0]], forward_only=True)

  def test_round_quantization_gradient(self):
    helper_test_op(None, lambda x: x + 0.125 * (x.round() - x), vals=[[-1.2, -0.7, -0.2, 0.2, 0.7, 1.2]])

  def test_isinf(self):
    val = [float('-inf'), 0., float('inf'), float('nan'), 1.1]
    helper_test_op(None, torch.isinf, Tensor.isinf, vals=[val], forward_only=True)
    if not COMPILE_ONLY:
      np.testing.assert_equal(Tensor(val).isinf(detect_positive=True, detect_negative=False).numpy(), [False, False, True, False, False])
      np.testing.assert_equal(Tensor(val).isinf(detect_positive=False, detect_negative=True).numpy(), [True, False, False, False, False])

  def test_isnan(self):
    helper_test_op(None, torch.isnan, Tensor.isnan, vals=[[float('-inf'), 0., float('inf'), float('nan'), 1.1]], forward_only=True)

  def test_isfinite(self):
    helper_test_op(None, torch.isfinite, Tensor.isfinite, vals=[[float('-inf'), 0., float('inf'), float('nan'), 1.1]], forward_only=True)

  def test_lerp(self):
    helper_test_op([(45,35), (45,35), (45,35)], lambda x,y,z: x.lerp(y,z))
    helper_test_op(None, lambda x,y,z: x.lerp(y,z), vals=[[1.,2.,3.], [4.,5.,6.], 0.5])

  def test_tril(self):
    helper_test_op([(3,3)], lambda x: x.tril())
    helper_test_op([(3,3)], lambda x: x.tril(1))
    helper_test_op([(3,3)], lambda x: x.tril(2))
    helper_test_op([(3,3)], lambda x: x.tril(-1))
    helper_test_op([(3,3)], lambda x: x.tril(-2))
    helper_test_op([(4,5)], lambda x: x.tril(4))
    helper_test_op([(4,5)], lambda x: x.tril(5))
    helper_test_op([(4,5)], lambda x: x.tril(6))
    helper_test_op([(4,5)], lambda x: x.tril(-4))
    helper_test_op([(4,5)], lambda x: x.tril(-5))
    helper_test_op([(4,5)], lambda x: x.tril(-6))
    helper_test_op([(5,3,3)], lambda x: x.tril())
    helper_test_op([(5,0,3)], lambda x: x.tril())
    helper_test_op([(5,3,3)], lambda x: x.tril(1))
    helper_test_op(None, lambda x: x.tril(), vals=[[[True] * 3] * 3], forward_only=True)

  def test_triu(self):
    helper_test_op([(3,3)], lambda x: x.triu())
    helper_test_op([(3,3)], lambda x: x.triu(1))
    helper_test_op([(3,3)], lambda x: x.triu(2))
    helper_test_op([(3,3)], lambda x: x.triu(-1))
    helper_test_op([(3,3)], lambda x: x.triu(-2))
    helper_test_op([(4,5)], lambda x: x.triu(4))
    helper_test_op([(4,5)], lambda x: x.triu(5))
    helper_test_op([(4,5)], lambda x: x.triu(6))
    helper_test_op([(4,5)], lambda x: x.triu(-4))
    helper_test_op([(4,5)], lambda x: x.triu(-5))
    helper_test_op([(4,5)], lambda x: x.triu(-6))
    helper_test_op([(5,3,3)], lambda x: x.triu())
    helper_test_op([(5,0,3)], lambda x: x.triu())
    helper_test_op([(5,3,3)], lambda x: x.triu(1))
    helper_test_op(None, lambda x: x.triu(), vals=[[[True] * 3] * 3], forward_only=True)

  def test_maximum(self):
    helper_test_op([(45,65), (45,65)], torch.maximum, Tensor.maximum)
    helper_test_op([(), ()], torch.maximum, Tensor.maximum)
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[1., 0., 3., -4.], 3.])
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[1., 0., 3., -4.], [-1., -2., 3., 0.]])
    helper_test_op(None, torch.maximum, Tensor.maximum,
                   vals=[[-1234, 0, 1234, dtypes.int.max, dtypes.int.min], dtypes.int.max], forward_only=True)
    helper_test_op(None, torch.maximum, Tensor.maximum,
                   vals=[[-1234, 0, 1234, dtypes.int.max, dtypes.int.min], dtypes.int.min], forward_only=True)
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[True, False, False], True], forward_only=True)
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[True, False, False], [True, True, False]], forward_only=True)

    # test applying to different dtype
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[1, 2, 3], 1.2], forward_only=True)
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[True, False, False], 1.2], forward_only=True)
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[True, False, False], 3], forward_only=True)

    helper_test_op([], lambda: torch.where(torch.tensor([True, False]), torch.tensor(-1.).half(), 0.).maximum(torch.tensor(1.0001)).float(),
                   lambda: Tensor([True, False]).where(Tensor(-1., dtype=dtypes.half), 0.).maximum(1.0001).float(), forward_only=True)

  def test_minimum(self):
    helper_test_op([(45,65), (45,65)], torch.minimum, Tensor.minimum)
    helper_test_op([(), ()], torch.minimum, Tensor.minimum)
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[1., 0., 3., -4.], 3.])
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[1., 0., 3., -4.], [-1., -2., 3., 0.]])
    helper_test_op(None, torch.minimum, Tensor.minimum,
                   vals=[[-1234, 0, 1234, dtypes.int.max, dtypes.int.min], dtypes.int.max], forward_only=True)
    helper_test_op(None, torch.minimum, Tensor.minimum,
                   vals=[[-1234, 0, 1234, dtypes.int.max, dtypes.int.min], dtypes.int.min], forward_only=True)
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[True, False, False], True], forward_only=True)
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[True, False, False], [True, True, False]], forward_only=True)

    # test applying to different dtype
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[1, 2, 3], 1.2], forward_only=True)
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[True, False, False], 1.2], forward_only=True)
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[True, False, False], 3], forward_only=True)

  def test_add(self):
    helper_test_op([], lambda: torch.tensor(1)+0.5, lambda: Tensor(1)+0.5, forward_only=True)
    helper_test_op([(45,68), (45,68)], lambda x,y: x+y)
    helper_test_op([(), ()], lambda x,y: x+y)
    helper_test_op([], lambda: torch.tensor([5])+8, lambda: Tensor([5], dtype=dtypes.int64)+8, forward_only=True)
  def test_add3(self):
    helper_test_op([(45,65), (45,65), (45,65)], lambda x,y,z: x+y+z)
  def test_broadcasted_add(self):
    helper_test_op([(45,65), (45,1)], lambda x,y: x+y)
    helper_test_op([(45,65), ()], lambda x,y: x+y)
  def test_broadcasted_add_2(self):
    helper_test_op([(45,65), (65,)], lambda x,y: x+y)

  def test_sub(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x-y)
    helper_test_op([(), ()], lambda x,y: x-y)
    if not NAN_UNSAFE: helper_test_op(None, lambda x: x-x, vals=[[-1.0, math.inf, math.nan]])
    self.assertFalse(np.signbit(((x:=Tensor([-1.0]))-x).numpy()).any())
  def test_scalar_sub(self):
    helper_test_op([(45,65)], lambda x: x-2)
    helper_test_op([()], lambda x: x-2)
  def test_scalar_rsub(self):
    helper_test_op([(45,65)], lambda x: 2-x)
    helper_test_op([()], lambda x: 2-x)

  def test_neg(self):
    helper_test_op([(45,65)], lambda x: -x)
    helper_test_op([(45,65)], lambda x: x.neg())
    helper_test_op([()], lambda x: x.neg())
  def test_logical_not(self):
    helper_test_op(None, torch.logical_not, Tensor.logical_not, vals=[[True, False, True]], forward_only=True)
    helper_test_op(None, torch.logical_not, Tensor.logical_not, vals=[[1.,2.,0.,0.5]], forward_only=True)

  def test_mul(self):
    helper_test_op([(64,64), (64,64)], lambda x,y: x*y)
    helper_test_op([(), ()], lambda x,y: x*y)
  def test_scalar_mul(self):
    helper_test_op([(45,65)], lambda x: x*2)
    helper_test_op([(45,65)], lambda x: x*-1)
    helper_test_op([(45,65)], lambda x: 255*x)
    helper_test_op([(45,65)], lambda x: 2*x)
    helper_test_op([()], lambda x: x*2)
    helper_test_op([()], lambda x: 2*x)

  def test_div(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x/y)
    helper_test_op([(), ()], lambda x,y: x/y)

  @unittest.skipIf(Device.DEFAULT == "AMD" and DEV.renderer == "LLVM", "AMD with LLVM backend generate rcp in FP division causes trunc/floor errors")
  def test_div_rounding_mode(self):
    for denominator in [-10, -5, -3, -2, -1, 1, 2, 3, 5, 10]:
      # int numerator
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode=None), forward_only=True, vals=[[5, 6, 7, 0, -5, -6, -7], [denominator]])
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode="trunc"), forward_only=True, vals=[[5, 6, 7, 0, -5, -6, -7], [denominator]])
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode="floor"), forward_only=True, vals=[[5, 6, 7, 0, -5, -6, -7], [denominator]])
      # float numerator
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode=None), forward_only=True, vals=[[5.0, 6, 7, 0, -5, -6, -7], [denominator]])
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode="trunc"), forward_only=True, vals=[[5.0, 6, 7, 0, -5, -6, -7], [denominator]])
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode="floor"), forward_only=True, vals=[[5.0, 6, 7, 0, -5, -6, -7], [denominator]])

    for denominator in [-10.0, -5.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 5.0, 10.0]:
      # int numerator
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode=None), forward_only=True, vals=[[5, 6, 7, 0, -5, -6, -7], [denominator]])
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode="trunc"), forward_only=True, vals=[[5, 6, 7, 0, -5, -6, -7], [denominator]])
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode="floor"), forward_only=True, vals=[[5, 6, 7, 0, -5, -6, -7], [denominator]])
      # float numerator
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode=None), forward_only=True, vals=[[5.0, 6, 7, 0, -5, -6, -7], [denominator]])
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode="trunc"), forward_only=True, vals=[[5.0, 6, 7, 0, -5, -6, -7], [denominator]])
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode="floor"), forward_only=True, vals=[[5.0, 6, 7, 0, -5, -6, -7], [denominator]])

    for numerator in [110, 111, -110, -111]:
      for denominator in [55, -55]:
        helper_test_op(None, lambda x,y: x.div(y, rounding_mode=None), forward_only=True, vals=[[numerator], [denominator]])
        helper_test_op(None, lambda x,y: x.div(y, rounding_mode="trunc"), forward_only=True, vals=[[numerator], [denominator]])
        helper_test_op(None, lambda x,y: x.div(y, rounding_mode="floor"), forward_only=True, vals=[[numerator], [denominator]])

    self.helper_test_exception(None, lambda x,y: x.div(y, rounding_mode="typo"), forward_only=True, vals=[[5], [0]], expected=RuntimeError)

  def test_div_int(self):
    helper_test_op(None, lambda x,y: x/y, Tensor.div, forward_only=True, vals=[[5, 6, 7],[1, 2, 3]])
    helper_test_op(None, lambda x,y: x//y, forward_only=True, vals=[[5, 6, 7],[1, 2, 3]])
    helper_test_op(None, lambda x: x/2, forward_only=True, vals=[[3, 4, 5]])
    helper_test_op(None, lambda x: x//2, forward_only=True, vals=[[3, 4, 5]])
    helper_test_op(None, functools.partial(torch.div, rounding_mode="trunc"),
                   functools.partial(Tensor.div, rounding_mode="trunc"), forward_only=True,
                   vals=[[-4, 7, 5, 4, -7, 8], [2, -3, 8, -2, 3, 5]])
    if not COMPILE_ONLY:
      x = Tensor(2**64 - 1, dtype=dtypes.uint64).div(1, rounding_mode="trunc")
      np.testing.assert_equal(x.numpy(), 2**64 - 1)

  def test_scalar_div(self):
    helper_test_op([(45,65)], lambda x: x/255)
    helper_test_op([(45,65)], lambda x: x/1)
    helper_test_op([(45,65)], lambda x: 1/x)
    helper_test_op([(45,65)], lambda x: x/2)
    helper_test_op([(45,65)], lambda x: 2/x)
    helper_test_op([()], lambda x: x/2)
    helper_test_op([()], lambda x: 2/x)

  def test_mod(self):
    a = [-4, 7, 5, 4, -7, 8, -9]
    b = [2, -3, 8, -2, 3, 5, -5]
    for float_a in [True, False]:
      for float_b in [True, False]:
        va = [float(ai) for ai in a] if float_a else a
        vb = [float(bi) for bi in b] if float_b else b
        helper_test_op(None, lambda x,y: x%y, Tensor.mod, forward_only=True, vals=[va, vb])
        helper_test_op(None, lambda x,y: x%y, forward_only=True, vals=[va, vb])
        helper_test_op(None, lambda x: x%2, forward_only=True, vals=[va])
        helper_test_op(None, lambda x: x%3, forward_only=True, vals=[va])
        helper_test_op(None, lambda x: x%3.5, lambda x: (x%3.5).clone(), forward_only=True, vals=[va])
        helper_test_op(None, lambda x: 100%x, forward_only=True, vals=[va])
        helper_test_op(None, lambda x: 100.5%x, lambda x: (100.5%x).clone(), forward_only=True, vals=[va])

  def test_fmod(self):
    a = [-4, 7, 5, 4, -7, 8, -9]
    b = [2, -3, 8, -2, 3, 5, -5]
    for float_a in [True, False]:
      for float_b in [True, False]:
        va = [float(ai) for ai in a] if float_a else a
        vb = [float(bi) for bi in b] if float_b else b
        helper_test_op(None, lambda x,y: x.fmod(y), forward_only=True, vals=[va, vb])
        helper_test_op(None, lambda x: x.fmod(2), forward_only=True, vals=[va])
        helper_test_op(None, lambda x: x.fmod(3.5), lambda x: x.fmod(3.5).clone(), forward_only=True, vals=[va])

  def test_mul_naninf(self):
    helper_test_op([(45,65)], lambda x: x*math.inf)
    helper_test_op([(45,65)], lambda x: x*-math.inf)
    helper_test_op([(45,65)], lambda x: x*math.nan)
    if not NAN_UNSAFE: helper_test_op(None, lambda x: x*0, vals=[[1.0, math.inf, -math.inf, math.nan]])
  def test_div_naninf(self):
    helper_test_op([(45,65)], lambda x: x/math.inf)
    helper_test_op([(45,65)], lambda x: x/-math.inf)
    helper_test_op([(45,65)], lambda x: x/math.nan)
    helper_test_op([(45,65)], lambda x: math.inf/x)
    helper_test_op([(45,65)], lambda x: (-math.inf)/x)
    helper_test_op([(45,65)], lambda x: math.nan/x)

  def test_pow_full(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x**y)
    helper_test_op([(45,65), (45,65)], lambda x,y: x.pow(y))

  def test_pow(self):
    helper_test_op([(45,65)], lambda x: x**0)
    helper_test_op([(45,65)], lambda x: x**1)
    helper_test_op([(45,65)], lambda x: x**2)
    helper_test_op([(45,65)], lambda x: x**3)
    helper_test_op([(45,65)], lambda x: x**-2)
    helper_test_op([()], lambda x: x**2)
    helper_test_op([()], lambda x: x**-2)
    # Regression tests for https://github.com/tinygrad/tinygrad/issues/1151
    helper_test_op([(45,65)], lambda x: x**3, low=-30, high=-27)
    helper_test_op([()], lambda x: x**3, low=-30, high=-27)
    # Regression tests for https://github.com/tinygrad/tinygrad/issues/1251
    helper_test_op([(45,65)], lambda x: x**0.2, low=-30, high=-27)
    helper_test_op([(45,65)], lambda x: x**1.2, low=-30, high=-27)
    helper_test_op([()], lambda x: x**0.2, low=-30, high=-27)
    helper_test_op([()], lambda x: x**1.2, low=-30, high=-27)
    a, b = Tensor([0.0]), torch.tensor([0.0], requires_grad=True)
    helper_test_op([], lambda: b**1.1, lambda: a**1.1)

  def test_pow_const(self):
    helper_test_op([(45,65)], lambda x: x**0.0)
    helper_test_op([(45,65)], lambda x: x**1.0)
    helper_test_op([(45,65)], lambda x: x**-1.0)
    helper_test_op([(45,65)], lambda x: x**8.0)
    helper_test_op([(45,65)], lambda x: x**5.5)
    helper_test_op([(45,65)], lambda x: x**-5.5)
    # helper_test_op([(45,65)], lambda x: x**-8.0)  # TODO: fix this
    helper_test_op([(45,65)], lambda x: 1.0**x)
    helper_test_op(None, lambda x: 1.0**x, vals=[[math.nan, math.inf, -math.inf]])
    helper_test_op([(45,65)], lambda x: 5.5**x)
    helper_test_op([(45,65)], lambda x: (-5.5)**x)
    helper_test_op([(45,65)], lambda x: 8.0**x)
    helper_test_op([(45,65)], lambda x: x**2.0)
    helper_test_op([(45,65)], lambda x: 2.0**x)
    helper_test_op([()], lambda x: x**2.0)
    helper_test_op([()], lambda x: 2.0**x)
    helper_test_op(None, lambda x: 0**x, vals=[[-2.,-1,0,1,2,3]])
    helper_test_op(None, lambda x: 0.7**x, vals=[[-2.,-1,0,1,2,3]])
    helper_test_op(None, lambda x: (-2)**x, vals=[[-2.,-1,0,1,2,3]])
    # 2**52+2 - 0.5 rounds back to itself
    helper_test_op(None, lambda x: x**(2.0**52), vals=[[0.5, 1., 2.]], forward_only=True)
    helper_test_op(None, lambda x: x**(2.0**52+2), vals=[[0.5, 1., 2.]], forward_only=True)
    # float to power of int
    helper_test_op(None, lambda x: 0.7**x, lambda x: (0.7**x).clone(), vals=[[-2,-1,0,1,2,3]], forward_only=True)

  @unittest.skipIf(COMPILE_ONLY, "test requires runtime")
  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, NIRRenderer), "TODO: broken in LVP")
  def test_pow_const_direct(self):
    # x ** c
    def get_tiny_gradient(x, c):
      t = Tensor([x], dtype=dtypes.float)
      return (t ** c)[0].gradient(t)[0].item()
    def get_torch_gradient(x, c):
      t = torch.tensor([x], dtype=torch.float, requires_grad=True)
      return torch.autograd.grad(t ** c, t)[0].item()
    for x in [-math.inf, 0, 1, math.inf]:
      for c in [-1, 0, 0.3, 1, 2]:
        torch_out = get_torch_gradient(x, c)
        # the pow backward routes through exp2/log2, whose 0/inf behavior is undefined on WEBGPU
        if Device.DEFAULT == "WEBGPU" and not math.isfinite(torch_out): continue
        tiny_out = get_tiny_gradient(x, c)
        if math.isnan(tiny_out):
          assert math.isnan(torch_out)
        else:
          self.assertAlmostEqual(tiny_out, torch_out, msg=f"{x}, {c}")

  def test_pow_neg_inf_frac_exponent(self):
    # pow(-inf, 0.3) is +inf, so the gradient 0.3*pow(-inf, -0.7) is 0, never nan
    helper_test_op(None, lambda x: x**0.3, vals=[[-math.inf]])
    # is_odd truncates, so it calls 3.3 odd: only the non_int guard keeps pow(-inf, 3.3) from negating to -inf
    helper_test_op(None, lambda x: x**3.3, vals=[[-math.inf]])

  def test_pow_zero_exponent(self):
    # x ** 0 is the constant 1 for every x, so the gradient with respect to the base is 0, never nan
    # TODO: nan ** 0, failed on WEBGPU
    helper_test_op(None, lambda x,y: x**y, vals=[[-math.inf, math.inf, 0.0], [0.0, 0.0, 0.0]])

  def test_pow_zero_tensor(self):
    helper_test_op(None, lambda x,y: x**y, vals=[[0.0], [0.0]])
    # TODO: fix WEBGPU
    if Device.DEFAULT != "WEBGPU":
      helper_test_op(None, lambda x,y: x**y, vals=[[0.0], [0.3]])
      helper_test_op(None, lambda x,y: x**y, vals=[[0.0], [-0.3]])
  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "WEBGPU issue")
  def test_exp2_log2_zero_times_negative(self):
    # gallivm's exp2/log2 have "undefined behavior with infs, 0s and nans", so exp2(log2(0)*y) returns 0 instead of inf
    helper_test_op(None, lambda x,y: (x.log2()*y).exp2(), lambda x,y: (x.log2()*y).exp2(), vals=[[0.0], [-0.7]], forward_only=True)
  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "pow at 0 routes through exp2/log2, whose 0/inf behavior is undefined on WEBGPU")
  def test_pow_zero_const(self):
    helper_test_op(None, lambda x: x**0.3, vals=[[0.0]])
    helper_test_op(None, lambda x: x**0.0, vals=[[0.0]])
    helper_test_op(None, lambda x: x**-0.3, vals=[[0.0]])
    helper_test_op(None, lambda x: x**-1.0, vals=[[-1.0, 0.0, 1.0]])

  def test_int_pow_const_int(self):
    helper_test_op(None, lambda x: x**0, vals=[[-2,0,2]], forward_only=True, atol=0)
    helper_test_op(None, lambda x: x**1, vals=[[-2,0,2]], forward_only=True, atol=0)
    helper_test_op(None, lambda x: x**2, vals=[[-2,0,2]], forward_only=True, atol=0)
    helper_test_op(None, lambda x: x**7, vals=[[11,12,13]], forward_only=True, atol=0)
    helper_test_op(None, lambda x: x**29, vals=[[-2,0,2]], forward_only=True, atol=0)
    self.helper_test_exception(None, lambda x: x**-2, vals=[[-2,0,2]], forward_only=True, expected=RuntimeError)

  def test_pow_int_dtypes(self):
    for dt, tdt in ((dtypes.uint8, torch.uint8), (dtypes.int8, torch.int8), (dtypes.int16, torch.int16), (dtypes.int64, torch.int64)):
      base, exponent = [3, 2, 1, 0, 5], [5, 7, 100, 3, 9]
      if dt is not dtypes.uint8: base, exponent = base + [-3, -1, 2], exponent + [3, -7, -1]
      if dt is dtypes.int64: base, exponent = base + [3, 2], exponent + [2**40, 63]
      helper_test_op([], lambda: torch.tensor(base, dtype=tdt) ** torch.tensor(exponent, dtype=tdt),
                         lambda: Tensor(base, dtype=dt) ** Tensor(exponent, dtype=dt), forward_only=True, atol=0)

  def test_pow_int_base_float_exponent(self):
    for exponent in (0.5, 1.5, 2.0, -1.0, 0.0):
      helper_test_op([], lambda: torch.tensor([1, 2, 3, 4], dtype=torch.int) ** exponent,
                         lambda: (Tensor([1, 2, 3, 4], dtype=dtypes.int32) ** exponent).clone(), forward_only=True)

  def test_sqrt(self):
    helper_test_op([(45,65)], lambda x: x.sqrt())
    helper_test_op(None, lambda x: x.sqrt(), vals=[[0.0]])
    helper_test_op([()], lambda x: x.sqrt())
  def test_rsqrt(self):
    helper_test_op([(45,65)], lambda x: x.rsqrt())
    helper_test_op(None, lambda x: x.rsqrt(), vals=[[0.0]])
    helper_test_op([()], lambda x: x.rsqrt())

  def test_xor(self):
    data = [[1,-8,1],[32,1,6]]
    tor = torch.tensor(data, dtype=torch.int)
    ten = Tensor(data, dtype=dtypes.int32)
    # NOTE: this breaks assigns because it's folded to 0!
    helper_test_op([], lambda: tor^tor, lambda: ten^ten, forward_only=True)
    helper_test_op([], lambda: tor^0x1337, lambda: ten^0x1337, forward_only=True)
    helper_test_op([], lambda: 0x1337^tor, lambda: 0x1337^ten, forward_only=True)

  def test_and(self):
    data = [[1,-8,1],[32,1,6]]
    tor = torch.tensor(data, dtype=torch.int)
    ten = Tensor(data, dtype=dtypes.int32)
    helper_test_op([], lambda: tor&tor, lambda: ten&ten, forward_only=True)
    helper_test_op([], lambda: tor&0x1337, lambda: ten&0x1337, forward_only=True)
    helper_test_op([], lambda: 0x1337&tor, lambda: 0x1337&ten, forward_only=True)
    helper_test_op([], lambda: (tor&12)&tor, lambda: (ten&12)&ten, forward_only=True)

    data = [[True, True, False, False], [True, False, True, False]]
    tor0, tor1 = torch.tensor(data[0], dtype=torch.bool),  torch.tensor(data[1], dtype=torch.bool)
    ten0, ten1 = Tensor(data[0], dtype=dtypes.bool), Tensor(data[1], dtype=dtypes.bool)
    helper_test_op([], lambda: tor0&tor1, lambda: ten0&ten1, forward_only=True)

    helper_test_op(None, lambda x: (1 < x) & (x < 2), forward_only=True, vals=[[1.2, 1.2, 1.2, 3.2]])
    helper_test_op([(3000,)]*10, lambda *xs: (sum(xs[1:], xs[0]) > 5) & (xs[0] < 0.9), forward_only=True)

    if not COMPILE_ONLY:
      np.testing.assert_equal((Tensor(2**64-1, dtype=dtypes.uint64) & 0xFFFFFFFF).numpy(), 0xFFFFFFFF)

  def test_or(self):
    data = [[1,-8,1],[32,1,6]]
    tor = torch.tensor(data, dtype=torch.int)
    ten = Tensor(data, dtype=dtypes.int32)
    helper_test_op([], lambda: tor|tor, lambda: ten|ten, forward_only=True)
    helper_test_op([], lambda: tor|0x1337, lambda: ten|0x1337, forward_only=True)
    helper_test_op([], lambda: 0x1337|tor, lambda: 0x1337|ten, forward_only=True)

    data = [[True, True, False, False], [True, False, True, False]]
    tor0, tor1 = torch.tensor(data[0], dtype=torch.bool),  torch.tensor(data[1], dtype=torch.bool)
    ten0, ten1 = Tensor(data[0], dtype=dtypes.bool), Tensor(data[1], dtype=dtypes.bool)
    helper_test_op([], lambda: tor0|tor1, lambda: ten0|ten1, forward_only=True)

  def test_bitwise_not(self):
    data = [[1,-8,1],[32,1,6]]
    tor = torch.tensor(data, dtype=torch.int)
    ten = Tensor(data, dtype=dtypes.int32)
    helper_test_op([], lambda: tor.bitwise_not(), lambda: ten.bitwise_not(), forward_only=True)
    helper_test_op([], lambda: ~tor, lambda: ~ten, forward_only=True)

    data = [[True, False]]
    tor = torch.tensor(data, dtype=torch.bool)
    ten = Tensor(data, dtype=dtypes.bool)
    helper_test_op([], lambda: tor.bitwise_not(), lambda: ten.bitwise_not(), forward_only=True)
    helper_test_op([], lambda: ~tor, lambda: ~ten, forward_only=True)

  def test_lshift(self):
    data = [[0,1,2],[1<<8,1<<16,1<<31-1]]
    tor = torch.tensor(data, dtype=torch.int)
    ten = Tensor(data, dtype=dtypes.uint32)
    # cast to int32 because torch does not support uint32
    helper_test_op([], lambda: tor << 0, lambda: (ten << 0).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor << 2, lambda: (ten << 2).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor << 31, lambda: (ten << 31).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor << torch.tensor([0,2,4]).int(),
                   lambda: (ten << Tensor([0,2,4], dtype=dtypes.uint32)).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor.bitwise_left_shift(2), lambda: ten.lshift(2).cast(dtypes.int32), forward_only=True)
    self.helper_test_exception([], lambda: torch.tensor([1.0]) << 2, lambda: (Tensor([1.0]) << 2).realize(), expected=RuntimeError)
    self.helper_test_exception([], lambda: tor << torch.tensor([1.0]), lambda: (ten << Tensor([1.0])).realize(), expected=RuntimeError)
    self.helper_test_exception([], lambda: tor << 1.0, lambda: (ten << 1.0).realize(), expected=RuntimeError)

  def test_rshift(self):
    data = [[0,1,2],[1<<8,1<<16,1<<31-1]]
    tor = torch.tensor(data, dtype=torch.int)
    ten = Tensor(data, dtype=dtypes.uint32)
    # cast to int32 because torch does not support uint32
    helper_test_op([], lambda: tor >> 0, lambda: (ten >> 0).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor >> 2, lambda: (ten >> 2).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor >> 31, lambda: (ten >> 31).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor >> torch.tensor([0,2,4]).int(),
                   lambda: (ten >> Tensor([0,2,4], dtype=dtypes.uint32)).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor.bitwise_right_shift(2), lambda: ten.rshift(2).cast(dtypes.int32), forward_only=True)
    self.helper_test_exception([], lambda: torch.tensor([4.0]) >> 1, lambda: (Tensor([4.0]) >> 1).realize(), expected=RuntimeError)
    self.helper_test_exception([], lambda: tor >> torch.tensor([1.0]), lambda: (ten >> Tensor([1.0])).realize(), expected=RuntimeError)

  def test_lshift_signed(self):
    data = [[-1, -3, 1, 7], [0, -2147483648, 2147483647, -1]]
    tor = torch.tensor(data, dtype=torch.int32)
    ten = Tensor(data, dtype=dtypes.int)
    helper_test_op([], lambda: tor << 0, lambda: ten << 0, forward_only=True)
    helper_test_op([], lambda: tor << 2, lambda: ten << 2, forward_only=True)
    helper_test_op([], lambda: tor << 8, lambda: ten << 8, forward_only=True)
    helper_test_op([], lambda: tor << 31, lambda: ten << 31, forward_only=True)
    helper_test_op([], lambda: tor << torch.tensor([0,2,8,31]).int(), lambda: ten << Tensor([0,2,8,31], dtype=dtypes.int), forward_only=True)

  def test_rshift_signed(self):
    data = [[-1, -3, 1, 7], [0, -2147483648, 2147483647, -1]]
    tor = torch.tensor(data, dtype=torch.int32)
    ten = Tensor(data, dtype=dtypes.int)
    helper_test_op([], lambda: tor >> 0, lambda: ten >> 0, forward_only=True)
    helper_test_op([], lambda: tor >> 2, lambda: ten >> 2, forward_only=True)
    helper_test_op([], lambda: tor >> 8, lambda: ten >> 8, forward_only=True)
    helper_test_op([], lambda: tor >> 31, lambda: ten >> 31, forward_only=True)
    helper_test_op([], lambda: tor >> torch.tensor([0,2,8,31]).int(), lambda: ten >> Tensor([0,2,8,31], dtype=dtypes.int), forward_only=True)

  def test_idiv_shift_rewrite_negative(self):
    a = Tensor(-5).div(2, rounding_mode="trunc").item()
    b = Tensor(-5).contiguous().div(2, rounding_mode="trunc").item()
    self.assertEqual(a, b)
    self.assertEqual(Tensor(-1).contiguous().div(4, rounding_mode="trunc").item(), 0)  # NOTE this is trunc-div behaviour

  @Context(TRANSCENDENTAL=2)
  def test_sin_transcendental(self): self.test_sin()
  @Context(TRANSCENDENTAL=2)
  def test_cos_transcendental(self): self.test_cos()
  @Context(TRANSCENDENTAL=2)
  def test_tan_transcendental(self): self.test_tan()
  @Context(TRANSCENDENTAL=2)
  def test_exp_transcendental(self): self.test_exp()
  @Context(TRANSCENDENTAL=2)
  def test_log_transcendental(self): self.test_log()

  @unittest.skipIf(DEV.renderer == "NAK", "MUFU.SIN is not accurate enough")
  def test_sin(self):
    helper_test_op([(45,65)], lambda x: x.sin())
    helper_test_op([()], lambda x: x.sin())
    # works on real CUDA but not CI
    if not ((DEV.interface.startswith("MOCK") and Device.DEFAULT == "NV") or Device.DEFAULT == "WEBGPU"):
      helper_test_op(None, lambda x: x.sin(), vals=[[math.nan, math.inf, -math.inf, 0.0]])
      helper_test_op(None, lambda x: x.sin(), vals=[[1e1, 1e2, 1e3, 1e4, 1e5, 1e6, -1e1, -1e2, -1e3, -1e4, -1e5, -1e6]],
                    atol=3e-3, rtol=3e-3, grad_atol=3e-3, grad_rtol=3e-3)
  @unittest.skipIf(DEV.renderer == "NAK", "MUFU.SIN is not accurate enough")
  def test_cos(self):
    helper_test_op([(45,65)], lambda x: x.cos())
    helper_test_op([], lambda: torch.tensor(2.0).cos(), lambda: Tensor(2).cos(), forward_only=True)
    helper_test_op([()], lambda x: x.cos())
    if not ((DEV.interface.startswith("MOCK") and Device.DEFAULT == "NV") or Device.DEFAULT == "WEBGPU"):
      helper_test_op(None, lambda x: x.cos(), vals=[[math.nan, math.inf, -math.inf, 0.0]])
      helper_test_op(None, lambda x: x.cos(), vals=[[1e1, 1e2, 1e3, 1e4, 1e5, 1e6, -1e1, -1e2, -1e3, -1e4, -1e5, -1e6]],
                    atol=3e-3, rtol=3e-3, grad_atol=3e-3, grad_rtol=3e-3)
  @unittest.skipIf(DEV.renderer == "NAK", "MUFU.SIN is not accurate enough")
  def test_tan(self):
    # NOTE: backward has much higher diff with input close to pi/2 and -pi/2
    helper_test_op([(45,65)], lambda x: x.tan(), low=-1.5, high=1.5)
    helper_test_op([(45,65)], lambda x: x.tan(), low=-5, high=5)
    helper_test_op([()], lambda x: x.tan())
    if not ((DEV.interface.startswith("MOCK") and Device.DEFAULT == "NV") or Device.DEFAULT == "WEBGPU"):
      helper_test_op(None, lambda x: x.tan(), vals=[[math.nan, math.inf, -math.inf, 0.0]])
      helper_test_op(None, lambda x: x.tan(), vals=[[1e1, 1e2, 1e3, 1e4, 1e5, 1e6, -1e1, -1e2, -1e3, -1e4, -1e5, -1e6]],
                    atol=3e-3, rtol=3e-3, grad_atol=3e-3, grad_rtol=3e-3)

  def test_asin(self):
    helper_test_op([(45,65)], lambda x: x.asin(), low=-1, high=1)
    helper_test_op([(45,65)], lambda x: x.asin(), low=-300, high=-297)
    helper_test_op([(45,65)], lambda x: x.asin(), low=300, high=303)
    helper_test_op(None, lambda x: x.asin(), vals=[[-0.5, 0., 0.5]])
  def test_acos(self):
    # high grad atol
    helper_test_op([(45,65)], lambda x: x.acos(), low=-1, high=1)
    helper_test_op([(45,65)], lambda x: x.acos(), low=-300, high=-297)
    helper_test_op([(45,65)], lambda x: x.acos(), low=300, high=303)
    helper_test_op(None, lambda x: x.acos(), vals=[[-0.5, 0., 0.5]])
  def test_atan(self):
    helper_test_op([(45,65)], lambda x: x.atan())
    helper_test_op([(45,65)], lambda x: x.atan(), low=-300, high=-297)
    helper_test_op([(45,65)], lambda x: x.atan(), low=300, high=303)
    helper_test_op(None, lambda x: x.atan(), vals=[[-0.5, 0., 0.5]])

  def test_relu(self):
    helper_test_op([(64,64)], lambda x: x.relu())
    helper_test_op([()], lambda x: x.relu())
  def test_relu_exact(self):
    helper_test_op(None, lambda x: x.relu(), vals=[[-1.,0,1]])
  def test_relu_maximum_exact(self):
    helper_test_op(None, lambda x: torch.maximum(x, torch.zeros_like(x, requires_grad=False)), lambda x: Tensor.maximum(x, 0), vals=[[-1.,0,1]])
  def test_leaky_relu(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.leaky_relu(x,0.01), Tensor.leaky_relu)
    helper_test_op([()], lambda x: torch.nn.functional.leaky_relu(x,0.01), Tensor.leaky_relu)
  def test_celu(self):
    for val in range(1, 5):
      helper_test_op([(45,65)], lambda x: torch.nn.functional.celu(x,val), lambda x: x.celu(val))
      helper_test_op([(3,3)], lambda x: torch.nn.functional.celu(x,val), lambda x: x.celu(val), low=300, high=400)
      helper_test_op([()], lambda x: torch.nn.functional.celu(x,val), lambda x: x.celu(val))
  def test_selu(self):
    helper_test_op([(45,65)], torch.nn.functional.selu, Tensor.selu)
    helper_test_op([(3,3)], torch.nn.functional.selu, Tensor.selu, low=300, high=400)
    helper_test_op(None, torch.nn.functional.selu, Tensor.selu, vals=[[-1.,0.,1.]], atol=0, rtol=1e-6)
    helper_test_op([()], torch.nn.functional.selu, Tensor.selu)
  def test_silu(self):
    helper_test_op([(45,65)], torch.nn.functional.silu, Tensor.silu)
    helper_test_op([()], torch.nn.functional.silu, Tensor.silu)
  def test_swish(self):
    helper_test_op([(45,65)], torch.nn.functional.silu, Tensor.swish)
    helper_test_op([()], torch.nn.functional.silu, Tensor.swish)

  def test_abs(self):
    helper_test_op([(45,65)], torch.abs, Tensor.abs)
    helper_test_op([()], torch.abs, Tensor.abs)
  def test_abs_exact(self):
    for v in [-1., -0., 0., 1., math.inf, -math.inf, math.nan, -math.nan]:
      # abs(nan) gradient is undefined: torch=0, tinygrad=1, jax=-1
      helper_test_op(None, torch.abs, Tensor.abs, vals=[[v]], forward_only=math.isnan(v))

  def test_log(self):
    helper_test_op([(45,65)], torch.log, Tensor.log)
    helper_test_op(None, torch.log, Tensor.log, vals=[[math.inf, -math.inf, math.nan]])
    helper_test_op([()], torch.log, Tensor.log)
  def test_log10(self):
    helper_test_op([(45,65)], torch.log10, Tensor.log10)
    helper_test_op(None, torch.log10, Tensor.log10, vals=[[math.inf, -math.inf, math.nan]])
    helper_test_op([()], torch.log10, Tensor.log10)
  def test_log2(self):
    helper_test_op([(45,65)], torch.log2, Tensor.log2)
    helper_test_op(None, torch.log2, Tensor.log2, vals=[[math.inf, -math.inf, math.nan]])
    helper_test_op([()], torch.log2, Tensor.log2)

  def test_exp(self):
    helper_test_op([(45,65)], torch.exp, Tensor.exp)
    # int scalar input: the python coefficients in the decomposition must stay float
    helper_test_op([], lambda: torch.tensor(2.0).exp(), lambda: Tensor(2).exp(), forward_only=True)
    helper_test_op(None, torch.exp, Tensor.exp, vals=[[math.inf, -math.inf, math.nan]])
    helper_test_op([()], torch.exp, Tensor.exp)
  def test_exp2(self):
    helper_test_op([(45,65)], torch.exp2, Tensor.exp2)
    helper_test_op(None, torch.exp2, Tensor.exp2, vals=[[math.inf, -math.inf, math.nan]])
    helper_test_op([()], torch.exp2, Tensor.exp2)

  def test_sign(self):
    helper_test_op([(45,65)], torch.sign, Tensor.sign)
    helper_test_op([()], torch.sign, Tensor.sign)
  def test_sign_exact(self):
    helper_test_op(None, torch.sign, Tensor.sign, vals=[[-1.,0,1]])

  def test_copysign(self):
    helper_test_op([(45,65), (45,65)], torch.copysign, Tensor.copysign)
    helper_test_op([(45,65), (45,1)], torch.copysign, Tensor.copysign)
    helper_test_op([(45,1), (1,65)], torch.copysign, Tensor.copysign)
    helper_test_op([(), ()], torch.copysign, Tensor.copysign)

  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "fails locally")
  def test_copysign_exact(self):
    # NOTE: -nan (negative nan) is not tested because we can't detect its sign bit without bitcast
    v = [-1., -0., 0., 1., math.inf, -math.inf, math.nan]
    for i in v:
      for j in v:
        # torch returns nan gradient for copysign at inf, but mathematically (and per jax) it's ±1
        helper_test_op(None, torch.copysign, Tensor.copysign, vals=[[i], [j]], forward_only=math.isinf(i) or math.isnan(i))

  def test_logaddexp(self):
    helper_test_op([(45,65), (45,65)], torch.logaddexp, Tensor.logaddexp)
    helper_test_op(None, torch.logaddexp, Tensor.logaddexp, vals=[[-1.], [-1.0, 2, 3]])
    helper_test_op(None, torch.logaddexp, Tensor.logaddexp, vals=[[-100.0, -200, -300], [-1.0, 2, 3]])
    helper_test_op(None, torch.logaddexp, Tensor.logaddexp, vals=[[1.0, 2000, 30000], [-1.0, 2, 3]])
    helper_test_op(None, torch.logaddexp, Tensor.logaddexp, vals=[[-math.inf, math.inf, 1.0, -math.inf], [-math.inf, math.inf, -math.inf, 1.0]])

  def test_softsign(self):
    helper_test_op([(45,65)], torch.nn.functional.softsign, Tensor.softsign)
    helper_test_op([()], torch.nn.functional.softsign, Tensor.softsign)
  def test_softsign_exact(self):
    helper_test_op(None, torch.nn.functional.softsign, Tensor.softsign, vals=[[-1.,0,1]])

  def test_sigmoid(self):
    helper_test_op([(45,65)], torch.sigmoid, Tensor.sigmoid)
    helper_test_op([()], torch.sigmoid, Tensor.sigmoid)
  def test_sigmoid_extreme(self):
    helper_test_op([(45,65)], torch.sigmoid, Tensor.sigmoid, low=300, high=400)
    helper_test_op([(45,65)], torch.sigmoid, Tensor.sigmoid, low=-400, high=-300)
    x = Tensor([300.0])
    self.assertAlmostEqual(x.sigmoid()[0].gradient(x)[0].item(), 0.0)
    x = Tensor([-300.0])
    self.assertAlmostEqual(x.sigmoid()[0].gradient(x)[0].item(), 0.0)
  def test_sigmoid_alt_extreme(self):
    def sigmoid(x:Tensor): return x.exp() / (1 + x.exp())
    x = Tensor([300.0])
    self.assertAlmostEqual(sigmoid(x)[0].gradient(x)[0].item(), 0.0)
    x = Tensor([-300.0])
    self.assertAlmostEqual(sigmoid(x)[0].gradient(x)[0].item(), 0.0)

  def test_logsigmoid(self):
    helper_test_op([(45,65)], torch.nn.functional.logsigmoid, Tensor.logsigmoid)
    helper_test_op([()], torch.nn.functional.logsigmoid, Tensor.logsigmoid)

  def test_hardsigmoid(self):
    helper_test_op([(45,65)], torch.nn.functional.hardsigmoid, Tensor.hardsigmoid)
    helper_test_op([()], torch.nn.functional.hardsigmoid, Tensor.hardsigmoid)
  def test_hardsigmoid_extreme(self):
    helper_test_op([(45,65)], torch.nn.functional.hardsigmoid, Tensor.hardsigmoid, low=300, high=400)
    helper_test_op([(45,65)], torch.nn.functional.hardsigmoid, Tensor.hardsigmoid, low=-400, high=-300)
    helper_test_op(None, torch.nn.functional.hardsigmoid, Tensor.hardsigmoid, vals=[[1e7, 1e8, 2.68e8, 1e9]])
    helper_test_op(None, torch.nn.functional.hardsigmoid, Tensor.hardsigmoid, vals=[[-3.1, -3., -2.9, 2.9, 3., 3.1]])
  def test_softplus(self):
    helper_test_op([(45,65)], torch.nn.functional.softplus, Tensor.softplus, grad_atol=1e-6)
    helper_test_op([(45,65)], lambda t: torch.nn.functional.softplus(t, beta=3), lambda t: Tensor.softplus(t, beta=3), grad_atol=1e-6)
    helper_test_op([(45,65)], lambda t: torch.nn.functional.softplus(t, beta=1/3), lambda t: Tensor.softplus(t, beta=1/3), grad_atol=1e-6)
    helper_test_op([(45,65)], torch.nn.functional.softplus, Tensor.softplus, grad_atol=1e-6, low=300, high=400)
    helper_test_op([(45,65)], torch.nn.functional.softplus, Tensor.softplus, grad_atol=1e-6, low=-400, high=-300)
    helper_test_op([()], torch.nn.functional.softplus, Tensor.softplus, grad_atol=1e-6)
    helper_test_op(None, torch.nn.functional.softplus, Tensor.softplus, vals=[[-math.inf, math.inf, 0.0]], forward_only=True)

  def test_erf(self):
    helper_test_op([(45,65)], torch.erf, Tensor.erf)
    helper_test_op([(45,65)], torch.erf, Tensor.erf, low=300, high=400)
    helper_test_op([(45,65)], torch.erf, Tensor.erf, low=-400, high=-300)
    helper_test_op(None, torch.erf, Tensor.erf, vals=[[-1., 0., 1.]])
    helper_test_op([()], torch.erf, Tensor.erf)

  def test_gelu(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.gelu(x, approximate="tanh"), lambda x: Tensor.gelu(x, approximate="tanh"))
    helper_test_op([(45,65)], lambda x: torch.nn.functional.gelu(x, approximate="none"), lambda x: Tensor.gelu(x, approximate="none"))
  def test_gelu_extreme(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.gelu(x, approximate="tanh"), lambda x: Tensor.gelu(x, approximate="tanh"),
                   low=300, high=400)
    helper_test_op([(45,65)], lambda x: torch.nn.functional.gelu(x, approximate="tanh"), lambda x: Tensor.gelu(x, approximate="tanh"),
                   low=-400, high=-300)
    helper_test_op([(45,65)], lambda x: torch.nn.functional.gelu(x, approximate="none"), lambda x: Tensor.gelu(x, approximate="none"),
                   low=300, high=400)
    helper_test_op([(45,65)], lambda x: torch.nn.functional.gelu(x, approximate="none"), lambda x: Tensor.gelu(x, approximate="none"),
                   low=-400, high=-300)
  def test_quick_gelu(self):
    helper_test_op([(45,65)], lambda x: x * torch.sigmoid(1.702 * x), Tensor.quick_gelu)
    helper_test_op([()], lambda x: x * torch.sigmoid(1.702 * x), Tensor.quick_gelu)
  def test_quick_gelu_extreme(self):
    helper_test_op([(45,65)], lambda x: x * torch.sigmoid(1.702 * x), Tensor.quick_gelu, low=300, high=400)
    helper_test_op([(45,65)], lambda x: x * torch.sigmoid(1.702 * x), Tensor.quick_gelu, low=-400, high=-300)

  def test_relu6(self):
    helper_test_op([(45,65)], torch.nn.functional.relu6, Tensor.relu6)
    helper_test_op([()], torch.nn.functional.relu6, Tensor.relu6)
    helper_test_op(None, torch.nn.functional.relu6, Tensor.relu6, vals=[[6.71089e7, 2.68435e8, 1e9]])
    helper_test_op(None, torch.nn.functional.relu6, Tensor.relu6, vals=[[0., 6.]])
  def test_hardswish(self):
    helper_test_op([(45,65)], torch.nn.functional.hardswish, Tensor.hardswish, grad_atol=1e-6)
    helper_test_op([()], torch.nn.functional.hardswish, Tensor.hardswish, grad_atol=1e-6)
    helper_test_op(None, torch.nn.functional.hardswish, Tensor.hardswish, vals=[[-3., 3.]], grad_atol=1e-6)
  def test_mish(self):
    helper_test_op([(45,65)], torch.nn.functional.mish, Tensor.mish)
    helper_test_op([()], torch.nn.functional.mish, Tensor.mish)

  def test_small_cumsum(self):
    helper_test_op([(10)], lambda x: torch.cumsum(x, dim=0), lambda x: Tensor.cumsum(x, axis=0))
  @slow_test
  def test_simple_cumsum(self):
    helper_test_op([(512)], lambda x: torch.cumsum(x, dim=0), lambda x: Tensor.cumsum(x, axis=0))
    helper_test_op([(1022)], lambda x: torch.cumsum(x, dim=0), lambda x: Tensor.cumsum(x, axis=0))
  @slow_test
  def test_cumsum(self):
    helper_test_op([()], lambda x: torch.cumsum(x, dim=0), lambda x: Tensor.cumsum(x, axis=0))
    self.helper_test_exception([()], lambda x: torch.cumsum(x, dim=1), lambda x: Tensor.cumsum(x, axis=1), expected=IndexError)
    helper_test_op([(20,)], lambda x: torch.cumsum(x, dim=0), lambda x: Tensor.cumsum(x, axis=0))
    self.helper_test_exception([(20,)], lambda x: torch.cumsum(x, dim=1), lambda x: Tensor.cumsum(x, axis=1), expected=IndexError)
    self.helper_test_exception([(20,)], lambda x: torch.cumsum(x, dim=-2), lambda x: Tensor.cumsum(x, axis=-2), expected=IndexError)
    helper_test_op([(20,30)], lambda x: torch.cumsum(x, dim=0), lambda x: Tensor.cumsum(x, axis=0))
    helper_test_op([(20,30)], lambda x: torch.cumsum(x, dim=1), lambda x: Tensor.cumsum(x, axis=1))
    helper_test_op([(20,30,40)], lambda x: torch.cumsum(x, dim=2), lambda x: Tensor.cumsum(x, axis=2))
    helper_test_op([(20,30,40)], lambda x: torch.cumsum(x, dim=-1), lambda x: Tensor.cumsum(x, axis=-1))
  def test_cumsum_zero_axis(self):
    helper_test_op([(2,0,4)], lambda x: torch.cumsum(x, dim=1), lambda x: Tensor.cumsum(x, axis=1))
    helper_test_op([(0,3)], lambda x: torch.cumsum(x, dim=0), lambda x: Tensor.cumsum(x, axis=0))
    helper_test_op([(2,3,0)], lambda x: torch.cumsum(x, dim=2), lambda x: Tensor.cumsum(x, axis=2))

  def test_small_cumprod(self):
    helper_test_op([(10)],lambda x: torch.cumprod(x, dim=0),lambda x: Tensor.cumprod(x, axis=0))
  @slow_test
  def test_simple_cumprod(self):
    helper_test_op([(512)],lambda x: torch.cumprod(x, dim=0),lambda x: Tensor.cumprod(x, axis=0))
    helper_test_op([(1022)],lambda x: torch.cumprod(x, dim=0),lambda x: Tensor.cumprod(x, axis=0))
  def test_cumprod_zero_axis(self):
    helper_test_op([(2, 0, 4)],lambda x: torch.cumprod(x, dim=1),lambda x: Tensor.cumprod(x, axis=1))
    helper_test_op([(0, 3)],lambda x: torch.cumprod(x, dim=0),lambda x: Tensor.cumprod(x, axis=0))
    helper_test_op([(2, 3, 0)],lambda x: torch.cumprod(x, dim=2),lambda x: Tensor.cumprod(x, axis=2))

  def test_small_cummax(self):
    helper_test_op([(10)], lambda x: torch.cummax(x, dim=0).values, lambda x: Tensor.cummax(x, axis=0)[0])
    helper_test_op([(10)], lambda x: torch.cummax(x, dim=0).indices.int(), lambda x: Tensor.cummax(x, axis=0)[1], forward_only=True)
    helper_test_op(None, lambda x: torch.cummax(x, dim=0).indices.int(), lambda x: Tensor.cummax(x, axis=0)[1], vals=[[1,1,0,1]], forward_only=True)
  @slow_test
  def test_simple_cummax(self):
    helper_test_op([(512)], lambda x: torch.cummax(x, dim=0).values, lambda x: Tensor.cummax(x, axis=0)[0])
    helper_test_op([(512)], lambda x: torch.cummax(x, dim=0).indices.int(), lambda x: Tensor.cummax(x, axis=0)[1], forward_only=True)
    helper_test_op([(1022)], lambda x: torch.cummax(x, dim=0).values, lambda x: Tensor.cummax(x, axis=0)[0])
    helper_test_op([(1022)], lambda x: torch.cummax(x, dim=0).indices.int(), lambda x: Tensor.cummax(x, axis=0)[1], forward_only=True)
  def test_cummax_zero_axis(self):
    helper_test_op([(2,0,4)], lambda x: torch.cummax(x, dim=1).values, lambda x: Tensor.cummax(x, axis=1)[0])
    helper_test_op([(2,0,4)], lambda x: torch.cummax(x, dim=1).indices.int(), lambda x: Tensor.cummax(x, axis=1)[1], forward_only=True)
    helper_test_op([(0,3)], lambda x: torch.cummax(x, dim=0).values, lambda x: Tensor.cummax(x, axis=0)[0])
    helper_test_op([(0,3)], lambda x: torch.cummax(x, dim=0).indices.int(), lambda x: Tensor.cummax(x, axis=0)[1], forward_only=True)
    helper_test_op([(2,3,0)], lambda x: torch.cummax(x, dim=2).values, lambda x: Tensor.cummax(x, axis=2)[0])
    helper_test_op([(2,3,0)], lambda x: torch.cummax(x, dim=2).indices.int(), lambda x: Tensor.cummax(x, axis=2)[1], forward_only=True)

  def test_small_cummin(self):
    helper_test_op([(10)], lambda x: torch.cummin(x, dim=0).values, lambda x: Tensor.cummin(x, axis=0)[0])
    helper_test_op([(10)], lambda x: torch.cummin(x, dim=0).indices.int(), lambda x: Tensor.cummin(x, axis=0)[1], forward_only=True)
  @slow_test
  def test_simple_cummin(self):
    helper_test_op([(512)], lambda x: torch.cummin(x, dim=0).values, lambda x: Tensor.cummin(x, axis=0)[0])
    helper_test_op([(512)], lambda x: torch.cummin(x, dim=0).indices.int(), lambda x: Tensor.cummin(x, axis=0)[1], forward_only=True)
    helper_test_op([(1022)], lambda x: torch.cummin(x, dim=0).values, lambda x: Tensor.cummin(x, axis=0)[0])
    helper_test_op([(1022)], lambda x: torch.cummin(x, dim=0).indices.int(), lambda x: Tensor.cummin(x, axis=0)[1], forward_only=True)
  def test_cummin_zero_axis(self):
    helper_test_op([(2,0,4)], lambda x: torch.cummin(x, dim=1).values, lambda x: Tensor.cummin(x, axis=1)[0])
    helper_test_op([(2,0,4)], lambda x: torch.cummin(x, dim=1).indices.int(), lambda x: Tensor.cummin(x, axis=1)[1], forward_only=True)
    helper_test_op([(0,3)], lambda x: torch.cummin(x, dim=0).values, lambda x: Tensor.cummin(x, axis=0)[0])
    helper_test_op([(0,3)], lambda x: torch.cummin(x, dim=0).indices.int(), lambda x: Tensor.cummin(x, axis=0)[1], forward_only=True)
    helper_test_op([(2,3,0)], lambda x: torch.cummin(x, dim=2).values, lambda x: Tensor.cummin(x, axis=2)[0])
    helper_test_op([(2,3,0)], lambda x: torch.cummin(x, dim=2).indices.int(), lambda x: Tensor.cummin(x, axis=2)[1], forward_only=True)

  def test_argmax(self):
    # check if it returns the first index for multiple occurrences
    helper_test_op(None, lambda x: x.argmax().type(torch.int32), lambda x: x.argmax(), forward_only=True, vals=[[2, 2]])
    helper_test_op(None, lambda x: x.argmax().type(torch.int32), lambda x: x.argmax(), forward_only=True, vals=[[1, 2, 2]])
    helper_test_op([(10,20)], lambda x: x.argmax().type(torch.int32), lambda x: x.argmax(), forward_only=True)
    helper_test_op([(10,20)], lambda x: x.argmax(0, False).type(torch.int32), lambda x: x.argmax(0, False), forward_only=True)
    helper_test_op([(10,20)], lambda x: x.argmax(1, False).type(torch.int32), lambda x: x.argmax(1, False), forward_only=True)
    helper_test_op([(10,20)], lambda x: x.argmax(1, True).type(torch.int32), lambda x: x.argmax(1, True), forward_only=True)
    # regression test for bitwise_not then argmax
    helper_test_op(None, lambda x: (~x).argmax().type(torch.int32), lambda x: (~x).argmax(), forward_only=True, vals=[[2, 2]])

    helper_test_op(None, lambda x: x.argmax().type(torch.int32), lambda x: x.argmax(), forward_only=True, vals=[[0, -2**31]])
    helper_test_op(None, lambda x: x.argmax().type(torch.int32), lambda x: x.argmax(), forward_only=True, vals=[[-2**31, 0]])
    # NOTE: torch does not support this on bool
    helper_test_op(None, lambda x: x.type(torch.int32).argmax().type(torch.int32), lambda x: x.argmax(), forward_only=True, vals=[[False, True]])
    helper_test_op(None, lambda x: x.type(torch.int32).argmax().type(torch.int32), lambda x: x.argmax(), forward_only=True, vals=[[True, False]])

  def test_argmin(self):
    # check if it returns the first index for multiple occurrences
    helper_test_op(None, lambda x: x.argmin().type(torch.int32), lambda x: x.argmin(), forward_only=True, vals=[[2, 2]])
    helper_test_op(None, lambda x: x.argmin().type(torch.int32), lambda x: x.argmin(), forward_only=True, vals=[[3, 2, 2]])
    helper_test_op([(10,20)], lambda x: x.argmin().type(torch.int32), lambda x: x.argmin(), forward_only=True)
    helper_test_op([(10,20)], lambda x: x.argmin(0, False).type(torch.int32), lambda x: x.argmin(0, False), forward_only=True)
    helper_test_op([(10,20)], lambda x: x.argmin(1, False).type(torch.int32), lambda x: x.argmin(1, False), forward_only=True)
    helper_test_op([(10,20)], lambda x: x.argmin(1, True).type(torch.int32), lambda x: x.argmin(1, True), forward_only=True)

    helper_test_op(None, lambda x: x.argmin().type(torch.int32), lambda x: x.argmin(), forward_only=True, vals=[[0, -2**31]])
    helper_test_op(None, lambda x: x.argmin().type(torch.int32), lambda x: x.argmin(), forward_only=True, vals=[[-2**31, 0]])
    # NOTE: torch does not support this on bool
    helper_test_op(None, lambda x: x.type(torch.int32).argmin().type(torch.int32), lambda x: x.argmin(), forward_only=True, vals=[[False, True]])
    helper_test_op(None, lambda x: x.type(torch.int32).argmin().type(torch.int32), lambda x: x.argmin(), forward_only=True, vals=[[True, False]])

  def test_sort_independent_outputs(self):
    for output in [0, 1]:
      helper_test_op(None, lambda x: x.sort(stable=True).values if output == 0 else x.sort(stable=True).indices.int(),
                     lambda x: x.sort()[output], forward_only=True, vals=[[3., 1., 3., 2.]])

  def test_argsort(self):
    helper_test_op([(8,8,6)], lambda x: torch.argsort(x, dim=1, descending=True, stable=True).type(torch.int32),
                              lambda x: x.argsort(1, True), forward_only=True)

  def test_topk(self):
    def torch_topk(x, k, dim=-1, largest=True):
      values, indices = x.topk(k, dim, largest)
      return values, indices.int()

    helper_test_op([(8,)], lambda x: torch_topk(x, 3), lambda x: x.topk(3), forward_only=True)
    for dim, largest in [(0, True), (1, False)]:
      helper_test_op([(5,5,4)], lambda x: torch_topk(x, 4, dim, largest), lambda x: x.topk(4, dim, largest), forward_only=True)
    # repeated values
    if not COMPILE_ONLY:
      for largest, expected_values, expected_indices in [(True, [1, 1, 1], [0, 1, 3]), (False, [0, 0, 0], [2, 4, 6])]:
        values, indices = Tensor([1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]).topk(3, largest=largest)
        Tensor.realize(values, indices)
        np.testing.assert_equal(values.numpy(), expected_values)
        np.testing.assert_equal(indices.numpy(), expected_indices)
    self.helper_test_exception([(4)], lambda x: x.topk(5), expected=(RuntimeError, ValueError))

  def test_topk_independent_outputs(self):
    for output in [0, 1]:
      helper_test_op(None, lambda x: x.topk(2).values if output == 0 else x.topk(2).indices.int(),
                     lambda x: x.topk(2)[output], forward_only=True, vals=[[3., 1., 4., 2.]])

  @slow_test
  def test_einsum_ellipsis(self):
    """The expected behavior for einsum is described in the PyTorch docs: https://pytorch.org/docs/stable/generated/torch.einsum.html"""
    # test ellipsis
    helper_test_op([(3, 8, 9), (3, 8, 9)], lambda a, b: torch.einsum('...id, ...jd -> ...ij', [a, b]),
               lambda a, b: Tensor.einsum('...id, ...jd -> ...ij', [a, b]))
    # ellipsis will come first in the output before the subscript labels, if rhs is not specified
    helper_test_op([(3, 8, 9), (3, 8, 9)], lambda a, b: torch.einsum('...id, ...jd', [a, b]),
               lambda a, b: Tensor.einsum('...id, ...jd', [a, b]))
    # multiple ellipsis in different operands with different shapes are allowed
    helper_test_op([(2, 3, 4, 5), (5, 2, 4)], lambda a, b: torch.einsum('i...j,ji...->...', [a, b]),
               lambda a, b: Tensor.einsum('i...j,ji...->...', [a, b]))
    # match torch ellipsis handling
    helper_test_op([(8, 7, 16, 16, 16), (8, 7, 16, 16, 16)], lambda a, b: torch.einsum('ij...,ij...->ij', [a, b]),
               lambda a, b: Tensor.einsum('ij...,ij...->ij', [a, b]))
    # multiple ellipsis in one operand are not allowed
    self.helper_test_exception([(2, 3, 4), (2, 3, 4)], lambda a, b: torch.einsum('...ik..., ...jk ->', [a, b]),
                lambda a, b: Tensor.einsum('...ik..., ...jk ->', [a, b]), expected=(RuntimeError, ValueError))
    # multiple ellipsis must broadcast together
    self.helper_test_exception([(2, 3, 4), (2, 3, 4)], lambda a, b: torch.einsum('i...j,ji...->...', [a, b]),
                lambda a, b: Tensor.einsum('i...j,ji...->...', [a, b]), expected=RuntimeError)

  def test_einsum_trace(self):
    # inner product
    helper_test_op([(5,), (5,)], lambda a, b: torch.einsum('i,i', a, b), lambda a, b: Tensor.einsum('i,i', a, b))
    # simple diagonal
    helper_test_op([(4, 4)], lambda a: torch.einsum('ii->i', a), lambda a: Tensor.einsum('ii->i', a))
    # trace (sum of diagonal)
    helper_test_op([(4, 4)], lambda a: torch.einsum('ii->', a), lambda a: Tensor.einsum('ii->', a))
    # batch diagonal
    helper_test_op([(3, 5, 5)], lambda a: torch.einsum('...ii->...i', a), lambda a: Tensor.einsum('...ii->...i', a))
    # batch trace
    helper_test_op([(3, 5, 5)], lambda a: torch.einsum('...ii->...', a), lambda a: Tensor.einsum('...ii->...', a))
    # diagonal not at the end
    helper_test_op([(3, 3, 4)], lambda a: torch.einsum('iij->ij', a), lambda a: Tensor.einsum('iij->ij', a))
    helper_test_op([(3, 4, 3)], lambda a: torch.einsum('iji->ij', a), lambda a: Tensor.einsum('iji->ij', a))
    # two repeated letters, and a letter repeated three times
    helper_test_op([(3, 4, 3, 4)], lambda a: torch.einsum('ijij->ji', a), lambda a: Tensor.einsum('ijij->ji', a))
    helper_test_op([(3, 3, 4, 3)], lambda a: torch.einsum('iiji->ij', a), lambda a: Tensor.einsum('iiji->ij', a))

  @unittest.skipIf(IMAGE>0, "no 1d dot for images")
  def test_dot_1d(self):
    helper_test_op([(65), (65)], lambda x,y: x.matmul(y), Tensor.dot)
    helper_test_op([(65), (65,45)], lambda x,y: x.matmul(y), Tensor.dot)
    helper_test_op([(45,65), (65)], lambda x,y: x.matmul(y), Tensor.dot)
    helper_test_op([(8,45,65), (65)], lambda x,y: x.matmul(y), Tensor.dot)
    helper_test_op([(65), (8,65,45)], lambda x,y: x.matmul(y), Tensor.dot)
    self.helper_test_exception([(4), (1,2)], lambda x, y: x.matmul(y), Tensor.dot, expected=RuntimeError)
    self.helper_test_exception([(2,1), (4)], lambda x, y: x.matmul(y), Tensor.dot, expected=RuntimeError)
    self.helper_test_exception([(1), (4)], lambda x, y: x.matmul(y), Tensor.dot, expected=RuntimeError)
  @slow_test
  def test_dot(self):
    helper_test_op([(45,65), (65,100)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-5)
    helper_test_op([(8,45,65), (8,65,100)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-5)
    self.helper_test_exception([(2, 4), (1, 3)], lambda x, y: x.matmul(y), Tensor.dot, expected=RuntimeError)
    self.helper_test_exception([(2, 1), (4, 3)], lambda x, y: x.matmul(y), Tensor.dot, expected=RuntimeError)
    with self.assertRaises(RuntimeError):
      a = Tensor(3.14)
      a.matmul(a)
  def test_mulacc_with_zero_strides(self):
    helper_test_op(
      [],
      lambda: torch.tensor(1.0).reshape((1,1,1)).expand(2,4,3).mul(torch.tensor(1.0).reshape((1,1,1)).expand(2,4,3)).sum(-1),
      lambda: Tensor(1.0).reshape((1,1,1)).expand(2,4,3).mul(Tensor(1.0).reshape((1,1,1)).expand(2,4,3)).sum(-1),
      forward_only=True
    )
    a = [[1.,1.,1.,1.], [1.,1.,1.,1.]]
    b = [1.,1.,1.,1.]
    helper_test_op(
      [],
      lambda: torch.tensor(a).reshape((2,4,1)).expand(2,4,3).mul(torch.tensor(b).reshape((1,4,1)).expand(2,4,3)).sum([0,2]),
      lambda: Tensor(a).reshape((2,4,1)).expand(2,4,3).mul(Tensor(b).reshape((1,4,1)).expand(2,4,3)).sum([0,2]),
      forward_only=True
    )
    helper_test_op(
      [],
      lambda: torch.ones((1,2)).matmul(torch.ones((2,3))), lambda: Tensor.ones((1,2)).dot(Tensor.ones((2,3))),
      forward_only=True
    )

  def test_matmul_simple(self):
    helper_test_op([(4), (4,4)], lambda x,y: x.matmul(y), Tensor.dot)
  def test_matmul(self):
    helper_test_op([(64), (64,99)], lambda x,y: x.matmul(y), Tensor.dot)

  @unittest.skipIf(IMAGE>0, "no batched matmul on images")
  def test_matmul_batched(self):
    helper_test_op([(3), (1,3,3,5)], lambda x,y: x.matmul(y), Tensor.dot)

  @unittest.skipIf(IMAGE>0, "no batched matmul on images")
  def test_matmul_batched_vector(self):
    helper_test_op([(4,3), (1,3,3,5)], lambda x,y: x.matmul(y), Tensor.dot)
  def test_small_gemm(self):
    helper_test_op([(8,8), (8,8)], lambda x,y: x.matmul(y), lambda x,y: x@y)
  def test_9_gemm(self):
    helper_test_op([(9,9), (9,9)], lambda x,y: x.matmul(y), lambda x,y: x@y)
  def test_small_gemm_padded(self):
    helper_test_op([(9,9), (9,9)],
                   lambda x,y: torch.nn.functional.pad(x, (0,7,0,7)).matmul(torch.nn.functional.pad(y, (0,7,0,7))),
                   lambda x,y: x.pad(((0,7),(0,7)))@y.pad(((0,7),(0,7))))
  def test_small_gemm_range(self):
    helper_test_op(None, lambda x,y: x.matmul(y), lambda x,y: x@y, vals=[np.arange(0,64,dtype=np.float32).reshape(8,8),
                                                                         np.arange(64,128,dtype=np.float32).reshape(8,8)])
  def test_small_gemm_eye(self):
    helper_test_op(None, lambda x,y: x.matmul(y), lambda x,y: x@y, vals=[np.eye(8).astype(np.float32), np.eye(8).astype(np.float32)])
  @unittest.skipUnless(dtypes.half in Device[Device.DEFAULT].renderer.supported_dtypes(), "not precise enough when emulating")
  @unittest.skipIf(IMAGE>0, "image does math in float32")
  def test_gemm_fp16(self):
    helper_test_op([(64,64), (64,64)], lambda x,y: x.half().matmul(y.half()), atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3)
  def test_gemm(self):
    helper_test_op([(64,64), (64,64)], lambda x,y: x.matmul(y))
  @slow_test
  def test_big_gemm(self):
    helper_test_op([(256,256), (256,256)], lambda x,y: x.matmul(y), atol=1e-4)
  @unittest.skipIf(IMAGE>0, "no 0 in shape matmul on images")
  def test_gemm_with_zeros_shape(self):
    helper_test_op([(8,8), (8,0)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-7)
    helper_test_op([(0,8), (8,8)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-7)
    helper_test_op([(0,8), (8,0)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-7)
    helper_test_op([(8,0), (0,8)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-7)
    helper_test_op([(0,0), (0,0)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-7)
    helper_test_op([(0), (0,8)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-7)
    helper_test_op([(0), (0)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-7)
  @slow_test
  def test_broadcastdot(self):
    helper_test_op([(10,45,65), (65,45)], lambda x,y: x @ y, Tensor.dot, atol=1e-4)
    with self.assertRaises(RuntimeError):
      a = Tensor(3.14)
      b = Tensor.ones(3,3)
      a @ b
  @slow_test
  def test_multidot(self):
    helper_test_op([(10,45,65), (10,65,45)], lambda x,y: x @ y, Tensor.dot, atol=1e-4)
    helper_test_op([(3,3,45,65), (3,3,65,45)], lambda x,y: x @ y, Tensor.dot, atol=1e-4)

  def test_sum_simple(self):
    helper_test_op(None, lambda x: x.sum(), vals=[[1.,1.]])
  # NOTE: simple test for locals
  # FORWARD_ONLY=1 DEBUG=4 python3 test/test_ops.py TestOps.test_sum_full
  def test_sum_full(self):
    helper_test_op([(16384)], lambda x: x.sum())
  def test_sum_relu(self):
    helper_test_op([(3,4,5)], lambda x: x.relu().sum().relu())
  def test_sum_tiny(self):
    helper_test_op([(4,2,2)], lambda x: x.sum(axis=(0,2)))
  def test_sum(self):
    helper_test_op([(45,3)], lambda x: x.sum())
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=3))
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=(1,3)))
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=(0,2)))
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=(1,2)))
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=1))
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=1, keepdim=True))
    helper_test_op([()], lambda x: x.sum())
    helper_test_op([()], lambda x: x.sum(0))
    helper_test_op([()], lambda x: x.sum(-1))
    helper_test_op([()], lambda x: x.sum(()))
    self.helper_test_exception([(3,4,5,6)], lambda x: x.sum(5), expected=IndexError)
    self.helper_test_exception([()], lambda x: x.sum(1), expected=IndexError)
    self.helper_test_exception([()], lambda x: x.sum((1,)), expected=IndexError)

  def test_sum_dtype_arg(self):
    helper_test_op([(45,3)], lambda x: x.sum(), lambda x: x.sum(dtype=dtypes.float32))
    if dtypes.float64 in Device[Device.DEFAULT].renderer.supported_dtypes():
      helper_test_op([(45,3)], lambda x: x.sum(dtype=torch.float64), lambda x: x.sum(dtype=dtypes.float64))

    with self.assertRaises(AttributeError): Tensor([1.0, 2.0]).sum(dtype="")

  def test_sum_with_zeros_shape(self):
    helper_test_op([(4, 0)], lambda x: x.sum(axis=(0,)))
    helper_test_op([(4, 0)], lambda x: x.sum(axis=(1,)))
    helper_test_op([(4, 0)], lambda x: x.sum(axis=(0,1)))

  def test_prod(self):
    helper_test_op(None, lambda x: x.prod(), vals=[[1.0, 2.0, 3.0]])
    helper_test_op(None, lambda x: x.prod(), vals=[[0.0, 2.0, 3.0]])
    helper_test_op(None, lambda x: x.prod(), vals=[[0.0, 0.0, 3.0]])
    with Context(NOOPT=1): helper_test_op(None, lambda x: x.prod(), vals=[[1.0, 2.0, 3.0]])
    helper_test_op([(3,4,5,6)], lambda x: x.prod(dim=3), lambda x: x.prod(axis=3))
    helper_test_op([(3,4,5,6)], lambda x: x.prod(dim=1), lambda x: x.prod(axis=1))
    helper_test_op([(3,4,5,6)], lambda x: x.prod(dim=1, keepdim=True), lambda x: x.prod(axis=1, keepdim=True))
    helper_test_op([()], lambda x: x.prod())
    helper_test_op([()], lambda x: x.prod(0))
    helper_test_op([()], lambda x: x.prod(-1))

  def test_min(self):
    helper_test_op([(3,3)], lambda x: x.min())
    helper_test_op([(45,3)], lambda x: x.min())
    helper_test_op([(45,3)], lambda x: x.min().mul(0.5))
    helper_test_op([()], lambda x: x.min())

    helper_test_op(None, lambda x: x.min(), forward_only=True, vals=[[0, -2**31]])
    helper_test_op(None, lambda x: x.min(), forward_only=True, vals=[[-2**31, 0]])
    helper_test_op(None, lambda x: x.min(), forward_only=True, vals=[[False, True]])
    helper_test_op(None, lambda x: x.min(), forward_only=True, vals=[[True, False]])

  def test_max(self):
    helper_test_op([(45,3)], lambda x: x.max())
    helper_test_op([(45,3)], lambda x: x.max().mul(0.5))
    helper_test_op(None, lambda x: x.max().mul(0.5), vals=[[[1.0,1.0,0.0,1.0]],])
    helper_test_op([(3,4,5,6)], lambda x: x.max(axis=1)[0], lambda x: x.max(axis=1))
    helper_test_op([()], lambda x: x.max())

    helper_test_op(None, lambda x: x.max(), forward_only=True, vals=[[0, -2**31]])
    helper_test_op(None, lambda x: x.max(), forward_only=True, vals=[[-2**31, 0]])
    helper_test_op(None, lambda x: x.max(), forward_only=True, vals=[[False, True]])
    helper_test_op(None, lambda x: x.max(), forward_only=True, vals=[[True, False]])

  def test_const_reduce(self):
    helper_test_op([(3,3)], lambda x: torch.full_like(x, 2).sum(), lambda x: (x.full_like(2)).sum(), forward_only=True)
    helper_test_op([(3,3)], lambda x: torch.full_like(x, 2).prod(), lambda x: (x.full_like(2)).prod(), forward_only=True)
    helper_test_op([(3,3)], lambda x: torch.full_like(x, 2).max(), lambda x: (x.full_like(2)).max(), forward_only=True)

  def test_any(self):
    helper_test_op([(3,4,5,6)], lambda x: x.any(), forward_only=True)
    helper_test_op(None, lambda x: x.any(), vals=[[True, True]], forward_only=True)
    helper_test_op(None, lambda x: x.any(), vals=[[True, False]], forward_only=True)
    helper_test_op(None, lambda x: x.any(), vals=[[False, False]], forward_only=True)
    helper_test_op([()], lambda x: x.any(), forward_only=True)
  def test_any_axis(self):
    helper_test_op([(3,4,5,6)], lambda x: x.any(axis=(1,2)), forward_only=True)
  def test_any_zero_axis(self):
    helper_test_op([(1,0,3,0,5)], lambda x: x.any(axis=(1,3)), forward_only=True)

  def test_all(self):
    helper_test_op([(3,4,5,6)], lambda x: x.all(), forward_only=True)
    helper_test_op(None, lambda x: x.all(), vals=[[True, True]], forward_only=True)
    helper_test_op(None, lambda x: x.all(), vals=[[True, False]], forward_only=True)
    helper_test_op(None, lambda x: x.all(), vals=[[False, False]], forward_only=True)
    helper_test_op([()], lambda x: x.all(), forward_only=True)
    helper_test_op(None, lambda: torch.ones(2**15).bool().all(), lambda: Tensor.ones(2**15).bool().all(), vals=[], forward_only=True)
  def test_all_axis(self):
    helper_test_op([(3,4,5,6)], lambda x: x.all(axis=(1,2)), forward_only=True)
  def test_all_zero_axis(self):
    helper_test_op([(1,0,3,0,5)], lambda x: x.all(axis=(1,3)), forward_only=True)
  def test_isclose_edge_cases(self):
    for a in [math.inf, -math.inf, math.nan, 0.0]:
      for b in [math.inf, -math.inf, math.nan, 0.0]:
        helper_test_op(None, lambda x,y: x.isclose(y), vals=[[a], [b]], forward_only=True)
        helper_test_op(None, lambda x,y: x.isclose(y, equal_nan=True), vals=[[a], [b]], forward_only=True)

  def test_isclose_scalar(self):
    # torch needs a tensor
    helper_test_op([(3, 4, 5, 6)], lambda x: x.isclose(torch.tensor(1.0)), lambda x: x.isclose(1.0), forward_only=True)
    helper_test_op(None, lambda x: x.isclose(torch.tensor(1.0)), lambda x: x.isclose(1.0),
                   vals=[[1.0, 1.0 + 1e-7, 2.0, math.inf, -math.inf, math.nan]], forward_only=True)

  def test_mean(self):
    helper_test_op([(3,4,5,6)], lambda x: x.mean())
    helper_test_op([()], lambda x: x.mean())
  def test_mean_axis(self):
    helper_test_op([(3,4,5,6)], lambda x: x.mean(axis=(1,2)))
  def test_mean_zero_axis(self):
    helper_test_op([(1,0,3,0,5)], lambda x: x.mean(axis=(1,3)))

  @slow_test
  def test_var(self):
    helper_test_op([(15, 25, 35)], lambda x: x.var())
    helper_test_op([(15, 25, 35)], lambda x: x.var(correction=0))
    helper_test_op([(15, 25, 35)], lambda x: x.var(correction=5))
    helper_test_op(None, lambda x: x.float().var(), lambda x: x.var(), vals=[[1, 2, 3, 4]], forward_only=True)
    # TODO: fix this
    # helper_test_op([(10, 2)], lambda x: x.var(correction=50))
  def test_var_zero_in_axis(self):
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", message="var\\(\\): degrees of freedom is <= 0")
      helper_test_op([(1,0,3,0,5)], lambda x: x.var(axis=(1,3)))
      helper_test_op([(1,0,3,0,5)], lambda x: x.var(axis=(1,3), correction=0))
      helper_test_op([(1,0,3,0,5)], lambda x: x.var(axis=(1,3), correction=5))
  def test_var_one_in_axis(self):
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", message="var\\(\\): degrees of freedom is <= 0")
      # a single element var is 0/0
      if not NAN_UNSAFE: helper_test_op([(1,2,3,1,5)], lambda x: x.var(axis=(0,3)))
      # TODO: fix backward
      if not NAN_UNSAFE: helper_test_op([(1,2,3,1,5)], lambda x: x.var(axis=(0,3), correction=5), forward_only=True)
      helper_test_op([(1,2,3,1,5)], lambda x: x.var(axis=(0,4), correction=5), forward_only=True)
    helper_test_op([(1,)], lambda x: x.var(axis=(0,), correction=0))
    helper_test_op([(1,2,3,1,5)], lambda x: x.var(axis=(0,3), correction=0))
    helper_test_op([(1,2,3,1,5)], lambda x: x.var(axis=(0,4)))
    helper_test_op([(1,2,3,1,5)], lambda x: x.var(axis=(0,4), correction=0))
  def test_var_keepdim(self):
    helper_test_op([(15, 25, 35)], lambda x: x.var(keepdim=True))
    helper_test_op([(15, 25, 35)], lambda x: x.var(0, keepdim=True, correction=0))

  @slow_test
  def test_std(self):
    helper_test_op([(15, 25, 35)], lambda x: x.std())
    helper_test_op([(15, 25, 35)], lambda x: x.std(correction=0))
    helper_test_op([(15, 25, 35)], lambda x: x.std(correction=5))
  def test_std_zero_in_axis(self):
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom is <= 0")
      helper_test_op([(1,0,3,0,5)], lambda x: x.std(axis=(1,3)))
      helper_test_op([(1,0,3,0,5)], lambda x: x.std(axis=(1,3), correction=0))
      helper_test_op([(1,0,3,0,5)], lambda x: x.std(axis=(1,3), correction=5))
  def test_std_one_in_axis(self):
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom is <= 0")
      if not NAN_UNSAFE: helper_test_op([(1,2,3,1,5)], lambda x: x.std(axis=(0,3)))
      if not NAN_UNSAFE: helper_test_op([(1,2,3,1,5)], lambda x: x.std(axis=(0,3), correction=5))
      helper_test_op([(1,2,3,1,5)], lambda x: x.std(axis=(0,4), correction=5))
    # TODO: fix backward
    helper_test_op([(1,)], lambda x: x.std(axis=(0,), correction=0), forward_only=True)
    helper_test_op([(1,2,3,1,5)], lambda x: x.std(axis=(0,3), correction=0), forward_only=True)
    helper_test_op([(1,2,3,1,5)], lambda x: x.std(axis=(0,4)))
    helper_test_op([(1,2,3,1,5)], lambda x: x.std(axis=(0,4), correction=0))
  def test_std_keepdim(self):
    helper_test_op([(15, 25, 35)], lambda x: x.std(keepdim=True))
    helper_test_op([(15, 25, 35)], lambda x: x.std(0, keepdim=True, correction=0))
  @slow_test
  def test_std_mean(self):
    helper_test_op([(15,25,35)], lambda x: torch.stack(torch.std_mean(x)),
                                 lambda x: Tensor.stack(*x.std_mean()))
    helper_test_op([(15,25,35)], lambda x: torch.stack(torch.std_mean(x, correction=5)),
                                 lambda x: Tensor.stack(*x.std_mean(correction=5)))
    helper_test_op([(15,25,35)], lambda x: torch.stack(torch.std_mean(x, keepdim=True, correction=0)),
                                 lambda x: Tensor.stack(*x.std_mean(keepdim=True, correction=0)))
    helper_test_op([(3,4,5,6)], lambda x: torch.stack(torch.std_mean(x, axis=(1,2))),
                                lambda x: Tensor.stack(*x.std_mean(axis=(1,2))))

  def test_std_mean_loaded_nan(self):
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", message="std_mean\\(\\): degrees of freedom is <= 0")
      helper_test_op([(1,0,3,0,5)], lambda x: torch.stack(torch.std_mean(x, axis=(1,3))),
                                    lambda x: Tensor.stack(*x.std_mean(axis=(1,3))))
  def test_softmax(self):
    helper_test_op([(45,65)], torch.nn.Softmax(dim=1), Tensor.softmax, atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45)], torch.nn.Softmax(dim=0), Tensor.softmax, atol=1e-7, grad_atol=1e-7)
    helper_test_op([()], torch.nn.Softmax(dim=0), Tensor.softmax, atol=1e-7, grad_atol=1e-7)
    helper_test_op([()], torch.nn.Softmax(dim=-1), Tensor.softmax, atol=1e-7, grad_atol=1e-7)
  def test_softmax_other_axis(self):
    helper_test_op([(10,10,10)], lambda x: x.softmax(0), atol=1e-7, grad_atol=2e-7)
    helper_test_op([(10,10,10)], lambda x: x.softmax(1), atol=1e-7, grad_atol=2e-7)
    helper_test_op([(10,10,10)], lambda x: x.softmax(2), atol=1e-7, grad_atol=2e-7)
  def test_softmax_argmax(self):
    helper_test_op([(45,65)], lambda x: x.softmax(0).argmax().type(torch.int32),
                              lambda x: x.softmax(0).argmax(), forward_only=True, atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45,65)], lambda x: x.softmax(1).argmax().type(torch.int32),
                              lambda x: x.softmax(1).argmax(), forward_only=True, atol=1e-7, grad_atol=1e-7)
  def test_log_softmax(self):
    helper_test_op([(45,65)], torch.nn.LogSoftmax(dim=1), Tensor.log_softmax, atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45)], torch.nn.LogSoftmax(dim=0), Tensor.log_softmax, atol=1e-7, grad_atol=1e-7)
    helper_test_op([()], torch.nn.LogSoftmax(dim=0), Tensor.log_softmax, atol=1e-7, grad_atol=1e-7)
    helper_test_op([()], torch.nn.LogSoftmax(dim=-1), Tensor.log_softmax, atol=1e-7, grad_atol=1e-7)
  def test_softmin(self):
    helper_test_op([(45,65)], torch.nn.Softmin(dim=1), Tensor.softmin, atol=1e-7, grad_atol=1e-7)

  def test_normalize(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.normalize(x), lambda x: x.normalize(), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45,65)], lambda x: torch.nn.functional.normalize(x, dim=0), lambda x: x.normalize(dim=0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(10,10,10)], lambda x: torch.nn.functional.normalize(x, dim=2), lambda x: x.normalize(dim=2), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45,65)], lambda x: torch.nn.functional.normalize(x, p=1), lambda x: x.normalize(p=1), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45,65)], lambda x: torch.nn.functional.normalize(x, p=3, dim=0), lambda x: x.normalize(p=3, dim=0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45,65)], lambda x: torch.nn.functional.normalize(x, p=0), lambda x: x.normalize(p=0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45,65)], lambda x: torch.nn.functional.normalize(x, p=-1), lambda x: x.normalize(p=-1), atol=1e-7, grad_atol=1e-7)
  def test_rmsnorm(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.rms_norm(x, (65,), eps=1e-6), lambda x: x.rmsnorm(), atol=1e-6, grad_atol=1e-6)
    helper_test_op([(10,10,10)], lambda x: torch.nn.functional.rms_norm(x, (10,10), eps=1e-3), lambda x: x.rmsnorm((1,2), eps=1e-3),
                   atol=1e-6, grad_atol=1e-6)
  def test_normalize_int(self):
    helper_test_op(None, lambda x: torch.nn.functional.normalize(x.float(), p=2), lambda x: x.normalize(p=2), forward_only=True,
                   vals=[[[3, 4], [6, 8]]])

  def test_logsumexp(self):
    helper_test_op([(45,65)], lambda x: torch.logsumexp(x, dim=0), lambda x: x.logsumexp(0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45,65)], lambda x: torch.logsumexp(x, dim=0, keepdim=True), lambda x: x.logsumexp(0, True), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45,65)], lambda x: torch.logsumexp(x, dim=1), lambda x: x.logsumexp(1), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45,65)], lambda x: torch.logsumexp(x, dim=1, keepdim=True), lambda x: x.logsumexp(1, True), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(6,6,6)], lambda x: torch.logsumexp(x, dim=2), lambda x: x.logsumexp(2), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(6,6,6,6)], lambda x: torch.logsumexp(x, dim=2), lambda x: x.logsumexp(2), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(6,6,6,6)], lambda x: torch.logsumexp(x, dim=3), lambda x: x.logsumexp(3), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(6,6,6,6)], lambda x: torch.logsumexp(x, dim=(1,3)), lambda x: x.logsumexp((1,3)), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45)], lambda x: torch.logsumexp(x, dim=0), lambda x: x.logsumexp(0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([()], lambda x: torch.logsumexp(x, dim=0), lambda x: x.logsumexp(0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([()], lambda x: torch.logsumexp(x, dim=-1), lambda x: x.logsumexp(-1), atol=1e-7, grad_atol=1e-7)
    helper_test_op(None, lambda x: torch.logsumexp(x, dim=0), lambda x: x.logsumexp(0), vals=[[-math.inf, -math.inf]], forward_only=True)

  @slow_test
  def test_logcumsumexp(self):
    helper_test_op([(45,65)], lambda x: torch.logcumsumexp(x, dim=0), lambda x: x.logcumsumexp(0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45,65)], lambda x: torch.logcumsumexp(x, dim=1), lambda x: x.logcumsumexp(1), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(6,6,6)], lambda x: torch.logcumsumexp(x, dim=2), lambda x: x.logcumsumexp(2), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(6,6,6,6)], lambda x: torch.logcumsumexp(x, dim=2), lambda x: x.logcumsumexp(2), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(6,6,6,6)], lambda x: torch.logcumsumexp(x, dim=3), lambda x: x.logcumsumexp(3), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45)], lambda x: torch.logcumsumexp(x, dim=0), lambda x: x.logcumsumexp(0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([()], lambda x: torch.logcumsumexp(x, dim=0), lambda x: x.logcumsumexp(0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([()], lambda x: torch.logcumsumexp(x, dim=0), lambda x: x.logcumsumexp(), atol=1e-7, grad_atol=1e-7)
    helper_test_op([()], lambda x: torch.logcumsumexp(x, dim=-1), lambda x: x.logcumsumexp(-1), atol=1e-7, grad_atol=1e-7)

  def test_logcumsumexp_numerical(self):
    helper_test_op(None, lambda x: torch.logcumsumexp(x, dim=0), lambda x: x.logcumsumexp(), atol=1e-7, grad_atol=1e-7, vals=[[0.0, 100.0]])
    helper_test_op(None, lambda x: torch.logcumsumexp(x, dim=0), lambda x: x.logcumsumexp(), vals=[[-math.inf, 0.0, 1.0]], forward_only=True)

  def test_sinh(self):
    helper_test_op([(45,65)], lambda x: x.sinh(), grad_atol=1e-6)
    # TODO: backward nan instead of inf
    helper_test_op([(45,65)], lambda x: x.sinh(), grad_atol=1e-6, low=-300, high=-297, forward_only=True)
    helper_test_op([(45,65)], lambda x: x.sinh(), grad_atol=1e-6, low=300, high=303, forward_only=True)
  def test_cosh(self):
    helper_test_op([(45,65)], lambda x: x.cosh(), grad_atol=1e-6)
    # TODO: backward nan instead of inf
    helper_test_op([(45,65)], lambda x: x.cosh(), grad_atol=1e-6, low=-300, high=-297, forward_only=True)
    helper_test_op([(45,65)], lambda x: x.cosh(), grad_atol=1e-6, low=300, high=303, forward_only=True)
  def test_tanh(self):
    helper_test_op([(45,65)], lambda x: x.tanh(), grad_atol=1e-6)
  def test_tanh_extreme(self):
    helper_test_op([(45,65)], lambda x: x.tanh(), grad_atol=1e-6, low=-300, high=-297)
    helper_test_op([(45,65)], lambda x: x.tanh(), grad_atol=1e-6, low=300, high=303)
  def test_hardtanh(self):
    for val in range(10, 30, 5):
      helper_test_op([(45,65)], lambda x: torch.nn.functional.hardtanh(x, -val, val), lambda x: x.hardtanh(-val, val), grad_atol=1e-6)
      helper_test_op([()], lambda x: torch.nn.functional.hardtanh(x, -val, val), lambda x: x.hardtanh(-val, val), grad_atol=1e-6)
  def test_asinh(self):
    helper_test_op([(45,65)], lambda x: x.asinh(), grad_atol=1e-6)
    helper_test_op([(45,65)], lambda x: x.asinh(), grad_atol=1e-6, low=-300, high=-297)
    helper_test_op([(45,65)], lambda x: x.asinh(), grad_atol=1e-6, low=300, high=303)
    helper_test_op([(45,65)], lambda x: x.asinh(), grad_atol=1e-6, low=-1e10, high=-1e9)
    helper_test_op(None, lambda x: x.asinh(), grad_atol=1e-6, vals=[[-1.0, 0.0, 1.0]])
  def test_acosh(self):
    helper_test_op([(45,65)], lambda x: x.acosh(), grad_atol=1e-6)
    helper_test_op([(45,65)], lambda x: x.acosh(), grad_atol=1e-3, grad_rtol=1e-2, low=-300, high=-297)
    helper_test_op([(45,65)], lambda x: x.acosh(), grad_atol=1e-6, low=300, high=303)
  def test_atanh(self):
    helper_test_op([(45,65)], lambda x: x.atanh(), grad_atol=1e-6)
    helper_test_op([(45,65)], lambda x: x.atanh(), grad_atol=1e-6, low=-300, high=-297)
    helper_test_op([(45,65)], lambda x: x.atanh(), grad_atol=1e-6, low=300, high=303)

  def test_flip_eye_crash(self):
    helper_test_op([], lambda: (torch.eye(10)@torch.eye(10).flip(0)),
                       lambda: (Tensor.eye(10)@Tensor.eye(10).flip(0)), forward_only=True)

  def test_broadcast_simple(self):
    helper_test_op([(1,65), (45,1)], lambda x,y: x/y)
    helper_test_op([(45,65), ()], lambda x,y: x/y)

  def test_slice_in_bounds_1dim(self):
    helper_test_op([(3)], lambda x: x[1:3])
    helper_test_op([(3)], lambda x: x[0:2])
    helper_test_op([(3)], lambda x: x[-2:2])

  def test_slice_on_0dim_tensor(self):
    helper_test_op([()], lambda x: x[None])

    with self.assertRaises(IndexError):
      a = Tensor(3.14)
      a[0]

  def test_slice_int_indexing(self):
    helper_test_op([(3)], lambda x: x[0])
    helper_test_op([(3)], lambda x: x[2])
    helper_test_op([(3)], lambda x: x[-1])
    helper_test_op([(3)], lambda x: x[-3])
    helper_test_op([(10,10)], lambda x: x[1])
    helper_test_op([(3,3,3)], lambda x: x[1,1,1])

  def test_slice_in_bounds_multidim(self):
    helper_test_op([(3,3,3)], lambda x: x[1:2])
    helper_test_op([(3,3,3)], lambda x: x[1:2, 2])
    helper_test_op([(3,3,3)], lambda x: x[1:2, 1:2])
    helper_test_op([(3,3,3)], lambda x: x[1:2, 1:2, 0:-1])

  def test_slice_with_none(self):
    helper_test_op([(3,3,3)], lambda x: x[None])
    helper_test_op([(3,3,3)], lambda x: x[1:2, None])
    helper_test_op([(3,3,3)], lambda x: x[1:2, None, 1:2])
    helper_test_op([(3,3,3)], lambda x: x[1:2, 1:2, None, -1])
    helper_test_op([(3,3,3)], lambda x: x[None, None, 1, None, 2, 0:2])

  def test_slice_with_const_tensor(self):
    t = Tensor.zeros(1, dtype=dtypes.int)
    helper_test_op([(3,3,3)], lambda x: x[:, [0], :], lambda x: x[:, t, :])
    helper_test_op([(3,3,3)], lambda x: x[:, [0], :], lambda x: x[:, t.contiguous(), :])

  def test_slice_one_endpoint_out_of_bounds(self):
    helper_test_op([(3,3,3)], lambda x: x[0:4])
    helper_test_op([(3,3,3)], lambda x: x[-6:4])
    helper_test_op([(3,3,3)], lambda x: x[1:50])
    helper_test_op([(3,3,3)], lambda x: x[1:50, 1:2, -1])

  def test_slice_stride_gt_one(self):
    helper_test_op([(7,5,10)], lambda x: x[::2, ::3, ::4])
    helper_test_op([(7,5,10)], lambda x: x[1:5:2, ::3, ::4])
    helper_test_op([(7,5,10)], lambda x: x[1:5:2, 3, ::4])
    helper_test_op([(7,5,10)], lambda x: x[1:5:2, None, None, 3, None, ::4])

  @unittest.skipIf(COMPILE_ONLY, "test requires runtime")
  def test_slice_negative_strides(self):
    # Torch doesn't support slicing with negative steps
    a = np.random.randn(10, 10, 10).astype(np.float32)
    t = Tensor(a)
    np.testing.assert_allclose(a[::-1], t[::-1].numpy())
    np.testing.assert_allclose(a[::-2], t[::-2].numpy())
    np.testing.assert_allclose(a[:, 2:0:-1], t[:, 2:0:-1].numpy())
    np.testing.assert_allclose(a[:, 2:0:-1, 3:1:-2], t[:, 2:0:-1, 3:1:-2].numpy())
    np.testing.assert_allclose(a[4:0:-3, 2:0:-1, -1:-5:-2], t[4:0:-3, 2:0:-1, -1:-5:-2].numpy())
    np.testing.assert_allclose(a[2:5:-1, :, :], t[2:5:-1, :, :].numpy())  # shape = (0, 10, 10)
    np.testing.assert_allclose(a[:, 2:5:-1, :], t[:, 2:5:-1, :].numpy())  # shape = (0, 10, 10)
    np.testing.assert_allclose(a[:, :, 2:5:-1], t[:, :, 2:5:-1].numpy())  # shape = (0, 10, 10)

  def test_slice_both_endpoints_out_of_bounds(self):
    helper_test_op([(3,3,3)], lambda x: x[5:10])
    helper_test_op([(3,3,3)], lambda x: x[-15:-7])

  def test_slice_start_gt_end(self):
    helper_test_op([(3,3,3)], lambda x: x[-2:2])
    helper_test_op([(3,3,3)], lambda x: x[-2:-5])

  def test_slice_zero_in_shape(self):
    helper_test_op([(10,10)], lambda x: x[1:1])  # x.shape = (0, 10)

  def test_slice_ellipsis(self):
    helper_test_op([(3,3,3,3)], lambda x: x[..., 0])
    helper_test_op([(3,3,3,3)], lambda x: x[0, ...])
    helper_test_op([(3,3,3,3)], lambda x: x[0, ..., 0])
    helper_test_op([(3,3,3,3)], lambda x: x[0:3, ..., 2:3])
    helper_test_op([(3,3,3,3)], lambda x: x[None, 0:3, ..., 0, None])

  # this was the failure in llama early realizing freqs_cis
  def test_double_slice(self):
    helper_test_op([(4,4)], lambda x: x[:, 1:2][1:2])
    helper_test_op([(4,4)], lambda x: x[1:3][1:2])
    helper_test_op([(4,4)], lambda x: x[:, 1:2][0:1])
    helper_test_op([(4,4)], lambda x: x[:, 1:2][:, 0:1])

  def test_pad(self):
    helper_test_op([(3,3,3,3)], lambda x: torch.nn.functional.pad(x, (1,2,3,4)), lambda x: x.pad(padding=(1,2,3,4)))
    helper_test_op([(3,3,3,3)], lambda x: torch.nn.functional.pad(x, (-1,2,-3,4)), lambda x: x.pad(padding=(-1,2,-3,4)))
    helper_test_op([(3,3,3,3)], lambda x: torch.nn.functional.pad(x, (1,2,3,4), value=5), lambda x: x.pad(padding=(1,2,3,4),value=5))
    helper_test_op([(3,3,3,3)], lambda x: torch.nn.functional.pad(x, (-1,2,-3,4), value=5), lambda x: x.pad(padding=(-1,2,-3,4),value=5))
    helper_test_op([(3,3,3,3)], lambda x: torch.nn.functional.pad(x, (1,2,3,4), value=math.inf), lambda x: x.pad(padding=(1,2,3,4),value=math.inf))
    helper_test_op([(3,3,3,3)], lambda x: torch.nn.functional.pad(x, (-1,2,-3,4), value=-math.inf),
                                lambda x: x.pad(padding=(-1,2,-3,4),value=-math.inf))
    helper_test_op([(3,3)], lambda x: torch.nn.functional.pad(x, (1,2,3,4)),lambda x: x.pad(((3,4),(1,2))))
    helper_test_op([(3,3)], lambda x: torch.nn.functional.pad(x, (-1,2,-3,4)), lambda x: x.pad(((-3,4), (-1,2))))
    helper_test_op([(3,3)], lambda x: torch.nn.functional.pad(x, (1,2,3,4), value=5), lambda x: x.pad(((3,4), (1,2)), value=5))
    helper_test_op([(3,3)], lambda x: torch.nn.functional.pad(x, (0,0,3,4), value=1), lambda x: x.pad(((3,4), None), value=1))
    helper_test_op([(3,3)], lambda x: torch.nn.functional.pad(x, (0,0,0,0), value=1), lambda x: x.pad((None, None), value=1))
    # raise error for uneven pads
    self.helper_test_exception([(3,3)], lambda x: torch.nn.functional.pad(x, (2,0,2)), lambda x: x.pad((2,0,2)),
                               expected=(RuntimeError, ValueError))
    # raise error for too many or too little pads
    self.helper_test_exception([(3,3)], lambda x: torch.nn.functional.pad(x, (0,0,0,0,1,0,3,0)), lambda x: x.pad((0,0,0,0,1,0,3,0)),
                               expected=(RuntimeError, ValueError))
    # raise error for mode string typo
    self.helper_test_exception([(3,3,3)], lambda x: torch.nn.functional.pad(x, (3,0), mode="typo"), lambda x: x.pad((3,0), mode="typo"),
                               expected=NotImplementedError)
    x = Tensor.ones(3,3)
    with self.assertRaises(ValueError): x.pad((None,(0,1),(3,0)))
    with self.assertRaises(ValueError): x.pad(((0,1),))

  def test_pad_value_after_cast(self):
    for v in [1.5, 3e9, -math.inf, math.nan]:
      helper_test_op(None, lambda x: torch.nn.functional.pad(x.float(), (0,3), value=v), lambda x: x.float().pad((0,3), value=v),
                     forward_only=True, vals=[[1,2,3,4,5]])

  def test_pad_reflect_mode(self):
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (0,2,3,2), mode="reflect"), lambda x: x.pad((0,2,3,2), mode="reflect"))
    helper_test_op([(5,5,5)], lambda x: torch.nn.functional.pad(x, (0,2), mode="reflect"), lambda x: x.pad((0,2), mode="reflect"))
    helper_test_op([(1,1,5,5,5)], lambda x: torch.nn.functional.pad(x, (1,2,3,4,1,2), mode="reflect"),
                                  lambda x: x.pad((1,2,3,4,1,2), mode="reflect"))
    helper_test_op([(3,3,3,3)], lambda x: torch.nn.functional.pad(x, (-1,2,2,-1), mode="reflect"), lambda x: x.pad((-1,2,2,-1), mode="reflect"))
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (3,-3,0,-3), mode="reflect"), lambda x: x.pad((3,-3,0,-3), mode="reflect"))
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (3,-5,1,-5), mode="reflect"), lambda x: x.pad((3,-5,1,-5), mode="reflect"))
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (0,0,0,-5), mode="reflect"), lambda x: x.pad((0,0,0,-5), mode="reflect"))

    # max pad size for reflect is exactly once: pad < input size
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (4,4,0,4), mode="reflect"), lambda x:x.pad((4,4,0,4),mode="reflect"))
    # raise error for relfection padding when: pad >= input size
    self.helper_test_exception([(1,1,5,5)],
                                lambda x: torch.nn.functional.pad(x, (3,5,0,0),mode="reflect"), lambda x: x.pad((3,5,0,0),mode="reflect"),
                                expected=(RuntimeError, ValueError))

  def test_pad_replicate_mode(self):
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (0,2,3,2), mode="replicate"), lambda x: x.pad((0,2,3,2), mode="replicate"))
    helper_test_op([(5,5,5)], lambda x: torch.nn.functional.pad(x, (0,2), mode="replicate"), lambda x: x.pad((0,2), mode="replicate"))
    helper_test_op([(1,1,5,5,5)], lambda x: torch.nn.functional.pad(x, (1,2,3,4,1,2),mode="replicate"),lambda x:x.pad((1,2,3,4,1,2),mode="replicate"))
    helper_test_op([(3,3,3,3)], lambda x: torch.nn.functional.pad(x, (-1,2,2,-1), mode="replicate"), lambda x: x.pad((-1,2,2,-1), mode="replicate"))
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (3,-3,0,-3), mode="replicate"), lambda x: x.pad((3,-3,0,-3), mode="replicate"))
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (3,-5,1,-5), mode="replicate"), lambda x: x.pad((3,-5,1,-5), mode="replicate"))
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (0,0,0,-5), mode="replicate"), lambda x: x.pad((0,0,0,-5), mode="replicate"))

    # no max pad sizes for replicate
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (3,11,0,30), mode="replicate"), lambda x: x.pad((3,11,0,30), mode="replicate"))

  def test_pad_circular_mode(self):
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (0,2,3,2), mode="circular"), lambda x: x.pad((0,2,3,2), mode="circular"))
    helper_test_op([(5,5,5)], lambda x: torch.nn.functional.pad(x, (0,2), mode="circular"), lambda x: x.pad((0,2), mode="circular"))
    helper_test_op([(1,1,5,5,5)], lambda x: torch.nn.functional.pad(x, (1,2,3,5,1,2),mode="circular"),lambda x:x.pad((1,2,3,5,1,2),mode="circular"))
    # circular pad cannot wrap around more than once
    self.helper_test_exception([(1,1,5,5)],
                                lambda x: torch.nn.functional.pad(x, (3,6,0,0), mode="circular"), lambda x: x.pad((3,6,0,0), mode="circular"),
                                expected=(RuntimeError, ValueError))
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (1,-2,2,-1), mode="circular"), lambda x: x.pad((1,-2,2,-1), mode="circular"))

  def test_pad_reshape(self):
    helper_test_op([(1, 2)],
                   lambda x: torch.nn.functional.pad(x, (0, 1, 1, 0)).reshape((3, 2)),
                   lambda x: x.pad((0, 1, 1, 0)).reshape((3, 2)))
    helper_test_op([(1, 2)],
                   lambda x: torch.nn.functional.pad(x, (0, 2, 1, 1)).reshape((4, 3)),
                   lambda x: x.pad((0, 2, 1, 1)).reshape((4, 3)))
    helper_test_op([(1, 1, 1, 2)],
                   lambda x: torch.nn.functional.pad(x, (0, 4, 2, 2, 1, 2, 0, 2)).reshape((4, 3, 6, 5)),
                   lambda x: x.pad(((0, 2), (1, 2), (2, 2), (0, 4))).reshape((4, 3, 6, 5)))

  def test_stack_slice(self):
    helper_test_op([(4)], lambda x: torch.stack([x for i in range(3)])[0,:], lambda x: Tensor.stack(*[x for i in range(3)])[0,:])
    helper_test_op([(5)], lambda x: torch.stack([x for i in range(3)])[0,0], lambda x: Tensor.stack(*[x for i in range(3)])[0,0])
    helper_test_op([(4,4)], lambda x: torch.stack([x for i in range(4)])[3], lambda x: Tensor.stack(*[x for i in range(4)])[3])

  def test_transpose(self):
    helper_test_op([(3,3)], lambda x: x.T)
    helper_test_op([(3,3,3)], lambda x: x.transpose(1,2))
    helper_test_op([(3,3,3)], lambda x: x.transpose(0,2))

  def test_permute(self):
    helper_test_op([(1,2,3,4)], lambda x: x.permute((3,0,2,1)))
    helper_test_op([(3,4,5,6)], lambda x: x.permute((3,2,1,0)))
    helper_test_op([(3,4,5,6)], lambda x: x.permute((-2,-1,1,0)))
    helper_test_op([()], lambda x: x.permute(()))
    self.helper_test_exception([(3,4,5,6)], lambda x: x.permute((0,2)), expected=RuntimeError)
    self.helper_test_exception([(3,4,5,6)], lambda x: x.permute((0,1,2,3,3,3)), expected=RuntimeError)
    self.helper_test_exception([(3,4,5,6)], lambda x: x.permute((0,0,1,2,3)), expected=RuntimeError)

  def test_reshape(self):
    helper_test_op([(4,3,6,6)], lambda x: x.reshape((12,6,6)))
    helper_test_op([(4,3,6,6)], lambda x: x.reshape((-1,3,6,6)))
    helper_test_op([(4,3,6,6)], lambda x: x.reshape((-1,1,6,6)))
    helper_test_op([(4,3,6,6)], lambda x: x.reshape((4,3,6,6)), lambda x: x.reshape((None,None,6,6)))
    helper_test_op([()], lambda x: x.reshape(()))
    helper_test_op([(1,)], lambda x: x.reshape(()))
    helper_test_op([()], lambda x: x.reshape((1,)))
    helper_test_op([()], lambda x: x.reshape((1,1,1)))
    self.helper_test_exception([(3,4)], lambda x: x.reshape((-1,-1,2)), expected=RuntimeError)
    self.helper_test_exception([(3,4)], lambda x: x.reshape((-1,-1,-1,2)), expected=RuntimeError)

    with self.assertRaises(ValueError):
      x = Tensor.ones((4,3,6,6))
      x.reshape([])

  def test_view(self):
    helper_test_op([(4,3,6,6)], lambda x: x.view((12,6,6)))
    helper_test_op([(4,3,6,6)], lambda x: x.view((-1,3,6,6)))
    helper_test_op([(6,)], lambda x: x.view(2, 3))
    helper_test_op([(6,1)], lambda x: x.view([2, 3]))
    helper_test_op([(1,6)], lambda x: x.view((3, 2)))
    helper_test_op([(3,2)], lambda x: x.view((2, 3)))
    helper_test_op([(3,2)], lambda x: x.view(6))

  def test_flip(self):
    helper_test_op([(4,3,6,6)], lambda x: x.flip((0,)))
    helper_test_op([(4,3,6,6)], lambda x: x.flip((0,1)))
    helper_test_op([(4,3,6,6)], lambda x: x.flip((0,1,3)))
    helper_test_op([(4,3,6,6)], lambda x: x.flip((3,)))
    helper_test_op([(4,3,6,6)], lambda x: x.flip((0,1,3)).flip(0))
    helper_test_op([(4,3,6,6)], lambda x: x.flip((-1,)))
    helper_test_op([()], lambda x: x.flip(()))
    helper_test_op([(1,)], lambda x: x.flip(()))
    helper_test_op([(4,3,6,6)], lambda x: x.flip(()))
    self.helper_test_exception([(3,4)], lambda x: x.flip((0,0)), expected=RuntimeError)
    self.helper_test_exception([(3,4)], lambda x: x.flip((1,1)), expected=RuntimeError)
    self.helper_test_exception([(3,4)], lambda x: x.flip((1,-1)), expected=RuntimeError)

  def test_squeeze(self):
    helper_test_op([(1,3,6,6)], lambda x: x.squeeze(0))
    helper_test_op([(4,3,1,6)], lambda x: x.squeeze(1))
    helper_test_op([(4,3,6,6)], lambda x: x.squeeze(3))
    self.helper_test_exception([(4,3,6,6)], lambda x: x.squeeze(50), expected=IndexError)
    helper_test_op([(4,3,6,1)], lambda x: x.squeeze(-1))
    helper_test_op([(1,3,1,6)], lambda x: x.squeeze((0,2)))
    helper_test_op([(1,3,1,6)], lambda x: x.squeeze((-2,1)))
    helper_test_op([(1,3,1,6)], lambda x: x.squeeze(()))
    helper_test_op([(4,3,6,6)], lambda x: x.squeeze())
    helper_test_op([(1,3,6,6)], lambda x: x.squeeze())
    helper_test_op([(2,3,1)], lambda x: x.squeeze())
    helper_test_op([()], lambda x: x.squeeze(-1))
    helper_test_op([()], lambda x: x.squeeze(0))
    helper_test_op([()], lambda x: x.squeeze())
    self.helper_test_exception([()], lambda x: x.squeeze(10), expected=IndexError)
    self.helper_test_exception([()], lambda x: x.squeeze(1), expected=IndexError)
    self.helper_test_exception([()], lambda x: x.squeeze(-2), expected=IndexError)

  def test_unsqueeze(self):
    helper_test_op([(4,3,6,6)], lambda x: x.unsqueeze(0))
    helper_test_op([(4,3,6,6)], lambda x: x.unsqueeze(4))
    helper_test_op([(4,3,6,6)], lambda x: x.unsqueeze(-1))
    helper_test_op([(4,3,6,6)], lambda x: x.unsqueeze(-3))
    helper_test_op([()], lambda x: x.unsqueeze(0))

  def test_flatten(self):
    for axis in range(3):
      helper_test_op([(4,3,6,6)], lambda x: x.flatten(start_dim=axis))
    for axis in range(3):
      helper_test_op([(4,3,6,6)], lambda x: x.flatten(end_dim=axis))
    helper_test_op([(4,3,6,6)], lambda x: x.flatten(start_dim=1, end_dim=3))
    helper_test_op([()], lambda x: x.flatten())
    helper_test_op([(1,)], lambda x: x.flatten())

  def test_unflatten(self):
    helper_test_op([(4,3,6,6)], lambda x: x.unflatten(0, (2, 2)))
    helper_test_op([(4,3,6,6)], lambda x: x.unflatten(3, (3, 2)))
    helper_test_op([(4,3,6,6)], lambda x: x.unflatten(-1, (3, 2, 1)))

  def test_diag(self):
    helper_test_op([(5,)], lambda x: x.diag())

  def test_diagonal(self):
    helper_test_op([(5,5)], lambda x: x.diagonal())
    helper_test_op([(3,4)], lambda x: x.diagonal())  # rectangular
    helper_test_op([(4,3)], lambda x: x.diagonal())  # rectangular (other way)
    helper_test_op([(3,3,3)], lambda x: x.diagonal(dim1=-2, dim2=-1))  # batched
    helper_test_op([(4,5,6)], lambda x: x.diagonal(dim1=-2, dim2=-1))  # batched rectangular
    helper_test_op([(2,3,4,5)], lambda x: x.diagonal(dim1=-2, dim2=-1))  # 4D batched
    helper_test_op([(5,5)], lambda x: x.diagonal(offset=1))  # positive offset
    helper_test_op([(5,5)], lambda x: x.diagonal(offset=-1))  # negative offset
    helper_test_op([(3,5)], lambda x: x.diagonal(offset=2))  # offset on rectangular
    self.helper_test_exception([(3,3)], lambda x: x.diagonal(dim1=0, dim2=0), expected=RuntimeError)

  def test_roll(self):
    helper_test_op([(2, 4)], lambda x: x.roll(1))
    helper_test_op([(2, 4)], lambda x: x.roll((1,)))
    helper_test_op([(0,)], lambda x: x.roll(1, 0))
    helper_test_op([(2, 0, 3)], lambda x: x.roll(1, 0))
    helper_test_op([(2, 0, 3)], lambda x: x.roll(1, 1))
    helper_test_op([(2, 0, 3)], lambda x: x.roll(1))
    self.helper_test_exception([(2, 4)], lambda x: x.roll((1, 2)), expected=RuntimeError)
    helper_test_op([(2, 4)], lambda x: x.roll(1, 0))
    helper_test_op([(2, 4)], lambda x: x.roll((1, 2), (1, 1)))
    helper_test_op([(2, 4)], lambda x: x.roll(-1, 0))
    helper_test_op([(2, 4)], lambda x: x.roll(shifts=(2, 1), dims=(0, 1)))
    helper_test_op([(2, 4, 6)], lambda x: x.roll(1, 0))
    helper_test_op([(2, 4)], lambda x: x.roll(1, -1))
    helper_test_op([(2, 4)], lambda x: x.roll(-1, -1))
    helper_test_op([(2, 4)], lambda x: x.roll(5, 0))
    helper_test_op([(2, 4)], lambda x: x.roll(-5, 0))
    helper_test_op([(2, 4, 6)], lambda x: x.roll(shifts=(2, -3), dims=(0, 2)))
    helper_test_op([(2, 4, 6)], lambda x: x.roll(shifts=(1, 2, -1), dims=(0, 1, 2)))
    helper_test_op([(2, 4)], lambda x: x.roll(0, 0))
    helper_test_op([(2, 4, 6)], lambda x: x.roll(shifts=(0, 0), dims=(0, 1)))
    helper_test_op([(2, 4, 6)], lambda x: x.roll(shifts=(0, 2), dims=(0, 1)))
    self.helper_test_exception([(3, 3)], lambda x: x.roll(shifts=1, dims=(0, 1)), expected=RuntimeError)
    self.helper_test_exception([(10,)], lambda x: x.roll(shifts=(1, 2), dims=0), expected=RuntimeError)

  def test_detach(self):
    helper_test_op([(4,3,6,6)], lambda x: x.detach(), forward_only=True)
    helper_test_op([()], lambda x: x.detach(), forward_only=True)

  def test_expand(self):
    helper_test_op([(4,3,1,6)], lambda x: x.expand((4,3,2,6)))
    helper_test_op([(1,1,1,1)], lambda x: x.expand((4,3,2,6)))
    helper_test_op([(4,3,1,6)], lambda x: x.expand((6,1,4,3,2,6)))
    helper_test_op([(4,3,1,6)], lambda x: x.expand((0,1,4,3,2,6)))
    helper_test_op([(4,3,1,6)], lambda x: x.expand((4,3,0,6)))
    helper_test_op([()], lambda x: x.expand((4,3,2,6)))
    helper_test_op([()], lambda x: x.expand([]))

    with self.assertRaises((ValueError, RuntimeError)): Tensor.ones(4,3,1,6).expand(4,1,1,6)
    with self.assertRaises((ValueError, RuntimeError)): Tensor.ones(4,3,1,6).expand(4,6,1,6)
    with self.assertRaises((ValueError, RuntimeError)): Tensor.ones(4,3,1,6).expand(3,1,6)
    with self.assertRaises((ValueError, RuntimeError)): Tensor.ones(4,3,2,6).expand(4,3,0,6)

  @unittest.skip("very slow")
  def test_sd_big_conv(self):
    # internal shape (1, 1, 512, 62, 62, 512, 3, 3) overflows a int
    helper_test_op([(1,256,64,64), (512,256,3,3)],
                    lambda x,w: torch.nn.functional.conv2d(x, w),
                    lambda x,w: x.conv2d(w), atol=1e-3)

  @unittest.skip("slow")
  def test_large_bs_conv(self):
    # large batch size can cause OpenCL image to exceed max image height on macOS
    # (or cause the conv kernel to overflow short sampling coords)
    helper_test_op([(4096,3,3,3), (1,3,3,3)],
                    lambda x,w: torch.nn.functional.conv2d(x, w),
                    lambda x,w: x.conv2d(w), atol=1e-3)

  @unittest.skip("slow")
  def test_large_ic_conv(self):
    # large input channel count can cause OpenCL image to exceed max image width on macOS
    helper_test_op([(1,2048,3,3), (1,2048,3,3)],
                    lambda x,w: torch.nn.functional.conv2d(x, w),
                    lambda x,w: x.conv2d(w))

  def test_simple_conv2d(self):
    helper_test_op([(1,4,9,9), (4,4,3,3)],
      lambda x,w: torch.nn.functional.conv2d(x,w),
      lambda x,w: Tensor.conv2d(x,w), grad_rtol=1e-5)

  def test_simple_conv2d_bias(self):
    helper_test_op([(1,4,9,9), (4,4,3,3), (4,)],
      lambda x,w,b: torch.nn.functional.conv2d(x,w,b),
      lambda x,w,b: Tensor.conv2d(x,w,b), grad_rtol=1e-5)

  @slow_test
  @unittest.skipIf(IMAGE>0, "no conv3d on images")
  def test_simple_conv3d(self):
    helper_test_op([(1,4,9,9,9), (4,4,3,3,3)],
      lambda x,w: torch.nn.functional.conv3d(x,w),
      lambda x,w: Tensor.conv2d(x,w), grad_rtol=1e-5)

  def test_simple_conv2d_1x1(self):
    helper_test_op([(1,4,9,9), (4,4,1,1)],
      lambda x,w: torch.nn.functional.conv2d(x,w),
      lambda x,w: Tensor.conv2d(x,w), grad_rtol=1e-5)

  def test_simple_conv2d_batched(self):
    helper_test_op([(2,4,9,9), (4,4,3,3)],
      lambda x,w: torch.nn.functional.conv2d(x,w),
      lambda x,w: Tensor.conv2d(x,w), grad_rtol=1e-5)

  # conv transpose

  def test_simple_conv_transpose2d(self):
    helper_test_op([(2,4,9,9), (4,4,3,3)],
      lambda x,w: torch.nn.functional.conv_transpose2d(x,w),
      lambda x,w: Tensor.conv_transpose2d(x,w), grad_rtol=1e-5)

  def test_grouped_conv_transpose2d(self):
    helper_test_op([(2,4,9,9), (4,4,3,3)],
      lambda x,w: torch.nn.functional.conv_transpose2d(x,w,groups=2,padding=(2,1),dilation=(2,1)),
      lambda x,w: Tensor.conv_transpose2d(x,w,groups=2,padding=(2,1),dilation=(2,1)), grad_rtol=1e-5)

  @slow_test
  def test_output_padded_conv_transpose2d(self):
    for output_padding, stride in [((1,1), (2,3)), ((2,1), (3,2))]:
      helper_test_op([(2,4,6,5), (4,4,3,3),(4,)],
        lambda x,w,b: torch.nn.functional.conv_transpose2d(x,w,b,output_padding=output_padding,stride=stride),
        lambda x,w,b: Tensor.conv_transpose2d(x,w,b,output_padding=output_padding,stride=stride), grad_rtol=1e-5)

  @slow_test
  @unittest.skipIf(IMAGE>0, "no conv3d on images")
  def test_simple_conv_transpose3d(self):
    helper_test_op([(2,4,9,9,9), (4,4,3,3,3)],
      lambda x,w: torch.nn.functional.conv_transpose3d(x,w),
      lambda x,w: Tensor.conv_transpose2d(x,w), grad_rtol=1e-5)

  @unittest.skipIf(IMAGE>0, "no conv1d on images")
  def test_simple_padding_conv1d(self):
    bs = 6
    cin = 2
    groups = 1
    H = 5
    p = (1,1)
    helper_test_op([(bs,cin,11), (6,cin//groups,H)],
      lambda x,w: torch.nn.functional.conv1d(torch.nn.functional.pad(x, p),w),
      lambda x,w: Tensor.conv2d(x,w,padding=p))

  @unittest.skipIf(IMAGE>0, "no conv1d on images")
  def test_strided_conv1d_simple(self):
    bs, H = 2, 3
    helper_test_op([(bs,1,5), (1,1,H)],
      lambda x,w: torch.nn.functional.conv1d(x,w,stride=2),
      lambda x,w: Tensor.conv2d(x,w,stride=2))

  @unittest.skipIf(IMAGE>0, "no conv1d on images")
  def test_asymmetric_padding_conv1d(self):
    for p in [(0,1), (2,1), (2,0)]:
      with self.subTest(p):
        for n in [3,4]:
          for k in [2]:
            helper_test_op([(1,1,n), (1,1,k)],
              lambda x,w: torch.nn.functional.conv1d(torch.nn.functional.pad(x, p),w),
              lambda x,w: Tensor.conv2d(x,w,padding=p))

  def test_simple_grouped_conv2d(self):
    bs = 1
    groups = 2
    rcout = 1
    cin = 2
    helper_test_op([(bs,groups*cin,1,1), (groups*rcout,cin,1,1)],
      lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups),
      lambda x,w: Tensor.conv2d(x,w,groups=groups), grad_rtol=1e-5)

  def test_medium_grouped_conv2d(self):
    bs = 1
    groups = 2
    rcout = 2
    cin = 2
    helper_test_op([(bs,groups*cin,1,1), (groups*rcout,cin,1,1)],
      lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups),
      lambda x,w: Tensor.conv2d(x,w,groups=groups), grad_rtol=1e-5)

  def test_depthwise_conv2d(self):
    bs = 1
    groups = 32
    rcout = 1
    cin = 1
    helper_test_op([(bs,groups*cin,32,32), (groups*rcout,cin,1,1)],
      lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups),
      lambda x,w: Tensor.conv2d(x,w,groups=groups), grad_rtol=1e-5)

  @slow_test
  def test_strided_conv2d_simple(self):
    bs,H,W = 2,3,1
    helper_test_op([(bs,1,5,1), (1,1,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,stride=2),
      lambda x,w: Tensor.conv2d(x,w,stride=2))

  def test_negative_padding_conv2d(self):
    n,k = 10, 3
    helper_test_op([(1,1,n,n), (1,1,k,k)],
      lambda x,w: torch.nn.functional.conv2d(x[:, :, 1:-1, 1:-1],w),
      lambda x,w: Tensor.conv2d(x,w,padding=-1))
    helper_test_op([(1,1,n,n), (1,1,k,k)],
      lambda x,w: torch.nn.functional.conv2d(x[:, :, 1:, 1:],w),
      lambda x,w: Tensor.conv2d(x,w,padding=(-1,0,-1,0)))

  def test_simple_padding_conv2d(self):
    p = (0,1,1,0)
    helper_test_op(None,
      lambda x,w: torch.nn.functional.conv2d(torch.nn.functional.pad(x, p),w),
      lambda x,w: Tensor.conv2d(x,w,padding=p), vals=[[[[[2.,3.]]]], [[[[1.]]]]])

  def test_dilated_conv2d(self):
    bs = 4
    cin = 3
    H,W = 3,3
    for d in [2, (2,1)]:
      with self.subTest(dilation := d):
        helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
          lambda x,w: torch.nn.functional.conv2d(x,w,dilation=dilation),
          lambda x,w: Tensor.conv2d(x,w,dilation=dilation))

  @slow_test
  def test_max_pool2d_asymmetric_padding(self):
    for p in [(0,1,0,1), (2,1,2,1), (2,0,2,1)]:
      with self.subTest(padding=p):
        helper_test_op([(4,2,111,28)],
          lambda x: torch.nn.functional.max_pool2d(torch.nn.functional.pad(x, p, value=float("-inf")), kernel_size=(5,5)),
          lambda x: Tensor.max_pool2d(x, kernel_size=(5,5), padding=p))

  @slow_test
  def test_max_pool2d_padding_int(self):
    ksz = (2,2)
    helper_test_op([(4,2,11,28)],
      lambda x: torch.nn.functional.max_pool2d(x.int(), kernel_size=ksz, padding=1),
      lambda x: Tensor.max_pool2d(x.int(), kernel_size=ksz, padding=1), forward_only=True)

  @unittest.skipIf( Device.DEFAULT in {"CUDA", "NV"}, "CUDA fails on this")
  def test_max_pool2d_unit_stride(self):
    helper_test_op([(3, 2, 17, 14)],
      lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(5,5), stride=1),
      lambda x: Tensor.max_pool2d(x, kernel_size=(5,5), stride=1))

  @slow_test
  def test_max_pool2d_dilation(self):
    for dilation in [(2, 3), (3, 2), 2, 3]:
      helper_test_op([(3, 2, 17, 14)],
        lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(5,5), dilation=dilation),
        lambda x: Tensor.max_pool2d(x, kernel_size=(5,5), dilation=dilation))

  def test_max_pool2d_ceil_mode_output_size_reduce_by_one(self):
    # sliding window ignored from end region
    helper_test_op([(1,1,5,5)],
      lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(3,3), stride=3, padding=1, ceil_mode=True),
      lambda x: Tensor.max_pool2d(x, kernel_size=(3,3), stride=3, padding=1, ceil_mode=True))

  def test_max_pool2d_return_indices(self):
    # batch and multi-channel
    helper_test_op([(2,3,6,6)],
      lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(2,2), return_indices=True)[1].type(torch.int32),
      lambda x: Tensor.max_pool2d(x, kernel_size=(2,2), return_indices=True)[1], forward_only=True)
    # dilation
    helper_test_op([(1,1,10,10)],
      lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(3,2), dilation=(2,3), return_indices=True)[1].type(torch.int32),
      lambda x: Tensor.max_pool2d(x, kernel_size=(3,2), dilation=(2,3), return_indices=True)[1], forward_only=True)
    # padding
    helper_test_op([(1,1,5,5)],
      lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(3,3), padding=1, return_indices=True)[1].type(torch.int32),
      lambda x: Tensor.max_pool2d(x, kernel_size=(3,3), padding=1, return_indices=True)[1], forward_only=True)
    # ceil mode padding
    helper_test_op([(1, 1, 7, 7)],
      lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2), ceil_mode=True, return_indices=True)[1].type(torch.int32),
      lambda x: Tensor.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2), ceil_mode=True, return_indices=True)[1],
      forward_only=True)
    # global maxpool
    helper_test_op([(1,1,12,13)],
      lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(12, 13), return_indices=True)[1].type(torch.int32),
      lambda x: Tensor.max_pool2d(x, kernel_size=(12, 13), return_indices=True)[1],
      forward_only=True)
    # multiple identical values in same window and overlapping windows
    helper_test_op(None,
      lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(3,3), stride=1, return_indices=True)[1].type(torch.int32),
      lambda x: Tensor.max_pool2d(x, kernel_size=(3,3), stride=1, return_indices=True)[1],
      vals=[[[[[1]*6]*6]]], forward_only=True)  # Tensor.ones(1,1,6,6)
    # overlapping max indices
    helper_test_op(None,
      lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(2,2), stride=1, return_indices=True)[1].type(torch.int32),
      lambda x: Tensor.max_pool2d(x, kernel_size=(2,2), stride=1, return_indices=True)[1],
      vals=[[[[[1,2]*3]*6]]], forward_only=True)  # Tensor([1,2,1,2,1,2]).expand(1,1,6,6)

  @slow_test
  def test_max_unpool2d(self):
    args = {"kernel_size":(5,5), "stride":(6,5)}
    helper_test_op([(8,3,50,50)],
      lambda x: torch.nn.functional.max_unpool2d(*torch.nn.functional.max_pool2d(x, return_indices=True, **args), **args),
      lambda x: Tensor.max_unpool2d(*Tensor.max_pool2d(x, return_indices=True, **args), **args), forward_only=True)
    args = {"kernel_size":(3,3), "stride":(6,7), "padding":1}
    helper_test_op([(8,3,30,30)],
      lambda x: torch.nn.functional.max_unpool2d(*torch.nn.functional.max_pool2d(x, return_indices=True, **args), **args, output_size=(30,30)),
      lambda x: Tensor.max_unpool2d(*Tensor.max_pool2d(x, return_indices=True, **args), **args, output_size=(30,30)), forward_only=True)
    # batch_size and channel_size of output_size are ignored
    helper_test_op([(1,3,7,6)],
      lambda x: torch.nn.functional.max_unpool2d(*torch.nn.functional.max_pool2d(x, kernel_size=(2,2), return_indices=True),
                                                 kernel_size=(2,2), output_size=(99,99,7,6)),
      lambda x: Tensor.max_unpool2d(*Tensor.max_pool2d(x, kernel_size=(2,2), return_indices=True),
                                    kernel_size=(2,2), output_size=(99,99,7,6)), forward_only=True)

  def test_max_unpool2d_inf(self):
    data = [[[[math.inf, -math.inf, math.nan], [1.0, 2.0, 3.0]]]]
    ksz = (2,2)
    helper_test_op((),
      lambda: torch.nn.functional.max_unpool2d(
        *torch.nn.functional.max_pool2d(torch.tensor(data), kernel_size=ksz, return_indices=True),
        kernel_size=ksz
      ),
      lambda: Tensor.max_unpool2d(
        *Tensor.max_pool2d(Tensor(data), kernel_size=ksz, return_indices=True),
        kernel_size=ksz
      ),
      forward_only=True)

  def test_avg_pool2d_asymmetric_padding(self):
    shape = (32,2,11,28)
    for p in [(0,1,0,1), (2,1,2,1), (2,0,2,1)]:
      with self.subTest(padding=p):
        helper_test_op([shape],
          lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=(5,5), padding=1),
          lambda x: Tensor.avg_pool2d(x, kernel_size=(5,5), padding=1), rtol=1e-5)
    self.helper_test_exception([shape], lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=(2,2), padding=(1,1,1)),
                               lambda x: Tensor.avg_pool2d(x, kernel_size=(2,2), padding=(1,1,1)), expected=(RuntimeError, ValueError))
    # regression test for https://github.com/tinygrad/tinygrad/pull/7581
    helper_test_op([(1,1,8,8)],
      lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=(1,2), padding=(0,1), stride=(5,1)),
      lambda x: Tensor.avg_pool2d(x, kernel_size=(1,2), padding=(0,1), stride=(5,1)), rtol=1e-5)

  @slow_test
  def test_avg_pool2d_padding_not_counted(self):
    shape = (32,2,11,28)
    for ksz in [(3,3), 2, (3,2)]:
      with self.subTest(kernel_size=ksz):
        helper_test_op([shape],
          lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=ksz, padding=1, count_include_pad=False),
          lambda x: Tensor.avg_pool2d(x, kernel_size=ksz, padding=1, count_include_pad=False), rtol=1e-5)

  def test_avg_pool2d_ceil_mode_output_size_reduce_by_one(self):
    # sliding window ignored from end region
    helper_test_op([(1,1,5,5)],
      lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=(3,3), stride=3, padding=1, ceil_mode=True),
      lambda x: Tensor.avg_pool2d(x, kernel_size=(3,3), stride=3, padding=1, ceil_mode=True))

  def test_avg_pool2d_ceil_mode_include_pad_output_size_reduce_by_one(self):
    # sliding window ignored from end region
    helper_test_op([(1,1,5,5)],
      lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=(3,3), stride=3, padding=1, ceil_mode=True, count_include_pad=True),
      lambda x: Tensor.avg_pool2d(x, kernel_size=(3,3), stride=3, padding=1, ceil_mode=True, count_include_pad=True))

  def test_global_avg_pool2d(self):
    helper_test_op([(32,2,11,28)],
      lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=(11,28)),
      lambda x: Tensor.avg_pool2d(x, kernel_size=(11,28)), rtol=1e-5)

  def test_avg_pool3d(self):
    # TODO: AMD:LLVM has larger atol
    # TODO: DEV=PYTHON backward hangs?
    atol = 1e-2 if DEV.device == "AMD" and DEV.renderer == "LLVM" else 1e-6
    helper_test_op([(1,1,16,16,16)],
      lambda x: torch.nn.functional.avg_pool3d(x, kernel_size=(8,8,8), stride=5, padding=1, count_include_pad=False),
      lambda x: Tensor.avg_pool2d(x, kernel_size=(8,8,8), stride=5, padding=1, count_include_pad=False), atol=atol, rtol=1e-5, forward_only=True)

  def test_interpolate_linear(self):
    for in_sz, out_sz in [((52,),(29,)), ((29,),(52,))]:
      helper_test_op([(2,3)+in_sz],
        lambda x: torch.nn.functional.interpolate(x, size=out_sz, mode="linear"),
        lambda x: Tensor.interpolate(x, size=out_sz, mode="linear"))

  def test_interpolate_linear_corners_aligned(self):
    for in_sz, out_sz in [((52,),(29,)), ((29,),(52,)), ((29,),(1,))]:
      helper_test_op([(2,3)+in_sz],
        lambda x: torch.nn.functional.interpolate(x, size=out_sz, mode="linear", align_corners=True),
        lambda x: Tensor.interpolate(x, size=out_sz, mode="linear", align_corners=True))

  def test_interpolate_nearest(self, mode="nearest"):
    for in_sz, out_sz in [((13,),(9,)), ((9,),(13,))]:
      helper_test_op([(2,3)+in_sz],
        lambda x: torch.nn.functional.interpolate(x, size=out_sz, mode=mode),
        lambda x: Tensor.interpolate(x, size=out_sz, mode=mode))
    for in_sz, out_sz in [((13,10),(9,11)), ((13,9),(11,10)), ((9,11),(10,13))]:
      helper_test_op([(2,3)+in_sz],
        lambda x: torch.nn.functional.interpolate(x, size=out_sz, mode=mode),
        lambda x: Tensor.interpolate(x, size=out_sz, mode=mode))
    for in_sz, out_sz in [((5,2,8),(3,6,4))]:
      helper_test_op([(2,3)+in_sz],
        lambda x: torch.nn.functional.interpolate(x, size=out_sz, mode=mode),
        lambda x: Tensor.interpolate(x, size=out_sz, mode=mode))

  def test_interpolate_nearest_exact(self): self.test_interpolate_nearest("nearest-exact")

  @slow_test
  def test_interpolate_bilinear(self):
    for in_sz, out_sz in [((12,20),(9,31)), ((12,9),(31,20)), ((9,31),(20,12))]:
      helper_test_op([(2,3)+in_sz],
        lambda x: torch.nn.functional.interpolate(x, size=out_sz, mode="bilinear"),
        lambda x: Tensor.interpolate(x, size=out_sz, mode="linear"), atol=1e-4)

  @slow_test
  def test_interpolate_bilinear_corners_aligned(self):
    for in_sz, out_sz in [((12,20),(9,31)), ((12,9),(31,20)), ((9,31),(20,12))]:
      helper_test_op([(2,3)+in_sz],
        lambda x: torch.nn.functional.interpolate(x, size=out_sz, mode="bilinear", align_corners=True),
        lambda x: Tensor.interpolate(x, size=out_sz, mode="linear", align_corners=True), atol=1e-4)

  def test_interpolate_trilinear(self):
    for in_sz, out_sz in [((5,2,8),(3,6,4))]:
      helper_test_op([(2,3)+in_sz],
        lambda x: torch.nn.functional.interpolate(x, size=out_sz, mode="trilinear"),
        lambda x: Tensor.interpolate(x, size=out_sz, mode="linear"), atol=1e-4)

  def test_interpolate_trilinear_corners_aligned(self):
    for in_sz, out_sz in [((5,2,8),(3,6,4))]:
      helper_test_op([(2,3)+in_sz],
        lambda x: torch.nn.functional.interpolate(x, size=out_sz, mode="trilinear", align_corners=True),
        lambda x: Tensor.interpolate(x, size=out_sz, mode="linear", align_corners=True), atol=1e-4)

  @slow_test
  def test_cat(self):
    for dim in range(-2, 3):
      helper_test_op([(45,65,9), (45,65,9), (45,65,9)], lambda x,y,z: torch.cat((x,y,z), dim), lambda x,y,z: x.cat(y, z, dim=dim))

    # zero in non-cat axis
    helper_test_op([(45,0,9), (45,0,9), (45,0,9)], lambda x,y,z: torch.cat((x,y,z), 0), lambda x,y,z: x.cat(y, z, dim=0))

    # zero in cat axis
    helper_test_op([(45,0,9), (45,1,9), (45,2,9)], lambda x,y,z: torch.cat((x,y,z), 1), lambda x,y,z: x.cat(y, z, dim=1))
    helper_test_op([(45,0,9), (45,0,9), (45,0,9)], lambda x,y,z: torch.cat((x,y,z), 1), lambda x,y,z: x.cat(y, z, dim=1))

    with self.assertRaises(IndexError):
      a = Tensor(3.14)
      a.cat(a)

  def test_multicat(self):
    helper_test_op(None, lambda x,y: torch.cat((x,y)), lambda x,y: x.cat(y), vals=[[True],[2.,3.]], forward_only=True)
    for dim in range(-1, 2):
      helper_test_op([(45,65), (45,65), (45,65)], lambda x,y,z: torch.cat((x,y,z), dim), lambda x,y,z: x.cat(y, z, dim=dim))

  @unittest.skipIf(COMPILE_ONLY, "test requires runtime")
  def test_stack(self):
    for dim in range(-1, 3):
      helper_test_op([(5,6,3), (5,6,3), (5,6,3)], lambda x, y, z: torch.stack((x, y, z), dim), lambda x, y, z: Tensor.stack(x, y, z, dim=dim))
      helper_test_op([(5,6,3), (5,6,3), (5,6,3)], lambda x, y, z: torch.stack((x, y, z), dim), lambda x, y, z: Tensor.stack((x, y, z), dim=dim))

    with self.assertRaises(IndexError):
      Tensor.stack(Tensor.randn(45, 65, 3), dim=77)
    with self.assertRaises(ValueError):
      Tensor.stack((Tensor([1, 2]), Tensor([3, 4])), Tensor([5, 6]))

    np.testing.assert_allclose(Tensor.stack(Tensor(3.14), Tensor(3.14)).numpy(), np.array([3.14, 3.14]))

  def test_stack_max(self):
    helper_test_op(None, lambda x, y: torch.stack((x, y)).max(axis=0)[0], lambda x, y: Tensor.stack(x, y).max(axis=0), vals=[[1.], [2.]])

  def test_repeat(self):
    x = Tensor.randn(4, 6, 3)
    base_repeats = [2, 4, 3]

    for reps in [[], [4], [2, 1], [3, 2, 2]]:
      repeats = base_repeats + reps
      helper_test_op([(4, 6, 3)], lambda x: x.repeat(*repeats), lambda x: x.repeat(repeats))
      helper_test_op([()], lambda x: x.repeat(*repeats), lambda x: x.repeat(repeats))

    with self.assertRaises(ValueError):
      x.repeat((2, 4))

    np.testing.assert_allclose(x.repeat((2, 0, 4)).numpy(), Tensor.zeros(8, 0, 12).numpy())

  def test_repeat_interleave(self):
    helper_test_op([(3, 3)], lambda x: x.repeat_interleave(6))
    helper_test_op([(3, 3)], lambda x: x.repeat_interleave(2, 1))
    helper_test_op([(3, 3)], lambda x: x.repeat_interleave(2, 0))
    helper_test_op([(3, 3)], lambda x: x.repeat_interleave(2, -1))
    helper_test_op([(3, 3)], lambda x: x.repeat_interleave(2, -2))

  def test_clip(self):
    helper_test_op([(45,65)], lambda x: x.clip(-2.3, 1.2))
    # NOTE: torch set backward to 1 at the boundaries
    # https://github.com/pytorch/pytorch/blob/7a41b66367c38d0af3e8a90f7be48d6b281e7bca/tools/autograd/derivatives.yaml#L421
    helper_test_op(None, lambda x: x.clip(-2.5, 1.5), vals=[[-3.0, -2.5, 0, 1.5, 2]])
    helper_test_op([(45,65)], lambda x: x.clip(0, 0))
    helper_test_op([(45,65)], lambda x: x.clip(10, 100))
    helper_test_op([(45,65)], lambda x: x.clip(0, 0.1))
    helper_test_op([(45,65)], lambda x: x.clip(-0.3, -0.2))
    helper_test_op([(45,65)], lambda x: x.clip(3, 0))  # min > max
    helper_test_op([(45,65)], lambda x: x.clip(None, 0))
    helper_test_op([(45,65)], lambda x: x.clip(0, None))
    self.helper_test_exception([(45,65)], lambda x: x.clip(None, None), expected=RuntimeError)

  def test_matvecmat(self):
    helper_test_op([(1,128), (128,128), (128,128)], lambda x,y,z: (x@y).relu()@z)

  def test_matvec(self):
    helper_test_op([(1,128), (128,128)], lambda x,y: (x@y).relu())

  @unittest.skipIf(COMPILE_ONLY, "test requires runtime")
  def test_inf_where(self):
    x = Tensor.full((3, 3), float("inf"))
    n = (x < 0).where(x, 1).numpy()
    assert np.all(n == 1.)

  def _get_index_randoms(self):
    # indices cannot have gradient
    a = torch.randint(low=-1, high=1, size=(2,1,1,1,1,1), dtype=torch.int64, requires_grad=False)
    b = torch.randint(high=1, size=(1,3,1,1,1,1), dtype=torch.int64, requires_grad=False)
    c = torch.randint(low=-5, high=5, size=(1,1,4,1,1,1), dtype=torch.int64, requires_grad=False)
    d = torch.randint(high=4, size=(2,1,1,5,1,1), dtype=torch.int64, requires_grad=False)
    e = torch.randint(high=1, size=(1,1,1,1,6,1), dtype=torch.int64, requires_grad=False)
    i, j, k, o, p = [Tensor(tor.detach().cpu().numpy().astype(np.int32)) for tor in [a,b,c,d,e]]
    return a,b,c,d,e,i,j,k,o,p

  def test_fancy_indexing_inf(self):
    data = [math.inf, -math.inf, math.nan]
    helper_test_op((), lambda: torch.tensor(data)[torch.tensor([0, 1, 2])], lambda: Tensor(data)[Tensor([0, 1, 2])])

  def test_fancy_indexing_index_dtypes(self):
    helper_test_op((), lambda: torch.tensor([10., 20., 30., 40.])[torch.tensor([1, 2, 3, 0])],
                       lambda: Tensor([10., 20., 30., 40.])[Tensor([1, 2, 3, 0], dtype=dtypes.uint8)])

  def test_slice_fancy_indexing_with_tensors(self):
    # indexing using idx with different dim
    helper_test_op([(2,3)], lambda x: x[torch.tensor([[0,0,0],[0,0,0]]), torch.tensor(1)],
                            lambda x: x[Tensor([[0,0,0],[0,0,0]]), Tensor(1)])
    helper_test_op([(2,3)], lambda x: x[torch.tensor([1]), torch.tensor([[0,0,0],[0,0,0]])],
                            lambda x: x[Tensor([1]), Tensor([[0,0,0],[0,0,0]])])
    helper_test_op([(2,3)], lambda x: x[torch.tensor([[0,0,0],[0,0,0]]), torch.tensor([2,1,1])],
                            lambda x: x[Tensor([[0,0,0],[0,0,0]]), Tensor([2,1,1])])
    helper_test_op([(2,3)], lambda x: x[torch.tensor([[0,1,-1],[-1,-2,0]]), torch.tensor([2,1,-1])],
                            lambda x: x[Tensor([[0,1,-1],[-1,-2,0]]), Tensor([2,1,-1])])

  @slow_test
  def test_slice_fancy_indexing_list_with_tensors(self):
    a,b,c,d,e,i,j,k,o,p = self._get_index_randoms()
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[(a,)], lambda x: x[(i,)])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[(a,1)], lambda x: x[(i,1)])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[(a,(1,1))], lambda x: x[(i,(1,1))])

  def test_slice_fancy_indexing_errors(self):
    a = Tensor.ones(10,11,12)
    # tensors used as indices must be int tensors
    with self.assertRaises(IndexError): a[Tensor(1.1)]
    with self.assertRaises(IndexError): a[[1.1]]
    with self.assertRaises(IndexError): a[Tensor([True, False])]
    with self.assertRaises(IndexError): a[[True, False]]
    # shape mismatch, cannot broadcast. either exception is okay
    with self.assertRaises((IndexError, ValueError)): a[Tensor.randint(3,1,1,1), Tensor.randint(1,4,1,1), Tensor.randint(2,4,4,1)]
    with self.assertRaises((IndexError, ValueError)): a[Tensor.randint(3,1,1,1), Tensor.randint(1,4,1,1,1)]
    # this is fine
    helper_test_op([(5, 6)], lambda x: x[[True, False, 2]])

  def test_gather(self):
    # indices cannot have gradient
    # indices cannot be negative (torch gather)
    b = torch.randint(3, size=[3,4,5], dtype=torch.int64, requires_grad=False)
    a = Tensor(b.detach().cpu().numpy().astype(np.int32), dtype=dtypes.int32)
    helper_test_op([(4,5,6)], lambda x: x.gather(dim=0, index=b), lambda x: x.gather(dim=0, index=a))
    helper_test_op([(4,5,6)], lambda x: x.gather(dim=1, index=b), lambda x: x.gather(dim=1, index=a))
    helper_test_op([(4,5,6)], lambda x: x.gather(dim=2, index=b), lambda x: x.gather(dim=2, index=a))
    helper_test_op([(3,4,5)], lambda x: x.gather(dim=0, index=b), lambda x: x.gather(dim=0, index=a))
    helper_test_op([(4,5,6)], lambda x: x.gather(dim=-1, index=b), lambda x: x.gather(dim=-1, index=a))
    helper_test_op([(4,5,6)], lambda x: x.gather(dim=-2, index=b), lambda x: x.gather(dim=-2, index=a))
    helper_test_op([(4,5,6)], lambda x: x.gather(dim=-3, index=b), lambda x: x.gather(dim=-3, index=a))
    self.helper_test_exception([(4,5,6)], lambda x: x.gather(dim=0, index=torch.tensor([1], dtype=torch.int64)),
                                          lambda x: x.gather(dim=0, index=Tensor([1], dtype=dtypes.int32)), expected=(RuntimeError, AssertionError))
    self.helper_test_exception([(2,1,1)], lambda x: x.gather(dim=0, index=b),
                                          lambda x: x.gather(dim=0, index=a), expected=(RuntimeError, AssertionError))
    helper_test_op(None, lambda x: x.gather(dim=0, index=torch.tensor([2, 1, 0, 1, 2], requires_grad=False)),
                         lambda x: x.gather(dim=0, index=Tensor([2, 1, 0, 1, 2])),
                         vals=[[1., 2., 3.]])
    # gather with inf values
    helper_test_op(None, lambda x: x.gather(dim=0, index=torch.tensor([2, 1, 0, 1, 2], requires_grad=False)),
                         lambda x: x.gather(dim=0, index=Tensor([2, 1, 0, 1, 2])),
                         vals=[[-float("inf"), 2., 3.]])

  def test_gather_bool_index(self):
    helper_test_op(None, lambda x,y: x.gather(dim=0, index=y.bool().long()),
                         lambda x,y: x.gather(dim=0, index=y.cast(dtypes.bool).cast(dtypes.int)),
                         vals=[[1., 2., 3.], [0.5, 0., 2.]], forward_only=True)
    helper_test_op(None, lambda x,y: x[y.bool().long()], lambda x,y: x[y.cast(dtypes.bool).cast(dtypes.int)],
                         vals=[[1., 2., 3.], [0.5, 0., 2.]], forward_only=True)

  def test_scatter(self):
    b = torch.randint(3, size=[3,4,5], dtype=torch.int64, requires_grad=False)
    a = Tensor(b.detach().cpu().numpy().astype(np.int32), dtype=dtypes.int32)
    for dim in (0,1,2,-1,-2,-3):
      helper_test_op([(4,5,6), (4,5,6)], lambda x,src: x.scatter(dim=dim, index=b, src=src),
                                         lambda x,src: x.scatter(dim=dim, index=a, src=src), forward_only=True)

    helper_test_op([(3,4,5), (3,4,5)], lambda x,src: x.scatter(dim=1, index=b, src=src),
                                       lambda x,src: x.scatter(dim=1, index=a, src=src), forward_only=True)
    helper_test_op([(10,3,10), (10,10,10)], lambda x,src: x.scatter(dim=1, index=b, src=src),
                                            lambda x,src: x.scatter(dim=1, index=a, src=src), forward_only=True)

    self.helper_test_exception([(2,3,10), (10,10,10)], lambda x,src: x.scatter(dim=1, index=b, src=src),
                                                       lambda x,src: x.scatter(dim=1, index=a, src=src), expected=(RuntimeError, AssertionError))
    self.helper_test_exception([(10,3,10), (10,3,10)], lambda x,src: x.scatter(dim=1, index=b, src=src),
                                                       lambda x,src: x.scatter(dim=1, index=a, src=src), expected=(RuntimeError, AssertionError))
    self.helper_test_exception([(3,4,5), (3,4,5)], lambda x,src: x.scatter(dim=1, index=b, src=src, mode="typo"),
                                                   lambda x,src: x.scatter(dim=1, index=a, src=src, mode="typo"), expected=TypeError)
    self.helper_test_exception([(3,4,5), (3,4,5)], lambda x,src: x.half().scatter(dim=1, index=b, src=src),
                                                   lambda x,src: x.half().scatter(dim=1, index=a, src=src), expected=RuntimeError)

    helper_test_op([(4,5,6)], lambda x: x.scatter(dim=1, index=b, value=3), lambda x: x.scatter(dim=1, index=a, src=3), forward_only=True)
    helper_test_op([(4,5,6)], lambda x: x.scatter(dim=1, index=b, value=float("inf")),
      lambda x: x.scatter(dim=1, index=a, src=float("inf")), forward_only=True)

    # overlapping indices with 0s
    b = torch.tensor([0,0], requires_grad=False)
    a = Tensor(b.detach().cpu().numpy().astype(np.int32), dtype=dtypes.int32)
    helper_test_op(None,
      lambda x,src: x.scatter(0, b, src),
      lambda x,src: x.scatter(0, a, src), forward_only=True,
      vals=[[1.,2.,3.,4.], [1.,0.]])

  def test_scatter_add(self):
    b = torch.randint(3, size=[3,4,5], dtype=torch.int64, requires_grad=False)
    a = Tensor(b.detach().cpu().numpy().astype(np.int32), dtype=dtypes.int32)
    helper_test_op([(4,5,6)], lambda x: x.scatter(dim=1, index=b, value=float("inf"), reduce="add"),
      lambda x: x.scatter(dim=1, index=a, src=float("inf"), reduce="add"), forward_only=True)

    # TODO: fails for webgpu
    if Device.DEFAULT != "WEBGPU":
      helper_test_op([(4,5,6)],
        lambda x: x.scatter(1, b, float("nan"), reduce="add"),
        lambda x: x.scatter(1, a, float("nan"), reduce="add"), forward_only=True)

  def test_scatter_mul(self):
    b = torch.randint(3, size=[3,4,5], dtype=torch.int64, requires_grad=False)
    a = Tensor(b.detach().cpu().numpy().astype(np.int32), dtype=dtypes.int32)
    helper_test_op([(4,5,6)], lambda x: x.scatter(dim=1, index=b, value=float("inf"), reduce="multiply"),
      lambda x: x.scatter(dim=1, index=a, src=float("inf"), reduce="multiply"), forward_only=True)

    # TODO: fails for webgpu
    if Device.DEFAULT != "WEBGPU":
      helper_test_op([(4,5,6)],
        lambda x: x.scatter(1, b, float("nan"), reduce="multiply"),
        lambda x: x.scatter(1, a, float("nan"), reduce="multiply"), forward_only=True)

  @slow_test
  def test_scatter_reduce(self):
    b = torch.randint(3, size=[3,4,5], dtype=torch.int64, requires_grad=False)
    a = Tensor(b.detach().cpu().numpy().astype(np.int32))
    for reduce in ("sum", "prod", "mean", "amin", "amax"):
      for dim in (-1,1,-3):
        helper_test_op([(3,4,5), (3,4,5)],
          lambda x,src: x.scatter_reduce(dim=dim, index=b, src=src, reduce=reduce),
          lambda x,src: x.scatter_reduce(dim=dim, index=a, src=src, reduce=reduce), forward_only=True)
        helper_test_op([(3,4,5), (3,4,5)],
          lambda x,src: x.scatter_reduce(dim=dim, index=b, src=src, reduce=reduce, include_self=False),
          lambda x,src: x.scatter_reduce(dim=dim, index=a, src=src, reduce=reduce, include_self=False), forward_only=True)

  def test_scatter_reduce_prod_zeros(self):
    b = torch.randint(3, size=[3,4,5], dtype=torch.int64, requires_grad=False)
    a = Tensor(b.detach().cpu().numpy().astype(np.int32), dtype=dtypes.int32)
    x = Tensor.zeros([4,5,6]).float()
    y = torch.zeros([4,5,6]).float()
    helper_test_op([(4,5,6)],
      lambda src: y.scatter_reduce(dim=1, index=b, src=src, reduce="prod"),
      lambda src: x.scatter_reduce(dim=1, index=a, src=src, reduce="prod"), forward_only=True)

  @slow_test
  def test_scaled_dot_product_attention_causal(self):
    helper_test_op([(32,8,16,64), (32,8,16,64), (32,8,16,64)],
                   lambda x,y,z: torch.nn.functional.scaled_dot_product_attention(x,y,z,is_causal=True),
                   lambda x,y,z: Tensor.scaled_dot_product_attention(x,y,z,is_causal=True))

    self.helper_test_exception([(32,8,16,64), (32,8,16,64), (32,8,16,64), (32,8,16,16)],
      lambda x,y,z,m: torch.nn.functional.scaled_dot_product_attention(x,y,z,is_causal=True,attn_mask=m),
      lambda x,y,z,m: Tensor.scaled_dot_product_attention(x,y,z,is_causal=True,attn_mask=m),
      expected=RuntimeError)

  @slow_test
  def test_scaled_dot_product_attention_gqa(self):
    helper_test_op([(32,32,16,64), (32,8,16,64), (32,8,16,64)],
                   lambda x,y,z: torch.nn.functional.scaled_dot_product_attention(x,y,z,enable_gqa=True),
                   lambda x,y,z: Tensor.scaled_dot_product_attention(x,y,z,enable_gqa=True))

  def test_binary_crossentropy_reductions(self):
    for kwargs in ({}, {"reduction": "sum"}, {"reduction": "none"}):
      helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy(x.sigmoid(), y.clip(0,1), **kwargs),
                                         lambda x,y: x.sigmoid().binary_crossentropy(y.clip(0,1), **kwargs))
      helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy_with_logits(x, y.clip(0,1), **kwargs),
                                         lambda x,y: x.binary_crossentropy_logits(y.clip(0,1), **kwargs))
  def test_binary_crossentropy_logits_pos_weights(self):
    pos_weight = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy_with_logits(x,y.clip(0,1),
                                                                                                        pos_weight=torch.tensor(pos_weight)),
                                       lambda x,y: x.binary_crossentropy_logits(y.clip(0,1),pos_weight=Tensor(pos_weight)))

  def test_cross_entropy_class_probabilities(self):
    helper_test_op([(32,), (32,)], lambda x,y: torch.nn.functional.cross_entropy(x, y), lambda x,y: x.cross_entropy(y))
    helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.cross_entropy(x, y, label_smoothing=0.3),
                                       lambda x,y: x.cross_entropy(y, label_smoothing=0.3))
    helper_test_op([(32,4,4,4), (32,4,4,4)], lambda x,y: torch.nn.functional.cross_entropy(x, y), lambda x,y: x.cross_entropy(y))

  def test_cross_entropy_class_indices(self):
    classes = np.random.randint(0, 10, (32,), dtype=np.int32).tolist()
    helper_test_op([(32,10)], lambda x: torch.nn.functional.cross_entropy(x, torch.tensor(classes)),
                              lambda x: x.cross_entropy(Tensor(classes)))
    self.helper_test_exception([(32,10), (32,1)], lambda x,y: torch.nn.functional.cross_entropy(x, y),
                                                  lambda x,y: x.cross_entropy(y), expected=(AssertionError, RuntimeError))

  def test_cross_entropy_reductions(self):
    for r in ("mean", "sum", "none"):
      helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.cross_entropy(x, y, reduction=r),
                                         lambda x,y: x.cross_entropy(y, reduction=r))
    self.helper_test_exception([(32,10), (32,10)], lambda x,y: torch.nn.functional.cross_entropy(x, y, reduction="typo"),
                                                   lambda x,y: x.cross_entropy(y, reduction="typo"), expected=ValueError)

  def test_sparse_categorical_crossentropy(self):
    classes = np.random.randint(0, 10, (12,), dtype=np.int32).tolist()
    helper_test_op([(12,10)], lambda x: torch.nn.CrossEntropyLoss()(x, torch.tensor(classes)),
                              lambda x: x.sparse_categorical_crossentropy(Tensor(classes)))

    # combine args
    helper_test_op([(12,10)],
                   lambda x: torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=classes[0], label_smoothing=0.3)(x, torch.tensor(classes)),
                   lambda x: x.sparse_categorical_crossentropy(Tensor(classes), reduction="mean", ignore_index=classes[0], label_smoothing=0.3))

    # with batch. somehow this does not match torch
    classes = np.random.randint(0, 10, (3,12), dtype=np.int32).tolist()
    helper_test_op([(3,12,10)], lambda x: torch.nn.CrossEntropyLoss()(x.permute(0,2,1), torch.tensor(classes)),
                                lambda x: x.sparse_categorical_crossentropy(Tensor(classes)))

  def test_sparse_categorical_crossentropy_reductions(self):
    for r in ("mean", "sum", "none"):
      classes = np.random.randint(0, 10, (12,), dtype=np.int32).tolist()
      helper_test_op([(12,10)], lambda x: torch.nn.CrossEntropyLoss(reduction=r)(x, torch.tensor(classes)),
                                lambda x: x.sparse_categorical_crossentropy(Tensor(classes), reduction=r))

  def test_nll_loss_3d(self):
    target = np.random.randint(0, 10, (32,3,3,3), dtype=np.int32).tolist()
    helper_test_op([(32,10,3,3,3)],
                   lambda x: torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x, dim=1), torch.tensor(target)),
                   lambda x: x.log_softmax(axis=1).nll_loss(Tensor(target)))

  def test_nll_loss_unbatched(self):
    helper_test_op([(2,)], lambda x: torch.nn.functional.nll_loss(x, torch.tensor(1)),
                          lambda x: x.nll_loss(Tensor(1)))

  def test_nll_loss_reductions(self):
    target = np.random.randint(0, 10, (32,), dtype=np.int32).tolist()
    for kwargs in ({}, {"reduction": "sum"}, {"reduction": "none"}):
      helper_test_op([(32,10)],
        lambda x: torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x, dim=1), torch.tensor(target), **kwargs),
        lambda x: x.log_softmax(axis=1).nll_loss(Tensor(target), **kwargs))
    self.helper_test_exception([(32,10)],
      lambda x: torch.nn.functional.nll_loss(x, torch.tensor(target), reduction="typo"),
      lambda x: x.nll_loss(Tensor(target), reduction="typo"), expected=ValueError)

  def test_nll_loss_weight(self):
    target = np.random.randint(0, 10, (32,), dtype=np.int32).tolist()
    weight = np.random.normal(0, 1, (10,)).astype(np.float32).tolist()
    for r in ("mean", "sum", "none"):
      helper_test_op([(32,10)],
        lambda x: torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x, dim=1), torch.tensor(target), torch.tensor(weight), reduction=r),
        lambda x: x.log_softmax(axis=1).nll_loss(Tensor(target), Tensor(weight), reduction=r))

  def test_nll_loss_ignore_index(self):
    logits = [[2.0, 0.5, -1.0],
              [1.5, 2.5, -0.5],
              [0.0, -2.0, 1.0]]
    target = [0, 1, 2]
    helper_test_op(None, lambda x: torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x, dim=1), torch.tensor(target), ignore_index=1),
                         lambda x: x.log_softmax().nll_loss(Tensor(target), ignore_index=1),
                         vals=[logits])

  def test_one_hot(self):
    data = [1, 2, 4]
    helper_test_op([], lambda: torch.nn.functional.one_hot(torch.tensor(data), 6).type(torch.int32),
                       lambda: Tensor(data).one_hot(6), forward_only=True)
    # like jax.nn.one_hot, num_classes must be non-negative (torch accepts -1 for auto-inference, we don't)
    with self.assertRaises(ValueError): Tensor(data).one_hot(-1)
    data = [[[1, 2, 3], [0, 3, 5]], [[1, 2, 3], [0, 3, 5]]]
    helper_test_op([], lambda: torch.nn.functional.one_hot(torch.tensor(data), 8).type(torch.int32),
                       lambda: Tensor(data).one_hot(8), forward_only=True)

  def test_masked_fill(self):
    helper_test_op([(32,10)], lambda x: x.masked_fill((x>0.1).detach(), -math.inf))
    helper_test_op([(32,10)], lambda x: x.masked_fill((x<0.1).detach(), -math.inf))

  @unittest.skipIf((DEV.interface.startswith("MOCK") or Device.DEFAULT == "PYTHON"), "very slow on MOCKGPU because reduce does not fold")
  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "webgpu runtime issue")
  @unittest.skipIf(Device.DEFAULT == "QCOM", "QCOM fails with: Resource deadlock avoided")
  @unittest.skipIf(Device.DEFAULT == "CPU" and DEV.renderer == "LVP", "extremely slow with LVP")
  def test_masked_select(self):
    helper_test_op([(32, 10)], lambda x: x.masked_select(x>0.5), lambda x: x.masked_select(x>0.5), forward_only=True)
    helper_test_op([(32, 10)], lambda x: x.masked_select(torch.tensor(True)), lambda x: x.masked_select(Tensor(True)), forward_only=True)

  @unittest.skipIf(COMPILE_ONLY, "test requires runtime")
  def test_masked_select_size(self):
    t = Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
    mask = Tensor([True, False, True, False, True, False, False, False, True])
    np.testing.assert_equal(t.masked_select(mask, size=4).numpy(), [0, 2, 4, 8])
    np.testing.assert_equal(t.masked_select(mask, size=6, fill_value=-1).numpy(), [0, 2, 4, 8, -1, -1])
    np.testing.assert_equal(t.masked_select(mask, size=2).numpy(), [0, 2])
    np.testing.assert_equal(Tensor([], dtype=dtypes.int32).masked_select(Tensor([], dtype=dtypes.bool), size=2, fill_value=-1).numpy(), [-1, -1])
    # fill_value must not alter output dtype
    self.assertEqual(Tensor([1.0, 2.0]).masked_select(Tensor([True, False]), size=3, fill_value=-1).dtype, dtypes.default_float)

  @slow_test
  def test_nonzero(self):
    helper_test_op([(32, 10)], lambda x: (x>0.5).nonzero().int(), lambda x: (x>0.5).nonzero(), forward_only=True)
    helper_test_op([(20,)], lambda x: (x>0.5).nonzero().int(), lambda x: (x>0.5).nonzero(), forward_only=True)
    helper_test_op([(10, 5, 3)], lambda x: (x>0.5).nonzero().int(), lambda x: (x>0.5).nonzero(), forward_only=True)
    for v in (0, 1, 0.0, 2.5, True, False):
      helper_test_op(None, lambda x: x.nonzero().int(), lambda x: x.nonzero(), vals=[v], forward_only=True)

  @unittest.skipIf(COMPILE_ONLY, "test requires runtime")
  def test_nonzero_size(self):
    np.testing.assert_equal(Tensor([1, 0, 2, 0, 3]).nonzero(size=3).numpy(), [[0], [2], [4]])
    np.testing.assert_equal(Tensor([1, 0, 2, 0, 3]).nonzero(size=5, fill_value=-1).numpy(), [[0], [2], [4], [-1], [-1]])
    np.testing.assert_equal(Tensor([[1, 0], [0, 2]]).nonzero(size=2).numpy(), [[0, 0], [1, 1]])
    self.assertEqual(Tensor(5).nonzero(size=4).shape, (4, 0))
    np.testing.assert_equal(Tensor([], dtype=dtypes.int32).nonzero(size=3, fill_value=-1).numpy(), [[-1], [-1], [-1]])
    # fill_value must not promote dtype to float
    self.assertEqual(Tensor([1, 0]).nonzero(size=3, fill_value=-1.5).dtype, dtypes.default_int)

  def test_cast(self):
    helper_test_op([(3, 3)], lambda x: x.float())
    helper_test_op(None, lambda x: x.float(), vals=[[0, 1, 2, 3]], forward_only=True)
    helper_test_op(None, lambda x: x.float(), vals=[[True, False]], forward_only=True)
    helper_test_op([(3, 3)], lambda x: x.int(), forward_only=True)
    helper_test_op([(3, 3)], lambda x: x.bool(), forward_only=True)

  def test_bitcast(self):
    helper_test_op([(3, 3)], lambda x: x.view(torch.int32), lambda x: x.bitcast(dtypes.int32), forward_only=True)

  def test_int_or(self):
    t = (Tensor([0], dtype='int') | 0xFFFFFFFF).item()
    if not COMPILE_ONLY: assert t == -1

class TestOpsUint8(unittest.TestCase):
  def test_cast(self):
    helper_test_op([(2,3,64,64)], lambda x: x.type(torch.uint8), lambda x: x.cast('uint8'), forward_only=True, low=0, high=255)

  def test_cast_relu(self):
    helper_test_op([(2,3,64,64)], lambda x: x.relu().type(torch.uint8), lambda x: x.relu().cast('uint8'), forward_only=True)

  def test_interpolate_bilinear(self):
    out_sz = (10, 10)
    helper_test_op([(2,3,64,64)],
      lambda x: torch.nn.functional.interpolate((10*x).relu().type(torch.uint8), size=out_sz, mode="bilinear"),
      lambda x: Tensor.interpolate((10*x).relu().cast('uint8'), size=out_sz, mode="linear"), forward_only=True)

  def test_interpolate_nearest(self):
    out_sz = (10, 10)
    helper_test_op([(2,3,64,64)],
      lambda x: torch.nn.functional.interpolate((10*x).relu().type(torch.uint8), size=out_sz, mode="nearest"),
      lambda x: Tensor.interpolate((10*x).relu().cast('uint8'), size=out_sz, mode="nearest"), forward_only=True)

  def test_interpolate_nearest_exact(self):
    out_sz = (10, 10)
    helper_test_op([(2,3,64,64)],
      lambda x: torch.nn.functional.interpolate((10*x).relu().type(torch.uint8), size=out_sz, mode="nearest-exact"),
      lambda x: Tensor.interpolate((10*x).relu().cast('uint8'), size=out_sz, mode="nearest-exact"), forward_only=True)

  def test_min(self):
    helper_test_op(None,
      lambda x: x.type(torch.uint8).min(),
      lambda x: x.cast(dtypes.uint8).min(), forward_only=True, vals=[[[0, 1, 2], [3, 4, 5]]])
    helper_test_op(None,
      lambda x: x.type(torch.uint8).min(),
      lambda x: x.cast(dtypes.uint8).min(), forward_only=True, vals=[[0, 128, 255, 64, 32, 16]])

if __name__ == '__main__':
  np.random.seed(1337)
  unittest.main(verbosity=2)
