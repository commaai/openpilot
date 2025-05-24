import time, math, unittest, functools
import numpy as np
from typing import List, Callable
import torch
import warnings
from tinygrad.helpers import getenv, IMAGE, DEBUG, CI, Context, TRANSCENDENTAL, DEVECTORIZE, OSX
from tinygrad import Tensor, Device, dtypes
from tinygrad.tensor import _to_np_dtype
from tinygrad.device import is_dtype_supported

if getenv("TINY_BACKEND"):
  import tinygrad.frontend.torch # noqa: F401 # pylint: disable=unused-import
  torch.set_default_device("tiny")

if CI:
  warnings.filterwarnings("ignore", message="Non-empty compiler output encountered")

FORWARD_ONLY = getenv("FORWARD_ONLY", 0)
PRINT_TENSORS = getenv("PRINT_TENSORS", 0)

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
  ret = tinygrad_fxn(*tst).realize()
  tinygrad_fp = time.monotonic() - st

  def compare(s, tinygrad_output, torch_output, atol, rtol):
    if PRINT_TENSORS: print(s, tinygrad_output, torch_output)
    try:
      assert tinygrad_output.shape == torch_output.shape, f"shape mismatch: tinygrad={tinygrad_output.shape} | torch={torch_output.shape}"
      assert tinygrad_output.dtype == torch_output.dtype, f"dtype mismatch: tinygrad={tinygrad_output.dtype} | torch={torch_output.dtype}"
      if np.issubdtype(tinygrad_output.dtype, np.floating):
        np.testing.assert_allclose(tinygrad_output, torch_output, atol=atol, rtol=rtol)
      else:
        np.testing.assert_equal(tinygrad_output, torch_output)
    except Exception as e:
      raise Exception(f"{s} failed shape {tinygrad_output.shape}: {e}")

  if DEBUG >= 6:
    np.set_printoptions(linewidth=200, suppress=True)
    print(ret.numpy())
    print(out.detach().cpu().numpy())
  compare("forward pass", ret.numpy(), out.detach().cpu().numpy(), atol=atol, rtol=rtol)

  torch_fbp, tinygrad_fbp = np.nan, np.nan
  if not forward_only and not FORWARD_ONLY and ts and tst:
    st = time.monotonic()
    torch_grads = torch.autograd.grad(torch_fxn(*ts).sum(), ts)
    torch_fbp = time.monotonic() - st

    st = time.monotonic()
    # NOTE: we now have to recompute the forward pass since we realized it
    tiny_grads = tinygrad_fxn(*tst).sum().gradient(*tst)
    Tensor.realize(*tiny_grads)
    tinygrad_fbp = time.monotonic() - st

    for i, (t, torch_grad) in enumerate(zip(tiny_grads, torch_grads)):
      compare(f"backward pass tensor {i}", t.numpy(), torch_grad.detach().cpu().numpy(), atol=grad_atol, rtol=grad_rtol)

  if not CI:
    print("\ntesting %40r   torch/tinygrad fp: %.2f / %.2f ms  bp: %.2f / %.2f ms " % \
          (shps, torch_fp*1000, tinygrad_fp*1000, torch_fbp*1000, tinygrad_fbp*1000), end="")

def prepare_test_op(low, high, shps, vals, forward_only=False):
  if shps is None:
    ts = [torch.tensor(x, requires_grad=(not forward_only)) for x in vals]
  else:
    np.random.seed(0)
    np_data = [np.random.uniform(low=low, high=high, size=size).astype(_to_np_dtype(dtypes.default_float)) for size in shps]
    ts = [torch.tensor(data, requires_grad=(not forward_only)) for data in np_data]
  for i in range(len(ts)):
    # NOTE: torch default int64 for python ints input
    if ts[i].dtype == torch.int64: ts[i] = ts[i].type(torch.int32)
  tst = [Tensor(x.detach().cpu().numpy(), requires_grad=(not forward_only and not FORWARD_ONLY)) for x in ts]
  return ts, tst

class TestOps(unittest.TestCase):

  def helper_test_exception(self, shps, torch_fxn, tinygrad_fxn, expected, forward_only=False, exact=False, vals=None, low=-1.5, high=1.5):
    if getenv("MOCKGPU") and Device.DEFAULT == "NV": self.skipTest('helper_test_exception fails in CI CUDA')
    ts, tst = prepare_test_op(low, high, shps, vals, forward_only)
    with self.assertRaises(expected) as torch_cm:
      torch_fxn(*ts)
    with self.assertRaises(expected) as tinygrad_cm:
      tinygrad_fxn(*tst)
    if exact: self.assertEqual(str(torch_cm.exception), str(tinygrad_cm.exception))
    if not CI: print("\ntesting %40r   torch/tinygrad exception: %s / %s" % (shps, torch_cm.exception, tinygrad_cm.exception), end="")

  def test_full_like(self):
    a = Tensor([[1,2,3],[4,5,6]], dtype=dtypes.float32)
    b = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32)
    helper_test_op([], lambda: torch.full_like(b, 4), lambda: Tensor.full_like(a, 4), forward_only=True)

    a = Tensor([[1,2,3],[4,5,6]], dtype=dtypes.int32)
    b = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.int32)
    helper_test_op([], lambda: torch.full_like(b, 4), lambda: Tensor.full_like(a, 4), forward_only=True)

  def test_full(self):
    helper_test_op([], lambda: torch.full((45,65), 4, dtype=torch.int32), lambda: Tensor.full((45,65), 4), forward_only=True)

  def test_negative_dims(self):
    creation_methods: List[Callable[..., Tensor]] = [
      Tensor.empty,
      Tensor.rand,
      Tensor.zeros,
      Tensor.ones,
      Tensor.randn,
      Tensor.randint,
      Tensor.normal,
      Tensor.uniform,
      Tensor.scaled_uniform,
      Tensor.glorot_uniform
    ]

    for method in creation_methods:
      with self.assertRaises(ValueError): method(-3, 2)
      with self.assertRaises(ValueError): method((2, -3))
      with self.assertRaises(ValueError): method((2, -3, 0))

  def test_negative_dims_full(self):
    with self.assertRaises(ValueError): Tensor.full((-3,), 2)
    with self.assertRaises(ValueError): Tensor.full((2, -3), 4)
    with self.assertRaises(ValueError): Tensor.full((2, -3, 0), 4)

  def test_negative_dims_eye(self):
    with self.assertRaises(ValueError): Tensor.eye(-3, 3)
    with self.assertRaises(ValueError): Tensor.eye(3, -3)
    with self.assertRaises(ValueError): Tensor.eye(-3, -3)

  def test_negative_dims_kaiming(self):
    creation_methods = [Tensor.kaiming_uniform, Tensor.kaiming_normal]
    for method in creation_methods:
      with self.assertRaises(ValueError): method(-3, 3)
      with self.assertRaises(ValueError): method((-3, 3), 3)
      with self.assertRaises(ValueError): method((-3, -3), 3)

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

  def test_meshgrid(self):
    x, xt = torch.tensor([0.,1.,2.], requires_grad=True), Tensor([0.,1.,2.], requires_grad=True)
    y, yt = torch.tensor([3.,4.,5.,6.], requires_grad=True), Tensor([3.,4.,5.,6.], requires_grad=True)
    z, zt = torch.tensor([7.,8.,9.], requires_grad=True), Tensor([7.,8.,9.], requires_grad=True)
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
      ten = xt.meshgrid(Tensor(10., requires_grad=True), yt, zt, indexing=indexing)
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

  # this is more complex and won't fold for a while
  def test_sum_cat_collapse(self):
    helper_test_op([], lambda: torch.cat([torch.ones(256,256), torch.zeros(256,64)], dim=1).sum(axis=1),
                       lambda: Tensor.cat(Tensor.ones(256,256), Tensor.zeros(256,64), dim=1).sum(axis=1), forward_only=True)

  def test_max_dont_collapse(self):
    helper_test_op([], lambda: torch.ones(256,256).max(1)[0], lambda: Tensor.ones(256,256).max(1), forward_only=True)

  def test_where(self):
    helper_test_op(
      [(100,)],
      lambda x: torch.where(x > 0.5, 4, 2).type(torch.int32),
      lambda x: (x > 0.5).where(4, 2), forward_only=True)

    for shps in [[(8,),(1,),(1,)], [(10,10),(10,),(10,)], [(100,)]*3, [(10,10)]*3]:
      helper_test_op(
        shps,
        lambda x, a, b: torch.where(x > 0.5, a, b),
        lambda x, a, b: (x > 0.5).where(a, b), forward_only=True)

  def test_where_permute(self):
    helper_test_op(
      [(5, 5)],
      lambda x: torch.where(x > 0.5, 4, 2).type(torch.int32).permute((1, 0)),
      lambda x: (x > 0.5).where(4, 2).permute((1, 0)), forward_only=True)

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
    tt1 = Tensor.ones(4, requires_grad=True)
    tt2 = Tensor.ones(4, requires_grad=True)
    self.assertRaises(RuntimeError, (tt1 != tt2).sum().backward)
    """
    tt = Tensor.randn(4, requires_grad=True)
    (tt*(tt != 0)).sum().backward()
    t = torch.tensor(tt.numpy(), requires_grad=True)
    (t*(t != 0)).sum().backward()
    np.testing.assert_allclose(t.grad.cpu().numpy(), tt.grad.numpy(), rtol=1e-5)

  def test_cmp_lt_backwards(self):
    # new grad zeroes these out
    """
    t1 = torch.ones(4, requires_grad=True)
    t2 = torch.ones(4, requires_grad=True)
    self.assertRaises(RuntimeError, (t1 < t2).sum().backward)
    tt1 = Tensor.ones(4, requires_grad=True)
    tt2 = Tensor.ones(4, requires_grad=True)
    self.assertRaises(RuntimeError, (tt1 < tt2).sum().backward)
    """
    tt = Tensor.randn(4, requires_grad=True)
    (tt*(tt < 0)).sum().backward()
    t = torch.tensor(tt.numpy(), requires_grad=True)
    (t*(t < 0)).sum().backward()
    np.testing.assert_allclose(t.grad.cpu().numpy(), tt.grad.numpy(), rtol=1e-5)

  # TODO: fix backward of these functions
  def test_trunc(self):
    helper_test_op([()], lambda x: x.trunc(), forward_only=True)
    helper_test_op([(45,35)], lambda x: x.trunc(), forward_only=True)
    helper_test_op(None, lambda x: x.trunc(), vals=[[1.499, 1.5, 1.501, 1.0, 2.1, 0.0, -5.0, -2.499, -2.5, -2.501]], forward_only=True)
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

  @unittest.skipIf(Device.DEFAULT == "WEBGPU" and CI, "isinf check of 'nan' fails on CI software-based vulkan")
  def test_isinf(self):
    val = [float('-inf'), 0., float('inf'), float('nan'), 1.1]
    helper_test_op(None, torch.isinf, Tensor.isinf, vals=[val], forward_only=True)
    np.testing.assert_equal(Tensor(val).isinf(detect_positive=True, detect_negative=False).numpy(), [False, False, True, False, False])
    np.testing.assert_equal(Tensor(val).isinf(detect_positive=False, detect_negative=True).numpy(), [True, False, False, False, False])

  def test_isnan(self):
    helper_test_op(None, torch.isnan, Tensor.isnan, vals=[[float('-inf'), 0., float('inf'), float('nan'), 1.1]], forward_only=True)

  def test_isfinite(self):
    helper_test_op(None, torch.isfinite, Tensor.isfinite, vals=[[float('-inf'), 0., float('inf'), float('nan'), 1.1]], forward_only=True)

  def test_lerp(self):
    helper_test_op([(45,35), (45,35), (45,35)], lambda x,y,z: x.lerp(y,z))
    helper_test_op(None, lambda x,y,z: x.lerp(y,z), vals=[[1.,2.,3.], [4.,5.,6.], 0.5])

  @unittest.skipIf(Device.DEFAULT == "QCOM", "OpenCL fails to compile this (both on GPU(qcom)/QCOM backends)")
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

  @unittest.skipIf(Device.DEFAULT == "QCOM", "OpenCL fails to compile this (both on GPU(qcom)/QCOM backends)")
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
                   vals=[[-1234, 0, 1234, dtypes.max(dtypes.int), dtypes.min(dtypes.int)], dtypes.max(dtypes.int)], forward_only=True)
    helper_test_op(None, torch.maximum, Tensor.maximum,
                   vals=[[-1234, 0, 1234, dtypes.max(dtypes.int), dtypes.min(dtypes.int)], dtypes.min(dtypes.int)], forward_only=True)
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[True, False, False], True], forward_only=True)
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[True, False, False], [True, True, False]], forward_only=True)

    # test applying to different dtype
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[1, 2, 3], 1.2], forward_only=True)
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[True, False, False], 1.2], forward_only=True)
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[True, False, False], 3], forward_only=True)

  def test_minimum(self):
    helper_test_op([(45,65), (45,65)], torch.minimum, Tensor.minimum)
    helper_test_op([(), ()], torch.minimum, Tensor.minimum)
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[1., 0., 3., -4.], 3.])
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[1., 0., 3., -4.], [-1., -2., 3., 0.]])
    helper_test_op(None, torch.minimum, Tensor.minimum,
                   vals=[[-1234, 0, 1234, dtypes.max(dtypes.int), dtypes.min(dtypes.int)], dtypes.max(dtypes.int)], forward_only=True)
    helper_test_op(None, torch.minimum, Tensor.minimum,
                   vals=[[-1234, 0, 1234, dtypes.max(dtypes.int), dtypes.min(dtypes.int)], dtypes.min(dtypes.int)], forward_only=True)
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[True, False, False], True], forward_only=True)
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[True, False, False], [True, True, False]], forward_only=True)

    # test applying to different dtype
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[1, 2, 3], 1.2], forward_only=True)
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[True, False, False], 1.2], forward_only=True)
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[True, False, False], 3], forward_only=True)

  def test_tiny_add(self):
    helper_test_op([(3), (3)], lambda x,y: x+y, Tensor.add, forward_only=True)
  def test_tiny_mul(self):
    helper_test_op([(64), (64)], lambda x,y: x*y, Tensor.mul, forward_only=True)

  def test_add(self):
    helper_test_op([(45,68), (45,68)], lambda x,y: x+y, Tensor.add)
    helper_test_op([(45,68), (45,68)], lambda x,y: x+y)
    helper_test_op([(), ()], lambda x,y: x+y)
  def test_add3(self):
    helper_test_op([(45,65), (45,65), (45,65)], lambda x,y,z: x+y+z)
  def test_broadcasted_add(self):
    helper_test_op([(45,65), (45,1)], lambda x,y: x+y)
    helper_test_op([(45,65), ()], lambda x,y: x+y)
  def test_broadcasted_add_2(self):
    helper_test_op([(45,65), (65,)], lambda x,y: x+y)

  def test_sub(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x-y, Tensor.sub)
    helper_test_op([(45,65), (45,65)], lambda x,y: x-y)
    helper_test_op([(), ()], lambda x,y: x-y)
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
    helper_test_op([(64,64), (64,64)], lambda x,y: x*y, Tensor.mul)
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
    helper_test_op([(45,65), (45,65)], lambda x,y: x/y, Tensor.div)
    helper_test_op([(45,65), (45,65)], lambda x,y: x/y)
    helper_test_op([(), ()], lambda x,y: x/y)

  @unittest.skipIf(getenv("AMD_LLVM", 0), "AMD with LLVM backend generate rcp in FP division causes trunc/floor errors")
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

    self.helper_test_exception(None, lambda x,y: x.div(y, rounding_mode="typo"), lambda x,y: x.div(y, rounding_mode="typo"), forward_only=True,
                               vals=[[5], [0]], expected=RuntimeError)

  def test_div_int(self):
    helper_test_op(None, lambda x,y: x/y, Tensor.div, forward_only=True, vals=[[5, 6, 7],[1, 2, 3]])
    helper_test_op(None, lambda x,y: x//y, forward_only=True, vals=[[5, 6, 7],[1, 2, 3]])
    helper_test_op(None, lambda x: x/2, forward_only=True, vals=[[3, 4, 5]])
    helper_test_op(None, lambda x: x//2, forward_only=True, vals=[[3, 4, 5]])
    helper_test_op(None, functools.partial(torch.div, rounding_mode="trunc"), Tensor.idiv, forward_only=True,
                   vals=[[-4, 7, 5, 4, -7, 8], [2, -3, 8, -2, 3, 5]])
    if is_dtype_supported(dtypes.uint64):
      x = Tensor(2**64 - 1, dtype=dtypes.uint64).idiv(1)
      np.testing.assert_equal(x.numpy(), 2**64 - 1)
    # 1 // 0 is device dependent, but it should not raise
    Tensor([1]).idiv(1).realize()
    if not (CI and getenv("PTX")):  # TODO: crashed in PTX CI
      # ... because if might be in a where branch that the output is well defined
      t = Tensor([-1, 0, 1, 2])
      np.testing.assert_equal((t > 0).where(1//t, t).numpy(), [-1, 0, 1, 0])

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
        helper_test_op(None, lambda x: x%3.5, forward_only=True, vals=[va])
        helper_test_op(None, lambda x: 100%x, forward_only=True, vals=[va])
        helper_test_op(None, lambda x: 100.5%x, forward_only=True, vals=[va])

  def test_mul_naninf(self):
    helper_test_op([(45,65)], lambda x: x*math.inf)
    helper_test_op([(45,65)], lambda x: x*-math.inf)
    helper_test_op([(45,65)], lambda x: x*math.nan)
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
    a, b = Tensor([0.0], requires_grad=True), torch.tensor([0.0], requires_grad=True)
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
    helper_test_op([(45,65)], lambda x: 5.5**x)
    helper_test_op([(45,65)], lambda x: (-5.5)**x)
    helper_test_op([(45,65)], lambda x: 8.0**x)
    helper_test_op([(45,65)], lambda x: x**2.0)
    helper_test_op([(45,65)], lambda x: 2.0**x)
    helper_test_op([()], lambda x: x**2.0)
    helper_test_op([()], lambda x: 2.0**x)
    helper_test_op(None, lambda x: 0**x, vals=[[-2.,-1,0,1,2,3]])
    helper_test_op(None, lambda x: (-2)**x, vals=[[-2.,-1,0,1,2,3]])

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
        tiny_out = get_tiny_gradient(x, c)
        torch_out = get_torch_gradient(x, c)
        if math.isnan(tiny_out):
          assert math.isnan(torch_out)
        else:
          self.assertAlmostEqual(tiny_out, torch_out, msg=f"{x}, {c}")

  def test_pow_zero_tensor(self):
    helper_test_op(None, lambda x,y: x**y, vals=[[0.0], [0.0]])
    # TODO: fix WEBGPU
    if Device.DEFAULT != "WEBGPU":
      helper_test_op(None, lambda x,y: x**y, vals=[[0.0], [0.3]])
      helper_test_op(None, lambda x,y: x**y, vals=[[0.0], [-0.3]])
  def test_pow_zero_const(self):
    helper_test_op(None, lambda x: x**0.3, vals=[[0.0]])
    helper_test_op(None, lambda x: x**0.0, vals=[[0.0]])
    helper_test_op(None, lambda x: x**-0.3, vals=[[0.0]])
    helper_test_op(None, lambda x: x**-1.0, vals=[[-1.0, 0.0, 1.0]])

  @unittest.skip("not supported")
  def test_pow_int(self):
    def _test(base, exponent): helper_test_op(None, lambda x,y: x**y, vals=[base, exponent], forward_only=True)

    for base in ([1, 2, 3], [-1, -2, -3]):
      for exponent in ([2, 3, 4], [-2, -3, -4]):
        _test(base, exponent)
    # NOTE: torch 0 ** -1 is 0
    _test([0, 0, 0], [0, 1, 2])

    np.testing.assert_equal((Tensor(11) ** Tensor(7)).item(), 11 ** 7)
    np.testing.assert_equal((Tensor([11]) ** Tensor(7)).item(), 11 ** 7)
    # TODO: fix non-precise int pow
    with self.assertRaises(AssertionError): np.testing.assert_equal((Tensor(11) ** Tensor([7])).item(), 11 ** 7)
    with self.assertRaises(AssertionError): np.testing.assert_equal((Tensor([11]) ** Tensor([7])).item(), 11 ** 7)

    # pow to a const int
    helper_test_op([], lambda: torch.tensor([2], dtype=torch.int) ** torch.tensor(-2, dtype=torch.int),
                       lambda: Tensor([2]) ** Tensor(-2), forward_only=True)

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
    helper_test_op([], lambda: tor^tor, lambda: ten^ten, forward_only=True)
    helper_test_op([], lambda: tor^0x1337, lambda: ten^0x1337, forward_only=True)
    helper_test_op([], lambda: 0x1337^tor, lambda: 0x1337^ten, forward_only=True)

    self.helper_test_exception([(4), (4)], torch.bitwise_xor, Tensor.bitwise_xor, expected=RuntimeError)

  def test_and(self):
    data = [[1,-8,1],[32,1,6]]
    tor = torch.tensor(data, dtype=torch.int)
    ten = Tensor(data, dtype=dtypes.int32)
    helper_test_op([], lambda: tor&tor, lambda: ten&ten, forward_only=True)
    helper_test_op([], lambda: tor&0x1337, lambda: ten&0x1337, forward_only=True)
    helper_test_op([], lambda: 0x1337&tor, lambda: 0x1337&ten, forward_only=True)

    data = [[True, True, False, False], [True, False, True, False]]
    tor0, tor1 = torch.tensor(data[0], dtype=torch.bool),  torch.tensor(data[1], dtype=torch.bool)
    ten0, ten1 = Tensor(data[0], dtype=dtypes.bool), Tensor(data[1], dtype=dtypes.bool)
    helper_test_op([], lambda: tor0&tor1, lambda: ten0&ten1, forward_only=True)

    helper_test_op(None, lambda x: (1 < x) & (x < 2), forward_only=True, vals=[[1.2, 1.2, 1.2, 3.2]])

    self.helper_test_exception([(4), (4)], torch.bitwise_and, Tensor.bitwise_and, expected=RuntimeError)

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

    self.helper_test_exception([(4), (4)], torch.bitwise_or, Tensor.bitwise_or, expected=RuntimeError)

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

    self.helper_test_exception([(4)], torch.bitwise_not, Tensor.bitwise_not, expected=RuntimeError)

  def test_lshift(self):
    data = [[0,1,2],[1<<8,1<<16,1<<31-1]]
    tor = torch.tensor(data, dtype=torch.int)
    ten = Tensor(data, dtype=dtypes.uint32)
    # cast to int32 because torch does not support uint32
    helper_test_op([], lambda: tor << 0, lambda: (ten << 0).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor << 2, lambda: (ten << 2).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor << 31, lambda: (ten << 31).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor.__lshift__(2), lambda: ten.__lshift__(2).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor.bitwise_left_shift(2), lambda: ten.lshift(2).cast(dtypes.int32), forward_only=True)

  def test_rshift(self):
    data = [[0,1,2],[1<<8,1<<16,1<<31-1]]
    tor = torch.tensor(data, dtype=torch.int)
    ten = Tensor(data, dtype=dtypes.uint32)
    # cast to int32 because torch does not support uint32
    helper_test_op([], lambda: tor >> 0, lambda: (ten >> 0).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor >> 2, lambda: (ten >> 2).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor >> 31, lambda: (ten >> 31).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor.__rshift__(2), lambda: ten.__rshift__(2).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor.bitwise_right_shift(2), lambda: ten.rshift(2).cast(dtypes.int32), forward_only=True)

  def test_idiv_shift_rewrite_negative(self):
    a = Tensor(-5).idiv(2).item()
    b = Tensor(-5).contiguous().idiv(2).item()
    self.assertEqual(a, b)
    self.assertEqual(Tensor(-1).contiguous().idiv(4).item(), 0)  # NOTE this is trunc-div behaviour

  def test_sin(self):
    helper_test_op([(45,65)], lambda x: x.sin())
    helper_test_op([()], lambda x: x.sin())
    # works on real CUDA but not CI
    if not ((getenv("MOCKGPU") and Device.DEFAULT == "NV") or Device.DEFAULT == "WEBGPU"):
      helper_test_op(None, lambda x: x.sin(), vals=[[math.nan, math.inf, -math.inf, 0.0]])
      helper_test_op(None, lambda x: x.sin(), vals=[[1e1, 1e2, 1e3, 1e4, 1e5, 1e6, -1e1, -1e2, -1e3, -1e4, -1e5, -1e6]],
                    atol=3e-3, rtol=3e-3, grad_atol=3e-3, grad_rtol=3e-3)
  def test_cos(self):
    helper_test_op([(45,65)], lambda x: x.cos())
    helper_test_op([()], lambda x: x.cos())
    if not ((getenv("MOCKGPU") and Device.DEFAULT == "NV") or Device.DEFAULT == "WEBGPU"):
      helper_test_op(None, lambda x: x.sin(), vals=[[math.nan, math.inf, -math.inf, 0.0]])
      helper_test_op(None, lambda x: x.cos(), vals=[[1e1, 1e2, 1e3, 1e4, 1e5, 1e6, -1e1, -1e2, -1e3, -1e4, -1e5, -1e6]],
                    atol=3e-3, rtol=3e-3, grad_atol=3e-3, grad_rtol=3e-3)
  def test_tan(self):
    # NOTE: backward has much higher diff with input close to pi/2 and -pi/2
    helper_test_op([(45,65)], lambda x: x.tan(), low=-1.5, high=1.5)
    helper_test_op([(45,65)], lambda x: x.tan(), low=-5, high=5)
    helper_test_op([()], lambda x: x.tan())
    if not ((getenv("MOCKGPU") and Device.DEFAULT == "NV") or Device.DEFAULT == "WEBGPU"):
      helper_test_op(None, lambda x: x.sin(), vals=[[math.nan, math.inf, -math.inf, 0.0]])
      helper_test_op(None, lambda x: x.cos(), vals=[[1e1, 1e2, 1e3, 1e4, 1e5, 1e6, -1e1, -1e2, -1e3, -1e4, -1e5, -1e6]],
                    atol=3e-3, rtol=3e-3, grad_atol=3e-3, grad_rtol=3e-3)

  def test_asin(self):
    helper_test_op([(45,65)], lambda x: x.asin(), low=-1, high=1)
    helper_test_op([(45,65)], lambda x: x.asin(), low=-300, high=-297)
    helper_test_op([(45,65)], lambda x: x.asin(), low=300, high=303)
  def test_acos(self):
    # high grad atol
    helper_test_op([(45,65)], lambda x: x.acos(), low=-1, high=1)
    helper_test_op([(45,65)], lambda x: x.acos(), low=-300, high=-297)
    helper_test_op([(45,65)], lambda x: x.acos(), low=300, high=303)
  def test_atan(self):
    helper_test_op([(45,65)], lambda x: x.atan())
    helper_test_op([(45,65)], lambda x: x.atan(), low=-300, high=-297)
    helper_test_op([(45,65)], lambda x: x.atan(), low=300, high=303)

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
      helper_test_op([()], lambda x: torch.nn.functional.celu(x,val), lambda x: x.celu(val))
  def test_selu(self):
    helper_test_op([(45,65)], torch.nn.functional.selu, Tensor.selu)
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
    helper_test_op(None, torch.abs, Tensor.abs, vals=[[-1.,0,1]])

  @unittest.skipIf(TRANSCENDENTAL and Device.DEFAULT=="AMD", "TODO: remu crashes")
  def test_log(self):
    helper_test_op([(45,65)], torch.log, Tensor.log)
    helper_test_op(None, torch.log, Tensor.log, vals=[[math.inf, -math.inf, math.nan]])
    helper_test_op([()], torch.log, Tensor.log)
  def test_log2(self):
    helper_test_op([(45,65)], torch.log2, Tensor.log2)
    helper_test_op(None, torch.log2, Tensor.log2, vals=[[math.inf, -math.inf, math.nan]])
    helper_test_op([()], torch.log2, Tensor.log2)

  @unittest.skipIf(TRANSCENDENTAL and Device.DEFAULT=="AMD", "TODO: remu crashes")
  def test_exp(self):
    helper_test_op([(45,65)], torch.exp, Tensor.exp)
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
  def test_copysign_exact(self):
    for i in [-1.,0.,1.]:
      for j in [-1., 0., 1.]:
        helper_test_op(None, torch.copysign, Tensor.copysign, vals=[[i], [j]])

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
  def test_hardsigmoid(self):
    helper_test_op([(45,65)], torch.nn.functional.hardsigmoid, Tensor.hardsigmoid)
    helper_test_op([()], torch.nn.functional.hardsigmoid, Tensor.hardsigmoid)
  def test_hardsigmoid_extreme(self):
    helper_test_op([(45,65)], torch.sigmoid, Tensor.sigmoid, low=300, high=400)
    helper_test_op([(45,65)], torch.sigmoid, Tensor.sigmoid, low=-400, high=-300)
  def test_softplus(self):
    helper_test_op([(45,65)], torch.nn.functional.softplus, Tensor.softplus, grad_atol=1e-6)
    helper_test_op([(45,65)], lambda t: torch.nn.functional.softplus(t, beta=3), lambda t: Tensor.softplus(t, beta=3), grad_atol=1e-6)
    helper_test_op([(45,65)], lambda t: torch.nn.functional.softplus(t, beta=1/3), lambda t: Tensor.softplus(t, beta=1/3), grad_atol=1e-6)
    # # TODO: support threshold and enable this
    # helper_test_op([(45,65)], torch.nn.functional.softplus, Tensor.softplus, grad_atol=1e-6, low=300, high=400)
    helper_test_op([(45,65)], torch.nn.functional.softplus, Tensor.softplus, grad_atol=1e-6, low=-400, high=-300)
    helper_test_op([()], torch.nn.functional.softplus, Tensor.softplus, grad_atol=1e-6)

  def test_erf(self):
    helper_test_op([(45,65)], torch.erf, Tensor.erf)
    helper_test_op([(45,65)], torch.erf, Tensor.erf, low=300, high=400)
    helper_test_op([(45,65)], torch.erf, Tensor.erf, low=-400, high=-300)
    helper_test_op([()], torch.erf, Tensor.erf)

  def test_gelu(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.gelu(x, approximate="tanh"), Tensor.gelu)
  def test_gelu_extreme(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.gelu(x, approximate="tanh"), Tensor.gelu, low=300, high=400)
    helper_test_op([(45,65)], lambda x: torch.nn.functional.gelu(x, approximate="tanh"), Tensor.gelu, low=-400, high=-300)
  def test_quick_gelu(self):
    helper_test_op([(45,65)], lambda x: x * torch.sigmoid(1.702 * x), Tensor.quick_gelu)
    helper_test_op([()], lambda x: x * torch.sigmoid(1.702 * x), Tensor.quick_gelu)
  def test_quick_gelu_extreme(self):
    helper_test_op([(45,65)], lambda x: x * torch.sigmoid(1.702 * x), Tensor.quick_gelu, low=300, high=400)
    helper_test_op([(45,65)], lambda x: x * torch.sigmoid(1.702 * x), Tensor.quick_gelu, low=-400, high=-300)

  def test_elu(self):
    helper_test_op([(45,65)], torch.nn.functional.elu, Tensor.elu)
    helper_test_op([(45,65)], lambda x: torch.nn.functional.elu(x, alpha=0.1), lambda x: Tensor.elu(x, alpha=0.1))
    helper_test_op([()], torch.nn.functional.elu, Tensor.elu)
  def test_relu6(self):
    helper_test_op([(45,65)], torch.nn.functional.relu6, Tensor.relu6)
    helper_test_op([()], torch.nn.functional.relu6, Tensor.relu6)
  def test_hardswish(self):
    helper_test_op([(45,65)], torch.nn.functional.hardswish, Tensor.hardswish, grad_atol=1e-6)
    helper_test_op([()], torch.nn.functional.hardswish, Tensor.hardswish, grad_atol=1e-6)
  def test_mish(self):
    helper_test_op([(45,65)], torch.nn.functional.mish, Tensor.mish)
    helper_test_op([()], torch.nn.functional.mish, Tensor.mish)

  def test_small_cumsum(self):
    helper_test_op([(10)], lambda x: torch.cumsum(x, dim=0), lambda x: Tensor.cumsum(x, axis=0))
  def test_simple_cumsum(self):
    helper_test_op([(512)], lambda x: torch.cumsum(x, dim=0), lambda x: Tensor.cumsum(x, axis=0))
    helper_test_op([(1022)], lambda x: torch.cumsum(x, dim=0), lambda x: Tensor.cumsum(x, axis=0))
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
  def test_simple_cumprod(self):
    helper_test_op([(512)],lambda x: torch.cumprod(x, dim=0),lambda x: Tensor.cumprod(x, axis=0))
    helper_test_op([(1022)],lambda x: torch.cumprod(x, dim=0),lambda x: Tensor.cumprod(x, axis=0))
  def test_cumprod(self):
    helper_test_op([()],lambda x: torch.cumprod(x, dim=0),lambda x: Tensor.cumprod(x, axis=0))
    self.helper_test_exception([()],lambda x: torch.cumprod(x, dim=1),lambda x: Tensor.cumprod(x, axis=1),expected=IndexError)
    helper_test_op([(20,)],lambda x: torch.cumprod(x, dim=0),lambda x: Tensor.cumprod(x, axis=0))
    self.helper_test_exception([(20,)],lambda x: torch.cumprod(x, dim=1),lambda x: Tensor.cumprod(x, axis=1),expected=IndexError)
    self.helper_test_exception([(20,)],lambda x: torch.cumprod(x, dim=-2),lambda x: Tensor.cumprod(x, axis=-2),expected=IndexError)
    helper_test_op([(20, 30)],lambda x: torch.cumprod(x, dim=0),lambda x: Tensor.cumprod(x, axis=0))
    helper_test_op([(20, 30)],lambda x: torch.cumprod(x, dim=1),lambda x: Tensor.cumprod(x, axis=1))
    helper_test_op([(20, 30, 40)],lambda x: torch.cumprod(x, dim=2),lambda x: Tensor.cumprod(x, axis=2))
    helper_test_op([(20, 30, 40)],lambda x: torch.cumprod(x, dim=-1),lambda x: Tensor.cumprod(x, axis=-1))
  def test_cumprod_zero_axis(self):
    helper_test_op([(2, 0, 4)],lambda x: torch.cumprod(x, dim=1),lambda x: Tensor.cumprod(x, axis=1))
    helper_test_op([(0, 3)],lambda x: torch.cumprod(x, dim=0),lambda x: Tensor.cumprod(x, axis=0))
    helper_test_op([(2, 3, 0)],lambda x: torch.cumprod(x, dim=2),lambda x: Tensor.cumprod(x, axis=2))

  def test_small_cummax(self):
    helper_test_op([(10)], lambda x: torch.cummax(x, dim=0).values, lambda x: Tensor.cummax(x, axis=0))
  def test_simple_cummax(self):
    helper_test_op([(512)], lambda x: torch.cummax(x, dim=0).values, lambda x: Tensor.cummax(x, axis=0))
    helper_test_op([(1022)], lambda x: torch.cummax(x, dim=0).values, lambda x: Tensor.cummax(x, axis=0))
  def test_cummax(self):
    helper_test_op([()], lambda x: torch.cummax(x, dim=0).values, lambda x: Tensor.cummax(x, axis=0))
    # TODO: torch allows this?
    # self.helper_test_exception([()], lambda x: torch.cummax(x, dim=1).values, lambda x: Tensor.cummax(x, axis=1), expected=IndexError)
    helper_test_op([(20,)], lambda x: torch.cummax(x, dim=0).values, lambda x: Tensor.cummax(x, axis=0))
    self.helper_test_exception([(20,)], lambda x: torch.cummax(x, dim=1).values, lambda x: Tensor.cummax(x, axis=1), expected=IndexError)
    self.helper_test_exception([(20,)], lambda x: torch.cummax(x, dim=-2).values, lambda x: Tensor.cummax(x, axis=-2), expected=IndexError)
    helper_test_op([(20,30)], lambda x: torch.cummax(x, dim=0).values, lambda x: Tensor.cummax(x, axis=0))
    helper_test_op([(20,30)], lambda x: torch.cummax(x, dim=1).values, lambda x: Tensor.cummax(x, axis=1))
    helper_test_op([(20,30,40)], lambda x: torch.cummax(x, dim=2).values, lambda x: Tensor.cummax(x, axis=2))
    helper_test_op([(20,30,40)], lambda x: torch.cummax(x, dim=-1).values, lambda x: Tensor.cummax(x, axis=-1))
  def test_cummax_zero_axis(self):
    helper_test_op([(2,0,4)], lambda x: torch.cummax(x, dim=1).values, lambda x: Tensor.cummax(x, axis=1))
    helper_test_op([(0,3)], lambda x: torch.cummax(x, dim=0).values, lambda x: Tensor.cummax(x, axis=0))
    helper_test_op([(2,3,0)], lambda x: torch.cummax(x, dim=2).values, lambda x: Tensor.cummax(x, axis=2))

  def test_argmax(self):
    # check if it returns the first index for multiple occurences
    helper_test_op(None, lambda x: x.argmax().type(torch.int32), lambda x: x.argmax(), forward_only=True, vals=[[2, 2]])
    helper_test_op(None, lambda x: x.argmax().type(torch.int32), lambda x: x.argmax(), forward_only=True, vals=[[1, 2, 2]])
    np.testing.assert_equal(Tensor([2,2]).argmax().numpy(), 0)
    np.testing.assert_equal(Tensor([1,2,2]).argmax().numpy(), 1)
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
    # check if it returns the first index for multiple occurences
    helper_test_op(None, lambda x: x.argmin().type(torch.int32), lambda x: x.argmin(), forward_only=True, vals=[[2, 2]])
    helper_test_op(None, lambda x: x.argmin().type(torch.int32), lambda x: x.argmin(), forward_only=True, vals=[[3, 2, 2]])
    np.testing.assert_equal(Tensor([2,2]).argmin().numpy(), 0)
    np.testing.assert_equal(Tensor([3,2,2]).argmin().numpy(), 1)
    helper_test_op([(10,20)], lambda x: x.argmin().type(torch.int32), lambda x: x.argmin(), forward_only=True)
    helper_test_op([(10,20)], lambda x: x.argmin(0, False).type(torch.int32), lambda x: x.argmin(0, False), forward_only=True)
    helper_test_op([(10,20)], lambda x: x.argmin(1, False).type(torch.int32), lambda x: x.argmin(1, False), forward_only=True)
    helper_test_op([(10,20)], lambda x: x.argmin(1, True).type(torch.int32), lambda x: x.argmin(1, True), forward_only=True)

    helper_test_op(None, lambda x: x.argmin().type(torch.int32), lambda x: x.argmin(), forward_only=True, vals=[[0, -2**31]])
    helper_test_op(None, lambda x: x.argmin().type(torch.int32), lambda x: x.argmin(), forward_only=True, vals=[[-2**31, 0]])
    # NOTE: torch does not support this on bool
    helper_test_op(None, lambda x: x.type(torch.int32).argmin().type(torch.int32), lambda x: x.argmin(), forward_only=True, vals=[[False, True]])
    helper_test_op(None, lambda x: x.type(torch.int32).argmin().type(torch.int32), lambda x: x.argmin(), forward_only=True, vals=[[True, False]])

  def test_sort(self):
    for dim in [-1, 0, 1]:
      for descending in [True, False]:
        helper_test_op([(8,45,65)], lambda x: x.sort(dim, descending).values, lambda x: x.sort(dim, descending)[0], forward_only=True)
        helper_test_op([(8,45,65)], lambda x: x.sort(dim, descending).indices.type(torch.int32), lambda x: x.sort(dim, descending)[1],
                       forward_only=True)
    # repeated values
    helper_test_op(None, lambda x: x.sort(stable=True).values, lambda x: x.sort()[0], forward_only=True, vals=[[0, 1] * 9])
    helper_test_op(None, lambda x: x.sort(stable=True).indices.type(torch.int32), lambda x: x.sort()[1], forward_only=True, vals=[[0, 1] * 9])
    helper_test_op(None, lambda x: x.sort(stable=True, descending=True).values,
                   lambda x: x.sort(descending=True)[0], forward_only=True, vals=[[0, 1] * 9])
    helper_test_op(None, lambda x: x.sort(stable=True, descending=True).indices.type(torch.int32),
                   lambda x: x.sort(descending=True)[1], forward_only=True, vals=[[0, 1] * 9])

  def test_topk(self):
    helper_test_op([(10)], lambda x: x.topk(3).values, lambda x: x.topk(3)[0], forward_only=True)
    helper_test_op([(10)], lambda x: x.topk(3).indices.type(torch.int32), lambda x: x.topk(3)[1], forward_only=True)
    for dim in [0, 1, -1]:
      for largest in [True, False]:
        for sorted_ in [True]: # TODO support False
          helper_test_op([(10,20,30)],
                          lambda x: x.topk(5, dim, largest, sorted_).values,
                          lambda x: x.topk(5, dim, largest, sorted_)[0], forward_only=True)
          helper_test_op([(10,20,30)],
                          lambda x: x.topk(5, dim, largest, sorted_).indices.type(torch.int32),
                          lambda x: x.topk(5, dim, largest, sorted_)[1], forward_only=True)
    # repeated values
    value, indices = Tensor([1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]).topk(3)
    np.testing.assert_equal(value.numpy(), [1, 1, 1])
    np.testing.assert_equal(indices.numpy(), [0, 1, 3])
    value, indices = Tensor([1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]).topk(3, largest=False)
    np.testing.assert_equal(value.numpy(), [0, 0, 0])
    np.testing.assert_equal(indices.numpy(), [2, 4, 6])
    self.helper_test_exception([(4)], lambda x: x.topk(5), lambda x: x.topk(5), expected=(RuntimeError, ValueError))

  def test_einsum(self):
    # matrix transpose
    helper_test_op([(150,150)], lambda a: torch.einsum('ij->ji', a), lambda a: Tensor.einsum('ij->ji', a))
    helper_test_op([(150,150)], lambda a: torch.einsum('ij -> ji', a), lambda a: Tensor.einsum('ij -> ji', a))
    helper_test_op([(150,150)], lambda a: torch.einsum('ji', a), lambda a: Tensor.einsum('ji', a))
    helper_test_op([(20,30,40)], lambda a: torch.einsum('jki', a), lambda a: Tensor.einsum('jki', a))
    helper_test_op([(20,30,40)], lambda a: torch.einsum('dog', a), lambda a: Tensor.einsum('dog', a))
    # no -> and empty rhs
    helper_test_op([(20,30),(30,40)], lambda a, b: torch.einsum('ij,jk', a, b), lambda a, b: Tensor.einsum('ij,jk', a, b))
    # sum all elements
    helper_test_op([(20,30,40)], lambda a: torch.einsum('ijk->', a), lambda a: Tensor.einsum('ijk->', a))
    # column sum
    helper_test_op([(50,50)], lambda a: torch.einsum('ij->j', a), lambda a: Tensor.einsum('ij->j', a))
    # row sum
    helper_test_op([(15,15)], lambda a: torch.einsum('ij->i', a), lambda a: Tensor.einsum('ij->i', a))
    # matrix-vector multiplication
    helper_test_op([(15,20), (20,)], lambda a,b: torch.einsum('ik,k->i', a,b), lambda a,b: Tensor.einsum('ik,k->i', a, b))
    # matrix-matrix multiplication
    helper_test_op([(15,20), (20,30)], lambda a,b: torch.einsum('ik,kj->ij', a,b), lambda a,b: Tensor.einsum('ik,kj->ij', a, b))
    # matrix-matrix multiplication, different letter order
    helper_test_op([(15,20), (20,30)], lambda a,b: torch.einsum('jk,ki->ji', a,b), lambda a,b: Tensor.einsum('jk,ki->ji', a, b))
    # dot product
    helper_test_op([(30),(30)], lambda a,b: torch.einsum('i,i->i', [a,b]), lambda a,b: Tensor.einsum('i,i->i', [a,b]))
    # hadamard product
    helper_test_op([(30,40),(30,40)], lambda a,b: torch.einsum('ij,ij->ij', a,b), lambda a,b: Tensor.einsum('ij,ij->ij', a,b))
    # outer product
    helper_test_op([(15,), (15,)], lambda a,b: torch.einsum('i,j->ij', a,b), lambda a,b: Tensor.einsum('i,j->ij',a,b))
    # batch matrix multiplication
    helper_test_op([(10,20,30),(10,30,40)], lambda a,b: torch.einsum('ijk,ikl->ijl', [a, b]), lambda a,b: Tensor.einsum('ijk,ikl->ijl', [a, b]))
    # batch matrix multiplication, result permuted
    helper_test_op([(10,20,25),(10,25,32)], lambda a,b: torch.einsum('ijk,ikl->jil', [a, b]), lambda a,b: Tensor.einsum('ijk,ikl->jil', [a, b]))
    # batch matrix multiplication, result & input permuted
    helper_test_op([(20,10,25),(10,25,32)], lambda a,b: torch.einsum('jik,ikl->jil', [a, b]), lambda a,b: Tensor.einsum('jik,ikl->jil', [a, b]))
    # batch matrix multiplication, result with different letters
    helper_test_op([(10,20,30),(10,30,40)], lambda a,b: torch.einsum('ijk,ika->ija', [a, b]), lambda a,b: Tensor.einsum('ijk,ika->ija', [a, b]))
    # tensor contraction
    helper_test_op([(3,5,8,10),(11,13,5,16,8)], lambda a,b: torch.einsum('pqrs,tuqvr->pstuv', a,b),
                                                lambda a,b: Tensor.einsum('pqrs,tuqvr->pstuv', a,b), atol=1e-5)
    # tensor contraction, input permuted
    helper_test_op([(3,8,10,5),(11,5,13,16,8)], lambda a,b: torch.einsum('prsq,tquvr->pstuv', a,b),
                                                lambda a,b: Tensor.einsum('prsq,tquvr->pstuv', a,b), atol=1e-5)
    # tensor contraction, result with different letters
    helper_test_op([(3,5,8,10),(11,13,5,16,8)], lambda a,b: torch.einsum('zqrs,tuqvr->zstuv', a,b),
                                                lambda a,b: Tensor.einsum('zqrs,tuqvr->zstuv', a,b), atol=1e-5)
    # bilinear transformation
    helper_test_op([(2,3),(5,3,7),(2,7)], lambda a,b,c: torch.einsum('ik,jkl,il->ij', [a,b,c]), lambda a,b,c: Tensor.einsum('ik,jkl,il->ij', [a,b,c]))

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
    helper_test_op([(32, 7, 24, 24, 24), (32, 7, 24, 24, 24)], lambda a, b: torch.einsum('ij...,ij...->ij', [a, b]),
               lambda a, b: Tensor.einsum('ij...,ij...->ij', [a, b]))
    # multiple ellipsis in one operand are not allowed. This test shall raise an exception.
    with self.assertRaises(RuntimeError):
      helper_test_op([(2, 3, 4), (2, 3, 4)], lambda a, b: torch.einsum('...ik..., ...jk ->', [a, b]),
                       lambda a, b: Tensor.einsum('...ik..., ...jk ->', [a, b]))
    # multiple ellipsis must broadcast together. This test shall raise an exception.
    with self.assertRaises(RuntimeError):
      helper_test_op([(2, 3, 4, 5), (5, 2, 7)], lambda a, b: torch.einsum('i...j,ji...->...', [a, b]),
                       lambda a, b: Tensor.einsum('i...j,ji...->...', [a, b]))

  def test_einsum_shape_check(self):
    a = Tensor.zeros(3,8,10,5)
    b = Tensor.zeros(11,5,13,16,8)
    with self.assertRaises(AssertionError):
      Tensor.einsum('pqrs,tuqvr->pstuv',a,b)

  def test_einsum_arity_check1(self):
    a = Tensor.zeros(10,15)
    b = Tensor.zeros(15,20)
    c = Tensor.zeros(20,10)
    with self.assertRaises(AssertionError):
      Tensor.einsum('ij,jk->ij', a,b,c)

  def test_einsum_arity_check2(self):
    a = Tensor.zeros(10,10)
    with self.assertRaises(AssertionError):
      Tensor.einsum('ij,jk->ij', a)

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
  @unittest.skipIf(CI and Device.DEFAULT in ["NV", "LLVM", "GPU", "CUDA"] or IMAGE, "not supported on these in CI/IMAGE")
  def test_gemm_fp16(self):
    helper_test_op([(64,64), (64,64)], lambda x,y: x.half().matmul(y.half()), atol=5e-3, rtol=5e-3)
  def test_gemm(self):
    helper_test_op([(64,64), (64,64)], lambda x,y: x.matmul(y))
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
  def test_broadcastdot(self):
    helper_test_op([(10,45,65), (65,45)], lambda x,y: x @ y, Tensor.dot, atol=1e-4)
    with self.assertRaises(RuntimeError):
      a = Tensor(3.14)
      b = Tensor.ones(3,3)
      a @ b
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
    self.helper_test_exception([(3,4,5,6)], lambda x: x.sum(5), lambda x: x.sum(5), expected=IndexError)
    self.helper_test_exception([()], lambda x: x.sum(1), lambda x: x.sum(1), expected=IndexError)
    self.helper_test_exception([()], lambda x: x.sum((1,)), lambda x: x.sum((1,)), expected=IndexError)

  def test_sum_dtype_arg(self):
    helper_test_op([(45,3)], lambda x: x.sum(), lambda x: x.sum(dtype=dtypes.float32))
    if is_dtype_supported(dtypes.float64): helper_test_op([(45,3)], lambda x: x.sum(dtype=torch.float64), lambda x: x.sum(dtype=dtypes.float64))

    with self.assertRaises(AttributeError): Tensor([1.0, 2.0]).sum(dtype="")

  def test_sum_with_zeros_shape(self):
    helper_test_op([(4, 0)], lambda x: x.sum(axis=(0,)))
    helper_test_op([(4, 0)], lambda x: x.sum(axis=(1,)))
    helper_test_op([(4, 0)], lambda x: x.sum(axis=(0,1)))

  def test_prod(self):
    helper_test_op(None, lambda x: x.prod(), vals=[[1.0, 2.0, 3.0]])
    with Context(NOOPT=1): helper_test_op(None, lambda x: x.prod(), vals=[[1.0, 2.0, 3.0]])
    helper_test_op([(3,4,5,6)], lambda x: x.prod(dim=3), lambda x: x.prod(axis=3))
    helper_test_op([(3,4,5,6)], lambda x: x.prod(dim=1), lambda x: x.prod(axis=1))
    helper_test_op([(3,4,5,6)], lambda x: x.prod(dim=1, keepdim=True), lambda x: x.prod(axis=1, keepdim=True))
    helper_test_op([()], lambda x: x.prod())
    helper_test_op([()], lambda x: x.prod(0))
    helper_test_op([()], lambda x: x.prod(-1))

  def test_prod_dtype_arg(self):
    with self.assertRaises(AttributeError): Tensor([1.0, 2.0]).prod(dtype="")

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

  @unittest.skipIf(Device.DEFAULT == "QCOM", "OpenCL fails to compile this (both on GPU(qcom)/QCOM backends)")
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

  @unittest.skipIf(Device.DEFAULT == "QCOM", "OpenCL fails to compile this (both on GPU(qcom)/QCOM backends)")
  def test_all(self):
    helper_test_op([(3,4,5,6)], lambda x: x.all(), forward_only=True)
    helper_test_op(None, lambda x: x.all(), vals=[[True, True]], forward_only=True)
    helper_test_op(None, lambda x: x.all(), vals=[[True, False]], forward_only=True)
    helper_test_op(None, lambda x: x.all(), vals=[[False, False]], forward_only=True)
    helper_test_op([()], lambda x: x.all(), forward_only=True)
  def test_all_axis(self):
    helper_test_op([(3,4,5,6)], lambda x: x.all(axis=(1,2)), forward_only=True)
  def test_all_zero_axis(self):
    helper_test_op([(1,0,3,0,5)], lambda x: x.all(axis=(1,3)), forward_only=True)

  def test_isclose(self):
    helper_test_op([(3, 4, 5, 6)], lambda x: x.isclose(x), forward_only=True)
    helper_test_op([(3, 4, 5, 6), (3, 4, 5, 6)], lambda x,y: x.isclose(y), forward_only=True)
    helper_test_op([(3, 4, 5, 6)], lambda x: x.isclose(x, equal_nan=True), forward_only=True)
    helper_test_op([(3, 4, 5, 6)], lambda x: x.isclose(x + 1e-6), forward_only=True)
    helper_test_op([(3, 4, 5, 6)], lambda x: x.isclose(x + 1e-9), forward_only=True)
    helper_test_op([(3, 4, 5, 6)], lambda x: x.isclose(x + 1e-6, atol=0.0), forward_only=True)
    helper_test_op([(3, 4, 5, 6)], lambda x: x.isclose(x + 1e-9, atol=0.0), forward_only=True)
    helper_test_op([(3, 4, 5, 6)], lambda x: x.isclose(x + 1e-6, rtol=0.01), forward_only=True)
    helper_test_op([(3, 4, 5, 6)], lambda x: x.isclose(x + 1e-9, rtol=0.01), forward_only=True)
    helper_test_op(None, lambda x,y: x.isclose(y), vals=[[1e-7, 1e-8, 1e-9], [0.0, 0.0, 0.0]], forward_only=True)

  @unittest.skipIf(Device.DEFAULT == "WEBGPU" and CI, "isinf check of 'nan' fails on CI software-based vulkan")
  def test_isclose_edge_cases(self):
    for a in [math.inf, -math.inf, math.nan, 0.0]:
      for b in [math.inf, -math.inf, math.nan, 0.0]:
        helper_test_op(None, lambda x,y: x.isclose(y), vals=[[a], [b]], forward_only=True)
        helper_test_op(None, lambda x,y: x.isclose(y, equal_nan=True), vals=[[a], [b]], forward_only=True)

  def test_mean(self):
    helper_test_op([(3,4,5,6)], lambda x: x.mean())
    helper_test_op([()], lambda x: x.mean())
  def test_mean_axis(self):
    helper_test_op([(3,4,5,6)], lambda x: x.mean(axis=(1,2)))
  def test_mean_zero_axis(self):
    helper_test_op([(1,0,3,0,5)], lambda x: x.mean(axis=(1,3)))

  def test_var(self):
    helper_test_op([(15, 25, 35)], lambda x: x.var())
    helper_test_op([(15, 25, 35)], lambda x: x.var(correction=0))
    helper_test_op([(15, 25, 35)], lambda x: x.var(correction=5))
    # TODO: fix this
    # helper_test_op([(10, 2)], lambda x: x.var(correction=50))
  def test_var_axis(self):
    helper_test_op([(15, 25, 35)], lambda x: x.var(0))
    helper_test_op([(15, 25, 35)], lambda x: x.var(2))
    helper_test_op([(15, 25, 35)], lambda x: x.var([1, 2]))
    helper_test_op([(15, 25, 35)], lambda x: x.var(0, correction=0))
    helper_test_op([(15, 25, 35)], lambda x: x.var(2, correction=0))
    helper_test_op([(15, 25, 35)], lambda x: x.var([1, 2], correction=0))
  def test_var_zero_in_axis(self):
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", message="var\\(\\): degrees of freedom is <= 0")
      helper_test_op([(1,0,3,0,5)], lambda x: x.var(axis=(1,3)))
      helper_test_op([(1,0,3,0,5)], lambda x: x.var(axis=(1,3), correction=0))
      helper_test_op([(1,0,3,0,5)], lambda x: x.var(axis=(1,3), correction=5))
  def test_var_one_in_axis(self):
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", message="var\\(\\): degrees of freedom is <= 0")
      helper_test_op([(1,2,3,1,5)], lambda x: x.var(axis=(0,3)))
      # TODO: fix backward
      helper_test_op([(1,2,3,1,5)], lambda x: x.var(axis=(0,3), correction=5), forward_only=True)
      helper_test_op([(1,2,3,1,5)], lambda x: x.var(axis=(0,4), correction=5), forward_only=True)
    helper_test_op([(1,)], lambda x: x.var(axis=(0,), correction=0))
    helper_test_op([(1,2,3,1,5)], lambda x: x.var(axis=(0,3), correction=0))
    helper_test_op([(1,2,3,1,5)], lambda x: x.var(axis=(0,4)))
    helper_test_op([(1,2,3,1,5)], lambda x: x.var(axis=(0,4), correction=0))
  def test_var_keepdim(self):
    helper_test_op([(15, 25, 35)], lambda x: x.var(keepdim=True))
    helper_test_op([(15, 25, 35)], lambda x: x.var(0, keepdim=True, correction=0))

  def test_std(self):
    helper_test_op([(15, 25, 35)], lambda x: x.std())
    helper_test_op([(15, 25, 35)], lambda x: x.std(correction=0))
    helper_test_op([(15, 25, 35)], lambda x: x.std(correction=5))
  def test_std_axis(self):
    helper_test_op([(15, 25, 35)], lambda x: x.std(0))
    helper_test_op([(15, 25, 35)], lambda x: x.std(2))
    helper_test_op([(15, 25, 35)], lambda x: x.std([1, 2]))
    helper_test_op([(15, 25, 35)], lambda x: x.std(0, correction=0))
    helper_test_op([(15, 25, 35)], lambda x: x.std(2, correction=0))
    helper_test_op([(15, 25, 35)], lambda x: x.std([1, 2], correction=0))
  def test_std_zero_in_axis(self):
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom is <= 0")
      helper_test_op([(1,0,3,0,5)], lambda x: x.std(axis=(1,3)))
      helper_test_op([(1,0,3,0,5)], lambda x: x.std(axis=(1,3), correction=0))
      helper_test_op([(1,0,3,0,5)], lambda x: x.std(axis=(1,3), correction=5))
  def test_std_one_in_axis(self):
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom is <= 0")
      helper_test_op([(1,2,3,1,5)], lambda x: x.std(axis=(0,3)))
      helper_test_op([(1,2,3,1,5)], lambda x: x.std(axis=(0,3), correction=5))
      helper_test_op([(1,2,3,1,5)], lambda x: x.std(axis=(0,4), correction=5))
    # TODO: fix backward
    helper_test_op([(1,)], lambda x: x.std(axis=(0,), correction=0), forward_only=True)
    helper_test_op([(1,2,3,1,5)], lambda x: x.std(axis=(0,3), correction=0), forward_only=True)
    helper_test_op([(1,2,3,1,5)], lambda x: x.std(axis=(0,4)))
    helper_test_op([(1,2,3,1,5)], lambda x: x.std(axis=(0,4), correction=0))
  def test_std_keepdim(self):
    helper_test_op([(15, 25, 35)], lambda x: x.std(keepdim=True))
    helper_test_op([(15, 25, 35)], lambda x: x.std(0, keepdim=True, correction=0))
  def test_std_mean(self):
    helper_test_op([(15,25,35)], lambda x: torch.stack(torch.std_mean(x)),
                                 lambda x: Tensor.stack(*x.std_mean()))
    helper_test_op([(15,25,35)], lambda x: torch.stack(torch.std_mean(x, correction=5)),
                                 lambda x: Tensor.stack(*x.std_mean(correction=5)))
    helper_test_op([(15,25,35)], lambda x: torch.stack(torch.std_mean(x, keepdim=True, correction=0)),
                                 lambda x: Tensor.stack(*x.std_mean(keepdim=True, correction=0)))
    helper_test_op([(3,4,5,6)], lambda x: torch.stack(torch.std_mean(x, axis=(1,2))),
                                lambda x: Tensor.stack(*x.std_mean(axis=(1,2))))

  @unittest.skip("TODO: this fails because of loaded nan in mul folding")
  def test_std_mean_loaded_nan(self):
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
  def test_log_softmax_other_axis(self):
    helper_test_op([(10,10,10)], lambda x: x.log_softmax(0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(10,10,10)], lambda x: x.log_softmax(1), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(10,10,10)], lambda x: x.log_softmax(2), atol=1e-7, grad_atol=1e-7)

  def test_logsumexp(self):
    helper_test_op([(45,65)], lambda x: torch.logsumexp(x, dim=0), lambda x: x.logsumexp(0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45,65)], lambda x: torch.logsumexp(x, dim=0, keepdim=True), lambda x: x.logsumexp(0, True), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45,65)], lambda x: torch.logsumexp(x, dim=1), lambda x: x.logsumexp(1), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45)], lambda x: torch.logsumexp(x, dim=0), lambda x: x.logsumexp(0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([()], lambda x: torch.logsumexp(x, dim=0), lambda x: x.logsumexp(0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([()], lambda x: torch.logsumexp(x, dim=-1), lambda x: x.logsumexp(-1), atol=1e-7, grad_atol=1e-7)

  def test_logcumsumexp(self):
    helper_test_op([(45,65)], lambda x: torch.logcumsumexp(x, dim=0), lambda x: x.logcumsumexp(0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45,65)], lambda x: torch.logcumsumexp(x, dim=1), lambda x: x.logcumsumexp(1), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45)], lambda x: torch.logcumsumexp(x, dim=0), lambda x: x.logcumsumexp(0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([()], lambda x: torch.logcumsumexp(x, dim=0), lambda x: x.logcumsumexp(0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([()], lambda x: torch.logcumsumexp(x, dim=0), lambda x: x.logcumsumexp(), atol=1e-7, grad_atol=1e-7)
    helper_test_op([()], lambda x: torch.logcumsumexp(x, dim=-1), lambda x: x.logcumsumexp(-1), atol=1e-7, grad_atol=1e-7)

  @unittest.skipIf(not DEVECTORIZE, "broken without DEVECTORIZE. TODO: fix this")
  def test_logcumsumexp_numerical(self):
    helper_test_op(None, lambda x: torch.logcumsumexp(x, dim=0), lambda x: x.logcumsumexp(), atol=1e-7, grad_atol=1e-7, vals=[[0.0, 100.0]])

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
    # TODO: this one has larger tol?
    helper_test_op([(45,65)], lambda x: x.asinh(), atol=1e-2, rtol=2e-2, grad_rtol=2e-2, low=-300, high=-297)
    helper_test_op([(45,65)], lambda x: x.asinh(), grad_atol=1e-6, low=300, high=303)
  def test_acosh(self):
    helper_test_op([(45,65)], lambda x: x.acosh(), grad_atol=1e-6)
    helper_test_op([(45,65)], lambda x: x.acosh(), grad_atol=1e-3, grad_rtol=1e-2, low=-300, high=-297)
    helper_test_op([(45,65)], lambda x: x.acosh(), grad_atol=1e-6, low=300, high=303)
  def test_atanh(self):
    helper_test_op([(45,65)], lambda x: x.atanh(), grad_atol=1e-6)
    helper_test_op([(45,65)], lambda x: x.atanh(), grad_atol=1e-6, low=-300, high=-297)
    helper_test_op([(45,65)], lambda x: x.atanh(), grad_atol=1e-6, low=300, high=303)

  def test_topo_sort(self):
    helper_test_op([(45,65)], lambda x: (x+x)*x, grad_atol=1e-6)
    helper_test_op([()], lambda x: (x+x)*x, grad_atol=1e-6)

  def test_flip_eye_crash(self):
    helper_test_op([], lambda: (torch.eye(10)@torch.eye(10).flip(0)),
                       lambda: (Tensor.eye(10)@Tensor.eye(10).flip(0)), forward_only=True)

  def test_broadcast_full(self):
    for torch_op, tinygrad_op in [(torch.add, Tensor.add), (torch.sub, Tensor.sub), (torch.mul, Tensor.mul),
                                  (torch.div, Tensor.div), (torch.pow, Tensor.pow)]:
      for shapes in [((5,13,24,16), (5,1,24,1)), ((1,3,1,7,1), (2,1,5,1,8))]:
        with self.subTest(op=torch_op.__name__, shapes=shapes):
          if tinygrad_op != Tensor.pow:
            helper_test_op(shapes, torch_op, tinygrad_op)
          else:
            helper_test_op(shapes, torch_op, tinygrad_op, low=0, high=3)

  def test_broadcast_simple(self):
    helper_test_op([(45,65), (45,1)], lambda x,y: x/y)
    helper_test_op([(45,65), ()], lambda x,y: x/y)

  def test_broadcast_partial(self):
    for torch_op, tinygrad_op in [(torch.add, Tensor.add), (torch.sub, Tensor.sub), (torch.mul, Tensor.mul),
                                  (torch.div, Tensor.div), (torch.pow, Tensor.pow)]:
      for shapes in [((1,32,32,32), (1,32,1,1)), ((5,13,24,16,2), (1,13,24,1,1)),
                     ((4,1), (4,5)), ((1,4), (5,4))]:
        with self.subTest(op=torch_op.__name__, shapes=shapes):
          # NOTE: ANE backwards?
          if tinygrad_op != Tensor.pow:
            helper_test_op(shapes, torch_op, tinygrad_op)
          else:
            helper_test_op(shapes, torch_op, tinygrad_op, low=0, high=3)

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

  def test_slice_empty(self):
    helper_test_op([(10,10)], lambda x: x[1:1])

  def test_slice_zero_in_shape(self):
    helper_test_op([(10,10)], lambda x: x[1:1])  # x.shape = (0, 10)
    helper_test_op([(3,3,3)], lambda x: x[-2:-5])  # x.shape = (0, 3, 3)

  def test_slice_errors(self):
    a = Tensor.ones(4, 3)
    b = Tensor(2)
    with self.assertRaisesRegex(IndexError, "too many"): a[1, 77, 77, 77] # IndexError: (finds too many indices before the out of bounds)
    with self.assertRaisesRegex(IndexError, "out of bounds"): a[1, 3] # IndexError: (out of bounds).
    with self.assertRaisesRegex(IndexError, "out of bounds"): a[1, -4]
    with self.assertRaisesRegex(IndexError, "single ellipsis"): a[..., ...] # IndexError: only single ellipsis
    with self.assertRaises(ValueError): a[::0, 1] # no 0 strides
    with self.assertRaises(TypeError): a[:Tensor([3]), 1] # Tensor can't be used as a slice parameter
    with self.assertRaises(IndexError): b[:] # slice cannot be applied to a 0-dim tensor

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
    with self.assertRaises(NotImplementedError):
      # negative pads with circular pads is not supported
      Tensor.randn(1,1,5,5).pad((3,-5,1,-5), mode="circular")

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

  def test_pad_slice(self):
    for value in 0., 3.456:
      helper_test_op([(1)], lambda x: torch.nn.functional.pad(x,(1,0), value=value)[0], lambda x: x.pad(((1,0),), value=value)[0])
      helper_test_op([(4)], lambda x: torch.nn.functional.pad(x,(1,0), value=value)[0], lambda x: x.pad(((1,0),), value=value)[0])
      helper_test_op([(4)], lambda x: torch.nn.functional.pad(x,(3,0), value=value)[0:1], lambda x: x.pad(((3,0),), value=value)[0:1])
      helper_test_op([(4)], lambda x: torch.nn.functional.pad(x,(0,3), value=value)[6], lambda x: x.pad(((0,3),), value=value)[6])
      helper_test_op([(4)], lambda x: torch.nn.functional.pad(x,(0,3), value=value)[4:6], lambda x: x.pad(((0,3),), value=value)[4:6])
      helper_test_op([(5,5)], lambda x: torch.nn.functional.pad(x,(0,0,1,0), value=value)[0], lambda x: x.pad(((1,0),(0,0)), value=value)[0])
      helper_test_op([(2,2)], lambda x: torch.nn.functional.pad(x,(0,1,0,0), value=value)[0,2], lambda x: x.pad(((0,0),(0,1)), value=value)[0,2])
      helper_test_op([(4,4)], lambda x: torch.nn.functional.pad(x,(0,0,1,0), value=value)[0,2], lambda x: x.pad(((1,0),(0,0)), value=value)[0,2])
      helper_test_op([(4,4)], lambda x: torch.nn.functional.pad(x,(0,0,0,2), value=value)[5], lambda x: x.pad(((0,2),(0,0)), value=value)[5])
      helper_test_op([(4,4)], lambda x: torch.nn.functional.pad(x,(0,0,0,2), value=value)[3:5], lambda x: x.pad(((0,2),(0,0)), value=value)[3:5])
      helper_test_op([(4,4)], lambda x: torch.nn.functional.pad(x,(3,0,0,0), value=value)[1,0], lambda x: x.pad(((0,0),(3,0)), value=value)[1,0])
      helper_test_op([(4,4)], lambda x: torch.nn.functional.pad(x,(3,0,0,0), value=value)[1,0:4], lambda x: x.pad(((0,0),(3,0)), value=value)[1,0:4])
      helper_test_op([(4,4)], lambda x: torch.nn.functional.pad(x,(3,4,1,2), value=value)[0], lambda x: x.pad(((1,2),(3,4)), value=value)[0])
      helper_test_op([(4,4)], lambda x: torch.nn.functional.pad(x,(3,4,1,2), value=value)[:,1], lambda x: x.pad(((1,2),(3,4)), value=value)[:,1])
      helper_test_op([(4,4)], lambda x: torch.nn.functional.pad(x,(3,4,1,2), value=value)[:,4], lambda x: x.pad(((1,2),(3,4)), value=value)[:,4])
      helper_test_op([(3,3)], lambda x: torch.nn.functional.pad(x,(0,3,0,0), value=value)[:,4:6], lambda x: x.pad(((0,0),(0,3)), value=value)[:,4:6])
      helper_test_op([(3,3)], lambda x: torch.nn.functional.pad(x,(0,1,3,2), value=value)[0:2,:], lambda x: x.pad(((3,2),(0,1)), value=value)[0:2,:])
      helper_test_op([(3,3,3)], lambda x: torch.nn.functional.pad(x,(1,1,0,1,3,2), value=value)[0:2,:,:],
                                lambda x: x.pad(((3,2),(0,1),(1,1)), value=value)[0:2,:,:])
      helper_test_op([(3,3,3)], lambda x: torch.nn.functional.pad(x,(1,1,0,1,3,2), value=value)[2:4,:,:],
                                lambda x: x.pad(((3,2),(0,1),(1,1)), value=value)[2:4,:,:])

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
    self.helper_test_exception([(3,4,5,6)], lambda x: x.permute((0,2)), lambda x: x.permute((0,2)), expected=RuntimeError)
    self.helper_test_exception([(3,4,5,6)], lambda x: x.permute((0,1,2,3,3,3)), lambda x: x.permute((0,1,2,3,3,3)), expected=RuntimeError)
    self.helper_test_exception([(3,4,5,6)], lambda x: x.permute((0,0,1,2,3)), lambda x: x.permute((0,0,1,2,3)), expected=RuntimeError)

  def test_reshape(self):
    helper_test_op([(4,3,6,6)], lambda x: x.reshape((12,6,6)))
    helper_test_op([(4,3,6,6)], lambda x: x.reshape((-1,3,6,6)))
    helper_test_op([(4,3,6,6)], lambda x: x.reshape((-1,1,6,6)))
    helper_test_op([(4,3,6,6)], lambda x: x.reshape((4,3,6,6)), lambda x: x.reshape((None,None,6,6)))
    helper_test_op([()], lambda x: x.reshape(()))
    helper_test_op([(1,)], lambda x: x.reshape(()))
    helper_test_op([()], lambda x: x.reshape((1,)))
    helper_test_op([()], lambda x: x.reshape((1,1,1)))
    self.helper_test_exception([(3,4)], lambda x: x.reshape((-1,-1,2)), lambda x: x.reshape((-1,-1,2)), expected=RuntimeError)
    self.helper_test_exception([(3,4)], lambda x: x.reshape((-1,-1,-1,2)), lambda x: x.reshape((-1,-1,-1,2)), expected=RuntimeError)

    with self.assertRaises(ValueError):
      x = Tensor.ones((4,3,6,6))
      x.reshape([])

  def test_view(self):
    helper_test_op([(4,3,6,6)], lambda x: x.view((12,6,6)))
    helper_test_op([(4,3,6,6)], lambda x: x.view((-1,3,6,6)))

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
    self.helper_test_exception([(3,4)], lambda x: x.flip((0,0)), lambda x: x.flip((0,0)), expected=RuntimeError)
    self.helper_test_exception([(3,4)], lambda x: x.flip((1,1)), lambda x: x.flip((1,1)), expected=RuntimeError)
    self.helper_test_exception([(3,4)], lambda x: x.flip((1,-1)), lambda x: x.flip((1,-1)), expected=RuntimeError)

  def test_squeeze(self):
    helper_test_op([(1,3,6,6)], lambda x: x.squeeze(0))
    helper_test_op([(4,3,1,6)], lambda x: x.squeeze(1))
    helper_test_op([(4,3,6,6)], lambda x: x.squeeze(3))
    self.helper_test_exception([(4,3,6,6)], lambda x: torch.squeeze(x, 50), lambda x: x.squeeze(dim=50), expected=IndexError)
    self.helper_test_exception([(4,3,6,6)], lambda x: torch.squeeze(x, -50), lambda x: x.squeeze(dim=-50), expected=IndexError)
    helper_test_op([(4,3,6,1)], lambda x: x.squeeze(-1))
    helper_test_op([(4,3,6,6)], lambda x: x.squeeze())
    helper_test_op([(1,3,6,6)], lambda x: x.squeeze())
    helper_test_op([(2,3,1)], lambda x: x.squeeze())
    helper_test_op([()], lambda x: x.squeeze(-1))
    helper_test_op([()], lambda x: x.squeeze(0))
    helper_test_op([()], lambda x: x.squeeze())
    self.helper_test_exception([()], lambda x: torch.squeeze(x, 10), lambda x: x.squeeze(dim=10), expected=IndexError)
    self.helper_test_exception([()], lambda x: torch.squeeze(x, 1), lambda x: x.squeeze(dim=1), expected=IndexError)
    self.helper_test_exception([()], lambda x: torch.squeeze(x, -2), lambda x: x.squeeze(dim=-2), expected=IndexError)

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

  def test_roll(self):
    helper_test_op([(2, 4)], lambda x: torch.roll(x, 1, 0), lambda x: x.roll(1, 0))
    helper_test_op([(2, 4)], lambda x: torch.roll(x, -1, 0), lambda x: x.roll(-1, 0))
    helper_test_op([(2, 4)], lambda x: torch.roll(x, shifts=(2, 1), dims=(0, 1)), lambda x: x.roll(shifts=(2, 1), dims=(0, 1)))
    helper_test_op([(2, 4, 6)], lambda x: torch.roll(x, 1, 0), lambda x: x.roll(1, 0))
    helper_test_op([(2, 4)], lambda x: torch.roll(x, 1, -1), lambda x: x.roll(1, -1))
    helper_test_op([(2, 4)], lambda x: torch.roll(x, -1, -1), lambda x: x.roll(-1, -1))
    helper_test_op([(2, 4)], lambda x: torch.roll(x, 5, 0), lambda x: x.roll(5, 0))
    helper_test_op([(2, 4)], lambda x: torch.roll(x, -5, 0), lambda x: x.roll(-5, 0))
    helper_test_op([(2, 4, 6)], lambda x: torch.roll(x, shifts=(2, -3), dims=(0, 2)), lambda x: x.roll(shifts=(2, -3), dims=(0, 2)))
    helper_test_op([(2, 4, 6)], lambda x: torch.roll(x, shifts=(1, 2, -1), dims=(0, 1, 2)), lambda x: x.roll(shifts=(1, 2, -1), dims=(0, 1, 2)))
    helper_test_op([(2, 4)], lambda x: torch.roll(x, 0, 0), lambda x: x.roll(0, 0))
    helper_test_op([(2, 4, 6)], lambda x: torch.roll(x, shifts=(0, 0), dims=(0, 1)), lambda x: x.roll(shifts=(0, 0), dims=(0, 1)))
    helper_test_op([(2, 4, 6)], lambda x: torch.roll(x, shifts=(0, 2), dims=(0, 1)), lambda x: x.roll(shifts=(0, 2), dims=(0, 1)))

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

  def test_biased_conv2d(self):
    C = 8
    helper_test_op([(1,C,5,5), (C,C,1,1), (C,)],
      lambda x,w,b: torch.nn.functional.conv2d(torch.nn.functional.conv2d(x,w,b).relu(),w,b),
      lambda x,w,b: Tensor.conv2d(x,w,b).relu().conv2d(w,b))

  def test_simple_conv2d(self):
    helper_test_op([(1,4,9,9), (4,4,3,3)],
      lambda x,w: torch.nn.functional.conv2d(x,w).relu(),
      lambda x,w: Tensor.conv2d(x,w).relu(), grad_rtol=1e-5)

  def test_simple_conv2d_bias(self):
    helper_test_op([(1,4,9,9), (4,4,3,3), (4,)],
      lambda x,w,b: torch.nn.functional.conv2d(x,w,b).relu(),
      lambda x,w,b: Tensor.conv2d(x,w,b).relu(), grad_rtol=1e-5)

  @unittest.skipIf(IMAGE>0, "no conv3d on images")
  def test_simple_conv3d(self):
    helper_test_op([(1,4,9,9,9), (4,4,3,3,3)],
      lambda x,w: torch.nn.functional.conv3d(x,w).relu(),
      lambda x,w: Tensor.conv2d(x,w).relu(), grad_rtol=1e-5)

  @unittest.skipIf(IMAGE>0, "no conv3d on images")
  def test_padded_conv3d(self):
    helper_test_op([(1,4,5,5,5), (4,4,3,3,3)],
      lambda x,w: torch.nn.functional.conv3d(x,w,padding=1).relu(),
      lambda x,w: Tensor.conv2d(x,w,padding=[1,1,1,1,1,1]).relu(), grad_rtol=1e-5)

  def test_simple_conv2d_m4(self):
    helper_test_op([(1,16,18,18), (16,16,3,3)],
      lambda x,w: torch.nn.functional.conv2d(x,w).relu(),
      lambda x,w: Tensor.conv2d(x,w).relu(), grad_rtol=1e-5)

  def test_simple_conv2d_1x1(self):
    helper_test_op([(1,4,9,9), (4,4,1,1)],
      lambda x,w: torch.nn.functional.conv2d(x,w).relu(),
      lambda x,w: Tensor.conv2d(x,w).relu(), grad_rtol=1e-5)

  def test_simple_conv2d_1x1_m4(self):
    helper_test_op([(1,16,32,32), (16,16,1,1)],
      lambda x,w: torch.nn.functional.conv2d(x,w).relu(),
      lambda x,w: Tensor.conv2d(x,w).relu(), grad_rtol=1e-5)

  def test_nested_conv2d(self):
    helper_test_op([(1,32,9,9), (32,32,3,3), (32,32,3,3)],
      lambda x,w1,w2: torch.nn.functional.conv2d(torch.nn.functional.conv2d(x,w1).relu(), w2).relu(),
      lambda x,w1,w2: x.conv2d(w1).relu().conv2d(w2).relu())

  # expect reduce nodes == 3
  def test_simple_conv2d_nhwc(self):
    # weights (from tf): filter_height x filter_width x in_channels x out_channels
    helper_test_op([(2,9,9,10), (3,3,10,20)],
      lambda x,w: torch.nn.functional.conv2d(x.permute(0,3,1,2),w.permute(3,2,0,1)).relu(),
      lambda x,w: Tensor.conv2d(x.permute(0,3,1,2),w.permute(3,2,0,1)).relu(), atol=1e-5, grad_rtol=1e-5)

  def test_simple_conv2d_batched(self):
    helper_test_op([(2,4,9,9), (4,4,3,3)],
      lambda x,w: torch.nn.functional.conv2d(x,w).relu(),
      lambda x,w: Tensor.conv2d(x,w).relu(), grad_rtol=1e-5)

  # conv transpose

  def test_simple_conv_transpose2d(self):
    helper_test_op([(2,4,9,9), (4,4,3,3)],
      lambda x,w: torch.nn.functional.conv_transpose2d(x,w).relu(),
      lambda x,w: Tensor.conv_transpose2d(x,w).relu(), grad_rtol=1e-5)

  def test_bias_conv_transpose2d(self):
    helper_test_op([(2,4,9,9), (4,4,3,3), (4,)],
      lambda x,w,b: torch.nn.functional.conv_transpose2d(x,w,b).relu(),
      lambda x,w,b: Tensor.conv_transpose2d(x,w,b).relu(), grad_rtol=1e-5)

  def test_grouped_conv_transpose2d(self):
    helper_test_op([(2,4,9,9), (4,4,3,3)],
      lambda x,w: torch.nn.functional.conv_transpose2d(x,w,groups=2).relu(),
      lambda x,w: Tensor.conv_transpose2d(x,w,groups=2).relu(), grad_rtol=1e-5)

  def test_padded_conv_transpose2d(self):
    for padding in [(1,2), (2,1), 2, 1, 0]:
      helper_test_op([(2,4,9,9), (4,4,3,3)],
        lambda x,w: torch.nn.functional.conv_transpose2d(x,w,padding=padding).relu(),
        lambda x,w: Tensor.conv_transpose2d(x,w,padding=padding).relu(), grad_rtol=1e-5)
    self.helper_test_exception([(2,16,2,2), (32,16,3,3)], lambda x,w: torch.nn.functional.conv_transpose2d(x,w,padding=(1,1,1)),
                   lambda x,w: Tensor.conv_transpose2d(x,w,padding=(1,1,1)), expected=(RuntimeError, ValueError))

  def test_dilated_conv_transpose2d(self):
    for dilation in [(1,2), (2,1), 2, 1]:
      helper_test_op([(2,4,9,9), (4,4,3,3)],
        lambda x,w: torch.nn.functional.conv_transpose2d(x,w,dilation=dilation).relu(),
        lambda x,w: Tensor.conv_transpose2d(x,w,dilation=dilation).relu(), grad_rtol=1e-5)

  def test_strided_conv_transpose2d(self):
    for stride in [(2,1), (1,2), 1]:
      helper_test_op([(2,4,4,5), (4,4,3,3)],
        lambda x,w: torch.nn.functional.conv_transpose2d(x,w, stride=stride).relu(),
        lambda x,w: Tensor.conv_transpose2d(x,w,stride=stride).relu(), atol=1e-5, grad_rtol=1e-5)

  def test_output_padded_conv_transpose2d(self):
    for output_padding, stride in [((1,1), (2,3)), ((2,1), (3,2))]:
      helper_test_op([(2,4,6,5), (4,4,3,3),(4,)],
        lambda x,w,b: torch.nn.functional.conv_transpose2d(x,w,b,output_padding=output_padding,stride=stride).relu(),
        lambda x,w,b: Tensor.conv_transpose2d(x,w,b,output_padding=output_padding,stride=stride).relu(), grad_rtol=1e-5)

  @unittest.skipIf(IMAGE>0, "no conv3d on images")
  def test_simple_conv_transpose3d(self):
    helper_test_op([(2,4,9,9,9), (4,4,3,3,3)],
      lambda x,w: torch.nn.functional.conv_transpose3d(x,w).relu(),
      lambda x,w: Tensor.conv_transpose2d(x,w).relu(), grad_rtol=1e-5)

  @unittest.skipIf((IMAGE>0), "no conv1d on images")
  def test_conv1d(self):
    for bs in [1,8]:
      for cin in [1,3]:
        for H in [1,2,5]:
          for groups in [1,3] if cin == 3 and H == 5 else [1]:
            with self.subTest(batch_size=bs, channels=cin, groups=groups, height=H):
              helper_test_op([(bs,cin,11), (6,cin//groups,H)],
                lambda x,w: torch.nn.functional.conv1d(x,w,groups=groups).relu(),
                lambda x,w: Tensor.conv2d(x,w,groups=groups).relu(), grad_rtol=1e-5)

  @unittest.skipIf(IMAGE>0, "no conv1d on images")
  def test_simple_padding_conv1d(self):
    bs = 6
    cin = 2
    groups = 1
    H = 5
    p = (1,1)
    helper_test_op([(bs,cin,11), (6,cin//groups,H)],
      lambda x,w: torch.nn.functional.conv1d(torch.nn.functional.pad(x, p),w).relu(),
      lambda x,w: Tensor.conv2d(x,w,padding=p).relu())

  @unittest.skipIf(IMAGE>0, "no conv1d on images")
  def test_strided_conv1d_simple(self):
    bs, H = 2, 3
    helper_test_op([(bs,1,5), (1,1,H)],
      lambda x,w: torch.nn.functional.conv1d(x,w,stride=2).relu(),
      lambda x,w: Tensor.conv2d(x,w,stride=2).relu())

  @unittest.skipIf(IMAGE>0, "no conv1d on images")
  def test_asymmetric_padding_conv1d(self):
    for p in [(0,1), (2,1), (2,0)]:
      with self.subTest(p):
        for n in [3,4]:
          for k in [2]:
            helper_test_op([(1,1,n), (1,1,k)],
              lambda x,w: torch.nn.functional.conv1d(torch.nn.functional.pad(x, p),w).relu(),
              lambda x,w: Tensor.conv2d(x,w,padding=p).relu())

  def _test_conv2d(self, bs=1, cin=1, cout=6):
    for H in [1,2,3]:
      for W in [1,2,3,5]:
        for groups in [1,3] if cin == 3 and cout == 6 and H == 3 and W == 3 else [1]:
          with self.subTest(batch_size=bs, channels=cin, groups=groups, height=H, width=W):
            helper_test_op([(bs,cin,5,7), (cout,cin//groups,H,W)],
              lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups).relu(),
              lambda x,w: Tensor.conv2d(x,w,groups=groups).relu(), grad_rtol=1e-5)
  def test_conv2d(self): self._test_conv2d(bs=1, cin=3)
  def test_conv2d_bs_4_cin_3(self): self._test_conv2d(bs=4, cin=3, cout=2)
  def test_conv2d_bs_1_cin_1(self): self._test_conv2d(bs=1, cin=1)
  def test_conv2d_bs_4_cin_1(self): self._test_conv2d(bs=4, cin=1)

  def test_conv2d_errors(self):
    # kernel size cannot be larger than input size
    self.helper_test_exception([(1,1,6,7), (6,1,3,3)],
                               lambda x,w:torch.nn.functional.conv2d(x,w,dilation=3),
                               lambda x,w: Tensor.conv2d(x,w,dilation=3), expected=(RuntimeError, AssertionError))
    # regression test for https://github.com/tinygrad/tinygrad/pull/7549/
    self.helper_test_exception([(2,16,2,2), (32,16,3,3)], lambda x,w:torch.nn.functional.conv2d(x,w), lambda x,w: Tensor.conv2d(x,w),
                               expected=(RuntimeError, AssertionError))
    self.helper_test_exception([(2,16,2,2), (32,16,3,3)], lambda x,w:torch.nn.functional.conv2d(x,w,padding=(1,1,1)),
                               lambda x,w: Tensor.conv2d(x,w,padding=(1,1,1)), expected=(RuntimeError, ValueError))

  def test_large_input_conv2d(self):
    bs = 4
    cin = 16
    groups = 1
    H = 5
    W = 2
    helper_test_op([(bs,cin,64,64), (6,cin//groups,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups).relu(),
      # needed to relax tolerance on NVIDIA
      lambda x,w: Tensor.conv2d(x,w,groups=groups).relu(), atol=1e-4, grad_atol=1e-4, grad_rtol=1e-4)

  def test_simple_grouped_conv2d(self):
    bs = 1
    groups = 2
    rcout = 1
    cin = 2
    helper_test_op([(bs,groups*cin,1,1), (groups*rcout,cin,1,1)],
      lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups).relu(),
      lambda x,w: Tensor.conv2d(x,w,groups=groups).relu(), grad_rtol=1e-5)

  def test_medium_grouped_conv2d(self):
    bs = 1
    groups = 2
    rcout = 2
    cin = 2
    helper_test_op([(bs,groups*cin,1,1), (groups*rcout,cin,1,1)],
      lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups).relu(),
      lambda x,w: Tensor.conv2d(x,w,groups=groups).relu(), grad_rtol=1e-5)

  def test_depthwise_conv2d(self):
    bs = 1
    groups = 32
    rcout = 1
    cin = 1
    helper_test_op([(bs,groups*cin,32,32), (groups*rcout,cin,1,1)],
      lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups).relu(),
      lambda x,w: Tensor.conv2d(x,w,groups=groups).relu(), grad_rtol=1e-5)

  def test_grouped_conv2d(self):
    bs = 4
    groups = 5
    rcout = 7
    cin = 3
    helper_test_op([(bs,groups*cin,5,5), (groups*rcout,cin,3,3)],
      lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups).relu(),
      lambda x,w: Tensor.conv2d(x,w,groups=groups).relu(), grad_rtol=1e-5)

  def test_fancy_conv2d(self):
    bs = 2
    cin = 3
    cout = 1
    groups = 3
    H,W = 3,3
    helper_test_op([(bs,cin,11,28), (groups*cout,cin//groups,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups).relu(),
      lambda x,w: Tensor.conv2d(x,w,groups=groups).relu(), grad_rtol=1e-5)

  def test_strided_conv2d_simple(self):
    bs,H,W = 2,3,1
    helper_test_op([(bs,1,5,1), (1,1,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,stride=2).relu(),
      lambda x,w: Tensor.conv2d(x,w,stride=2).relu())

  @unittest.skipIf(Device.DEFAULT != "LLVM", "DEVECTORIZE=0 only for LLVM")
  def test_strided_conv2d_simple_vec(self):
    with Context(DEVECTORIZE=0): self.test_strided_conv2d_simple()

  def test_strided_conv2d(self):
    bs = 4
    cin = 3
    H,W = 3,3
    with self.subTest(stride := 2):
      helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
        lambda x,w: torch.nn.functional.conv2d(x,w,stride=2).relu(),
        lambda x,w: Tensor.conv2d(x,w,stride=stride).relu())
    with self.subTest(stride := (2,1)):
      helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
        lambda x,w: torch.nn.functional.conv2d(x,w,stride=stride).relu(),
        lambda x,w: Tensor.conv2d(x,w,stride=(2,1)).relu())

  def test_negative_padding_conv2d(self):
    n,k = 10, 3
    helper_test_op([(1,1,n,n), (1,1,k,k)],
      lambda x,w: torch.nn.functional.conv2d(x[:, :, 1:-1, 1:-1],w).relu(),
      lambda x,w: Tensor.conv2d(x,w,padding=-1).relu())
    helper_test_op([(1,1,n,n), (1,1,k,k)],
      lambda x,w: torch.nn.functional.conv2d(x[:, :, 1:, 1:],w).relu(),
      lambda x,w: Tensor.conv2d(x,w,padding=(-1,0,-1,0)).relu())

  def test_simple_padding_conv2d(self):
    p = (1,1,1,1)
    helper_test_op(None,
      lambda x,w: torch.nn.functional.conv2d(torch.nn.functional.pad(x, p),w).relu(),
      lambda x,w: Tensor.conv2d(x,w,padding=p).relu(), vals=[[[[[2.,3.]]]], [[[[1.]]]]])

  def test_asymmetric_padding_conv2d(self):
    for p in [(0,1,0,1), (2,1,2,1), (2,0,2,1)]:
      with self.subTest(p):
        for n in [3,4]:
          for k in [2]:
            helper_test_op([(1,1,n,n), (1,1,k,k)],
              lambda x,w: torch.nn.functional.conv2d(torch.nn.functional.pad(x, p),w).relu(),
              lambda x,w: Tensor.conv2d(x,w,padding=p).relu())
            helper_test_op([(1,1,n,n), (1,1,k,k)],
              lambda x,w: torch.nn.functional.conv2d(torch.nn.functional.pad(x, p),w).relu(),
              lambda x,w: Tensor.conv2d(x,w,padding=p).relu())

  def test_padded_conv2d_p21(self):
    bs,cin,H,W,padding = 4, 3, 3, 3, (2,1)
    helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,padding=padding).relu(),
      lambda x,w: Tensor.conv2d(x,w,padding=padding).relu())

  def test_padded_conv2d_p22(self):
    bs,cin,H,W,padding = 4, 3, 3, 3, (2,2)
    helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,padding=padding).relu(),
      lambda x,w: Tensor.conv2d(x,w,padding=padding).relu())

  def test_padded_conv2d_1x1(self):
    bs,cin,H,W,padding = 4, 3, 1, 1, 2
    helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,padding=padding).relu(),
      lambda x,w: Tensor.conv2d(x,w,padding=padding).relu())

  def test_padded_conv2d_bs1(self):
    bs,cin,H,W,padding = 1, 3, 3, 3, 1
    helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,padding=padding).relu(),
      lambda x,w: Tensor.conv2d(x,w,padding=padding).relu())

  def test_padding_add(self):
    helper_test_op([(64,64), (60,60)],
      lambda x,w: x+torch.nn.functional.pad(w, (2,2,2,2)),
      lambda x,w: x+w.pad((2,2,2,2)))

  def test_dilated_conv2d(self):
    bs = 4
    cin = 3
    H,W = 3,3
    for d in [2, (2,1)]:
      with self.subTest(dilation := d):
        helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
          lambda x,w: torch.nn.functional.conv2d(x,w,dilation=dilation).relu(),
          lambda x,w: Tensor.conv2d(x,w,dilation=dilation).relu())

  def test_max_pool2d_simple(self):
    ksz = (2,2)
    helper_test_op([(1,1,2,3)],
      lambda x: torch.nn.functional.max_pool2d(x, kernel_size=ksz),
      lambda x: Tensor.max_pool2d(x, kernel_size=ksz))

  def test_max_pool2d(self):
    for ksz in [(2,2), (3,3), 2, 3, (3,2), (5,5), (5,1)]:
      with self.subTest(kernel_size=ksz):
        helper_test_op([(32,2,110,28)],
          lambda x: torch.nn.functional.max_pool2d(x, kernel_size=ksz),
          lambda x: Tensor.max_pool2d(x, kernel_size=ksz))

  def test_max_pool2d_padding(self):
    for ksz in [(2,2), (3,3), 2, 3, (3,2)]:
      for p in [1, (1,0), (0,1)]:
        with self.subTest(kernel_size=ksz, padding=p):
          helper_test_op([(32,2,110,28)],
            lambda x: torch.nn.functional.max_pool2d(x, kernel_size=ksz, padding=p),
            lambda x: Tensor.max_pool2d(x, kernel_size=ksz, padding=p))
    self.helper_test_exception([(32,2,110,28)], lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(2,2), padding=(1,1,1)),
                   lambda x: Tensor.max_pool2d(x, kernel_size=(2,2), padding=(1,1,1)), expected=(RuntimeError, ValueError))

  def test_max_pool2d_asymmetric_padding(self):
    shape = (32,2,111,28)
    for p in [(0,1,0,1), (2,1,2,1), (2,0,2,1)]:
      with self.subTest(padding=p):
        helper_test_op([shape],
          lambda x: torch.nn.functional.max_pool2d(torch.nn.functional.pad(x, p, value=float("-inf")), kernel_size=(5,5)),
          lambda x: Tensor.max_pool2d(x, kernel_size=(5,5), padding=p))

  def test_max_pool2d_padding_int(self):
    ksz = (2,2)
    helper_test_op([(32,2,110,28)],
      lambda x: torch.nn.functional.max_pool2d(x.int(), kernel_size=ksz, padding=1),
      lambda x: Tensor.max_pool2d(x.int(), kernel_size=ksz, padding=1), forward_only=True)

  def test_max_pool2d_bigger_stride(self):
    for stride in [(2,3), (3,2), 2, 3]:
      with self.subTest(stride=stride):
        helper_test_op([(32,2,110,28)],
          lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(2,2), stride=stride),
          lambda x: Tensor.max_pool2d(x, kernel_size=(2,2), stride=stride))

  def test_max_pool2d_bigger_stride_dilation(self):
    for stride, dilation in zip([(2,3), (3,2), 2, 3, 4], [(3,2), (2,3), 2, 3, 6]):
      with self.subTest(stride=stride):
        helper_test_op([(32,2,110,28)],
          lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(2,2), stride=stride, dilation=dilation),
          lambda x: Tensor.max_pool2d(x, kernel_size=(2,2), stride=stride, dilation=dilation))

  @unittest.skipIf( Device.DEFAULT in {"CUDA", "NV"}, "CUDA fails on this")
  def test_max_pool2d_unit_stride(self):
    helper_test_op([(8, 2, 17, 14)],
      lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(5,5), stride=1),
      lambda x: Tensor.max_pool2d(x, kernel_size=(5,5), stride=1))

  def test_max_pool2d_smaller_stride(self):
    for stride in [(2,3), (3,2), 2, 3]:
      with self.subTest(stride=stride):
        helper_test_op([(8, 2, 17, 14)],
          lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(5,5), stride=stride),
          lambda x: Tensor.max_pool2d(x, kernel_size=(5,5), stride=stride))

  def test_max_pool2d_dilation(self):
    for dilation in [(2, 3), (3, 2), 2, 3]:
      helper_test_op([(8, 2, 17, 14)],
        lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(5,5), dilation=dilation),
        lambda x: Tensor.max_pool2d(x, kernel_size=(5,5), dilation=dilation))

  def test_max_pool2d_ceil_mode(self):
    shape = (1,1,6,6)
    for ksz in [(3,3), 3, (3,2), 4]:
      with self.subTest(kernel_size=ksz):
        helper_test_op([shape],
          lambda x: torch.nn.functional.max_pool2d(x, kernel_size=ksz, padding=1, stride=3, ceil_mode=True),
          lambda x: Tensor.max_pool2d(x, kernel_size=ksz, padding=1, stride=3, ceil_mode=True))

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

  def test_avg_pool2d(self):
    shape = (32,2,111,28)
    for ksz in [(2,2), (3,3), (3,2), (5,5), (5,1)]:
      with self.subTest(kernel_size=ksz):
        helper_test_op([shape],
          lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=ksz),
          lambda x: Tensor.avg_pool2d(x, kernel_size=ksz), rtol=1e-5)

    # regression test for https://github.com/tinygrad/tinygrad/pull/7581
    helper_test_op([(1,1,8,8)],
      lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=(1,2), padding=(0,1), stride=(5,1)),
      lambda x: Tensor.avg_pool2d(x, kernel_size=(1,2), padding=(0,1), stride=(5,1)), rtol=1e-5)

  def test_avg_pool2d_padding(self):
    shape = (32,2,111,28)
    for ksz in [(2,2), (3,3), 2, 3, (3,2)]:
      for p in [1, (1,0), (0,1)]:
        with self.subTest(kernel_size=ksz, padding=p):
          helper_test_op([shape],
            lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=ksz, padding=p),
            lambda x: Tensor.avg_pool2d(x, kernel_size=ksz, padding=p), rtol=1e-5)
    with self.assertRaises(ValueError):
      Tensor.avg_pool2d(Tensor.randn((32,2,111,28)), kernel_size=(2,2), padding=(1,1,1))

  def test_avg_pool2d_asymmetric_padding(self):
    shape = (32,2,111,28)
    for p in [(0,1,0,1), (2,1,2,1), (2,0,2,1)]:
      with self.subTest(padding=p):
        helper_test_op([shape],
          lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=(5,5), padding=1),
          lambda x: Tensor.avg_pool2d(x, kernel_size=(5,5), padding=1), rtol=1e-5)
    self.helper_test_exception([shape], lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=(2,2), padding=(1,1,1)),
                               lambda x: Tensor.avg_pool2d(x, kernel_size=(2,2), padding=(1,1,1)), expected=(RuntimeError, ValueError))

  def test_avg_pool2d_padding_not_counted(self):
    shape = (32,2,111,28)
    for ksz in [(2,2), (3,3), 2, 3, (3,2)]:
      with self.subTest(kernel_size=ksz):
        helper_test_op([shape],
          lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=ksz, padding=1, count_include_pad=False),
          lambda x: Tensor.avg_pool2d(x, kernel_size=ksz, padding=1, count_include_pad=False), rtol=1e-5)

  def test_avg_pool2d_ceil_mode(self):
    shape = (1,1,6,6)
    for ksz in [(3,3), 3, (3,2), 4]:
      with self.subTest(kernel_size=ksz):
        helper_test_op([shape],
          lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=ksz, padding=1, stride=3, ceil_mode=True),
          lambda x: Tensor.avg_pool2d(x, kernel_size=ksz, padding=1, stride=3, ceil_mode=True), rtol=1e-5)

  def test_avg_pool2d_ceil_mode_padding_not_counted(self):
    shape = (1,1,6,6)
    for ksz in [(3,3), 3, (3,2), 4]:
      with self.subTest(kernel_size=ksz):
        helper_test_op([shape],
          lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=ksz, padding=1, stride=3, ceil_mode=True, count_include_pad=False),
          lambda x: Tensor.avg_pool2d(x, kernel_size=ksz, padding=1, stride=3, ceil_mode=True, count_include_pad=False), rtol=1e-5)

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
    helper_test_op([(32,2,111,28)],
      lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=(111,28)),
      lambda x: Tensor.avg_pool2d(x, kernel_size=(111,28)), rtol=1e-5)

  def test_avg_pool3d_failure(self):
    with Context(NOOPT=0):
      helper_test_op([(1,1,16,16,16)],
        lambda x: torch.nn.functional.avg_pool3d(x, kernel_size=(8,8,8), stride=5, padding=1, count_include_pad=False),
        lambda x: Tensor.avg_pool2d(x, kernel_size=(8,8,8), stride=5, padding=1, count_include_pad=False), rtol=1e-5, forward_only=True)

  def test_avg_pool3d_noopt(self):
    with Context(NOOPT=1):
      helper_test_op([(1,1,16,16,16)],
        lambda x: torch.nn.functional.avg_pool3d(x, kernel_size=(8,8,8), stride=5, padding=1, count_include_pad=False),
        lambda x: Tensor.avg_pool2d(x, kernel_size=(8,8,8), stride=5, padding=1, count_include_pad=False), rtol=1e-5, forward_only=True)

  def test_interpolate_linear(self):
    for in_sz, out_sz in [((52,),(29,)), ((29,),(52,))]:
      helper_test_op([(2,3)+in_sz],
        lambda x: torch.nn.functional.interpolate(x, size=out_sz, mode="linear"),
        lambda x: Tensor.interpolate(x, size=out_sz, mode="linear"))

  def test_interpolate_linear_corners_aligned(self):
    for in_sz, out_sz in [((52,),(29,)), ((29,),(52,))]:
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

  def test_interpolate_bilinear(self):
    for in_sz, out_sz in [((52,40),(29,31)), ((52,29),(31,40)), ((29,31),(40,52))]:
      helper_test_op([(2,3)+in_sz],
        lambda x: torch.nn.functional.interpolate(x, size=out_sz, mode="bilinear"),
        lambda x: Tensor.interpolate(x, size=out_sz, mode="linear"), atol=1e-4)

  def test_interpolate_bilinear_corners_aligned(self):
    for in_sz, out_sz in [((52,40),(29,31)), ((52,29),(31,40)), ((29,31),(40,52))]:
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
    for dim in range(-1, 2):
      helper_test_op([(45,65), (45,65), (45,65)], lambda x,y,z: torch.cat((x,y,z), dim), lambda x,y,z: x.cat(y, z, dim=dim))

  def test_stack(self):
    for dim in range(-1, 3):
      helper_test_op([(45,65,3), (45,65,3), (45,65,3)], lambda x, y, z: torch.stack((x, y, z), dim), lambda x, y, z: Tensor.stack(x, y, z, dim=dim))

    with self.assertRaises(IndexError):
      Tensor.stack(Tensor.randn(45, 65, 3), dim=77)

    a = Tensor(3.14)
    np.testing.assert_allclose(Tensor.stack(a, a).numpy(), Tensor([3.14, 3.14]).numpy())

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

  def test_simple_repeat(self):
    repeats = [3, 3, 4]
    helper_test_op([(3, 3)], lambda x: x.repeat(*repeats), lambda x: x.repeat(repeats))

  def test_clip(self):
    helper_test_op([(45,65)], lambda x: x.clip(-2.3, 1.2))
    helper_test_op([(45,65)], lambda x: x.clip(0, 0))
    helper_test_op([(45,65)], lambda x: x.clip(10, 100))
    helper_test_op([(45,65)], lambda x: x.clip(0, 0.1))
    helper_test_op([(45,65)], lambda x: x.clip(-0.3, -0.2))
    helper_test_op([(45,65)], lambda x: x.clip(3, 0))  # min > max
    helper_test_op([(45,65)], lambda x: x.clip(None, 0))
    helper_test_op([(45,65)], lambda x: x.clip(0, None))
    self.helper_test_exception([(45,65)], lambda x: x.clip(None, None), lambda x: x.clip(None, None), RuntimeError)

  def test_matvecmat(self):
    helper_test_op([(1,128), (128,128), (128,128)], lambda x,y,z: (x@y).relu()@z)

  def test_matvec(self):
    helper_test_op([(1,128), (128,128)], lambda x,y: (x@y).relu())

  @unittest.skip("this test is broken #862")
  def test_max_nan(self):
    n = Tensor([1, float("nan")]).max().numpy()
    assert math.isnan(n.item()), f"{n.item()} is not nan"

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
    i, j, k, o, p = [Tensor(tor.detach().cpu().numpy().astype(np.int32), requires_grad=False) for tor in [a,b,c,d,e]]
    return a,b,c,d,e,i,j,k,o,p

  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "WEBGPU can only run kernels with up to 10 buffers")
  def test_slice_fancy_indexing_no_dim_collapse(self):
    a,b,c,d,e,i,j,k,o,p = self._get_index_randoms()
    # no dim collapse from int or dim injection from None
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,b,c,d,e], lambda x: x[i,j,k,o,p])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[:,b,c,d,:], lambda x: x[:,j,k,o,:])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,b,...], lambda x: x[i,j,...])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,...,e], lambda x: x[i,...,p])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[...,c,:,e], lambda x: x[...,k,:,p])

  def test_slice_fancy_indexing_dim_collapse_int(self):
    a,b,c,d,e,i,j,k,o,p = self._get_index_randoms()
    # dim collapse from int
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[1,b,c,d,e], lambda x: x[1,j,k,o,p])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,b,3,d,e], lambda x: x[i,j,3,o,p])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[1,b,2,d,2], lambda x: x[1,j,2,o,2])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,2,2,2,e], lambda x: x[i,2,2,2,p])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[1,:,3:11:2,d,0:2], lambda x: x[1,:,3:11:2,o,0:2])

  def test_slice_fancy_indexing_dim_inject_none(self):
    a,b,c,d,e,i,j,k,o,p = self._get_index_randoms()
    # dim injection from None
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[None,b,c,d,e], lambda x: x[None,j,k,o,p])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,b,c,d,None], lambda x: x[i,j,k,o,None])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,b,None,d,e], lambda x: x[i,j,None,o,p])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[None,b,c,d,None], lambda x: x[None,j,k,o,None])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,:,None,d,e], lambda x: x[i,:,None,o,p])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[None,None,None,None,None], lambda x: x[None,None,None,None,None])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[None,None,b,c,d,e], lambda x: x[None,None,j,k,o,p])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[None,None,b,c,None,None], lambda x: x[None,None,j,k,None,None])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,None,None,c,d,e], lambda x: x[i,None,None,k,o,p])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,None,None,c,None,None], lambda x: x[i,None,None,k,None,None])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[None,None,b,None,d,e], lambda x: x[None,None,j,None,o,p])

  def test_slice_fancy_indexing_dim_inject_and_collapse(self):
    a,b,c,d,e,i,j,k,o,p = self._get_index_randoms()  # noqa
    # dim injection and collapse
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[1,b,None,d,1], lambda x: x[1,j,None,o,1])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[None,b,2,d,None], lambda x: x[None,j,2,o,None])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[...,1,d,None], lambda x: x[...,1,o,None])

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

  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "WEBGPU can only run kernels with up to 10 buffers")
  def test_slice_fancy_indexing_list_indices(self):
    a,b,c,d,e,i,j,k,o,p = self._get_index_randoms()
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[[[0]]], lambda x: x[[[0]]])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[[0],b,c,d,:], lambda x: x[[0],j,k,o,:])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[[[[0]]],b,c,d,[[1]]], lambda x: x[[[[0]]],j,k,o,[[1]]])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[[1,0],b,c,d,:], lambda x: x[[1,0],j,k,o,:])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,b,c,[1,2,3],...], lambda x: x[i,j,k,[1,2,3],...])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,b,c,[[1],[2],[3]],...], lambda x: x[i,j,k,[[1],[2],[3]],...])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,[2,1,0],c,[2,1,0],e], lambda x: x[i,[2,1,0],k,[2,1,0],p])

  def test_slice_fancy_indexing_tuple_indices(self):
    a,b,c,d,e,i,j,k,o,p = self._get_index_randoms()
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[(((0,),),)], lambda x: x[(((0,),),)])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[(0,),b,c,d,:], lambda x: x[(0,),j,k,o,:])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[(1,0),b,c,d,:], lambda x: x[(1,0),j,k,o,:])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,b,c,(1,2,3),...], lambda x: x[i,j,k,(1,2,3),...])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,((2,),(1,),(0,)),c,(2,1,0)], lambda x: x[i,((2,),(1,),(0,)),k,(2,1,0)])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[1,(2,1,0),None,c,(2,1,0),e], lambda x: x[1,(2,1,0),None,k,(2,1,0),p])

  @unittest.skipIf(Device.DEFAULT == "WEBGPU" and not OSX, "WEBGPU Vulkan can only run kernels with up to 10 buffers")
  def test_slice_fancy_indexing_list_with_tensors(self):
    a,b,c,d,e,i,j,k,o,p = self._get_index_randoms()
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[[a]], lambda x: x[[i]])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[[a,1]], lambda x: x[[i,1]])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[[a,[1,1]]], lambda x: x[[i,[1,1]]])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[[a,(1,1)]], lambda x: x[[i,(1,1)]])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[[a,b,c,d,e]], lambda x: x[[i,j,k,o,p]])

  def test_slice_fancy_indexing_errors(self):
    a = Tensor.ones(10,11,12)
    # tensors used as indices must be int tensors
    with self.assertRaises(IndexError): a[Tensor(1.1)]
    with self.assertRaises(IndexError): a[Tensor([True, True])]
    # shape mismatch, cannot broadcast. either exception is okay
    with self.assertRaises((IndexError, ValueError)): a[Tensor.randint(3,1,1,1), Tensor.randint(1,4,1,1), Tensor.randint(2,4,4,1)]
    with self.assertRaises((IndexError, ValueError)): a[Tensor.randint(3,1,1,1), Tensor.randint(1,4,1,1,1)]

  def test_gather(self):
    # indices cannot have gradient
    # indices cannot be negative (torch gather)
    b = torch.randint(3, size=[3,4,5], dtype=torch.int64, requires_grad=False)
    a = Tensor(b.detach().cpu().numpy().astype(np.int32), dtype=dtypes.int32, requires_grad=False)
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

  @unittest.expectedFailure
  @unittest.skipIf(torch._C._get_privateuse1_backend_name() == "tiny", 'results in a success instead of a failure')
  def test_gather_failure(self):
    # gather with inf values do not work, other values results in nan
    helper_test_op(None, lambda x: x.gather(dim=0, index=torch.tensor([2, 1, 0, 1, 2], requires_grad=False)),
                         lambda x: x.gather(dim=0, index=Tensor([2, 1, 0, 1, 2])),
                         vals=[[-float("inf"), 2., 3.]])

  def test_scatter(self):
    b = torch.randint(3, size=[3,4,5], dtype=torch.int64, requires_grad=False)
    a = Tensor(b.detach().cpu().numpy().astype(np.int32), dtype=dtypes.int32, requires_grad=False)
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
    a = Tensor(b.detach().cpu().numpy().astype(np.int32), dtype=dtypes.int32, requires_grad=False)
    helper_test_op(None,
      lambda x,src: x.scatter(0, b, src),
      lambda x,src: x.scatter(0, a, src), forward_only=True,
      vals=[[1.,2.,3.,4.], [1.,0.]])

  def test_scatter_add(self):
    b = torch.randint(3, size=[3,4,5], dtype=torch.int64, requires_grad=False)
    a = Tensor(b.detach().cpu().numpy().astype(np.int32), dtype=dtypes.int32, requires_grad=False)
    helper_test_op([(4,5,6)], lambda x: x.scatter(dim=1, index=b, value=float("inf"), reduce="add"),
      lambda x: x.scatter(dim=1, index=a, src=float("inf"), reduce="add"), forward_only=True)

    # TODO: fails for webgpu
    if Device.DEFAULT != "WEBGPU":
      helper_test_op([(4,5,6)],
        lambda x: x.scatter(1, b, float("nan"), reduce="add"),
        lambda x: x.scatter(1, a, float("nan"), reduce="add"), forward_only=True)

  def test_scatter_mul(self):
    b = torch.randint(3, size=[3,4,5], dtype=torch.int64, requires_grad=False)
    a = Tensor(b.detach().cpu().numpy().astype(np.int32), dtype=dtypes.int32, requires_grad=False)
    helper_test_op([(4,5,6)], lambda x: x.scatter(dim=1, index=b, value=float("inf"), reduce="multiply"),
      lambda x: x.scatter(dim=1, index=a, src=float("inf"), reduce="multiply"), forward_only=True)

    # TODO: fails for webgpu
    if Device.DEFAULT != "WEBGPU":
      helper_test_op([(4,5,6)],
        lambda x: x.scatter(1, b, float("nan"), reduce="multiply"),
        lambda x: x.scatter(1, a, float("nan"), reduce="multiply"), forward_only=True)

  def test_scatter_no_reduce_tensor_src(self):
    with self.assertRaises(TypeError):
      Tensor.ones(4).scatter(dim=1, index=Tensor([0]), src=Tensor.ones(4), reduce="add")

  def test_scatter_reduce(self):
    b = torch.randint(3, size=[3,4,5], dtype=torch.int64, requires_grad=False)
    a = Tensor(b.detach().cpu().numpy().astype(np.int32), dtype=dtypes.int32, requires_grad=False)
    for reduce in ("sum", "prod", "mean", "amin", "amax"):
      for dim in (0,1,2,-1,-2,-3):
        helper_test_op([(4,5,6), (4,5,6)],
          lambda x,src: x.scatter_reduce(dim=dim, index=b, src=src, reduce=reduce),
          lambda x,src: x.scatter_reduce(dim=dim, index=a, src=src, reduce=reduce), forward_only=True)
        helper_test_op([(4,5,6), (4,5,6)],
          lambda x,src: x.scatter_reduce(dim=dim, index=b, src=src, reduce=reduce, include_self=False),
          lambda x,src: x.scatter_reduce(dim=dim, index=a, src=src, reduce=reduce, include_self=False), forward_only=True)

  def test_scatter_reduce_prod_zeros(self):
    b = torch.randint(3, size=[3,4,5], dtype=torch.int64, requires_grad=False)
    a = Tensor(b.detach().cpu().numpy().astype(np.int32), dtype=dtypes.int32, requires_grad=False)
    x = Tensor.zeros([4,5,6]).float()
    y = torch.zeros([4,5,6]).float()
    helper_test_op([(4,5,6)],
      lambda src: y.scatter_reduce(dim=1, index=b, src=src, reduce="prod"),
      lambda src: x.scatter_reduce(dim=1, index=a, src=src, reduce="prod"), forward_only=True)

  def test_scatter_reduce_errors(self):
    b = torch.randint(3, size=[3,4,5], dtype=torch.int64, requires_grad=False)
    a = Tensor(b.detach().cpu().numpy().astype(np.int32), dtype=dtypes.int32, requires_grad=False)
    # invalid reduce arg
    self.helper_test_exception([(4,5,6), (4,5,6)],
      lambda x,src: x.scatter_reduce(dim=0, index=b, src=src, reduce="INVALID"),
      lambda x,src: x.scatter_reduce(dim=0, index=a, src=src, reduce="INVALID"),
      RuntimeError)
    # dtype mismatch
    self.helper_test_exception([(4,5,6), (4,5,6)],
      lambda x,src: x.half().scatter_reduce(dim=0, index=b, src=src, reduce="sum"),
      lambda x,src: x.half().scatter_reduce(dim=0, index=a, src=src, reduce="sum"),
      RuntimeError)

  def test_scaled_dot_product_attention(self):
    helper_test_op([(32,8,16,64), (32,8,16,64), (32,8,16,64)], torch.nn.functional.scaled_dot_product_attention, Tensor.scaled_dot_product_attention)
    helper_test_op([(32,8,16,64), (32,8,16,64), (32,8,16,64), (32,8,16,16)],
                   lambda x,y,z,m: torch.nn.functional.scaled_dot_product_attention(x,y,z,attn_mask=m),
                   lambda x,y,z,m: Tensor.scaled_dot_product_attention(x,y,z,attn_mask=m))

  def test_scaled_dot_product_attention_mismatch_ls(self):
    helper_test_op([(32,8,4,64), (32,8,16,64), (32,8,16,64)], torch.nn.functional.scaled_dot_product_attention, Tensor.scaled_dot_product_attention)

  def test_scaled_dot_product_attention_causal(self):
    helper_test_op([(32,8,16,64), (32,8,16,64), (32,8,16,64)],
                   lambda x,y,z: torch.nn.functional.scaled_dot_product_attention(x,y,z,is_causal=True),
                   lambda x,y,z: Tensor.scaled_dot_product_attention(x,y,z,is_causal=True))

    self.helper_test_exception([(32,8,16,64), (32,8,16,64), (32,8,16,64), (32,8,16,16)],
      lambda x,y,z,m: torch.nn.functional.scaled_dot_product_attention(x,y,z,is_causal=True,attn_mask=m),
      lambda x,y,z,m: Tensor.scaled_dot_product_attention(x,y,z,is_causal=True,attn_mask=m),
      expected=RuntimeError)

  def test_binary_crossentropy(self):
    helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy(x.sigmoid(),torch.clip(y,0,1)),
                                       lambda x,y: x.sigmoid().binary_crossentropy(y.clip(0,1)))
    helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy_with_logits(x,torch.clip(y,0,1)),
                                       lambda x,y: x.binary_crossentropy_logits(y.clip(0,1)))
    helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy_with_logits(x,torch.clip(y,0,1)),
                                       lambda x,y: x.sigmoid().binary_crossentropy(y.clip(0,1)))
    helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy(x.sigmoid(),torch.clip(y,0,1)),
                                       lambda x,y: x.binary_crossentropy_logits(y.clip(0,1)))
  def test_binary_crossentropy_reductions(self):
    for r in ("mean", "sum", "none"):
      helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy(x.sigmoid(), torch.clip(y,0,1), reduction=r),
                                         lambda x,y: x.sigmoid().binary_crossentropy(y.clip(0,1), reduction=r))
      helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy_with_logits(x, torch.clip(y,0,1), reduction=r),
                                         lambda x,y: x.binary_crossentropy_logits(y.clip(0,1), reduction=r))
  def test_cross_entropy(self):
    helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.cross_entropy(x, y),
                                       lambda x,y: x.cross_entropy(y))
    helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.cross_entropy(x, torch.argmax(y, dim=1)),
                                       lambda x,y: x.cross_entropy(y.argmax(axis=1)), forward_only=True)
  def test_cross_entropy_reductions(self):
    for r in ("mean", "sum", "none"):
      helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.cross_entropy(x, y, reduction=r),
                                         lambda x,y: x.cross_entropy(y, reduction=r))
    self.helper_test_exception([(32,10), (32,10)], lambda x,y: torch.nn.functional.cross_entropy(x, y, reduction="typo"),
                                                   lambda x,y: x.cross_entropy(y, reduction="typo"), expected=ValueError)

  def test_cross_entropy_smoothing(self):
    for ls in (0., 0.3, 0.7, 1.):
      helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.cross_entropy(x, y, label_smoothing=ls),
                                         lambda x,y: x.cross_entropy(y, label_smoothing=ls))

  def test_nll_loss(self):
    helper_test_op([(32,10), (32)],
                   lambda x,y: torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x, dim=1), torch.clip(y,0).type(torch.long)),
                   lambda x,y: x.log_softmax(axis=1).nll_loss(y.clip(0).cast(dtypes.int32)), forward_only=True)

  def test_nll_loss_3d(self):
    helper_test_op([(32,10,3,3,3), (32,3,3,3)],
                   lambda x,y: torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x, dim=1), torch.clip(y,0).type(torch.long)),
                   lambda x,y: x.log_softmax(axis=1).nll_loss(y.clip(0).cast(dtypes.int32)), forward_only=True)

  def test_nll_loss_reductions(self):
    for r in ("mean", "sum", "none"):
      helper_test_op([(32,10), (32)],
        lambda x,y: torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x, dim=1), torch.clip(y,0).type(torch.long), reduction=r),
        lambda x,y: x.log_softmax(axis=1).nll_loss(y.clip(0).cast(dtypes.int32), reduction=r), forward_only=True)
    self.helper_test_exception([(32,10), (32)],
      lambda x,y: torch.nn.functional.nll_loss(x, torch.clip(y,0).type(torch.long), reduction="typo"),
      lambda x,y: x.nll_loss(y.clip(0).cast(dtypes.int32), reduction="typo"), expected=ValueError)

  def test_nll_loss_weight(self):
    for r in ("mean", "sum", "none"):
      helper_test_op([(32,10), (32), (10)],
        lambda x,y,z: torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x, dim=1), torch.clip(y,0).type(torch.long),
                                                   weight=z, reduction=r),
        lambda x,y,z: x.log_softmax(axis=1).nll_loss(y.clip(0).cast(dtypes.int32), weight=z, reduction=r), forward_only=True)

  def test_nll_loss_3d_weight(self):
    for r in ("mean", "sum", "none"):
      helper_test_op([(32,10,3,3,3), (32,3,3,3), (10)],
          lambda x,y,z: torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x, dim=1), torch.clip(y,0).type(torch.long),
                                                    weight=z, reduction=r),
          lambda x,y,z: x.log_softmax(axis=1).nll_loss(y.clip(0).cast(dtypes.int32), weight=z, reduction=r), forward_only=True)

  def test_nll_loss_ignore_index(self):
    logits = [[2.0, 0.5, -1.0],
              [1.5, 2.5, -0.5],
              [0.0, -2.0, 1.0]]
    targets = [0, 1, 2]
    helper_test_op(None, lambda x,y: torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x, dim=1),
                                                                  torch.clip(y,0).type(torch.long), ignore_index=1),
                         lambda x,y: x.log_softmax().nll_loss(y.clip(0), ignore_index=1),
                         forward_only=True, vals=[logits, targets])

  def test_one_hot(self):
    data = [1, 2, 4]
    helper_test_op([], lambda: torch.nn.functional.one_hot(torch.tensor(data), 6).type(torch.int32),
                       lambda: Tensor(data).one_hot(6), forward_only=True)
    helper_test_op([], lambda: torch.nn.functional.one_hot(torch.tensor(data)).type(torch.int32),
                       lambda: Tensor(data).one_hot(), forward_only=True)
    data = [[[1, 2, 3], [0, 3, 5]], [[1, 2, 3], [0, 3, 5]]]
    helper_test_op([], lambda: torch.nn.functional.one_hot(torch.tensor(data), 8).type(torch.int32),
                       lambda: Tensor(data).one_hot(8), forward_only=True)
    helper_test_op([], lambda: torch.nn.functional.one_hot(torch.tensor(data)).type(torch.int32),
                       lambda: Tensor(data).one_hot(), forward_only=True)

  def test_masked_fill(self):
    helper_test_op([(32,10)], lambda x: x.masked_fill((x>0.1).detach(), -math.inf))
    helper_test_op([(32,10)], lambda x: x.masked_fill((x<0.1).detach(), -math.inf))

  def test_masked_select(self):
    helper_test_op([(32, 10)], lambda x: x.masked_select(x>0.5), lambda x: x.masked_select(x>0.5), forward_only=True)
    helper_test_op([(32, 10)], lambda x: x.masked_select(torch.tensor(True)), lambda x: x.masked_select(Tensor(True)), forward_only=True)

  @unittest.skipIf(Device.DEFAULT == "QCOM", "OpenCL fails to compile this (both on GPU(qcom)/QCOM backends)")
  def test_cast(self):
    helper_test_op([(3, 3)], lambda x: x.float())
    helper_test_op(None, lambda x: x.float(), vals=[[0, 1, 2, 3]], forward_only=True)
    helper_test_op(None, lambda x: x.float(), vals=[[True, False]], forward_only=True)
    helper_test_op([(3, 3)], lambda x: x.int(), forward_only=True)
    helper_test_op([(3, 3)], lambda x: x.bool(), forward_only=True)

  def test_bitcast(self):
    helper_test_op([(3, 3)], lambda x: x.view(torch.int32), lambda x: x.bitcast(dtypes.int32), forward_only=True)

@unittest.skipUnless(is_dtype_supported(dtypes.uchar), f"no uint8 on {Device.DEFAULT}")
class TestOpsUint8(unittest.TestCase):
  def test_cast(self):
    helper_test_op([(2,3,64,64)], lambda x: x.type(torch.uint8), lambda x: x.cast('uint8'), forward_only=True)

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

@unittest.skipUnless(is_dtype_supported(dtypes.bfloat16), f"no bfloat16 on {Device.DEFAULT}")
class TestOpsBFloat16(unittest.TestCase):
  def test_cast(self):
    # TODO: helper_test_op breaks in unrelated part
    # TODO: wrong output with GPU=1 / PYTHON=1 on mac
    data = [60000.0, 70000.0, 80000.0]
    np.testing.assert_allclose(Tensor(data).cast("bfloat16").numpy(), torch.tensor(data).type(torch.bfloat16).float().numpy())

if __name__ == '__main__':
  np.random.seed(1337)
  unittest.main(verbosity=2)
