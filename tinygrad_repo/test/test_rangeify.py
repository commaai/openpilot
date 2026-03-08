import unittest
from tinygrad import Tensor, nn, Device
from tinygrad.helpers import Context, GlobalCounters, getenv, PCONTIG, DEBUG
from tinygrad.uop.ops import graph_rewrite, PatternMatcher, UPat, Ops
from tinygrad.codegen.opt import OptOps, Opt
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.renderer.nir import NIRRenderer

@unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, (NIRRenderer, PTXRenderer)), "broken in LVP and PTX")
class TestDoubleMatmul(unittest.TestCase):
  def setUp(self):
    with Context(DEBUG=0):
      self.a, self.b, self.c = [Tensor.randn(16, 16).contiguous().realize() for _ in range(3)]
      self.ref = (self.a @ self.b @ self.c).realize()

  def _test(self, opts):
    with Context(PCONTIG=2, DEBUG=max(2, DEBUG.value)):
      out = (self.a @ self.b @ self.c).contiguous(arg=opts).realize()

    with Context(DEBUG=0):
      err = (out-self.ref).square()
      self.assertLess(err.max().item(), 1e-4)
      self.assertLess(err.mean().item(), 1e-6)

  def test_baseline(self): self._test(())
  def test_upcast_0(self): self._test((Opt(OptOps.UPCAST, 0, 4),))
  def test_upcast_1(self): self._test((Opt(OptOps.UPCAST, 1, 4),))
  def test_upcast_2(self): self._test((Opt(OptOps.UPCAST, 2, 4),))
  def test_upcast_01(self): self._test((Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4)))
  def test_upcast_01_mismatch(self): self._test((Opt(OptOps.UPCAST, 0, 2), Opt(OptOps.UPCAST, 1, 4)))
  def test_upcast_02(self): self._test((Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 2, 4)))
  def test_upcast_12(self): self._test((Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UPCAST, 2, 4)))

  def test_unroll_0(self): self._test((Opt(OptOps.UNROLL, 0, 4),))
  def test_unroll_1(self): self._test((Opt(OptOps.UNROLL, 1, 4),))
  def test_unroll_01(self): self._test((Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UNROLL, 1, 4)))

  def test_upcast_0_unroll_0(self): self._test((Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UNROLL, 0, 4)))
  def test_upcast_1_unroll_0(self): self._test((Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UNROLL, 0, 4)))
  def test_upcast_2_unroll_0(self): self._test((Opt(OptOps.UPCAST, 2, 4), Opt(OptOps.UNROLL, 0, 4)))

  def test_upcast_0_unroll_1(self): self._test((Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UNROLL, 1, 4)))
  def test_upcast_1_unroll_1(self): self._test((Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UNROLL, 1, 4)))
  def test_upcast_2_unroll_1(self): self._test((Opt(OptOps.UPCAST, 2, 4), Opt(OptOps.UNROLL, 1, 4)))

  def test_upcast_1_unroll_1_small(self): self._test((Opt(OptOps.UPCAST, 1, 2), Opt(OptOps.UNROLL, 1, 2)))
  def test_upcast_1_unroll_1_rev(self): self._test((Opt(OptOps.UNROLL, 1, 2), Opt(OptOps.UPCAST, 1, 2)))

  def test_upcast_01_unroll_01(self):
    self._test((Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UNROLL, 1, 4)))
  def test_upcast_12_unroll_01(self):
    self._test((Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UPCAST, 2, 4), Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UNROLL, 1, 4)))

class TestRangeifyAssign(unittest.TestCase):
  def test_assign_permuted(self):
    A = Tensor.empty(4, 4, dtype='int')
    B = Tensor.arange(16).reshape(4,4)
    ret = A.permute(1,0).assign(B)
    lst = ret.tolist()
    lst2 = A.tolist()
    lst3 = B.tolist()
    print(lst)
    print(lst2)
    print(lst3)
    self.assertListEqual(lst, lst3)
    self.assertListEqual(lst2, B.permute(1, 0).tolist())

class TestRangeifyEdgeCase(unittest.TestCase):
  def test_matmul_relu_cat(self):
    a = Tensor.ones(100, 512).contiguous().realize()
    c = Tensor.ones(1, 512).contiguous().realize()
    cm = Tensor.ones(512, 512)
    c = c @ cm
    c = c.relu()

    res = Tensor.cat(a, c, dim=0)
    self.assertEqual(res.numpy()[-1, :16].tolist(), [512] * 16)

  def test_pcontig_multi_gather(self):
    # regression test: local bufferize must have device set for const_like to work
    with Context(PCONTIG=2):
      # NOTE: with uint type, this will become a long and fail on WEBGPU
      forest = Tensor(list(range(8)), dtype='int')
      idx = Tensor([0, 0], dtype='int')
      node_val = forest.gather(0, idx)
      idx2 = idx * 2 + 1
      node_val2 = forest.gather(0, idx2)
      result = (node_val + node_val2).numpy()
    self.assertEqual(result.tolist(), [1, 1])

if getenv("BIG") > 2:
  # llama 8B (8192)
  BS, HEADS, SEQLEN, EMB = 4, 32, 8192, 128
elif getenv("BIG") > 1:
  # llama 8B
  BS, HEADS, SEQLEN, EMB = 4, 32, 2048, 128
elif getenv("BIG") > 0:
  # bigger
  BS, HEADS, SEQLEN, EMB = 4, 32, 128, 128
else:
  BS, HEADS, SEQLEN, EMB = 4, 2, 16, 8

def fa():
  Tensor.manual_seed(1337)
  with Context(DEBUG=0): q,k,v = [Tensor.rand(BS, HEADS, SEQLEN, EMB).contiguous().realize() for _ in range(3)]
  GlobalCounters.reset()
  return q.scaled_dot_product_attention(k, v)

def fa_bw():
  Tensor.manual_seed(1337)
  with Context(DEBUG=0):
    q,k,v = [Tensor.rand(BS, HEADS, SEQLEN, EMB).contiguous().realize().requires_grad_() for _ in range(3)]
    attn_output = nn.Linear(HEADS*EMB, HEADS*EMB, bias=False)
    attn_output.weight.requires_grad_().realize()
    target = Tensor.rand(BS, SEQLEN, HEADS*EMB).contiguous().realize()

  GlobalCounters.reset()
  attn = q.scaled_dot_product_attention(k, v).contiguous().contiguous_backward()
  attn = attn.transpose(1, 2).reshape(BS, SEQLEN, -1)
  out = attn_output(attn)
  loss = (out - target).square().mean()
  loss.backward()
  #ret = [out, Tensor.stack(q.grad, k.grad, v.grad, dim=-1)]
  #ret = [out, Tensor.stack(q.grad, k.grad, dim=-1), v.grad]
  ret = [out, q.grad, k.grad, v.grad]
  Tensor.realize(*ret)
  return ret

@unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, (NIRRenderer, PTXRenderer)), "broken in LVP and PTX")
class TestPcontig(unittest.TestCase):
  def test_flash_attention_bw(self):
    with Context(PCONTIG=max(2, PCONTIG.value), DEBUG=2):
      grads = fa_bw()
      print(f"{GlobalCounters.global_ops/1e9:.2f} GFLOPS")

    with Context(PCONTIG=0, DEBUG=2):
      cmp_grads = fa_bw()
      print(f"{GlobalCounters.global_ops/1e9:.2f} GFLOPS")

    with Context(DEBUG=0):
      mses = [((x-y)**2).sum().item() for x,y in zip(grads, cmp_grads)]
    mse = sum(mses)
    print(f"mse: {mse}")
    self.assertLessEqual(mse, 1e-6)

  def test_flash_attention(self, opts=None):
    with Context(PCONTIG=2, DEBUG=max(2, DEBUG.value)):
      ret = fa().realize() if opts is None else fa().contiguous(arg=opts).realize()
      print(f"{GlobalCounters.global_ops/1e9:.2f} GFLOPS")
    with Context(DEBUG=2):
      cmp = fa().realize()
      print(f"{GlobalCounters.global_ops/1e9:.2f} GFLOPS")
    with Context(DEBUG=0):
      mse = ((cmp-ret)**2).sum().item()
    print(f"mse: {mse}")
    self.assertLessEqual(mse, 1e-6)

  def test_flash_attention_opt(self):
    opts = ()
    # columns in top matrix
    opts += (Opt(OptOps.UPCAST, 0, 4),)
    # columns in bottom matrix
    opts += (Opt(OptOps.UPCAST, 3, 4),)
    # rows in all the matrix
    opts += (Opt(OptOps.UPCAST, 4, 4),)
    self.test_flash_attention(opts)

# contiguous + reduce can support ranges?

@unittest.skip("pm_rangeify no longer exists. test this in a different way")
class TestRangeifyPM(unittest.TestCase):
  def setUp(self): self.base = Tensor.empty(10*10).reshape(10, 10).contiguous()
  def assert_same(self, a, b):
    def run_pm_rangeify(t:Tensor):
      from tinygrad.schedule.rangeify import pm_rangeify, RangeifyContext
      sink = t.uop.sink()
      pm_realize = PatternMatcher([(UPat(Ops.CONTIGUOUS, name="x"), lambda x: x.replace(op=Ops.REALIZE))])
      sink = graph_rewrite(sink, pm_realize)
      return graph_rewrite(sink, pm_rangeify, ctx=RangeifyContext())
    self.assertIs(run_pm_rangeify(a.contiguous()), run_pm_rangeify(b.contiguous()))

  def test_nothing_match(self):
    a = self.base.pad(((0,0),(0,1)))
    b = self.base.pad(((0,0),(0,1)))
    self.assert_same(a, b)

  def test_reshape_match(self):
    a = self.base
    b = self.base.reshape(100).reshape(10, 10)
    self.assert_same(a, b)

  def test_permute_reshape_match(self):
    a = self.base
    b = self.base.permute(1,0).reshape(100).reshape(10, 10).permute(1,0)
    self.assert_same(a, b)

  def test_padded_permute_match(self):
    a = self.base.pad(((0,0),(0,1)))
    b = self.base.permute(1,0).pad(((0,1),(0,0))).permute(1,0)
    self.assert_same(a, b)

  @unittest.expectedFailure
  def test_padded_reshape_match(self):
    a = self.base.pad(((0,0),(0,1)))
    b = self.base.reshape(100).reshape(10, 10).pad(((0,0),(0,1)))
    self.assert_same(a, b)

  @unittest.expectedFailure
  def test_padded_permute_reshape_match(self):
    a = self.base.pad(((0,0),(0,1)))
    b = self.base.permute(1,0).reshape(100).reshape(10, 10).pad(((0,1),(0,0))).permute(1,0)
    self.assert_same(a, b)

  # why is this failing?
  @unittest.expectedFailure
  def test_cross_pad_match(self):
    a = self.base.pad(((0,0),(0,1))).pad(((0,1),(0,0)))
    b = self.base.pad(((0,1),(0,0))).pad(((0,0),(0,1)))
    self.assert_same(a, b)

if __name__ == '__main__':
  unittest.main()
