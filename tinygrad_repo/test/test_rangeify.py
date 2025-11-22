import unittest
from tinygrad import Tensor, nn, Device
from tinygrad.helpers import Context, GlobalCounters, CI, getenv, PCONTIG, DEBUG
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

# *** non CI rangeify tests below this line ***

N = 256

@unittest.skipIf(CI, "useless in CI, doesn't test anything")
class TestRangeifyOpt(unittest.TestCase):
  def test_randperm(self):
    Tensor.randperm(10000).realize()

  def test_one_getitem(self):
    X = Tensor.empty(10000)
    sel = Tensor.arange(1000).contiguous().realize()
    Xsel = X[sel]
    Tensor.realize(Xsel)

  def test_two_getitem(self):
    # this is splitting on the child even when it really shouldn't
    X = Tensor.empty(10000)
    Y = Tensor.empty(10000)
    sel = Tensor.arange(1000).contiguous().realize()
    Xsel, Ysel = X[sel], Y[sel]
    Tensor.realize(Xsel, Ysel)

  def test_resnetconv(self):
    conv1 = nn.Conv2d(3, 8, kernel_size=7, stride=2, bias=False, padding=3)
    conv1.weight.replace(conv1.weight.empty_like())
    x = Tensor.empty(1, 3, 56, 56)
    x = conv1(x).pad([1,1,1,1])+1
    x.realize()

  # CPU=1 NOOPT=1 DEBUG=4 RANGEIFY=1 python3 test/test_rangeify.py TestRangeifyOpt.test_matmul_reshaped
  def test_matmul_reshaped(self):
    A = Tensor.empty(N, N)
    B = Tensor.empty(N, N)
    (A@B).reshape(N*N).contiguous().realize()

  def test_reduce_reshapes(self):
    A = Tensor.empty(8,8,8,8).permute(1,0,3,2).flatten()
    A.sum().realize()

@unittest.skipIf(CI, "useless in CI, doesn't test anything")
class TestRangeify(unittest.TestCase):
  def test_groupnorm(self):
    # ranges 1 and 3 are merging
    x = nn.GroupNorm(32, 128)
    x(Tensor.empty(1, 128, 64, 64)).realize()

  def test_expand_children(self):
    A = Tensor.empty(N, N).sum(axis=1)
    ba = A.expand(N, N)
    ((ba+1).sum(axis=1) + (ba+2).sum(axis=0)).realize()

  def test_partial_contig(self):
    A = Tensor.empty(64, 64, 64)
    ret = A.sum(axis=2).contiguous(arg=(1,)).sum(axis=1)
    ret.realize()

  @unittest.skip("RANGEIFY=0 does nothing")
  def test_double_gemm_real(self):
    def go():
      with Context(DEBUG=0):
        Tensor.manual_seed(1337)
        A,B,C = [Tensor.randn(N, N) for _ in range(3)]
        Tensor.realize(A, B, C)
      GlobalCounters.reset()
      return (A@B@C).realize()
    rng = go()
    with Context(RANGEIFY=0, DEBUG=2):
      ref = go()
      mse = ((rng-ref)**2).sum().item()
    print(f"mse: {mse}")
    self.assertLessEqual(mse, 1e-2)

  def test_double_gemm(self):
    A = Tensor.empty(N, N)
    B = Tensor.empty(N, N)
    C = Tensor.empty(N, N)
    (A@B@C).realize()

  def test_double_gemm_exp(self):
    A = Tensor.empty(N, N)
    B = Tensor.empty(N, N)
    C = Tensor.empty(N, N)
    (((A@B).exp()@C).exp()).realize()

  def test_double_gemm_exp_child(self):
    A = Tensor.empty(N, N)
    B = Tensor.empty(N, N)
    C = Tensor.empty(N, N)
    # A@B is used with exp, and also on the sum. this is two kernels now, is this right?
    ret = A@B
    ((ret.exp()@C)+ret).realize()

  def test_double_gemm_relu(self):
    A = Tensor.empty(N, N)
    B = Tensor.empty(N, N)
    C = Tensor.empty(N, N)
    (((A@B).relu()@C).relu()).realize()

  def test_double_gemm_relu_half_contig(self):
    A = Tensor.empty(N, N)
    B = Tensor.empty(N, N)
    C = Tensor.empty(N, N)
    (((A@B).relu().contiguous(arg=(1,))@C).relu()).realize()

  def test_double_gemm_half_contig(self):
    A = Tensor.empty(N, N)
    B = Tensor.empty(N, N)
    C = Tensor.empty(N, N)
    ((A@B).contiguous(arg=(1,))@C).realize()

  def test_double_gemm_contig(self):
    A = Tensor.empty(N, N)
    B = Tensor.empty(N, N)
    C = Tensor.empty(N, N)
    ((A@B).contiguous()@C).realize()

  def test_many_gemm(self):
    A = Tensor.empty(N, N)
    B = Tensor.empty(N, N)
    C = Tensor.empty(N, N)
    D = Tensor.empty(N, N)
    E = Tensor.empty(N, N)
    F = Tensor.empty(N, N)
    (A@B@C@D@E@F).realize()

  def test_conv2d(self):
    x = Tensor.empty(1, 4, 32, 32)
    w1 = Tensor.empty(8, 4, 3, 3)
    x.conv2d(w1).realize()

  def test_conv2d_elu(self):
    x = Tensor.empty(1, 4, 32, 32)
    w1 = Tensor.empty(8, 4, 3, 3)
    x.conv2d(w1).elu().realize()

  def test_conv2d_t(self):
    x = Tensor.empty(1, 4, 32, 32)
    w1 = Tensor.empty(8, 4, 3, 3)
    (x*2).conv2d(w1).realize()

  def test_double_conv2d(self):
    x = Tensor.empty(1, 4, 32, 32)
    w1 = Tensor.empty(8, 4, 3, 3)
    w2 = Tensor.empty(12, 8, 3, 3)
    x.conv2d(w1).conv2d(w2).realize()

  def test_resnet_conv2d(self):
    x = Tensor.empty(1, 8, 32, 32)
    w1 = Tensor.empty(8, 8, 3, 3)
    w2 = Tensor.empty(8, 8, 1, 1)
    x.conv2d(w1).conv2d(w2).realize()

  def test_xception_conv2d(self):
    # NOTE: this fusion is bad, it's recomputing the inner many times
    x = Tensor.empty(1, 4, 32, 32)
    w1 = Tensor.empty(8, 4, 1, 1)
    w2 = Tensor.empty(8, 1, 3, 3)
    x.conv2d(w1).conv2d(w2, groups=8).realize()

  def test_conv_maxpool_contig(self): self.test_conv_maxpool(True)
  def test_conv_maxpool(self, contig=False):
    GlobalCounters.reset()
    x = Tensor.empty(32, 16, 64, 64)
    l1 = nn.Conv2d(16, 16, 3)
    for p in nn.state.get_parameters(l1): p.replace(Tensor.empty(p.shape))
    x = l1(x)
    if contig: x = x.contiguous()
    x.max_pool2d().realize()

  def test_double_conv2d_half_contig(self):
    x = Tensor.empty(1, 4, 32, 32)
    w1 = Tensor.empty(8, 4, 3, 3)
    w2 = Tensor.empty(12, 8, 3, 3)
    # NOTE: this contiguous doesn't help
    x.conv2d(w1).contiguous(arg=(1,)).conv2d(w2).permute(0,2,3,1).contiguous().realize()

  def test_double_conv2d_contig(self):
    x = Tensor.empty(1, 4, 32, 32)
    w1 = Tensor.empty(8, 4, 3, 3)
    w2 = Tensor.empty(12, 8, 3, 3)
    x.conv2d(w1).contiguous().conv2d(w2).realize()

  def test_transformer_ffn(self):
    from tinygrad.apps.llm import TransformerBlock
    from tinygrad import nn
    blk = TransformerBlock(1024, 4096, 1, 1, 1e-5)
    for p in nn.state.get_parameters(blk): p.replace(Tensor.empty(p.shape))

    x = Tensor.empty(128, 1024)
    out = blk._feed_forward(x)
    out.realize()

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
