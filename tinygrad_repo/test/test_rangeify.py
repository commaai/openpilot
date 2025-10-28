import unittest
from tinygrad import Tensor, nn
from tinygrad.helpers import RANGEIFY, Context, GlobalCounters
from tinygrad.uop.ops import UOp

@unittest.skipIf(RANGEIFY<1, "tests only for RANGEIFY")
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

N = 256

@unittest.skipIf(RANGEIFY<1, "tests only for RANGEIFY")
class TestRangeify(unittest.TestCase):
  def test_expand_children(self):
    A = Tensor.empty(N, N).sum(axis=1)
    ba = A.expand(N, N)
    ((ba+1).sum(axis=1) + (ba+2).sum(axis=0)).realize()

  def test_partial_contig(self):
    A = Tensor.empty(64, 64, 64)
    ret = A.sum(axis=2).contiguous(arg=(1,)).sum(axis=1)
    ret.realize()

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

  def test_conv2d_t(self):
    x = Tensor.empty(1, 4, 32, 32)
    w1 = Tensor.empty(8, 4, 3, 3)
    (x*2).conv2d(w1).realize()

  def test_double_conv2d(self):
    x = Tensor.empty(1, 4, 32, 32)
    w1 = Tensor.empty(8, 4, 3, 3)
    w2 = Tensor.empty(12, 8, 3, 3)
    x.conv2d(w1).conv2d(w2).realize()

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

  def test_flash_attention(self):
    BS, HEADS, SEQLEN, EMB = 4, 2, 16, 8

    # bigger
    #BS, HEADS, SEQLEN, EMB = 4, 32, 1024, 64

    # llama 8B
    #BS, HEADS, SEQLEN, EMB = 4, 32, 2048, 128

    def fa():
      Tensor.manual_seed(1337)
      with Context(DEBUG=0): q,k,v = [Tensor.rand(BS, HEADS, SEQLEN, EMB).contiguous().realize() for _ in range(3)]
      return q.scaled_dot_product_attention(k, v).realize()

    with Context(DEBUG=4):
      GlobalCounters.reset()
      ret = fa()
    with Context(RANGEIFY=0):
      with Context(DEBUG=2):
        GlobalCounters.reset()
        cmp = fa()
      with Context(DEBUG=0):
        mse = ((cmp-ret)**2).sum().item()
    print(f"mse: {mse}")
    self.assertLessEqual(mse, 1e-6)

# contiguous + reduce can support ranges?

@unittest.skip("okay to disable this for now")
@unittest.skipIf(RANGEIFY<1, "tests only for RANGEIFY")
class TestOuterworld(unittest.TestCase):
  def test_passthrough_range(self):
    t = Tensor.rand(10, 10).realize()

    # passthrough ranges
    a = UOp.range(10, -1)
    sel = t[a]
    cpy = sel.contiguous(a).realize()

    self.assertTrue((t==cpy).all().item())

  def test_flip_range(self):
    t = Tensor.rand(10, 10).realize()

    # passthrough ranges
    a = UOp.range(10, -1)
    sel = t[9-a]
    cpy = sel.contiguous(a).realize()

    self.assertTrue((t.flip(0)==cpy).all().item())

  def test_vmap(self):
    def f(x): return x.sum(axis=0)*2

    x = Tensor.ones(3, 10, 2).contiguous()

    # vmap across axis 0
    a = UOp.range(3, -1)
    out = f(x[a])
    out = out.contiguous(a)

    # 3x2 grid of 20
    out.realize()
    print(out.numpy())

  @unittest.skip("opts don't work")
  def test_triple_gemm(self):
    x = Tensor.rand(1, 16).realize()
    W = Tensor.rand(3, 16, 16).realize()

    manual = (x @ W[0] @ W[1] @ W[2]).contiguous().realize()

    a = UOp.range(3, -1)
    x = x.assign(x @ W[a])
    out = x.contiguous(a)[-1].contiguous().realize()

    self.assertTrue((manual==out).all().item())

  def test_setitem_pyrange(self):
    with Context(DEBUG=0):
      t = Tensor.rand(10).realize()
      o = Tensor.empty(10)
    GlobalCounters.reset()
    for i in range(10):
      o[i] = t[i]
    o.realize()
    self.assertTrue((t==o).all().item())

  @unittest.skip("TODO: fix this")
  def test_setitem(self):
    with Context(DEBUG=0):
      t = Tensor.rand(10).realize()
      o = Tensor.empty(10)
    GlobalCounters.reset()
    i = UOp.range(10, -1)
    o[i] = t[i]
    o.contiguous(i).realize()
    self.assertTrue((t==o).all().item())

if __name__ == '__main__':
  unittest.main()
