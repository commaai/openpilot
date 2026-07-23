import unittest, random
from tinygrad import Tensor, Device, nn, GlobalCounters, TinyJit, dtypes, Variable
from tinygrad.uop.ops import Ops, UOp
from tinygrad.helpers import getenv, prod, Context
from tinygrad.nn.state import get_parameters
from tinygrad.engine.realize import run_linear, compile_linear
import numpy as np
from hypothesis import given, strategies as strat, settings
from test.helpers import not_support_multi_device, needs_second_gpu, slow, call_is_graph

settings.register_profile("my_profile", max_examples=200, deadline=None, derandomize=getenv("DERANDOMIZE_CI", False))
settings.load_profile("my_profile")

d0 = f"{Device.DEFAULT}:0"
d1 = f"{Device.DEFAULT}:1"
d2 = f"{Device.DEFAULT}:2"
d3 = f"{Device.DEFAULT}:3"
d4 = f"{Device.DEFAULT}:4"
d5 = f"{Device.DEFAULT}:5"
devices_2 = (d1, d2)
devices_3 = (d1, d2, d3)
devices_4 = (d1, d2, d3, d4)
N = 128

# shard_x is "data parallel"
# shard_w is "model parallel"

def _test_allreduce(t:Tensor):
  aa = (t[0:64] + t[64:128] + t[128:192] + t[192:256]).repeat([4,1]).realize()
  ts = t.shard(devices_4, 0).realize()
  b = Tensor(UOp.allreduce(ts.uop, Ops.ADD, ts.device))
  b.realize()
  return aa, b

@unittest.skipIf(not_support_multi_device(), "no multi")
class TestMultiTensor(unittest.TestCase):
  @needs_second_gpu
  def setUp(self): pass

  def test_to(self):
    X = Tensor.ones(256).contiguous().realize()
    X.to_(devices_2)
    assert X.shape == (256,)
    (X + X).realize()

  def test_gradient(self):
    X = Tensor.ones(256).contiguous().realize()
    X.to_(devices_2)
    grad = X.sum().gradient(X)[0]
    grad.realize()

  def test_shard(self):
    X = Tensor.ones(256).contiguous().realize()
    X.shard_(devices_2, 0)
    for lb in X.uop.src:
      assert lb.shape == (128,)
    (X + X).realize()

  @unittest.expectedFailure # TODO: fix
  def test_shard_empty(self):
    GlobalCounters.reset()
    X = Tensor.empty(256).shard(devices_2, 0).realize()
    assert GlobalCounters.kernel_count == 0
    (X + X).realize()

  # TODO: fix this to not copy on the src device
  @unittest.expectedFailure
  def test_shard_no_recompile(self):
    X = Tensor.ones(256).contiguous().realize()
    X.shard_(devices_2, 0)
    out = (X + X)
    linear = compile_linear(out.schedule_linear())
    names = [call.src[0].src[0].arg.name for call in linear.src if call.src[0].op is Ops.PROGRAM]
    run_linear(linear)
    self.assertEqual(len(set(names)), 1, "function was relinearized")

  def test_shard_same_device(self):
    X = Tensor.ones(256).contiguous().realize()
    X.shard_((d1, X.device), 0)
    (X + X).realize()

  def test_numpy(self):
    X = Tensor.ones(256)
    X.shard_((d1, d2), 0)
    np.testing.assert_allclose(X.numpy(), 1)

  def test_four_add(self):
    X = Tensor.ones(256, 256).contiguous().realize()
    W = Tensor.ones(256, 256).contiguous().realize()
    X.shard_(devices_4, 1)
    W.shard_(devices_4, None)
    O = X + W
    np.testing.assert_allclose(O.numpy(), 2)

  def test_elementwise_dtype(self):
    Tensor.manual_seed(0)
    X = Tensor.randn(8, 8).realize()
    W = Tensor.randn(8, 8).realize()
    X.shard_(devices_4, 0)
    W.shard_(devices_4, 0)
    O = X.shrink(((0, 2), None)) * W.shrink(((0, 2), None)) < 2
    np.testing.assert_allclose(O.numpy(), X.numpy()[0:2]*W.numpy()[0:2] < 2)

  def test_shrink_on_shard_axis(self):
    X = Tensor.arange(4*4).reshape(4,4).clone().realize()
    X_np = X.numpy()
    X.shard_(devices_2, 0)
    # only shrink on the device that owns the shard, this is enabled by the mselect simplifier
    for i in range(2):
      xt = X[i*2:i*2+2].contiguous()
      linear, var_vals = xt.linear_with_vars()
      #kernels = [call for call in linear.src if call.src[0].op is Ops.SINK]
      #self.assertEqual(len(kernels), 1)
      #self.assertEqual(kernels[0].src[1].buffer.device, devices_2[i])
      run_linear(linear, var_vals)
      np.testing.assert_equal(xt.numpy(), X_np[i*2:i*2+2])

  @given(strat.sampled_from((devices_2, devices_3)),
         strat.sampled_from((Ops.ADD, Ops.MUL, Ops.MAX)),
         strat.sampled_from((None, 0, 1)), strat.sampled_from((None, 0, 1)))
  def test_simple_reduce(self, devices, rop, shard_axis, reduce_axis):
    N = 4 * len(devices)
    X = (Tensor.rand(N*N)-1).reshape(N, N).shard_(devices, shard_axis)
    n = X.numpy()
    f = {Ops.ADD: lambda x: x.sum(reduce_axis), Ops.MUL: lambda x: x.prod(reduce_axis), Ops.MAX: lambda x: x.max(reduce_axis)}[rop]
    fX = f(X)
    fn = f(n)
    np.testing.assert_allclose(fX.numpy(), fn, rtol=1e-6, atol=1e-6)

  def test_stack(self):
    X = Tensor.rand(4, 4).shard_(devices_2, 0)
    Y = Tensor.rand(4, 4).shard_(devices_2, 0)
    Z = Tensor.rand(4, 4).shard_(devices_2, 1)  # mismatched shard axis gets resharded
    for dim in (0, 1):
      np.testing.assert_allclose(Tensor.stack(X, Y, Z, dim=dim).numpy(), np.stack([X.numpy(), Y.numpy(), Z.numpy()], axis=dim))
    grad = Tensor.stack(X, Y).sum().gradient(X)[0]
    np.testing.assert_allclose(grad.numpy(), 1)

  def test_allreduce_naive(self):
    with Context(RING=0):
      a,b = _test_allreduce(Tensor.rand(256, 256))
      np.testing.assert_almost_equal(a.numpy(), b.numpy(), decimal=5)

  def test_allreduce_ring(self):
    with Context(RING=2):
      a,b = _test_allreduce(Tensor.rand(256, 256))
      np.testing.assert_almost_equal(a.numpy(), b.numpy(), decimal=5)

  def test_allreduce_all2all(self):
    with Context(ALL2ALL=2):
      a,b = _test_allreduce(Tensor.rand(256, 256))
      np.testing.assert_almost_equal(a.numpy(), b.numpy(), decimal=5)

  def test_copy_jit(self):
    @TinyJit
    def copy_tensor(x:Tensor): return (x.to(f"{x.device.split(':')[0]}:1") + 1)
    for _ in range(5):
      t = Tensor.rand(256).realize()
      x = copy_tensor(t)
      np.testing.assert_equal((t+1).numpy(), x.numpy())

  def test_allreduce_naive_jit(self):
    with Context(RING=0):
      jit_allreduce = TinyJit(_test_allreduce)
      for _ in range(5):
        a,b = jit_allreduce(Tensor.rand(256, 256))
        np.testing.assert_almost_equal(a.numpy(), b.numpy(), decimal=5)

  def test_allreduce_ring_jit(self):
    with Context(RING=2):
      jit_allreduce = TinyJit(_test_allreduce)
      for _ in range(5):
        a,b = jit_allreduce(Tensor.rand(256, 256))
        np.testing.assert_almost_equal(a.numpy(), b.numpy(), decimal=5)

  def test_multitensor_jit_input(self):
    @TinyJit
    def f(x): return (x+1).contiguous().sum()
    for _ in range(5):
      tt = Tensor.arange(0, 4).clone().realize().shard((d1,d2), 0).realize()
      out = f(tt)
      assert out.item() == 1+2+3+4

  def test_multitensor_inside_jit(self):
    @TinyJit
    def f(x): return (x.shard((d1,d2), 0)+1).contiguous().sum()
    for _ in range(5):
      tt = Tensor.arange(0, 4).clone().realize()
      out = f(tt)
      assert out.item() == 1+2+3+4

  def test_fuzz_allreduce(self):
    random.seed(41)
    for it in range(2):
      for n in range(2, 4+1):
        shape = tuple([(n if i == 0 else 1) * random.randint(1, 10) for i in range(random.randint(1, 4))])
        t = Tensor.rand(shape).shard_(tuple([d0, d1, d2, d3][:n]), 0)
        with Context(RING=0):
          a = Tensor(UOp.allreduce(t.uop, Ops.ADD, t.device))
        with Context(RING=2):
          b = Tensor(UOp.allreduce(t.uop, Ops.ADD, t.device))
        diff = a - b
        mean_err = diff.reshape((prod(diff.shape),)).abs().mean().numpy()
        max_err = diff.reshape((prod(diff.shape),)).abs().max().numpy()
        assert mean_err < 1e-6, f"big mean error, iteration {it}_{n}"
        assert max_err < 1e-6, f"big max error, iteration {it}_{n}"

  def _test_model_train_step(self, m, fake_image, labels):
    from tinygrad.nn.optim import LARS
    optimizer = LARS(get_parameters(m), 0.1)

    optimizer.zero_grad()
    m.load_from_pretrained()
    output = m(fake_image).sparse_categorical_crossentropy(labels, label_smoothing=0.1)
    output.backward()
    grad = m.conv1.weight.grad.numpy()

    fake_image_sharded = fake_image.shard(devices_2, axis=0)
    labels_sharded = labels.shard(devices_2, axis=0)
    for p in get_parameters(m): p.shard_(devices_2).realize()
    GlobalCounters.reset()
    optimizer.zero_grad()
    shard_output = m(fake_image_sharded).sparse_categorical_crossentropy(labels_sharded, label_smoothing=0.1)
    shard_output.backward()
    shard_grad = m.conv1.weight.grad.numpy()
    # sometimes there is zeros in these grads... why?
    np.testing.assert_allclose(grad, shard_grad, atol=1e-5, rtol=1e-5)

  @slow
  def test_data_parallel_resnet_train_step(self):
    from extra.models.resnet import ResNet18
    fake_image = Tensor.rand((2, 3, 224//16, 224//16))
    labels = Tensor.randint(2, low=0, high=1000)
    m = ResNet18()
    self._test_model_train_step(m, fake_image, labels)

  def test_data_parallel_simple_train_step(self):
    class Model:
      def __init__(self): self.conv1 = nn.Linear(128,128)
      def __call__(self, x): return self.conv1(x)
      def load_from_pretrained(self): pass

    fake_image = Tensor.rand((128,))
    labels = Tensor.randint(2, low=0, high=127)
    m = Model()
    self._test_model_train_step(m, fake_image, labels)

  def test_assign_kv_cache_multi(self):
    bsz, max_context = 2, 8

    class Attn:
      @TinyJit
      def __call__(self, xk:Tensor, start_pos:UOp):
        seqlen = xk.shape[1]
        if not hasattr(self, "cache_k"):
          self.cache_k = Tensor.zeros(bsz, max_context, 1, 1).shard(devices_2).contiguous().realize()
        keys = self.cache_k.shrink((None, (0, start_pos), None, None)).cat(xk, dim=1).contiguous() if start_pos > 0 else xk
        self.cache_k.assign(keys.pad((None,(0,max_context-start_pos-seqlen),None,None)).contiguous()).realize()

    attn = Attn()
    xk = Tensor.ones(bsz, 3, 1, 1).shard(devices_2).contiguous()
    attn(xk, 0)
    for i in range(3,6):
      # copied from LLaMA
      start_pos = Variable("start_pos", 1, max_context).bind(i)
      xk = Tensor.ones(bsz, 1, 1, 1).shard(devices_2).contiguous()
      attn(xk, start_pos)

    out = attn.cache_k.flatten().numpy()
    np.testing.assert_allclose(out, [1.,1.,1.,1.,1.,1.,0.,0.,1.,1.,1.,1.,1.,1.,0.,0.])

  def test_multi_tensor_jit_graph_assign_updates_each_shard(self):
    @TinyJit
    def jf(out: Tensor) -> Tensor:
      tmp = (Tensor.arange(4, dtype=dtypes.float).clone().shard(devices_2, 0) + 1).contiguous().realize()
      out.assign((tmp + 1).contiguous()).realize()
      return out

    out = Tensor.full((4,), -1.0).shard(devices_2, 0).contiguous().realize()
    expected = np.arange(4, dtype=np.float32) + 2
    for _ in range(5):
      out.assign(Tensor.full((4,), -1.0).shard(devices_2, 0).contiguous()).realize()
      jf(out)
      np.testing.assert_allclose(out.numpy(), expected, atol=1e-4, rtol=1e-5)
    assert jf.captured is not None

  @unittest.skip("test broken")
  def test_multi_device_jit_graph(self):
    if Device[d0].graph is None or Device[d1].graph is None: raise unittest.SkipTest("only test graphs")

    @TinyJit
    def jf(a: Tensor, b: Tensor, c: Tensor, d:Tensor):
      # Create 80 entries on device 0: 2 batches.
      for _ in range(40):
        a = ((a + b).realize() + (a * b).realize()).realize()
      # Create 80 entries on device 1: 2 batches.
      for _ in range(40):
        c = ((c + d).realize() + (c * d).realize()).realize()
      # Create a copy from device 0 to 1: 1 entry.
      a = a.to(d1).realize()
      # Creates one last entry on device 1: 1 batch.
      return (a + c).realize()

    a = Tensor.randn(10, 10, device=d0).realize()
    b = Tensor.randn(10, 10, device=d0).realize()
    c = Tensor.randn(10, 10, device=d1).realize()
    d = Tensor.randn(10, 10, device=d1).realize()

    ref = jf(a, b, c, d).numpy()
    for _ in range(5):
      o = jf(a, b, c, d).numpy()
      np.testing.assert_allclose(ref, o, atol=1e-4, rtol=1e-5)

    # Checking that 2 graphs per device, 1 copy and 1 last graph on device 1 are created.
    sis = jf.captured.linear.src
    assert len(sis) == 6
    for si in (sis[0], sis[1], sis[2], sis[3], sis[5]):
      assert call_is_graph(si)
    assert sis[4].src[0].op is Ops.COPY

  def test_rand_on_multiple_devices(self):
    # different devices generate different rand
    d0_rand = Tensor.rand(256, device=d0).realize()
    d1_rand = Tensor.rand(256, device=d1).realize()
    assert not np.allclose(d0_rand.numpy(), d1_rand.numpy())

  def test_rand_on_multiple_devices_manual_seed(self):
    Tensor.manual_seed(123)
    d0_rand = Tensor.rand(2, device=d0).tolist()
    d1_rand = Tensor.rand(2, device=d1).tolist()

    # manual_seed again gives the same values
    Tensor.manual_seed(123)
    d0_rand2 = Tensor.rand(2, device=d0).tolist()
    d1_rand2 = Tensor.rand(2, device=d1).tolist()
    self.assertEqual(d0_rand, d0_rand2)
    self.assertEqual(d1_rand, d1_rand2)

    # device seed is only determined by init order, so flipping init order flips rands
    Tensor.manual_seed(123)
    d1_rand_flip = Tensor.rand(2, device=d1).tolist()
    d0_rand_flip = Tensor.rand(2, device=d0).tolist()
    self.assertEqual(d0_rand, d1_rand_flip)
    self.assertEqual(d1_rand, d0_rand_flip)

  def test_const_like_shrink_on_shard_axis(self):
    t = Tensor.ones(16, 16, dtype=dtypes.int).shard(devices_2, axis=0)
    out = t.const_like(2)[:, :8]
    linear, var_vals = out.linear_with_vars()
    self.assertEqual(len(linear.src), 0)
    run_linear(linear, var_vals)
    self.assertEqual(out.tolist(), [[2]*8]*16)

@unittest.skipIf(not_support_multi_device(), "no multi")
class TestHandleData(unittest.TestCase):
  @needs_second_gpu
  def test_copied_to_device(self):
    device = (d0, d1, d2, d3)
    t = Tensor([1, 2, 3, 4]).shard(device).realize()
    not_covered = t.to(d5)
    sched = not_covered.schedule_linear().src
    assert len(sched) == 1
    # setup again because create_schedule has side effect
    t = Tensor([1, 2, 3, 4]).shard(device).realize()
    not_covered = t.to(d5)
    assert not_covered.realize().tolist() == [1, 2, 3, 4]

    for d in device:
      t = Tensor([1, 2, 3, 4]).shard(device).realize()
      covered = t.to(d)
      sched = covered.schedule_linear().src
      # TODO: this isn't optimized out anymore
      #assert len(sched) == 0
      # setup again because create_schedule has side effect
      t = Tensor([1, 2, 3, 4]).shard(device).realize()
      covered = t.to(d)
      assert covered.realize().tolist() == [1, 2, 3, 4]

@unittest.skipIf(not_support_multi_device(), "need multi")
class TestMultiBufferView(unittest.TestCase):
  @needs_second_gpu
  def setUp(self): pass

  def _check(self, a_ref:Tensor, a_multi:Tensor, view_fn):
    b_ref = view_fn(a_ref)
    b_multi = view_fn(a_multi).contiguous()
    linear, var_vals = b_multi.linear_with_vars()
    if all(not d.startswith(("WEBGPU", "CL")) for d in b_multi.device):
      compiled = [call for call in linear.src if call.src[0].op is Ops.SINK]
      self.assertEqual(len(compiled), 0, f"expected zero compiled kernels, got {len(compiled)}")
    run_linear(linear, var_vals)
    np.testing.assert_equal(b_multi.numpy(), b_ref.numpy())

  @unittest.skip("flaky on LLVM")
  def test_shrink_non_shard_axis(self):
    ref = Tensor.arange(8*4*10).reshape(8, 4, 10).clone().realize()
    a = Tensor.arange(8*4*10).reshape(8, 4, 10).clone().shard(devices_2, axis=1).realize()
    self._check(ref, a, lambda t: t[3])

  def test_shrink_2d(self):
    ref = Tensor.arange(6*4).reshape(6, 4).clone().realize()
    a = Tensor.arange(6*4).reshape(6, 4).clone().shard(devices_2, axis=1).realize()
    self._check(ref, a, lambda t: t.shrink(((1, 4), None)))

  def test_reshape_then_shrink(self):
    ref = Tensor.arange(8*6).reshape(8, 6).clone().realize()
    a = Tensor.arange(8*6).reshape(8, 6).clone().shard(devices_2, axis=1).realize()
    self._check(ref, a, lambda t: t.reshape(4, 2, 6)[1])

  def test_chained_shrink(self):
    ref = Tensor.arange(10*8).reshape(10, 8).clone().realize()
    a = Tensor.arange(10*8).reshape(10, 8).clone().shard(devices_2, axis=1).realize()
    self._check(ref, a, lambda t: t.shrink(((2, 8), None)).shrink(((1, 4), None)))

  def test_4_devices(self):
    ref = Tensor.arange(8*12).reshape(8, 12).clone().realize()
    a = Tensor.arange(8*12).reshape(8, 12).clone().shard(devices_4, axis=1).realize()
    out = a[5].contiguous()
    linear, var_vals = out.linear_with_vars()
    if all(not d.startswith(("WEBGPU", "CL")) for d in out.device):
      compiled = [call for call in linear.src if call.src[0].op is Ops.SINK]
      self.assertEqual(len(compiled), 0)
    run_linear(linear, var_vals)
    np.testing.assert_equal(out.numpy(), ref[5].numpy())

@unittest.skipIf(not_support_multi_device(), "need multi")
class TestMultiTransformer(unittest.TestCase):
  @needs_second_gpu
  def test_transformer(self):
    device = tuple(f"{Device.DEFAULT}:{i}" for i in range(2))

    from extra.models.llama import Transformer
    args = {"dim": 32, "n_heads": 1, "n_kv_heads": 1, "n_layers": 2, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 1024,
            "hidden_dim": 32, "max_context": 12}
    real_model = Transformer(**args)
    shard_model = Transformer(**args)

    # copy state
    nn.state.load_state_dict(shard_model, nn.state.get_state_dict(real_model))

    # shard
    for k,v in nn.state.get_state_dict(shard_model).items():
      if 'scale' in k: v.shard_(device, axis=None)  # from quantized
      elif '.attention.' in k: v.shard_(device, axis=-1)
      elif '.feed_forward.w1.' in k: v.shard_(device, axis=0)
      elif '.feed_forward.w3.' in k: v.shard_(device, axis=0)
      elif '.feed_forward.' in k: v.shard_(device, axis=-1)
      elif 'tok_embeddings.weight' in k: v.shard_(device, axis=0)
      elif 'output.weight' in k: v.shard_(device, axis=0)
      else: v.shard_(device, axis=None)

    last_tok = 0
    for i in range(10):
      real_tok = real_model(Tensor([[last_tok]], device=Device.DEFAULT), i).item()
      shard_tok = shard_model(Tensor([[last_tok]], device=device), i).item()

      # test kv cache
      kv1 = real_model.layers[0].attention.cache_kv.numpy()
      kv2 = shard_model.layers[0].attention.cache_kv.numpy()
      #print(np.concatenate([kv1[:, :, :, :, 0:1], kv2[:, :, :, :, 0:1]], axis=4))
      np.testing.assert_allclose(kv1, kv2, atol=1e-5, rtol=1e-5, err_msg=f"issue at token {i}")

      # test token
      self.assertEqual(real_tok, shard_tok, f"issue at token {i}")
      last_tok = real_tok

  @unittest.skip("super slow")
  def test_llama1b_full(self):
    from tinygrad.helpers import fetch
    fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model", "tokenizer.model", subdir="llama3-1b-instruct")
    model = fetch("https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf",
                  "Llama-3.2-1B-Instruct-Q6_K.gguf", subdir="llama3-1b-instruct")

    device = tuple(f"{Device.DEFAULT}:{i}" for i in range(2))
    from examples.llama3 import build_transformer
    real_model = build_transformer(model, model_size="1B", device=Device.DEFAULT)
    shard_model = build_transformer(model, model_size="1B", device=device)

    last_tok = 0
    real_tok = real_model(Tensor([[last_tok]], device=Device.DEFAULT), 0)
    shard_tok = shard_model(Tensor([[last_tok]], device=device), 0)
    self.assertEqual(real_tok.item(), shard_tok.item())

if __name__ == '__main__':
  unittest.main()
