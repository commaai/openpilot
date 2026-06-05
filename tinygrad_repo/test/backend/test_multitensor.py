import unittest, random
from tinygrad import Tensor, Device, nn, GlobalCounters, TinyJit, dtypes, Variable
from tinygrad.uop.ops import Ops, UOp
from tinygrad.helpers import getenv, prod, Context
from tinygrad.nn.state import get_parameters, get_state_dict
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

  def test_arange_shrink(self):
    x = Tensor.arange(4)
    self.assertEqual(x.shard(devices_2, 0).realize().shrink(((2, 4),)).tolist(), [2, 3])
    self.assertEqual(x.shard(devices_2, 0).realize().shrink(((0, 2),)).tolist(), [0, 1])

  def test_shard_like(self):
    X = Tensor.ones(256).shard(devices_2, 0)
    Y = Tensor.zeros(256).shard_like(X)
    self.assertEqual(Y.device, X.device)
    self.assertEqual(Y.uop.axis, 0)
    # also test with axis=None
    X2 = Tensor.ones(256).shard(devices_2, axis=None)
    Y2 = Tensor.zeros(256).shard_like(X2)
    self.assertEqual(Y2.device, X2.device)
    self.assertEqual(Y2.uop.axis, None)
    # test with single device
    X3 = Tensor.ones(256)
    Y3 = Tensor.zeros(256).shard_like(X3)
    self.assertEqual(Y3.device, X3.device)
    # cannot shard_like multi unless it's a no-op
    X4 = Tensor.ones(256).shard(devices_2, 0)
    Y4 = Tensor.ones(256).shard(devices_2, 0).shard_like(X4)
    self.assertEqual(Y4.device, X4.device)
    self.assertEqual(Y4.uop.axis, 0)
    with self.assertRaises(RuntimeError):
      Tensor.ones(256).shard(devices_2, None).shard_like(X4)

  def _test_shard_op(self, op, out, n=4):
    t = Tensor.ones(n).contiguous().realize().shard(devices_2, 0)
    r = op(t).realize()
    #assert t.uop.is_realized, "shard didn't realize"
    self.assertEqual(r.tolist(), out)
  def test_shard_reshape(self): self._test_shard_op(lambda t:t.reshape(2, 2), [[1.,1.],[1.,1.]])
  def test_shard_elementwise(self): self._test_shard_op(lambda t:(t+t).reshape(2, 2), [[2.,2.],[2.,2.]])
  def test_alu_deviceless_const(self):
    s = Tensor([1.0, 2, 3, 4]).shard((f"{Device.DEFAULT}:0", f"{Device.DEFAULT}:1"), axis=0)
    np.testing.assert_equal((s + Tensor(UOp.const(dtypes.float, 1.0))).numpy(), [2, 3, 4, 5])
    np.testing.assert_equal((s + Tensor(UOp.const(dtypes.float, 1.0)).reshape((1,)).expand((4,))).numpy(), [2, 3, 4, 5])

  def test_shard_reduce(self):
    self._test_shard_op(lambda t:t.reshape(2, 3).sum(axis=1), [3.,3.], n=6)
    self._test_shard_op(lambda t:t.reshape(2, 3).sum(axis=0), [2.,2.,2.], n=6)

  def test_shard_not_multiple(self):
    X = Tensor.ones(256).contiguous().realize()
    with self.assertRaises(RuntimeError):
      X.shard_(devices_3, 0)

  def test_tensor_from_multi(self):
    X = Tensor([1, 2], dtype=dtypes.int).shard_(devices_2, 0)
    Y = Tensor(X.uop)
    self.assertEqual(Y.device, devices_2)
    np.testing.assert_equal(X.numpy(), Y.numpy())

    with self.assertRaises(AssertionError):
      _ = Tensor(X.uop, dtype=dtypes.float)

  def test_sharded_arange(self):
    sharded_arange = Tensor.arange(1000).clone().shard(devices_2, 0)
    sharded_arange.realize()
    np.testing.assert_equal(sharded_arange.numpy(), np.arange(1000))

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

  def test_shard_plus_one_sum(self):
    X = Tensor.ones(256).contiguous().realize()
    X.shard_((d1, d2), 0)
    (X + 1).sum().realize()

  def test_shard_plus_one_sum_d0(self):
    X = Tensor.ones(256).contiguous().realize()
    X.shard_((d0, d2), 0)
    (X + 1).sum().realize()

  def test_numpy(self):
    X = Tensor.ones(256)
    X.shard_((d1, d2), 0)
    np.testing.assert_allclose(X.numpy(), 1)

  def _test_simple_add_axis(self, shard_x, shard_w):
    X = Tensor.ones(256).contiguous().realize()
    W = Tensor.ones(256).contiguous().realize()
    X.shard_((d1, d2), shard_x)
    W.shard_((d1, d2), shard_w)
    O = X + W
    np.testing.assert_allclose(O.numpy(), 2)

  def test_simple_add(self): return self._test_simple_add_axis(None, None)
  def test_simple_add_X(self): return self._test_simple_add_axis(0, None)
  def test_simple_add_W(self): return self._test_simple_add_axis(None, 0)
  def test_simple_add_XW(self): return self._test_simple_add_axis(0, 0)

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
    X = Tensor.arange(4*4).reshape(4,4).realize()
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

  def test_allreduce_shard_ring_sum(self):
    for axis in (0, 1, None):
      for use_ring in (0, 2):
        t = Tensor([1, 2, 3, 4]).reshape(2, 2)
        with Context(RING=use_ring):
          np.testing.assert_equal(t.shard(devices_2, axis=axis).sum().item(), 10)

  def test_allreduce_naive(self):
    with Context(RING=0):
      a,b = _test_allreduce(Tensor.rand(256, 256))
      np.testing.assert_almost_equal(a.numpy(), b.numpy(), decimal=5)

  def test_allreduce_ring(self):
    with Context(RING=2):
      a,b = _test_allreduce(Tensor.rand(256, 256))
      np.testing.assert_almost_equal(a.numpy(), b.numpy(), decimal=5)

  def test_multiple_to_single_device(self):
    kernel_counts = {}
    for ring in (0, 2):
      GlobalCounters.reset()
      with Context(RING=ring, SCACHE=0):
        t = Tensor.arange(32).clone().shard(devices_4, 0).to(Device.DEFAULT)
        t.realize()
      kernel_counts[ring] = GlobalCounters.kernel_count
      self.assertEqual(t.device, Device.DEFAULT)
      np.testing.assert_equal(t.numpy(), np.arange(32))
    self.assertNotEqual(kernel_counts[0], kernel_counts[2])

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
      tt = Tensor.arange(0, 4).contiguous().realize().shard((d1,d2), 0).realize()
      out = f(tt)
      assert out.item() == 1+2+3+4

  def test_multitensor_jit_input_reduce_shard_axis(self):
    @TinyJit
    def f(x): return x.sum(0).realize()
    for _ in range(5):
      tt = Tensor.ones(2, 64).contiguous().realize().shard((d1,d2), 0).realize()
      out = f(tt)
      np.testing.assert_allclose(out.numpy(), np.full(64, 2.0))

  def test_multitensor_inside_jit(self):
    @TinyJit
    def f(x): return (x.shard((d1,d2), 0)+1).contiguous().sum()
    for _ in range(5):
      tt = Tensor.arange(0, 4).contiguous().realize()
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

  def _test_matmul_shard_axis(self, shard_x, shard_w, device):
    X = Tensor.kaiming_uniform(N, N).realize()
    W = Tensor.kaiming_uniform(N, N).realize()
    Xs = X.shard(device, shard_x)
    Ws = W.shard(device, shard_w)
    O = (Xs@Ws)
    with np.errstate(all='ignore'):
      np.testing.assert_allclose(X.numpy() @ W.numpy(), O.to(Device.DEFAULT).numpy(), atol=1e-5)

  def _test_double_matmul_shard_axis(self, shard_x, shard_w, device):
    X = Tensor.kaiming_uniform(N, N).realize()
    W1 = Tensor.kaiming_uniform(N, N).realize()
    W2 = Tensor.kaiming_uniform(N, N).realize()
    Xs = X.shard(device, shard_x)
    W1s = W1.shard(device, shard_w)
    W2s = W2.shard(device, shard_w)
    O = (Xs@W1s)@W2s
    with np.errstate(all='ignore'):
      np.testing.assert_allclose((X.numpy() @ W1.numpy()) @ W2.numpy(), O.to(Device.DEFAULT).numpy(), atol=1e-5)

  def test_matmul_shard_none(self): return self._test_matmul_shard_axis(None, None, devices_2)
  def test_matmul_shard_X_0(self): return self._test_matmul_shard_axis(0, None, devices_2)
  def test_matmul_shard_X_1(self): return self._test_matmul_shard_axis(1, None, devices_2)
  def test_matmul_shard_W_0(self): return self._test_matmul_shard_axis(None, 0, devices_2)
  def test_matmul_shard_W_1(self): return self._test_matmul_shard_axis(None, 1, devices_2)

  def test_matmul_shard_0_0(self): return self._test_matmul_shard_axis(0, 0, devices_2)
  def test_matmul_shard_0_1(self): return self._test_matmul_shard_axis(0, 1, devices_2)
  def test_matmul_shard_1_0(self): return self._test_matmul_shard_axis(1, 0, devices_2)
  def test_matmul_shard_1_1(self): return self._test_matmul_shard_axis(1, 1, devices_2)

  def test_double_matmul_shard_X_0(self): return self._test_double_matmul_shard_axis(0, None, devices_2)
  def test_double_matmul_shard_X_1(self): return self._test_double_matmul_shard_axis(1, None, devices_2)
  def test_double_matmul_shard_W_0(self): return self._test_double_matmul_shard_axis(None, 0, devices_2)
  def test_double_matmul_shard_W_1(self): return self._test_double_matmul_shard_axis(None, 1, devices_2)

  def test_conv_data_shard(self):
    conv = nn.Conv2d(3, 16, 3, bias=False)
    for p in get_parameters(conv): p.shard_(devices_2)
    fake_image = Tensor.rand((2, 3, 32, 32)).shard(devices_2, axis=0)
    out = conv(fake_image)
    out.numpy()

  def test_conv_bias_data_shard(self):
    conv = nn.Conv2d(3, 16, 3)
    for p in get_parameters(conv): p.shard_(devices_2)
    fake_image = Tensor.rand((2, 3, 32, 32)).shard(devices_2, axis=0)
    out = conv(fake_image)
    out.numpy()

  def test_backprop_conv(self):
    with Tensor.train():
      conv = nn.Conv2d(3, 16, 3)
      for p in get_parameters(conv): p.shard_(devices_2)
      optim = nn.optim.Adam(get_parameters(conv))
      fake_image = Tensor.rand((2, 3, 32, 32)).shard(devices_2, axis=0)
      out = conv(fake_image)
      optim.zero_grad()
      out.mean().backward()
      #for p in get_parameters(conv): p.grad.realize()
      optim.step()
      out.numpy()

  def test_backprop_conv_wino(self):
    with Context(WINO=1): self.test_backprop_conv()

  def test_backward_sum(self):
    x = Tensor([[1.,2,3,4], [5,6,7,8]]).shard(devices_2, axis=0)
    w = Tensor([1.,2,3,4]).shard(devices_2)
    out = x * w
    out.mean().backward()
    tst = w.grad.numpy()
    np.testing.assert_allclose(tst, [0.75, 1., 1.25, 1.5])

  def test_lr_scheduler_OneCycleLR(self):
    from extra.lr_scheduler import OneCycleLR
    conv = nn.Conv2d(3, 16, 3)
    for p in get_parameters(conv): p.shard_(devices_2)
    optim = nn.optim.SGD(get_parameters(conv))
    lr_sched = OneCycleLR(optim, max_lr=0.1, pct_start=0.1, div_factor=100, final_div_factor=0.1, total_steps=10)
    lr_sched.step()

  def test_embedding(self):
    B, T, embed_size, vocab_size = 4, 10, 20, 28

    layer = nn.Embedding(vocab_size, embed_size)
    x = Tensor(np.random.randint(0, vocab_size, (B, T), dtype=np.int32))
    z = layer(x)

    layer_sharded = nn.Embedding(vocab_size, embed_size)
    layer_sharded.weight.replace(layer.weight.shard(devices_2, axis=1)).realize()
    x_sharded = x.shard(devices_2, axis=None)
    z_shard = layer_sharded(x_sharded)

    np.testing.assert_allclose(z.numpy(), z_shard.numpy(), atol=1e-6, rtol=1e-6)

  def test_embedding_backward(self, shard_weight_axis=None):
    B, T, embed_size, vocab_size = 4, 10, 20, 28

    layer = nn.Embedding(vocab_size, embed_size)
    x = Tensor(np.random.randint(0, vocab_size, (B, T), dtype=np.int32))
    z = layer(x)
    z.sum().backward()
    grad = layer.weight.grad.numpy()

    layer_sharded = nn.Embedding(vocab_size, embed_size)
    layer_sharded.weight.replace(layer.weight.shard(devices_2, axis=shard_weight_axis)).realize()
    x_sharded = x.shard(devices_2, axis=None)
    z_shard = layer_sharded(x_sharded)
    z_shard.sum().backward()
    grad_shard = layer_sharded.weight.grad.numpy()

    np.testing.assert_allclose(grad, grad_shard, atol=1e-6, rtol=1e-6)

  def test_embedding_backward_shard_weight(self): self.test_embedding_backward(shard_weight_axis=1)

  def test_rmsnorm(self):
    B, T, embed_size = 4, 10, 20

    norm = nn.RMSNorm(embed_size)
    x = Tensor.rand((B, T, embed_size)).contiguous().realize()
    y = norm(x)

    # for norm layers, the correct way to shard weights is duplication
    norm_sharded = nn.RMSNorm(embed_size)
    norm_sharded.weight.shard_(devices_2, axis=None).realize()

    # if x is being sharded, then all-reduce is involved
    x_sharded = x.shard(devices_2, axis=2).realize()
    y_shard = norm_sharded(x_sharded).realize()
    np.testing.assert_allclose(y.numpy(), y_shard.numpy(), atol=1e-6, rtol=1e-6)

    # if x is being duplicated, then the operations remain inside each GPU
    # which is the common case
    x_sharded = x.shard(devices_2, axis=None).realize()
    y_shard = norm_sharded(x_sharded).realize()
    np.testing.assert_allclose(y.numpy(), y_shard.numpy(), atol=1e-6, rtol=1e-6)

  def test_sdpa_causal_shard_batch(self):
    B, H, T, D = 4, 2, 10, 16
    q = Tensor.rand(B, H, T, D)
    k = Tensor.rand(B, H, T, D)
    v = Tensor.rand(B, H, T, D)
    q_shard = q.shard(devices_2, axis=0)
    k_shard = k.shard(devices_2, axis=0)
    v_shard = v.shard(devices_2, axis=0)
    Tensor.realize(q, k, v, q_shard, k_shard, v_shard)
    y = Tensor.scaled_dot_product_attention(q, k, v, is_causal=True).realize()
    y_shard = Tensor.scaled_dot_product_attention(q_shard, k_shard, v_shard, is_causal=True).realize()
    np.testing.assert_allclose(y_shard.numpy(), y.numpy(), atol=1e-6, rtol=1e-6)

  # NOTE: this is failing on LLVM CI, no idea why. Works locally.
  @slow
  def test_data_parallel_resnet(self):
    from extra.models.resnet import ResNet18

    fake_image = Tensor.rand((2, 3, 224//16, 224//16))
    fake_image_sharded = fake_image.shard(devices_2, axis=0)
    m = ResNet18()
    m.load_from_pretrained()
    real_output = m(fake_image).log_softmax().numpy()
    for p in get_parameters(m): p.shard_(devices_2).realize()
    GlobalCounters.reset()
    shard_output = m(fake_image_sharded).log_softmax().realize()
    shard_output_np = shard_output.numpy()
    np.testing.assert_allclose(real_output, shard_output_np, atol=1e-6, rtol=1e-6)

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

  def test_multi_tensor_jit_param(self):
    @TinyJit
    def jf(a, b) -> Tensor:
      return (a + b).realize()

    for _ in range(5):
      a = Tensor.ones(256).contiguous().realize()
      b = Tensor.ones(256).contiguous().realize()
      a.shard_(devices_2)
      b.shard_(devices_2)
      c = jf(a, b)
      np.testing.assert_allclose(c.numpy(), a.numpy()+b.numpy(), atol=1e-4, rtol=1e-5)
    assert jf.captured is not None

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

  def test_multi_tensor_jit_body(self):
    @TinyJit
    def jf() -> Tensor:
      a = Tensor.ones(256).contiguous().realize()
      b = Tensor.ones(256).contiguous().realize()
      a.shard_(devices_2)
      b.shard_(devices_2)
      return (a + b).realize()

    for _ in range(5):
      r = jf()
      np.testing.assert_allclose(r.numpy(), np.ones(256)+np.ones(256), atol=1e-4, rtol=1e-5)
    assert jf.captured is not None

  def test_multitensor_jit_in_list(self):
    # test MULTI tensor inside a list container - exercises the container unpacking + MULTI unpacking
    @TinyJit
    def f(a, arr): return (a + arr[0]).realize()
    for i in range(5):
      a = Tensor.full((4,), i).contiguous().realize().shard(devices_2, 0).realize()
      b = Tensor.ones(4).contiguous().realize().shard(devices_2, 0).realize()
      out = f(a, [b])
      np.testing.assert_allclose(out.numpy(), np.full(4, i) + np.ones(4), atol=1e-4, rtol=1e-5)

  def test_multitensor_jit_multiple_inputs(self):
    # test multiple MULTI tensors as inputs - each gets unpacked to component UOps
    @TinyJit
    def f(a, b, c): return (a + b + c).realize()
    for i in range(5):
      a = Tensor.full((4,), i).contiguous().realize().shard(devices_2, 0).realize()
      b = Tensor.full((4,), i*2).contiguous().realize().shard(devices_2, 0).realize()
      c = Tensor.ones(4).contiguous().realize().shard(devices_2, 0).realize()
      out = f(a, b, c)
      np.testing.assert_allclose(out.numpy(), np.full(4, i) + np.full(4, i*2) + np.ones(4), atol=1e-4, rtol=1e-5)

  def test_multitensor_jit_different_sharding(self):
    # test MULTI tensors with different sharding - one sharded on axis 0, one broadcast (axis=None)
    @TinyJit
    def f(a, b): return (a + b).realize()
    for i in range(5):
      a = Tensor.full((4, 4), i).contiguous().realize().shard(devices_2, 0).realize()
      b = Tensor.full((4, 4), i*2).contiguous().realize().shard(devices_2, None).realize()
      out = f(a, b)
      np.testing.assert_allclose(out.numpy(), np.full((4, 4), i) + np.full((4, 4), i*2), atol=1e-4, rtol=1e-5)

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

  def test_bn_ast_on_devices(self):
    t = Tensor.empty((16, 64, 112, 112)).shard(devices_4, axis=0)
    bn = nn.BatchNorm2d(64)
    for p in get_parameters(bn): p.shard_(devices_4).realize()

    out = bn(t)
    scheds = [call for call in out.schedule_linear().src if call.src[0].op is not Ops.COPY and set(call.device) <= set(devices_4)]
    self.assertEqual(set(scheds[0].device), set(devices_4), "should have ast on each shard device")
    self.assertEqual(len(set(s.src[0] for s in scheds)), 1)

  def test_flip(self):
    rng = Tensor.rand((10, 10, 10))
    t0 = rng.shard(devices_2, axis=1)
    out = t0.flip(0) + 1
    self.assertTrue((rng.flip(0)+1).allclose(out.to(rng.device)).item())

  @unittest.skip("flaky")
  def test_reshape_on_axis(self):
    t0 = Tensor.rand((26, 15, 7)).shard(devices_3, axis=1)

    # test split and rejoin to the right
    t1 = t0.reshape((26, 3, 5, 7))
    t2 = t0.reshape((26, 3, 35))
    t3 = t1.reshape((26, 15, 7))
    t4 = t2.reshape((26, 105,))

    for t in [t0, t1, t2, t3, t4]:
      assert t.uop.axis == 1
      np.testing.assert_allclose(t.numpy().flatten(), t0.numpy().flatten())

    # test shape-one axis
    t5 = t4.reshape((26, 1, 105))
    assert t5.uop.axis == 2
    np.testing.assert_allclose(t.numpy().flatten(), t5.numpy().flatten())

    # test split and rejoin to the right and reshape to the left
    t5 = t0.reshape((2, 13, 3, 5, 7))
    t6 = t0.reshape((13, 2, 3, 7, 5))
    t7 = t0.reshape((1, 13, 2, 3, 1, 7, 5))
    assert t5.uop.axis == 2
    assert t6.uop.axis == 2
    assert t7.uop.axis == 3
    np.testing.assert_allclose(t5.numpy().flatten(), t0.numpy().flatten())
    np.testing.assert_allclose(t6.numpy().flatten(), t0.numpy().flatten())
    np.testing.assert_allclose(t7.numpy().flatten(), t0.numpy().flatten())

    # test no left join
    with self.assertRaises((AssertionError, ValueError)):
      t0.reshape((26*15,7)).contiguous().schedule_linear()

  # it doesn't work like this anymore
  # NOTE: this never failed in assign_multi, it failed tensor spec because MULTI was never pushed in the graph
  @unittest.skip("this test is broken")
  def test_mlb_assign_change_axis(self):
    t_none = Tensor.zeros((16, 16)).shard(devices_2).contiguous().realize()
    t_zero = Tensor.ones((16, 16)).shard(devices_2, axis=0)
    with self.assertRaises(RuntimeError):
      # don't allow assigns that change axes
      t_none.assign(t_zero)
      t_none.schedule_linear()

  def test_init_rand_with_multiple_devices_fail(self):
    # init rand with multi device is not allowed
    with self.assertRaises(ValueError):
      Tensor.rand(256, device=devices_2)

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

  def test_rand_like_on_shard(self, axis=None):
    t = Tensor.empty((16, 16)).shard(devices_2, axis=axis)
    t2 = Tensor.rand_like(t)
    self.assertEqual(t.shape, t2.shape)
    self.assertEqual(t.device, t2.device)
    self.assertEqual(t.dtype, t2.dtype)
    self.assertEqual(t.uop.axis, t2.uop.axis)
    t2.realize()
  def test_rand_like_on_shard_axis(self): self.test_rand_like_on_shard(0)

  def test_rand_like_from_alu(self):
    a = Tensor.ones(4, 4).shard(devices_4, axis=0)
    aa = a + a
    self.assertEqual(aa.device, devices_4)
    self.assertEqual(aa.uop.axis, 0)
    raa = aa.rand_like()
    self.assertEqual(raa.device, devices_4)
    self.assertEqual(raa.uop.axis, 0)

    b = Tensor.empty(4, 4).shard(devices_4, axis=None)
    ab = a + b
    self.assertEqual(ab.device, devices_4)
    self.assertEqual(ab.uop.axis, 0)
    rab = ab.rand_like()
    self.assertEqual(rab.device, devices_4)
    self.assertEqual(rab.uop.axis, 0)

  def test_rand_like_none_shard(self):
    t = Tensor.empty((16, 16)).shard(devices_2)
    t2 = Tensor.rand_like(t)
    self.assertEqual(t.shape, t2.shape)
    self.assertEqual(t.device, t2.device)
    self.assertEqual(t.dtype, t2.dtype)
    self.assertEqual(t.uop.axis, t2.uop.axis)

  def test_rand_like_arg_dtype(self):
    t = Tensor.empty((16, 16), dtype=dtypes.int32).shard(devices_2, axis=1)
    t2 = Tensor.rand_like(t, dtype=dtypes.float32)
    self.assertEqual(t.dtype, dtypes.int32)
    self.assertEqual(t2.dtype, dtypes.float32)

  def test_rand_like_arg_device(self):
    # axis=None
    t = Tensor.empty((16, 16)).shard((d1, d2), axis=None)
    with self.assertRaises(RuntimeError):
      Tensor.rand_like(t, device=(d3, d4))

    # axis=1
    t = Tensor.empty((16, 16)).shard((d1, d2), axis=1)
    with self.assertRaises(RuntimeError):
      Tensor.rand_like(t, device=(d3, d4))

  def test_full_like_on_shard(self, axis=None):
    t = Tensor.empty((16, 16)).shard(devices_2, axis=axis)
    t2 = Tensor.full_like(t, 1.0)
    self.assertEqual(t.shape, t2.shape)
    self.assertEqual(t.device, t2.device)
    self.assertEqual(t.dtype, t2.dtype)
    self.assertEqual(t.uop.axis, t2.uop.axis)
    t2.realize()
  def test_full_like_on_shard_axis(self): self.test_full_like_on_shard(0)

  def test_full_like_shrink_on_shard_axis(self):
    t = Tensor.ones(16, 16, dtype=dtypes.int).shard(devices_2, axis=0)
    out = Tensor.full_like(t, 2)[:, :8]
    linear, var_vals = out.linear_with_vars()
    self.assertEqual(len(linear.src), 0)
    run_linear(linear, var_vals)
    self.assertEqual(out.tolist(), [[2]*8]*16)

  def test_dropout_on_shard(self):
    with Tensor.train():
      X = Tensor.ones(256).to(devices_2)
      output = X.dropout(0.5).numpy()
      unique, counts = np.unique(output, return_counts=True)
      assert set(unique) == {0, 2}, unique
      assert 96 < counts[0] < 160, counts[0]

  def test_dropout_on_shard_axis(self):
    with Tensor.train():
      X = Tensor.ones(512).shard(devices_2, axis=0)
      output = X.dropout(0.5).numpy()
      unique, counts = np.unique(output, return_counts=True)
      assert set(unique) == {0, 2}, unique
      assert 192 < counts[0] < 320, counts[0]

  @unittest.skip("TODO: this requires forced_realize to be deleted.")
  def test_shard_memory(self):
    devices = (d0, d1, d2, d3)
    t = Tensor.zeros(16, 16).contiguous()
    t.shard_(devices, axis=0).realize()
    assert all([lb is lb.base and lb.realized.base.size == 4 * 16 for lb in t.uop.src])

  def test_clone(self):
    for axis in (None, 0):
      t = Tensor.arange(16).reshape(4, 4).clone().shard(devices_2, axis=axis).contiguous().realize()
      t_clone = t.clone().realize()
      self.assertEqual(t_clone.device, t.device)
      self.assertEqual(t_clone.uop.axis, axis)
      self.assertEqual(t_clone.tolist(), t.tolist())
      t_clone += 1
      self.assertNotEqual(t_clone.tolist(), t.tolist())

  @unittest.skip("RANGEIFY doesn't support multi const folding")
  def test_multi_const_folding(self):
    with Context(TRACK_MATCH_STATS=0):
      a = Tensor.arange(3).realize()
      zeros = Tensor.zeros(3).realize()
    b = a.to(devices_2)*zeros.to(devices_2)
    sched = b.schedule_linear().src
    self.assertEqual(len(sched), 0)
    self.assertListEqual(b.tolist(), [0, 0, 0])

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

@unittest.skipIf(not_support_multi_device(), "no multi")
class TestShrinkMultiTensorShardedAxis(unittest.TestCase):
  @needs_second_gpu
  def setUp(self): pass

  # shrink a multitensor on sharded axis
  def test_shrink_bad_args(self):
    t = Tensor.arange(64).reshape(8, 8).contiguous().realize()
    t.shard_([f"{Device.DEFAULT}:{i}" for i in range(4)], axis=0)

    with self.assertRaises(AssertionError):
      # sharded axis shrink on non-device boundry is not allowed
      a = t.shrink(((0, 3), (0, 8))).contiguous()
      a.schedule_linear()
    a = t.shrink(((0, 2), (2, 4)))
    assert a.shape == (2, 2)
    ref = Tensor.arange(64).reshape(8, 8).shrink(((0, 2), (2, 4)))
    np.testing.assert_equal(a.numpy(), ref.numpy())

    a = t.shrink(((0, 2), (0, 8))).contiguous()
    a.schedule_linear()
    assert a.shape == (2, 8)

    p = a.pad(((0, 6), (0, 0))).contiguous()
    p.schedule_linear()
    assert p.shape == (8, 8)

  @given(strat.sampled_from([dtypes.float, dtypes.int, dtypes.int64, dtypes.int16]))
  def test_ops(self, dtype):
    if dtype not in Device[Device.DEFAULT].renderer.supported_dtypes(): return
    t = Tensor.arange(64).reshape(8, 8).contiguous().realize()
    t.shard_([f"{Device.DEFAULT}:{i}" for i in range(4)], axis=0)
    for i in range(4):
      print(f"{i=}")
      a = t.shrink(((0+2*i,2+2*i),None))
      b = Tensor(t.numpy()[0+2*i:2+2*i])
      assert a.shape == b.shape == (2, 8)
      np.testing.assert_allclose(a.numpy(), b.numpy())
      # cast
      np.testing.assert_allclose(a.float().numpy(), b.float().numpy())

      # elementwise
      np.testing.assert_allclose(a.exp().numpy(), b.exp().numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.reciprocal().numpy(), b.reciprocal().numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.pow(-0.5).numpy(), b.pow(-0.5).numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose((a+a).numpy(), (b+b).numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_equal((a+1).numpy(), (b+1).numpy())
      np.testing.assert_equal((1+a).numpy(), (1+b).numpy())
      np.testing.assert_allclose((a.where(a+a, a)).numpy(), (b.where(b+b, b)).numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose((a.where(1, 0)).numpy(), (b.where(1, 0)).numpy(), rtol=1e-7, atol=1e-3)

      # reduce
      np.testing.assert_allclose(a.max().numpy(), b.max().numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.sum().numpy(), b.sum().numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.mean().numpy(), b.mean().numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.max(0).numpy(), b.max(0).numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.sum(0).numpy(), b.sum(0).numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.mean(0).numpy(), b.mean(0).numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.max(1).numpy(), b.max(1).numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.sum(1).numpy(), b.sum(1).numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.mean(1).numpy(), b.mean(1).numpy(), rtol=1e-7, atol=1e-3)

      # pad it back
      np.testing.assert_allclose(a.pad(((2*i, 2*(4-i-1)), None)).numpy(), b.pad(((2*i, 2*(4-i-1)), None)).numpy(), rtol=1e-7, atol=1e-3)

      # other movement
      np.testing.assert_allclose(a.pad((None, (1, 1))).numpy(), b.pad((None, (1, 1))).numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.shrink((None, (1, 3))).numpy(), b.shrink((None, (1, 3))).numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.permute((1, 0)).numpy(), b.permute((1, 0)).numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.reshape((2, 2, 4)).numpy(), b.reshape((2, 2, 4)).numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.reshape((2, 1, 8)).expand((2, 5, 8)).numpy(), b.reshape((2, 1, 8)).expand((2, 5, 8)).numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.flip(-1).numpy(), b.flip(-1).numpy(), rtol=1e-7, atol=1e-3)

  def test_add_two_partitions(self):
    t = Tensor.arange(64).reshape(8, 8).contiguous().realize()
    t.shard_([f"{Device.DEFAULT}:{i}" for i in range(4)], axis=0)

    a = t.shrink(((2, 4), None))
    b = t.shrink(((6, 8), None))
    na = t.numpy()[2:4]
    nb = t.numpy()[6:8]
    np.testing.assert_equal(a.numpy(), na)
    np.testing.assert_equal(b.numpy(), nb)
    np.testing.assert_equal((a+b).numpy(), na+nb)
    c = a.pad(((2, 4), None)) + b.pad(((6, 0), None))
    c.realize()
    expected = np.concatenate([np.zeros_like(t.numpy()[0:2]), na, np.zeros_like(t.numpy()[4:6]), nb])
    np.testing.assert_equal(c.numpy(), expected)

  def test_add_different_tensors(self):
    devices = [f"{Device.DEFAULT}:{i}" for i in range(4)]
    x = Tensor.arange(64).reshape(8, 8).contiguous().realize().shard(devices, axis=0)

    to_add = []
    for i in range(len(devices)):
      to_add.append((Tensor.ones(2, 8) * i).shard(devices))

    added:list[Tensor] = []
    for bound, a in zip(x.uop.bounds, to_add):
      added.append(x[bound[0]:bound[1]] + a)

    output = added[0].cat(*added[1:])
    expected = np.arange(64).reshape((8,8)) + np.array([[0,0,1,1,2,2,3,3] for _ in range(8)]).T
    np.testing.assert_allclose(output.numpy(), expected)

@unittest.skipIf(not_support_multi_device(), "no multi")
class TestBatchNorm(unittest.TestCase):
  @needs_second_gpu
  def setUp(self): pass

  def test_unsynced_backprop_conv_bn(self):
    with Tensor.train():
      from extra.lr_scheduler import OneCycleLR

      convs = [nn.Conv2d(3, 16, 3), nn.Conv2d(3, 16, 3)]
      bns = [nn.BatchNorm2d(16), nn.BatchNorm2d(16)]

      for p in get_parameters(convs + bns):
        p.shard_((d1, d2))
      optim = nn.optim.Adam(get_parameters(convs + bns))
      lr_sched = OneCycleLR(optim, max_lr=0.1, pct_start=0.1, div_factor=100, final_div_factor=0.1, total_steps=10)
      lr_sched.step()

      fake_image = Tensor.rand((8, 3, 32, 32)).shard((d1, d2), axis=0)

      f1 = fake_image.shrink(((0, 4), None, None, None))
      f2 = fake_image.shrink(((4, 8), None, None, None))

      out1 = bns[0](convs[0](f1))
      out2 = bns[1](convs[1](f2))
      out = out1.cat(out2)
      optim.zero_grad()
      out.mean().backward()
      optim.step()
      out.numpy()

  def test_unsynced_backprop_standalone_bn(self):
    from extra.lr_scheduler import OneCycleLR
    GPUS = (d1, d2)

    class BatchNorm:
      def __init__(self, num_features):
        self.bns:list[nn.BatchNorm2d] = []
        for _ in GPUS:
          bn = nn.BatchNorm2d(num_features, track_running_stats=False, eps=1e-12, momentum=0.85, affine=True)
          self.bns.append(bn)

      def __call__(self, x:Tensor):
        bn_ts = []
        each = x.shape[0]//len(self.bns)
        for i, bn in enumerate(self.bns):
          xi = x.shrink(((each*(i), each*(i+1)), None, None, None))
          bni = bn(xi)
          bn_ts.append(bni)
        return bn_ts[0].cat(*bn_ts[1:])

    with Tensor.train():
      conv = nn.Conv2d(3, 16, 3)
      bn = BatchNorm(16)

      for p in get_parameters([conv, bn]):
        p.shard_(GPUS)
      optim = nn.optim.Adam(get_parameters([conv, bn]))
      lr_sched = OneCycleLR(optim, max_lr=0.1, pct_start=0.1, div_factor=100, final_div_factor=0.1, total_steps=10)
      lr_sched.step()

      fake_image = Tensor.rand((8, 3, 32, 32)).shard(GPUS, axis=0)

      out = bn(conv(fake_image))
      optim.zero_grad()
      out.mean().backward()
      optim.step()

  def test_unsynced_backprop_sync_weights(self):
    from extra.lr_scheduler import OneCycleLR
    from examples.hlb_cifar10 import UnsyncedBatchNorm
    GPUS = (d1, d2)

    with Tensor.train():
      conv = nn.Conv2d(3, 16, 3)
      bn = UnsyncedBatchNorm(16, num_devices=len(GPUS))

      for k, p in get_state_dict([conv, bn]).items():
        if 'running_mean' in k or 'running_var' in k:
          p.shard_(GPUS, axis=0)
        else:
          p.to_(GPUS)
      optim = nn.optim.Adam(get_parameters([conv, bn]))
      lr_sched = OneCycleLR(optim, max_lr=0.1, pct_start=0.1, div_factor=100, final_div_factor=0.1, total_steps=10)
      lr_sched.step()

      fake_image = Tensor.rand((8, 3, 32, 32)).shard(GPUS, axis=0)

      out = bn(conv(fake_image))
      optim.zero_grad()
      out.mean().backward()
      optim.step()

  @given(strat.sampled_from((False, True)))
  def test_batchnorm(self, is_training):
    devices = [f"{Device.DEFAULT}:{i}" for i in range(4)]
    x = Tensor.arange(4096).reshape(8, 8, 8, 8).contiguous().realize().shard(devices, axis=0)

    with Tensor.train(is_training):
      bns = []
      for _ in range(len(devices)):
        bn = nn.BatchNorm2d(8)
        for p in get_parameters(bn):
          p.shard_(devices)
        bns.append(bn)

      bn_ts = []
      for bound, bn in zip(x.uop.bounds, bns):
        bni = bn(x[bound[0]:bound[1]])
        bn_ts.append(bni)

      bn_ts[0].cat(*bn_ts[1:]).numpy()

  def test_synced_vs_unsynced_bn(self):
    from examples.hlb_cifar10 import UnsyncedBatchNorm
    from tinygrad.nn import BatchNorm2d
    devices = [f"{Device.DEFAULT}:{i}" for i in range(4)]
    x = Tensor.ones(8, 8, 8, 8).contiguous().realize().shard(devices, axis=0)

    with Tensor.train():
      synced_bn = BatchNorm2d(8)
      unsynced_bn = UnsyncedBatchNorm(8, num_devices=len(devices))

      for p in get_parameters(synced_bn):
        p.shard_(devices)
      for k, p in get_state_dict(unsynced_bn).items():
        if 'running_mean' in k or 'running_var' in k:
          p.shard_(devices, axis=0)
        else:
          p.to_(devices)

      synced_out = synced_bn(x)
      synced_si = list(synced_out.schedule_linear().src)
      unsynced_out = unsynced_bn(x)
      unsynced_si = list(unsynced_out.schedule_linear().src)

    # TODO: test synced / unsynced batchnorm cross device kernel and copies
    assert synced_si
    assert unsynced_si

def helper_test_shard_op(shps, fxn, atol=1e-6, rtol=1e-3):
  for shp in shps:
    single_in = Tensor.randn(shp)
    multi_in  = single_in.shard(devices_2, axis=0)

    single_out = fxn(single_in).numpy()
    multi_out  = fxn(multi_in).numpy()

    try:
      assert single_out.shape == multi_out.shape, f"shape mismatch: single={single_out.shape} | multi={multi_out.shape}"
      assert single_out.dtype == multi_out.dtype, f"dtype mismatch: single={single_out.dtype} | multi={multi_out.dtype}"
      np.testing.assert_allclose(single_out, multi_out, atol=atol, rtol=rtol)
    except Exception as e:
      raise Exception(f"Failed shape {single_out.shape}: {e}")

@unittest.skipIf(not_support_multi_device(), "no multi")
class TestTensorOps(unittest.TestCase):
  @needs_second_gpu
  def test_interpolate(self):
    helper_test_shard_op([(4,16,16),(4,24,24)], lambda x: Tensor.interpolate(x, (19,19)))

  @needs_second_gpu
  def test_bitcast(self):
    helper_test_shard_op([(256,), (256,)], lambda x: x.bitcast(dtypes.int))

@unittest.skipIf(not_support_multi_device(), "need multi")
class TestMultiBufferView(unittest.TestCase):
  @needs_second_gpu
  def setUp(self): pass

  def _check(self, a_ref:Tensor, a_multi:Tensor, view_fn):
    b_ref = view_fn(a_ref)
    b_multi = view_fn(a_multi).contiguous()
    linear, var_vals = b_multi.linear_with_vars()
    if all(hasattr(Device[d].allocator, "_offset") for d in b_multi.device):
      compiled = [call for call in linear.src if call.src[0].op is Ops.SINK]
      self.assertEqual(len(compiled), 0, f"expected zero compiled kernels, got {len(compiled)}")
    run_linear(linear, var_vals)
    np.testing.assert_equal(b_multi.numpy(), b_ref.numpy())

  @unittest.skip("flaky on LLVM")
  def test_shrink_non_shard_axis(self):
    ref = Tensor.arange(8*4*10).reshape(8, 4, 10).contiguous().realize()
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
    if all(hasattr(Device[d].allocator, "_offset") for d in out.device):
      compiled = [call for call in linear.src if call.src[0].op is Ops.SINK]
      self.assertEqual(len(compiled), 0)
    run_linear(linear, var_vals)
    np.testing.assert_equal(out.numpy(), ref[5].numpy())

@unittest.skipIf(not_support_multi_device(), "need multi")
class TestMultiFromUnrenderable(unittest.TestCase):
  @needs_second_gpu
  def test_from_npy(self):
    t = Tensor(np.arange(100, dtype=np.uint32))
    ll = t.shard((d0, d1), axis=0) + 1
    np.testing.assert_equal(ll.numpy(), np.arange(100)+1)

@unittest.skipIf(not_support_multi_device(), "need multi")
class TestMultiAssign(unittest.TestCase):
  device = tuple(f"{Device.DEFAULT}:{i}" for i in range(2))

  @needs_second_gpu
  def setUp(self): pass

  def test_multi_assign_realized(self):
    out = Tensor.zeros(4).shard(self.device, 0).contiguous().realize()
    ones = Tensor.ones(4).shard(self.device, 0).contiguous().realize()
    out.assign(ones).realize()
    self.assertListEqual(out.tolist(), [1,1,1,1])

  def test_multi_assign_unrealized(self):
    out = Tensor.zeros(4).contiguous().realize().shard(self.device, 0)
    ones = Tensor.ones(4).shard(self.device, 0).contiguous().realize()
    out.assign(ones).realize()
    self.assertListEqual(out.tolist(), [1,1,1,1])

  def test_multi_assign_both_unrealized(self):
    out = Tensor.zeros(4).contiguous().realize().shard(self.device, 0)
    ones = Tensor.ones(4).contiguous().realize().shard(self.device, 0)
    out.assign(ones).realize()
    self.assertListEqual(out.tolist(), [1,1,1,1])

  def test_multi_assign_piece(self):
    out = Tensor.zeros(4,4).shard(self.device, 0).contiguous().realize()
    ones = Tensor.ones(4,1).shard(self.device, 0).contiguous().realize()
    out[:, 2:3].assign(ones).realize()
    self.assertListEqual(out.tolist(), [[0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0]])

  def test_multi_assign_piece_noncontig(self):
    out = Tensor.zeros(4,4).contiguous().realize().shard(self.device, 0).realize()
    ones = Tensor.ones(4,1).shard(self.device, 0).contiguous().realize()
    out[:, 2:3].assign(ones).realize()
    self.assertListEqual(out.tolist(), [[0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0]])

  @unittest.expectedFailure
  def test_multi_assign_piece_unrealized(self):
    out = Tensor.zeros(4,4).contiguous().realize().shard(self.device, 0)
    ones = Tensor.ones(4,1).shard(self.device, 0).contiguous().realize()
    out[:, 2:3].assign(ones).realize()
    self.assertListEqual(out.tolist(), [[0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0]])

  def test_multi_assign_var_offset(self):
    out = Tensor.zeros(4,4).contiguous().realize().shard(self.device, 0).realize()
    ones = Tensor.ones(4,1).shard(self.device, 0).contiguous().realize()
    vi = Variable("i", 0, 3).bind(2)
    out[:, vi:vi+1].assign(ones).realize()
    self.assertListEqual(out.tolist(), [[0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0]])

  def test_multi_assign_var_offset_jit_none(self): self.test_multi_assign_var_offset_jit(None)
  def test_multi_assign_var_offset_jit(self, shard_axis=0):
    out = Tensor.zeros(4,6).contiguous().realize().shard(self.device, shard_axis).realize()
    ones = Tensor.ones(4,1).shard(self.device, shard_axis).contiguous().realize()

    @TinyJit
    def f(out:Tensor, vi):
      out[:, vi:vi+1].assign(ones).realize()
      ones.assign(ones+1).realize()

    vi = Variable("i", 0, 5)
    for i in range(1,5):
      GlobalCounters.reset()
      f(out, vi.bind(i))
    self.assertListEqual(out.tolist(), [[0,1,2,3,4,0]]*4)

@unittest.skipIf(not_support_multi_device(), "need multi")
class TestMultiSetitem(unittest.TestCase):
  device = tuple(f"{Device.DEFAULT}:{i}" for i in range(4))

  @needs_second_gpu
  def setUp(self): pass

  def _t(self, axis): return Tensor.arange(16).contiguous().realize().shard(self.device, axis=axis)

  def test_setitem_scalar_axis0(self):
    t = self._t(0)
    t[1] = 99
    self.assertListEqual(t.tolist(), [0,99,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

  def test_setitem_scalar_axis_none(self):
    t = self._t(None)
    t[1] = 99
    self.assertListEqual(t.tolist(), [0,99,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

  def test_setitem_slice_cross_shard(self):
    t = self._t(0)
    t[2:6] = 99
    self.assertListEqual(t.tolist(), [0,1,99,99,99,99,6,7,8,9,10,11,12,13,14,15])

  def test_setitem_full_slice(self):
    t = self._t(0)
    t[:] = 42
    self.assertListEqual(t.tolist(), [42]*16)

  def test_setitem_stride(self):
    t = self._t(0)
    t[::4] = 0
    self.assertListEqual(t.tolist(), [0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15])

  def test_setitem_single_shard(self):
    t = self._t(0)
    t[13] = 99
    self.assertListEqual(t.tolist(), [0,1,2,3,4,5,6,7,8,9,10,11,12,99,14,15])

  def test_setitem_tensor_value_replicated(self):
    t = self._t(0)
    t[2:6] = Tensor([90, 91, 92, 93]).shard(self.device)
    self.assertListEqual(t.tolist(), [0,1,90,91,92,93,6,7,8,9,10,11,12,13,14,15])

  def test_setitem_tensor_value_sharded_aligned(self):
    t = self._t(0)
    t[::4] = Tensor([90, 91, 92, 93]).shard(self.device, axis=0)
    self.assertListEqual(t.tolist(), [90,1,2,3,91,5,6,7,92,9,10,11,93,13,14,15])

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
