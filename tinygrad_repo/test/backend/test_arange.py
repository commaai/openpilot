import unittest
import numpy as np
from tinygrad import Tensor, GlobalCounters, dtypes, nn, Device, Variable
from tinygrad.helpers import Context, getenv, DEV
from tinygrad.engine.realize import run_linear, estimate_uop, compile_linear
from tinygrad.renderer.ptx import PTXRenderer
from test.helpers import needs_second_gpu

class TestArange(unittest.TestCase):
  def _get_flops(self, tensor, desired):
    GlobalCounters.reset()
    linear = compile_linear(tensor.schedule_linear())
    self.assertEqual(len(linear.src), 1)
    run_linear(linear)
    np.testing.assert_equal(tensor.numpy(), desired)
    return estimate_uop(linear.src[-1]).ops

  def test_arange_complexity(self):
    self.assertEqual(self._get_flops(Tensor.arange(256), np.arange(256)), 0)
    self.assertEqual(self._get_flops(Tensor.arange(2560), np.arange(2560)), 0)

  @unittest.skipIf(Device.DEFAULT == "CL", "TODO: fails on CI CL")
  def test_arange_cumsum(self):
    np.testing.assert_equal(Tensor.arange(513).cumsum(0).numpy(), np.arange(513).cumsum())

  def test_arange_cat(self):
    t = Tensor.arange(2, dtype=dtypes.int)+Tensor([3])
    self.assertEqual(t.cat(t).tolist(), [3, 4, 3, 4])

  def test_eye_complexity(self):
    with Context(NOOPT=1):
      # NOTE: not every backend supports CMPEQ
      self.assertLessEqual(self._get_flops(Tensor.eye(2560).contiguous(), np.eye(2560)), 2*2560*2560)

  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, PTXRenderer), "PTX indexing is weird")
  def test_tri_complexity(self):
    with Context(NOOPT=1):
      t = Tensor.ones(256, 256).contiguous().realize()
      linear = compile_linear(t.triu().schedule_linear())
      self.assertLessEqual(estimate_uop(linear.src[-1]).ops, 4 * 256 * 256)

DSET, DDIM = 2048, 32

class TestIndexing(unittest.TestCase):
  def test_arange_2_reduce(self):
    needle = Tensor.zeros(16384, dtype=dtypes.int).contiguous()
    needle[1337] = 1
    needle.realize()
    with Context(NOOPT=1):
      GlobalCounters.reset()
      out = ((Tensor.arange(1,16385)-1)*needle).sum()
      linear, var_vals = out.linear_with_vars()
      self.assertEqual(len(linear.src), 1)
      run_linear(linear, var_vals)
    self.assertEqual(out.item(), 1337)

  def test_manual_index(self):
    dataset = Tensor.rand(DSET, DDIM).realize()
    idxs = Tensor([0,3,5,6]).realize()
    real_index = dataset.numpy()[idxs.numpy()]
    print("*** indexing ***")
    with Context(NOOPT=1):
      GlobalCounters.reset()
      rng = Tensor.arange(DSET, dtype=dtypes.int).reshape(1, 1, DSET, 1).expand(4, DDIM, DSET, 1)
      idxs = idxs.reshape(4,1,1,1).expand(4, DDIM, DSET, 1)
      reshape_dataset = dataset.T.reshape(1, DDIM, DSET, 1).expand(4, DDIM, DSET, 1)
      full = (rng==idxs).where(reshape_dataset, Tensor.zeros(4, DDIM, DSET, 1, buffer=False))
      X = full.sum(axis=(2,3))
      linear, var_vals = X.linear_with_vars()
      self.assertEqual(len(linear.src), 1)
      run_linear(linear, var_vals)
      assert GlobalCounters.global_ops < 4*DSET, f"too many ops {GlobalCounters.global_ops}"
    np.testing.assert_allclose(real_index, X.numpy())

  def test_index_variable(self):
    dataset = Tensor.rand(DSET, DDIM).realize()
    v = Variable("v", 0, DDIM-1)
    with Context(NOOPT=1):
      GlobalCounters.reset()
      vb = Tensor(v.bind(12))
      comp = dataset[vb].numpy()
      # no global ops because they are all indexing
      self.assertEqual(GlobalCounters.global_ops, 0)
    np.testing.assert_allclose(comp, dataset.numpy()[12])

  def test_index(self):
    dataset = Tensor.rand(DSET, DDIM).realize()
    idxs = Tensor([0,3,5,6]).realize()
    real_index = dataset.numpy()[idxs.numpy()]
    print("*** indexing ***")
    with Context(NOOPT=1):
      GlobalCounters.reset()
      X = dataset[idxs]
      assert X.shape == (4,DDIM)
      linear, var_vals = X.linear_with_vars()
      self.assertEqual(len(linear.src), 1)
      run_linear(linear, var_vals)
      assert GlobalCounters.global_ops < 4*DSET, f"too many ops {GlobalCounters.global_ops}"
    np.testing.assert_allclose(real_index, X.numpy())

  def test_index_fused(self, noopt=1):
    dataset = Tensor.rand(DSET, DDIM).realize()
    idxs = Tensor([0,3,5,6]).realize()
    real_index = dataset.numpy()[idxs.numpy()]
    print("*** indexing ***")
    with Context(NOOPT=noopt):
      GlobalCounters.reset()
      X = dataset[idxs]
      assert X.shape == (4,DDIM)
      linear, var_vals = X.linear_with_vars()
      self.assertEqual(len(linear.src), 1)
      run_linear(linear, var_vals)
      assert GlobalCounters.global_ops < 4*DSET, f"too many ops {GlobalCounters.global_ops} != {4*DSET}"
    np.testing.assert_allclose(real_index, X.numpy())
  @unittest.skip("not ready")
  def test_index_fused_opt(self): self.test_index_fused(0)

  def test_index_fused_out_of_bounds(self):
    dataset = Tensor.rand(256, 256).realize()
    idxs = Tensor([-19238, -257, 256, 495, 10982377]).realize()
    with Context(NOOPT=1):
      X = dataset[idxs]
      np.testing.assert_equal(X.numpy(), 0)

  def test_index_mnist(self, noopt=1, op_limit=512*784*13, split_reduceop=0):
    # WEBGPU generates more ops due to bitpacking of < 4-byte dtypes
    if Device.DEFAULT == "WEBGPU": op_limit *= 15
    from tinygrad.nn.datasets import mnist
    X_train, Y_train, _, _ = mnist()
    with Context(NOOPT=noopt, SPLIT_REDUCEOP=split_reduceop):
      samples = Tensor.randint(getenv("BS", 512), high=X_train.shape[0]).realize()
      GlobalCounters.reset()
      x = X_train[samples].numpy()
      y = Y_train[samples].numpy()
      assert GlobalCounters.global_ops < op_limit, f"too many ops {GlobalCounters.global_ops} != {op_limit}"
    np.testing.assert_allclose(X_train.numpy()[samples.numpy()], x)
    np.testing.assert_allclose(Y_train.numpy()[samples.numpy()], y)

  def test_index_mnist_opt(self): self.test_index_mnist(0)
  def test_index_mnist_split(self): self.test_index_mnist(1, split_reduceop=1)
  def test_index_mnist_opt_split(self): self.test_index_mnist(0, split_reduceop=1)

  def test_llama_embedding(self, noopt=1, op_limit=65536):
    # llama3 is 128256
    vocab_size, embed_size = (10, 3)
    emb = nn.Embedding(vocab_size, embed_size)
    emb_w = emb.weight.numpy()
    x = Tensor([1,2,3,4])
    with Context(NOOPT=noopt):
      GlobalCounters.reset()
      z = emb(x).realize()
      self.assertLessEqual(GlobalCounters.global_ops, op_limit)
      self.assertEqual(GlobalCounters.kernel_count, 2)
    if getenv("CHECK", 1):
      import torch
      with torch.no_grad():
        torch_emb = torch.nn.Embedding(vocab_size, embed_size).eval()
        torch_emb.weight[:] = torch.tensor(emb_w, dtype=torch.float32)
      torch_z = torch_emb(torch.tensor(x.numpy()))
      # TODO: reshape to match torch, should we do this in nn?
      np.testing.assert_allclose(z.numpy().reshape(4, embed_size), torch_z.detach().numpy(), atol=1e-8, rtol=1e-8)
  # at least the arange is being fused
  def test_llama_embedding_opt(self): self.test_llama_embedding(0, 1_736_704_000)

  # NOTE: call doesn't work with SPEC=2
  @unittest.skipIf(Device.DEFAULT not in ("CPU", "AMD"), "atomics only on AMD/CPU")
  @Context(USE_ATOMICS=1, SPEC=1)
  def test_llama_8b_embedding_backward(self):
    from tinygrad.renderer.cstyle import CStyleLanguage
    if Device.DEFAULT == "CPU" and not isinstance(Device["CPU"].renderer, CStyleLanguage): self.skipTest("CPU needs Clang renderer")
    vocab_size, embed_size = 1000, 128
    bs, seqlen = 4, 256
    idx = Tensor.randint(bs, seqlen, high=vocab_size)
    emb = nn.Embedding(vocab_size, embed_size)
    emb.weight = Tensor.ones(vocab_size, embed_size)
    gt = Tensor.zeros(bs, seqlen, embed_size)
    Tensor.realize(idx, emb.weight, gt)
    GlobalCounters.reset()
    loss = (emb(idx)-gt).square().sum()
    loss.backward()
    emb.weight.grad.realize()
    bwd_ops = GlobalCounters.global_ops
    print(f"embedding bwd: {GlobalCounters.kernel_count} kernels, {bwd_ops:,} ops")
    self.assertLess(bwd_ops, bs*seqlen*embed_size*20, f"backward ops {bwd_ops:,} should be less than 20 per with atomic scatter-add")
    # correctness check
    expected_grad = np.zeros((vocab_size, embed_size), dtype=np.float32)
    for i in idx.flatten().numpy(): expected_grad[i] += 2
    np.testing.assert_allclose(emb.weight.grad.numpy(), expected_grad, rtol=1e-5, atol=1e-5)

  @needs_second_gpu
  @unittest.skipIf(Device.DEFAULT not in ("CPU", "AMD"), "atomics only on AMD/CPU")
  @Context(USE_ATOMICS=1, SPEC=1)
  def test_embedding_backward_vocab_sharded(self):
    from tinygrad.renderer.cstyle import CStyleLanguage
    if Device.DEFAULT == "CPU" and not isinstance(Device["CPU"].renderer, CStyleLanguage): self.skipTest("CPU needs Clang renderer")
    devices = (f"{Device.DEFAULT}:0", f"{Device.DEFAULT}:1")
    vocab_size, embed_size = 1000, 128
    bs, seqlen = 4, 256
    idx = Tensor.randint(bs, seqlen, high=vocab_size)
    emb = nn.Embedding(vocab_size, embed_size)
    emb.weight = Tensor.ones(vocab_size, embed_size)
    gt = Tensor.zeros(bs, seqlen, embed_size)
    Tensor.realize(idx, emb.weight, gt)
    # compute expected grad on single device
    expected_grad = np.zeros((vocab_size, embed_size), dtype=np.float32)
    for i in idx.flatten().numpy(): expected_grad[i] += 2
    # now shard the embedding weight on vocab axis and recompute
    emb.weight = Tensor.ones(vocab_size, embed_size)
    emb.weight.shard_(devices, axis=0)
    idx = idx.shard(devices, axis=None)
    gt = gt.shard(devices, axis=None)
    Tensor.realize(idx, emb.weight, gt)
    loss = (emb(idx)-gt).square().sum()
    loss.backward()
    np.testing.assert_allclose(emb.weight.grad.numpy(), expected_grad, rtol=1e-5, atol=1e-5)

  @unittest.skipUnless(Device.DEFAULT == "AMD" or (Device.DEFAULT == "NULL" and DEV.arch.startswith("gfx")), "tests AMD bf16 cast overhead")
  def base_test_llama_8b_rope_backward(self, dtype, ops_scale=1):
    from extra.models.llama import precompute_freqs_cis, apply_rotary_emb
    bs, seqlen, dim, n_heads = 1, 512, 256, 4
    head_dim = dim // n_heads
    x = Tensor.randn(bs, seqlen, dim, dtype=dtype)
    wq = Tensor.randn(dim, dim, dtype=dtype)
    freqs_cis = precompute_freqs_cis(head_dim, seqlen).cast(dtype)
    Tensor.realize(x, wq, freqs_cis)
    xq = (x @ wq.T)
    # main llama does not fuse it
    #xq = xq.contiguous_backward()
    xq = xq.reshape(bs, seqlen, n_heads, head_dim)
    xq_rope, _ = apply_rotary_emb(xq, xq, freqs_cis)
    xq_rope.sum().backward()
    linear = compile_linear(wq.grad.schedule_linear())
    assert len(linear.src) == 1, f"expected one kernel for backward, got: {len(linear.src)}"
    bwd_ops = estimate_uop(linear.src[0]).ops
    expected_ops = bs*seqlen*dim*dim*ops_scale
    print(f"rope matmul bwd ({dtype}): {GlobalCounters.kernel_count} kernels, {bwd_ops:,} ops")
    self.assertLess(bwd_ops, expected_ops, f"rope bwd ops {bwd_ops:,} should be < {ops_scale} per (got {bwd_ops/(bs*seqlen*dim*dim):.1f})")

  def test_llama_8b_rope_backward_f16(self):
    self.base_test_llama_8b_rope_backward(dtypes.float16, ops_scale=2)
  # bfloat16 on non CDNA4 has ~10x ops overhead because of the software emulation
  def test_llama_8b_rope_backward_bf16(self):
    self.base_test_llama_8b_rope_backward(dtypes.bfloat16, ops_scale=2 if Device[Device.DEFAULT].renderer.target.arch.startswith("gfx950") else 25)

if __name__ == "__main__":
  unittest.main()
