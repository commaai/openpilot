import unittest
import numpy as np
from tinygrad import Tensor, GlobalCounters, dtypes, nn, Device, Variable
from tinygrad.helpers import Context, getenv
from tinygrad.engine.realize import run_schedule
from tinygrad.engine.realize import CompiledRunner, get_program
from tinygrad.engine.schedule import ExecItem
from tinygrad.uop.ops import Ops
from tinygrad.renderer import Estimates
from tinygrad.renderer.ptx import PTXRenderer

class TestArange(unittest.TestCase):
  def _get_flops(self, tensor, desired):
    GlobalCounters.reset()
    sched = tensor.schedule()
    self.assertEqual(len(sched), 1)
    p = get_program(sched[-1].ast, renderer=Device[Device.DEFAULT].renderer)
    ExecItem(sched[-1].ast, [tensor.uop.buffer], prg=CompiledRunner(p)).run()
    np.testing.assert_equal(tensor.numpy(), desired)
    return p.estimates.ops

  def test_arange_complexity(self):
    self.assertEqual(self._get_flops(Tensor.arange(256), np.arange(256)), 0)
    self.assertEqual(self._get_flops(Tensor.arange(2560), np.arange(2560)), 0)

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
      sched = t.triu().schedule()
      p = get_program(sched[-1].ast, renderer=Device[Device.DEFAULT].renderer)
      self.assertLessEqual(Estimates.from_uops(p.uops).ops, 4 * 256 * 256)

DSET, DDIM = 2048, 32

class TestIndexing(unittest.TestCase):
  def test_arange_2_reduce(self):
    needle = Tensor.zeros(16384, dtype=dtypes.int).contiguous()
    needle[1337] = 1
    needle.realize()
    with Context(NOOPT=1):
      GlobalCounters.reset()
      out = ((Tensor.arange(1,16385)-1)*needle).sum()
      sched = out.schedule()
      self.assertEqual(len(sched), 1)
      run_schedule(sched)
    self.assertEqual(out.item(), 1337)

  def test_manual_index(self):
    dataset = Tensor.rand(DSET, DDIM).realize()
    idxs = Tensor([0,3,5,6]).realize()
    real_index = dataset.numpy()[idxs.numpy()]
    print("*** indexing ***")
    with Context(NOOPT=1):
      GlobalCounters.reset()
      rng = Tensor.ones(4, DDIM, DSET, dtype=dtypes.int)._cumalu(axis=-1, op=Ops.ADD, _include_initial=True).reshape(4, DDIM, DSET, 1)
      idxs = idxs.reshape(4,1,1,1).expand(4, DDIM, DSET, 1)
      reshape_dataset = dataset.T.reshape(1, DDIM, DSET, 1).expand(4, DDIM, DSET, 1)
      full = (rng==idxs).where(reshape_dataset, Tensor.zeros(4, DDIM, DSET, 1))
      X = full.sum(axis=(2,3))
      sched = X.schedule()
      self.assertEqual(len(sched), 1)
      run_schedule(sched)
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
      sched = X.schedule()
      self.assertEqual(len(sched), 1)
      run_schedule(sched)
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
      sched = X.schedule()
      self.assertEqual(len(sched), 1)
      run_schedule(sched)
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
    emb.weight = Tensor.ones(vocab_size, embed_size, requires_grad=True)
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

if __name__ == "__main__":
  unittest.main()
