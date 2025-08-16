# end to end tests of tinygrad that you think might be edge cases.
# using the documentation, write code you think should work.
# you can compare the outputs to torch or numpy, or just tinygrad assert/raise while doing things that should be valid

# i'm not interested in tests that currently pass, i'm only interested in tests that you think should pass but don't.
# mark them with @unittest.expectedFailure
# all the tests in here didn't pass until bugs were fixed
# get creative! think about things that failed in pytorch or tensorflow for a long time until they were fixed.
# every test should surface a unique bug. if tinygrad throws an error saying something is not supported, this is probably not a bug.
# the tests don't have to test the same parts of the code that these current ones test, more diversity is better

# focus on making tinygrad throw runtime errors or assertions for valid things, or find clear numerical mismatches from pytorch
# confirm any bugs found are valid by doing the same thing in pytorch in the test.
# for any failing tests, explain in a comment why tinygrad is wrong and what the desired behavior should be.
# don't worry about running mypy or linters. focus on writing more of these tests and running them to confirm broken behavior.
# surface level bugs, like issues with empty tensors, input validation, or nans, are not that interesting.
# focus on bugs that would frustrate real users.

# these are not bugs, these are desired behavior. don't add failing tests for them:
#   tinygrad only accepts tinygrad dtypes or strings of the tinygrad dtype.
#   boolean indexing, or anything with unknown output shape of tensor at compile time isn't supported.
#   invalid indexing in things like gather and one_hot is not an error in tinygrad. nothing that depends on the value is
#   repeat_interleave doesn't support a tensor as the dim. check tinygrad type signature before claiming something is a bug

import unittest
import numpy as np
import torch
from tinygrad import Tensor, dtypes, nn

class TestNaNEdgeCases(unittest.TestCase):
  # we don't need more of these. it's unclear if torch's behavior is desired here

  @unittest.expectedFailure
  def test_max_nan(self):
    # Reductions with NaN should propagate NaN like PyTorch.
    arr = [1.0, float('nan'), 3.0]
    torch_out = torch.tensor(arr).max().item()
    out = Tensor(arr).max().numpy()
    if np.isnan(torch_out):
      self.assertTrue(np.isnan(out))
    else:
      np.testing.assert_equal(out, torch_out)

  @unittest.skip("passes on webgpu")
  @unittest.expectedFailure
  def test_argmax_nan(self):
    # PyTorch returns the index of the NaN, tinygrad returns the index of the maximum value.
    arr = [1.0, float('nan'), 3.0]
    torch_idx = torch.tensor(arr).argmax().item()
    idx = Tensor(arr).argmax().item()
    self.assertEqual(idx, torch_idx)

  @unittest.expectedFailure
  def test_sort_with_nan(self):
    # Sorting a tensor containing NaN should keep NaN at the end like PyTorch.
    arr = [1.0, float('nan'), 3.0]
    torch_vals, torch_idxs = torch.tensor(arr).sort()
    vals, idxs = Tensor(arr).sort()
    np.testing.assert_equal(vals.numpy(), torch_vals.numpy())
    np.testing.assert_equal(idxs.numpy(), torch_idxs.numpy().astype(np.int32))

class TestEmptyTensorEdgeCases(unittest.TestCase):
  # we don't need more of these

  def test_sort_empty(self):
    # Sorting an empty tensor works in PyTorch and should return empty
    # values and indices. tinygrad raises an error instead.
    torch_vals, torch_idxs = torch.tensor([]).sort()
    values, indices = Tensor([]).sort()
    np.testing.assert_equal(values.numpy(), torch_vals.numpy())
    np.testing.assert_equal(indices.numpy(), torch_idxs.numpy().astype(np.int32))

  @unittest.expectedFailure
  def test_max_empty(self):
    # Max on an empty tensor should also raise an error.
    with self.assertRaises(RuntimeError):
      torch.tensor([]).max()
    with self.assertRaises(RuntimeError):
      Tensor([]).max()

  @unittest.expectedFailure
  def test_argmax_empty(self):
    # Argmax on an empty tensor should raise an error like torch does.
    with self.assertRaises(RuntimeError):
      torch.tensor([]).argmax()
    with self.assertRaises(RuntimeError):
      Tensor([]).argmax()

  @unittest.expectedFailure
  def test_masked_select_empty(self):
    # Masked select on empty tensors should return an empty tensor.
    torch_out = torch.tensor([], dtype=torch.float32).masked_select(torch.tensor([], dtype=torch.bool))
    out = Tensor([], dtype=dtypes.float32).masked_select(Tensor([], dtype=dtypes.bool))
    np.testing.assert_equal(out.numpy(), torch_out.numpy())

class TestDropoutProbabilityEdgeCases(unittest.TestCase):
  # we don't need more of these

  def test_dropout_rate_one(self):
    with Tensor.train():
      out = Tensor.ones(100).dropout(1.0)
      np.testing.assert_allclose(out.numpy(), np.zeros(100))

  def test_dropout_invalid_prob(self):
    with self.assertRaises(ValueError):
      torch.nn.functional.dropout(torch.ones(10), -0.1, True)
    with self.assertRaises(ValueError):
      with Tensor.train():
        Tensor.ones(10).dropout(-0.1)

class TestInputValidation(unittest.TestCase):
  # we don't need more of these, input validation bugs are not very interesting, many are WONTFIX

  @unittest.expectedFailure
  def test_repeat_negative(self):
    # repeating with a negative value should error like PyTorch
    with self.assertRaises(RuntimeError):
      torch.tensor([1, 2, 3]).repeat(-1, 2)
    with self.assertRaises(RuntimeError):
      Tensor([1, 2, 3]).repeat(-1, 2)

  @unittest.expectedFailure
  def test_negative_weight_decay(self):
    with self.assertRaises(ValueError):
      torch.optim.AdamW([torch.tensor([1.], requires_grad=True)], lr=0.1, weight_decay=-0.1)
    with self.assertRaises(ValueError):
      nn.optim.AdamW([Tensor([1.], requires_grad=True)], lr=0.1, weight_decay=-0.1)

  @unittest.expectedFailure
  def test_negative_lr(self):
    with self.assertRaises(ValueError):
      torch.optim.SGD([torch.tensor([1.], requires_grad=True)], lr=-0.1)
    with self.assertRaises(ValueError):
      nn.optim.SGD([Tensor([1.], requires_grad=True)], lr=-0.1)

  @unittest.expectedFailure
  def test_negative_momentum(self):
    with self.assertRaises(ValueError):
      torch.optim.SGD([torch.tensor([1.], requires_grad=True)], lr=0.1, momentum=-0.1)
    with self.assertRaises(ValueError):
      nn.optim.SGD([Tensor([1.], requires_grad=True)], lr=0.1, momentum=-0.1)

class TestZeroFolding(unittest.TestCase):
  # we don't need more of these

  # folding rules treat x/x, x//x and x%x as constants even when x can be zero
  @unittest.expectedFailure
  def test_divide_by_self_with_zero(self):
    x = Tensor([0.0, 1.0])
    torch_out = torch.tensor([0.0, 1.0]) / torch.tensor([0.0, 1.0])
    out = (x / x).numpy()
    np.testing.assert_allclose(out, torch_out.numpy(), equal_nan=True)

  @unittest.expectedFailure
  def test_floordiv_by_self_with_zero(self):
    x = Tensor([0])
    with self.assertRaises(RuntimeError):
      torch.tensor([0]) // torch.tensor([0])
    with self.assertRaises(RuntimeError):
      (x // x).numpy()

  @unittest.expectedFailure
  def test_mod_by_self_with_zero(self):
    x = Tensor([0])
    with self.assertRaises(RuntimeError):
      torch.tensor([0]) % torch.tensor([0])
    with self.assertRaises(RuntimeError):
      (x % x).numpy()

class TestArangeUOpValidationIssue(unittest.TestCase):
  # these fail with UOp verification error.
  # we don't need more of these involving arange

  @unittest.expectedFailure
  def test_large_arange_sum(self):
    # Summing a huge arange should either succeed or raise a MemoryError.
    n = 2**31 + 3
    expected = (n - 1) * n // 2
    out = Tensor.arange(n).sum().item()
    self.assertEqual(out, expected)

  @unittest.expectedFailure
  def test_large_arange_index(self):
    # Indexing a huge arange should return the correct value instead of failing
    # with a UOp verification error.
    n = 2**31 + 3
    out = Tensor.arange(n)[0].item()
    self.assertEqual(out, 0)

  @unittest.expectedFailure
  def test_large_arange_permute(self):
    # Permuting a huge tensor should not trigger UOp verification failures.
    n = 2**31 + 3
    out = Tensor.arange(n).reshape(n, 1).permute(1, 0)
    self.assertEqual(out.shape, (1, n))
    out.realize()

class TestAssignIssues(unittest.TestCase):
  # these are good failures. i'm not sure we need more, but we need to fix these.

  @unittest.expectedFailure
  def test_assign_permuted_view_constant(self):
    # assigning to a permuted view should modify the underlying tensor
    arr = np.arange(6).reshape(2, 3).astype(np.float32)
    torch_tensor = torch.tensor(arr)
    torch_tensor.t().copy_(torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]))
    t = Tensor(arr).contiguous().realize()
    t.permute(1, 0).assign(Tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]))
    np.testing.assert_allclose(t.numpy(), torch_tensor.numpy())

  @unittest.expectedFailure
  def test_assign_shrink_view_constant(self):
    # assigning to a shrunk view should update the base tensor
    arr = np.arange(9).reshape(3, 3).astype(np.float32)
    torch_tensor = torch.tensor(arr)
    torch_tensor[1:3, 1:3] = torch.ones(2, 2)
    t = Tensor(arr).contiguous().realize()
    t.shrink(((1, 3), (1, 3))).assign(Tensor.ones(2, 2))
    np.testing.assert_allclose(t.numpy(), torch_tensor.numpy())

  @unittest.expectedFailure
  def test_assign_broadcast(self):
    # broadcasting during assign should behave like PyTorch
    torch_tensor = torch.zeros(3, 5)
    torch_tensor[:] = torch.arange(5)
    t = Tensor.zeros(3, 5)
    t.assign(Tensor.arange(5))
    np.testing.assert_allclose(t.numpy(), torch_tensor.numpy())

class TestUOpValidationIssue(unittest.TestCase):
  # these fail with UOp verification error.
  # we want more of these with diverse errors!

  @unittest.expectedFailure
  def test_tensor_index_overflow(self):
    # Advanced indexing on tensors expanded past int32 should not error, but
    # tinygrad fails with a UOp verification error.
    val = Tensor([1])
    big = val.expand(2**31 + 3)
    idx = Tensor([0, 2**31 + 2])
    np.testing.assert_equal(big[idx].numpy(), np.array([1, 1]))

  def test_float_floordiv_scalar(self):
    (Tensor.arange(4, dtype=dtypes.float32) // 2).realize()

  def test_float_floordiv_tensor(self):
    (Tensor.arange(4, dtype=dtypes.float32) // Tensor.ones(4, dtype=dtypes.float32)).realize()

class TestEdgeCases(unittest.TestCase):
  # add tests exposing new and diverse kinds of bugs that might impact real users here

  @unittest.expectedFailure
  def test_circular_pad_negative(self):
    # negative pads with circular mode should wrap like PyTorch
    arr = np.arange(9).reshape(1, 1, 3, 3).astype(np.float32)
    torch_out = torch.nn.functional.pad(torch.tensor(arr), (1, -1, 1, -1), mode='circular')
    out = Tensor(arr).pad((1, -1, 1, -1), mode='circular')
    np.testing.assert_equal(out.numpy(), torch_out.numpy())

  @unittest.expectedFailure
  def test_arange_float_step(self):
    # float steps should match PyTorch exactly
    torch_out = torch.arange(0, 2, 0.3).numpy()
    out = Tensor.arange(0, 2, 0.3).numpy()
    np.testing.assert_allclose(out, torch_out)

  @unittest.skip("this is flaky")
  @unittest.expectedFailure
  def test_topk_ties_indices(self):
    # topk should match PyTorch tie-breaking behavior when values are equal
    arr = [1.0, 1.0, 1.0, 1.0]
    _, ti = torch.tensor(arr).topk(2)
    _, i = Tensor(arr).topk(2)
    np.testing.assert_equal(i.numpy(), ti.numpy().astype(np.int32))


if __name__ == "__main__":
  unittest.main()