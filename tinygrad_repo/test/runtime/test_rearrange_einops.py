# modified from
# https://github.com/arogozhnikov/einops/blob/master/tests/test_examples.py
# https://github.com/arogozhnikov/einops/blob/master/tests/test_ops.py
# https://github.com/arogozhnikov/einops/blob/master/tests/test_parsing.py

import numpy as np
import unittest
from tinygrad import Tensor


class test_rearrange_examples(unittest.TestCase):
  def test_tensor_train_example_numpy(self):
    # kept here just for a collection, only tested for numpy
    # https://arxiv.org/pdf/1509.06569.pdf, (5)
    x = Tensor.ones([3, 4, 5, 6])
    rank = 4

    # creating appropriate Gs
    Gs = [Tensor.ones([d, d, rank, rank]) for d in x.shape]
    Gs[0] = Gs[0][:, :, :1, :]
    Gs[-1] = Gs[-1][:, :, :, :1]

    # einsum way
    y = x.reshape((1,) + x.shape)
    for G in Gs:
      # taking partial results left-to-right
      # y = numpy.einsum('i j alpha beta, alpha i ...  -> beta ... j', G, y)
      y = Tensor(np.einsum("i j a b, a i ...  -> b ... j", G.numpy(), y.numpy()))
    y1 = y.reshape(-1)

    # alternative way
    y = x.reshape(-1)
    for G in Gs:
      i, j, alpha, beta = G.shape
      y = y.rearrange("(i rest alpha) -> rest (alpha i)", alpha=alpha, i=i)
      y = y @ G.rearrange("i j alpha beta -> (alpha i) (j beta)")
      y = y.rearrange("rest (beta j) -> (beta rest j)", beta=beta, j=j)
    y2 = y
    assert np.allclose(y1.numpy(), y2.numpy())

    # yet another way
    y = x
    for G in Gs:
      i, j, alpha, beta = G.shape
      y = y.rearrange("i ... (j alpha) -> ... j (alpha i)", alpha=alpha, i=i)
      y = y @ G.rearrange("i j alpha beta -> (alpha i) (j beta)")
    y3 = y.reshape(-1)
    assert np.allclose(y1.numpy(), y3.numpy())


class test_rearrange_ops(unittest.TestCase):
  def test_rearrange_ellipsis_ops(self):
    identity_patterns = [
      "...->...",
      "a b c d e-> a b c d e",
      "a b c d e ...-> ... a b c d e",
      "a b c d e ...-> a ... b c d e",
      "... a b c d e -> ... a b c d e",
      "a ... e-> a ... e",
      "a ... -> a ... ",
      "a ... c d e -> a (...) c d e",
    ]

    equivalent_rearrange_patterns = [
      ("a b c d e -> (a b) c d e", "a b ... -> (a b) ... "),
      ("a b c d e -> a b (c d) e", "... c d e -> ... (c d) e"),
      ("a b c d e -> a b c d e", "... -> ... "),
      ("a b c d e -> (a b c d e)", "... ->  (...)"),
      ("a b c d e -> b (c d e) a", "a b ... -> b (...) a"),
      ("a b c d e -> b (a c d) e", "a b ... e -> b (a ...) e"),
    ]

    xnp = np.arange(2 * 3 * 4 * 5 * 6, dtype=np.int32).reshape([2, 3, 4, 5, 6])
    x = Tensor(xnp)
    for pattern in identity_patterns:
      assert np.array_equal(xnp, x.rearrange(pattern).numpy()), pattern

    for pattern1, pattern2 in equivalent_rearrange_patterns:
      assert np.array_equal(x.rearrange(pattern1).numpy(), x.rearrange(pattern2).numpy())

  def test_rearrange_consistency(self):
    shape = [1, 2, 3, 5, 7, 11]
    xnp = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    x = Tensor(xnp)
    for pattern in [
      "a b c d e f -> a b c d e f",
      "b a c d e f -> a b d e f c",
      "a b c d e f -> f e d c b a",
      "a b c d e f -> (f e) d (c b a)",
      "a b c d e f -> (f e d c b a)",
    ]:
      result = x.rearrange(pattern).numpy()
      assert len(np.setdiff1d(xnp, result)) == 0
      assert result.dtype == xnp.dtype

    result = x.rearrange("a b c d e f -> a (b) (c d e) f").numpy()
    assert np.array_equal(xnp.flatten(), result.flatten())

    result = x.rearrange("a aa aa1 a1a1 aaaa a11 -> a aa aa1 a1a1 aaaa a11").numpy()
    assert np.array_equal(xnp, result)

    result1 = x.rearrange("a b c d e f -> f e d c b a").numpy()
    result2 = x.rearrange("f e d c b a -> a b c d e f").numpy()
    assert np.array_equal(result1, result2)

    result = x.rearrange("a b c d e f -> (f d) c (e b) a").rearrange("(f d) c (e b) a -> a b c d e f", b=2, d=5).numpy()
    assert np.array_equal(xnp, result)

    sizes = dict(zip("abcdef", shape))
    temp = x.rearrange("a b c d e f -> (f d) c (e b) a", **sizes)
    result = temp.rearrange("(f d) c (e b) a -> a b c d e f", **sizes).numpy()
    assert np.array_equal(xnp, result)

    x2 = np.arange(2 * 3 * 4, dtype=np.int32).reshape([2, 3, 4])
    result = Tensor(x2).rearrange("a b c -> b c a").numpy()
    assert x2[1, 2, 3] == result[2, 3, 1]
    assert x2[0, 1, 2] == result[1, 2, 0]

  def test_rearrange_permutations(self):
    # tests random permutation of axes against two independent numpy ways
    for n_axes in range(1, 10):
      x = np.arange(2**n_axes, dtype=np.int32).reshape([2] * n_axes)
      permutation = np.random.permutation(n_axes)
      left_expression = " ".join("i" + str(axis) for axis in range(n_axes))
      right_expression = " ".join("i" + str(axis) for axis in permutation)
      expression = left_expression + " -> " + right_expression
      result = Tensor(x).rearrange(expression).numpy()

      for pick in np.random.randint(0, 2, [10, n_axes]):
        assert x[tuple(pick)] == result[tuple(pick[permutation])]

    for n_axes in range(1, 10):
      x = np.arange(2**n_axes, dtype=np.int32).reshape([2] * n_axes)
      permutation = np.random.permutation(n_axes)
      left_expression = " ".join("i" + str(axis) for axis in range(n_axes)[::-1])
      right_expression = " ".join("i" + str(axis) for axis in permutation[::-1])
      expression = left_expression + " -> " + right_expression
      result = Tensor(x).rearrange(expression).numpy()
      assert result.shape == x.shape
      expected_result = np.zeros_like(x)
      for original_axis, result_axis in enumerate(permutation):
        expected_result |= ((x >> original_axis) & 1) << result_axis

      assert np.array_equal(result, expected_result)


class test_rearrange_parsing(unittest.TestCase):
  def test_unicode_ellipsis(self):
    equivalent_rearrange_patterns = [
      ("a b … -> (a b) … ", "a b ... -> (a b) ... "),
      ("… c d e -> … (c d) e", "... c d e -> ... (c d) e"),
      ("… -> … ", "... -> ... "),
      ("… ->  (…)", "... ->  (...)"),
      ("a b … -> b (…) a", "a b ... -> b (...) a"),
      ("a b … e -> b (a …) e", "a b ... e -> b (a ...) e"),
    ]

    xnp = np.arange(2 * 3 * 4 * 5 * 6, dtype=np.int32).reshape([2, 3, 4, 5, 6])
    x = Tensor(xnp)

    for pattern1, pattern2 in equivalent_rearrange_patterns:
      assert np.array_equal(x.rearrange(pattern1).numpy(), x.rearrange(pattern2).numpy())


if __name__ == "__main__":
  unittest.main()
