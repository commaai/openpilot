# modified from
# https://github.com/arogozhnikov/einops/blob/master/tests/test_examples.py
# https://github.com/arogozhnikov/einops/blob/master/tests/test_ops.py
# https://github.com/arogozhnikov/einops/blob/master/tests/test_parsing.py

import numpy as np
import unittest
from tinygrad import Tensor


class test_rearrange_examples(unittest.TestCase):
  def test1(self):
    # transpose
    x = Tensor(np.arange(10 * 20 * 30 * 40, dtype=np.int32).reshape([10, 20, 30, 40]))
    y = x.rearrange("b c h w -> b h w c")
    assert tuple(y.shape) == (10, 30, 40, 20)

  def test2(self):
    # view / reshape
    x = Tensor(np.arange(10 * 20 * 30 * 40, dtype=np.int32).reshape([10, 20, 30, 40]))
    y = x.rearrange("b c h w -> b (c h w)")
    assert tuple(y.shape) == (10, 20 * 30 * 40)

  def test3(self):
    # depth-to-space
    x = Tensor(np.arange(10 * 20 * 30 * 40, dtype=np.int32).reshape([10, 20, 30, 40]))
    y = x.rearrange("b (c h1 w1) h w -> b c (h h1) (w w1)", h1=2, w1=2)
    assert tuple(y.shape) == (10, 5, 30 * 2, 40 * 2)

  def test4(self):
    # space-to-depth
    x = Tensor(np.arange(10 * 20 * 30 * 40, dtype=np.int32).reshape([10, 20, 30, 40]))
    y = x.rearrange("b c (h h1) (w w1) -> b (h1 w1 c) h w", h1=2, w1=2)
    assert tuple(y.shape) == (10, 20 * 4, 30 // 2, 40 // 2)

  def test5(self):
    # simple transposition
    x = Tensor(np.arange(10 * 20 * 30 * 40, dtype=np.int32).reshape([10, 20, 30, 40]))
    y = x.rearrange("b1 sound b2 letter -> b1 b2 sound letter")
    assert tuple(y.shape) == (10, 30, 20, 40)

  def test6(self):
    # parsing parameters
    x = Tensor(np.arange(10 * 20 * 30 * 40, dtype=np.int32).reshape([10, 20, 30, 40]))
    t = x.rearrange("b c h w -> (b h w) c")
    t = t[:, ::2]  # replacement for dot-product, just changes size of second axis
    assert tuple(t.shape) == (10 * 30 * 40, 10)

  def test7(self):
    x = Tensor(np.arange(10 * 20 * 30 * 40, dtype=np.int32).reshape([10, 20, 30, 40]))
    # split of embedding into groups
    y1, y2 = x.rearrange("b (c g) h w -> g b c h w", g=2)
    assert tuple(y1.shape) == (10, 10, 30, 40)
    assert tuple(y2.shape) == (10, 10, 30, 40)

  def test8(self):
    x = Tensor(np.arange(10 * 20 * 1 * 1, dtype=np.int32).reshape([10, 20, 1, 1]))
    # squeeze - unsqueeze
    y = x.rearrange("b c () () -> b c")
    assert tuple(y.shape) == (10, 20)
    y = y.rearrange("b c -> c b () ()")
    assert tuple(y.shape) == (20, 10, 1, 1)

  def test9(self):
    x = Tensor(np.arange(10 * 20 * 1 * 1, dtype=np.int32).reshape([10, 20, 1, 1]))
    # squeeze - unsqueeze
    y = x.rearrange("b c 1 1 -> b c")
    assert tuple(y.shape) == (10, 20)
    y = y.rearrange("b1 c -> c b1 1 1")
    assert tuple(y.shape) == (20, 10, 1, 1)


class test_rearrange_ops(unittest.TestCase):
  def test_rearrange_errors(self):
    x = Tensor.zeros([1, 1, 1, 1, 1])
    x.rearrange("a b c d ... ->  a b c ... d")
    bad_patterns = [
      "a b c d (...) ->  a b c ... d",  # collapsed ellipsis on input
      "a b (c d ... ->  a b c ... d",   # unbalanced brackets
      "a b* c d ... ->  a b c ... d",   # not alphanumeric
      "a b c d ->  a b c d -> a b c d", # two "->"
      "a ... c ... ->  ... a ... c",    # two "..."
      "a b c d e -> f b c d e",         # name mismatch
    ]
    for pattern in bad_patterns:
      with self.assertRaises(AssertionError):
        x.rearrange(pattern)

    x.rearrange("... ->  (...)")
    with self.assertRaises(AssertionError):
      x.rearrange("(...) -> (...)")

    y = Tensor.zeros([8, 1])
    y.rearrange("(a1 a2 a3) b -> b a3 a2 a1", a1=2, a2=2)
    with self.assertRaises(RuntimeError):
      ## should fail as not enough dimensions specified
      y.rearrange("(a1 a2 a3) b -> b a3 a2 a1", a1=2)
    with self.assertRaises(ValueError):
      ## should fail as 6 does not divide 8
      y.rearrange("(a1 a2 a3) b -> b a3 a2 a1", a1=3, a2=2)
    with self.assertRaises(AssertionError):
      ## incorrect dimension provided for an axis that is only permuted
      y.rearrange("(a1 a2 a3) b -> b a3 a2 a1", a1=2, a2=2, b=2)
    with self.assertRaises(AssertionError):
      ## unused axis provided
      y.rearrange("(a b c) d -> a b c d", b=2, c=2, e=2)


class test_rearrange_parsing(unittest.TestCase):
  def test_elementary_axis_name(self):
    for name in [
      "a",
      "b",
      "h",
      "dx",
      "h1",
      "zz",
      "i9123",
      "somelongname",
      "Alex",
      "camelCase",
      "u_n_d_e_r_score",
      "unreasonablyLongAxisName",
    ]:
      Tensor.ones((1,)).rearrange(f"{name} -> {name}")

    for name in ["2b", "12", "_startWithUnderscore", "endWithUnderscore_", "_"]:
      with self.assertRaises(AssertionError):
        Tensor.ones((1,)).rearrange(f"{name} -> {name}")

    with self.assertRaises(RuntimeError):
      Tensor.ones((1,)).rearrange(" -> ")

  def test_invalid_expressions(self):
    # double ellipsis should raise an error
    def _test_expression(expression: str):
      Tensor.ones((2, 3, 4, 5, 6)).rearrange(f"{expression} -> {expression}")

    _test_expression("... a b c d")
    with self.assertRaises(AssertionError):
      _test_expression("... a b c d ...")
    with self.assertRaises(AssertionError):
      _test_expression("... a b c (d ...)")
    with self.assertRaises(AssertionError):
      _test_expression("(... a) b c (d ...)")

    # double/missing/enclosed parenthesis
    Tensor.ones((2, 3, 4, 5, 6)).rearrange("a b c d ... -> (a) b c (d ...)")
    with self.assertRaises(AssertionError):
      _test_expression("(a)) b c (d ...)")
    with self.assertRaises(AssertionError):
      _test_expression("(a b c (d ...)")
    with self.assertRaises(AssertionError):
      _test_expression("(a) (()) b c (d ...)")
    with self.assertRaises(AssertionError):
      _test_expression("(a) ((b c) (d ...))")

    # invalid identifiers
    _test_expression("camelCase under_scored cApiTaLs ß ...")
    with self.assertRaises(AssertionError):
      _test_expression("1a")
    with self.assertRaises(AssertionError):
      _test_expression("_pre")
    with self.assertRaises(AssertionError):
      _test_expression("...pre")
    with self.assertRaises(AssertionError):
      _test_expression("pre...")


if __name__ == "__main__":
  unittest.main()
