from __future__ import annotations
import unittest
from math import prod

from hypothesis import assume, given, settings, strategies as st
from hypothesis.extra import numpy as stn

import numpy as np
import torch
from tinygrad import Tensor
from tinygrad.helpers import CI, getenv


settings.register_profile(__file__, settings.default,
                          max_examples=100 if CI else 250, deadline=None, derandomize=getenv("DERANDOMIZE_CI", False))


# torch wraparound for large numbers
st_int32 = st.integers(-2147483648, 2147483647)

@st.composite
def st_shape(draw) -> tuple[int, ...]:
  s = draw(stn.array_shapes(min_dims=0, max_dims=6,
                            min_side=0, max_side=128))
  assume(prod(s) <= 1024 ** 2)
  assume(prod([d for d in s if d]) <= 1024 ** 4)
  return s

def tensors_for_shape(s:tuple[int, ...]) -> tuple[torch.tensor, Tensor]:
  x = np.arange(prod(s)).reshape(s)
  return torch.from_numpy(x), Tensor(x)

def apply(tor, ten, tor_fn, ten_fn=None):
  ok = True
  try: tor = tor_fn(tor)
  except: tor, ok = None, not ok  # noqa: E722
  try: ten = ten_fn(ten) if ten_fn is not None else tor_fn(ten)
  except: ten, ok = None, not ok  # noqa: E722
  return tor, ten, ok

class TestShapeOps(unittest.TestCase):
  @settings.get_profile(__file__)
  @given(st_shape(), st_int32, st.one_of(st_int32, st.lists(st_int32)))
  def test_split(self, s:tuple[int, ...], dim:int, sizes:int|list[int]):
    tor, ten = tensors_for_shape(s)
    tor, ten, ok = apply(tor, ten, lambda t: t.split(sizes, dim))
    assert ok
    if tor is None and ten is None: return

    assert len(tor) == len(ten)
    assert all([np.array_equal(tor.numpy(), ten.numpy()) for (tor, ten) in zip(tor, ten)])

  @settings.get_profile(__file__)
  @given(st_shape(), st_int32, st_int32)
  def test_chunk(self, s:tuple[int, ...], dim:int, num:int):
    # chunking on a 0 dim is cloning and leads to OOM if done unbounded.
    assume((0 <= (actual_dim := len(s)-dim if dim < 0 else dim) < len(s) and s[actual_dim] > 0) or
           (num < 16))

    tor, ten = tensors_for_shape(s)
    tor, ten, ok = apply(tor, ten, lambda t: t.chunk(num, dim))
    assert ok
    if tor is None and ten is None: return

    assert len(tor) == len(ten)
    assert all([np.array_equal(tor.numpy(), ten.numpy()) for (tor, ten) in zip(tor, ten)])

  @settings.get_profile(__file__)
  @given(st_shape(), st_int32)
  def test_squeeze(self, s:tuple[int, ...], dim:int):
    tor, ten = tensors_for_shape(s)
    tor, ten, ok = apply(tor, ten, lambda t: t.squeeze(dim))
    assert ok
    if tor is None and ten is None: return
    assert np.array_equal(tor.numpy(), ten.numpy())

  @settings.get_profile(__file__)
  @given(st_shape(), st_int32)
  def test_unsqueeze(self, s:tuple[int, ...], dim:int):
    tor, ten = tensors_for_shape(s)
    tor, ten, ok = apply(tor, ten, lambda t: t.unsqueeze(dim))
    assert ok
    if tor is None and ten is None: return
    assert np.array_equal(tor.numpy(), ten.numpy())

if __name__ == '__main__':
  unittest.main()
