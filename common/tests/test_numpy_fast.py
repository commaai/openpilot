import numpy as np
import unittest

from common.numpy_fast import clip, interp


class ClipTest(unittest.TestCase):
  def test_clip_too_low(self):
    hi = 9
    lo = 5
    x = 3
    assert clip(x, lo, hi) == 5

  def test_clip_between(self):
    hi = 9
    lo = 5
    x = 7
    assert clip(x, lo, hi) == 7

  def test_clip_too_high(self):
    hi = 9
    lo = 5
    x = 11
    assert clip(x, lo, hi) == 9


class InterpTest(unittest.TestCase):
  def test_correctness_controls(self):
    _A_CRUISE_MIN_BP = np.asarray([0., 5., 10., 20., 40.])
    _A_CRUISE_MIN_V = np.asarray([-1.0, -.8, -.67, -.5, -.30])
    v_ego_arr = [-1, -1e-12, 0, 4, 5, 6, 7, 10, 11, 15.2, 20, 21, 39,
                 39.999999, 40, 41]

    expected = np.interp(v_ego_arr, _A_CRUISE_MIN_BP, _A_CRUISE_MIN_V)
    actual = interp(v_ego_arr, _A_CRUISE_MIN_BP, _A_CRUISE_MIN_V)

    np.testing.assert_equal(actual, expected)

    for v_ego in v_ego_arr:
      expected = np.interp(v_ego, _A_CRUISE_MIN_BP, _A_CRUISE_MIN_V)
      actual = interp(v_ego, _A_CRUISE_MIN_BP, _A_CRUISE_MIN_V)
      np.testing.assert_equal(actual, expected)


if __name__ == "__main__":
  unittest.main()
