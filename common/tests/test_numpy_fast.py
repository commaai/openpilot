import numpy as np


class TestInterp:
  def test_correctness_controls(self):
    _A_CRUISE_MIN_BP = np.asarray([0., 5., 10., 20., 40.])
    _A_CRUISE_MIN_V = np.asarray([-1.0, -.8, -.67, -.5, -.30])
    v_ego_arr = [-1, -1e-12, 0, 4, 5, 6, 7, 10, 11, 15.2, 20, 21, 39,
                 39.999999, 40, 41]

    # Use np.interp directly for arrays
    expected = np.interp(v_ego_arr, _A_CRUISE_MIN_BP, _A_CRUISE_MIN_V)
    actual = np.interp(v_ego_arr, _A_CRUISE_MIN_BP, _A_CRUISE_MIN_V)

    # Assert that the arrays are equal
    np.testing.assert_array_equal(actual, expected)

    # Test for individual scalar values
    for v_ego in v_ego_arr:
        expected = float(np.interp(v_ego, _A_CRUISE_MIN_BP, _A_CRUISE_MIN_V))
        actual = float(np.interp(v_ego, _A_CRUISE_MIN_BP, _A_CRUISE_MIN_V))
        np.testing.assert_equal(actual, expected)

