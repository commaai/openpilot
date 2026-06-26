from openpilot.selfdrive.locationd.helpers import parabolic_peak_interp


class TestParabolicPeakInterp:
  def test_symmetric_peak_refines_to_center(self):
    assert parabolic_peak_interp([0.0, 1.0, 0.0], 1) == 1.0

  def test_endpoints_return_index(self):
    R = [1.0, 0.5, 0.2]
    assert parabolic_peak_interp(R, 0) == 0
    assert parabolic_peak_interp(R, len(R) - 1) == len(R) - 1

  def test_degenerate_inputs_do_not_raise(self):
    # an empty array or an out-of-range index must not raise IndexError
    assert parabolic_peak_interp([], 7) == 7
    assert parabolic_peak_interp([1.0, 2.0], 5) == 5
