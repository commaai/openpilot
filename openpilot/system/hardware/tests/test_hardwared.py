import math

from openpilot.system.hardware.hardwared import max_finite


def test_max_finite_ignores_non_finite_values():
  assert max_finite([math.nan, 42.0, math.inf, -math.inf]) == 42.0


def test_max_finite_returns_none_without_finite_values():
  assert max_finite([math.nan, math.inf, -math.inf]) is None
