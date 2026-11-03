from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.curve_speed import (get_curve_accel, CurveSpeed, A_LAT_MAX, CURVE_ACCEL_MIN, CURVE_JERK,
                                                          IN_CURVE_ACCEL_MIN, NO_LIMIT, RELEASE_JERK, YAW_RATE_STD_K)
from openpilot.selfdrive.modeld.constants import ModelConstants

T = np.array(ModelConstants.T_IDXS)


def model_msg(t, x, v, yaw_rate, yaw_rate_std=None):
  std = [] if yaw_rate_std is None else list(yaw_rate_std)
  return SimpleNamespace(position=SimpleNamespace(t=list(t), x=list(x)), velocity=SimpleNamespace(x=list(v)),
                         orientationRate=SimpleNamespace(z=list(yaw_rate), zStd=std))


def plan(v, curv_of_t):
  """A plan at constant speed v whose curvature at time t is curv_of_t(t)."""
  x = v * T
  return T, x, np.full_like(T, v), curv_of_t(T) * v


class TestCurveSpeed:
  def test_straight_road_has_no_limit(self):
    assert get_curve_accel(*plan(30.0, lambda t: 0.0 * t), 30.0) == NO_LIMIT

  def test_below_the_curve_speed_the_limit_is_an_accel_cap(self):
    curv = 0.01  # 100 m radius: 15.8 m/s at 2.5 m/s²
    v_max = np.sqrt(A_LAT_MAX / curv)
    far = get_curve_accel(*plan(10.0, lambda t: np.where(t > 2, curv, 0.0)), 10.0)
    near = get_curve_accel(*plan(15.0, lambda t: np.where(t > 2, curv, 0.0)), 15.0)
    assert 0.0 < near < far
    assert near <= (v_max ** 2 - 15.0 ** 2) / (2 * 15.0 * 2.0) + 1e-6

  def test_brakes_for_a_sharp_curve_ahead(self):
    v, curv = 25.0, 0.02  # 50 m radius: 12.2 m/s at 3 m/s²
    a = get_curve_accel(*plan(v, lambda t: np.where(t > 3, curv, 0.0)), v)
    v_max = np.sqrt(A_LAT_MAX / curv)
    assert CURVE_ACCEL_MIN <= a < 0.0
    # enough to reach the curve speed by the start of the curve
    assert a <= (v_max ** 2 - v ** 2) / (2 * v * 3.5) + 1e-6 or a == CURVE_ACCEL_MIN

  def test_brakes_gently_in_a_curve_that_does_not_get_tighter(self):
    v, curv = 20.0, 0.01  # 3 m/s² limit is 17.3 m/s, we are in the curve at 20 m/s
    a = get_curve_accel(*plan(v, lambda t: np.full_like(t, curv)), v)
    assert IN_CURVE_ACCEL_MIN <= a < 0.0

  def test_one_bad_plan_point_does_not_brake(self):
    v = 25.0
    t, x, vv, yaw_rate = plan(v, lambda t: 0.0 * t)
    yaw_rate[10] = 0.5  # one point with a huge yaw rate
    assert get_curve_accel(t, x, vv, yaw_rate, v) == NO_LIMIT

  def test_braking_stops_quickly_but_accel_comes_back_slowly(self):
    v = 25.0
    cs = CurveSpeed(DT_MDL)
    curve, straight = (model_msg(*plan(v, f)) for f in (lambda t: np.where(t > 3, 0.02, 0.0), lambda t: 0.0 * t))
    for _ in range(100):
      a_braking = cs.update(curve, v)
    assert a_braking < -1.0
    steps_to_zero, a = 0, a_braking
    while a < 0.0:
      a = cs.update(straight, v)
      steps_to_zero += 1
    assert steps_to_zero * DT_MDL <= -a_braking / CURVE_JERK + DT_MDL
    a_zero = a
    for _ in range(int(2.0 / DT_MDL)):
      a = cs.update(straight, v)
    # 2 s after the braking ends, the accel cap is up by RELEASE_JERK * 2 s, not at once
    assert a == pytest.approx(a_zero + 2.0 * RELEASE_JERK, abs=1e-6)

  def test_uncertain_curve_slows_down_more(self):
    v, curv = 20.0, 0.005  # 2.0 m/s² at 20 m/s: below the limit if the model is sure
    t, x, vv, yaw_rate = plan(v, lambda t: np.where(t > 3, curv, 0.0))
    sure, unsure = CurveSpeed(DT_MDL), CurveSpeed(DT_MDL)
    for _ in range(100):
      a_sure = sure.update(model_msg(t, x, vv, yaw_rate, np.full_like(t, 0.001)), v)
      a_unsure = unsure.update(model_msg(t, x, vv, yaw_rate, np.where(t > 3, 0.05, 0.001)), v)
    assert a_sure > 0.0
    assert a_unsure < 0.0
    assert YAW_RATE_STD_K > 0.0

  def test_no_std_in_the_message_is_the_same_as_a_sure_model(self):
    v = 25.0
    t, x, vv, yaw_rate = plan(v, lambda t: np.where(t > 3, 0.02, 0.0))
    a, b = CurveSpeed(DT_MDL), CurveSpeed(DT_MDL)
    for _ in range(50):
      assert a.update(model_msg(t, x, vv, yaw_rate), v) == b.update(model_msg(t, x, vv, yaw_rate, np.zeros_like(t)), v)
