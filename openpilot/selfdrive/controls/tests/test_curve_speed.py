from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.curve_speed import (get_curve_gap, get_curve_accel, CurveSpeed, A_LAT_MAX, BANK_MAX,
                                                          COAST_GAP, CURVE_ACCEL_MIN, CURVE_JERK, NO_LIMIT)
from openpilot.selfdrive.modeld.constants import ModelConstants

T = np.array(ModelConstants.T_IDXS)
NO_STD = np.zeros_like(T)


def plan(v, curv_of_t):
  """A plan at constant speed v whose curvature at time t is curv_of_t(t). Returns t, x, v, yaw rate."""
  return T, v * T, np.full_like(T, v), curv_of_t(T) * v


def model_msg(t, x, v, yaw_rate, yaw_rate_std=None):
  std = [] if yaw_rate_std is None else list(yaw_rate_std)
  return SimpleNamespace(position=SimpleNamespace(t=list(t), x=list(x)), velocity=SimpleNamespace(x=list(v)),
                         orientationRate=SimpleNamespace(z=list(yaw_rate), zStd=std))


def curve_ahead(curv, t_start=3.0):
  return lambda t: np.where(t > t_start, curv, 0.0)


def run(cs, msg, roll, a_e2e, steps):
  for _ in range(steps):
    a = cs.update(msg, roll, a_e2e)
  return a


class TestCurveGap:
  def test_straight_road_has_no_gap(self):
    t, _, v, yaw_rate = plan(30.0, lambda t: 0.0 * t)
    assert get_curve_gap(t, v, yaw_rate, NO_STD, 0.0) == 0.0

  def test_gap_is_the_speed_over_the_curve_speed_per_second(self):
    v, curv = 25.0, 0.02  # 50 m radius: 11.8 m/s at 2.8 m/s²
    t, _, vv, yaw_rate = plan(v, curve_ahead(curv))
    v_max = np.sqrt(A_LAT_MAX / curv)
    first = T[T > 3.0][0]
    assert get_curve_gap(t, vv, yaw_rate, NO_STD, 0.0) == pytest.approx((v - v_max) / first)

  def test_no_gap_when_the_plan_already_slows_for_the_curve(self):
    curv = 0.02
    v_max = np.sqrt(A_LAT_MAX / curv)
    # the model plans to be at the curve speed when it gets to the curve, even if the car is fast now
    v = np.where(T > 3.0, v_max * 0.99, 25.0)
    assert get_curve_gap(T, v, curve_ahead(curv)(T) * v, NO_STD, 0.0) == 0.0

  def test_bank_into_the_curve_lowers_the_gap(self):
    t, _, v, yaw_rate = plan(25.0, curve_ahead(0.02))
    flat = get_curve_gap(t, v, yaw_rate, NO_STD, 0.0)
    banked = get_curve_gap(t, v, yaw_rate, NO_STD, np.radians(3.0))
    off_camber = get_curve_gap(t, v, yaw_rate, NO_STD, -np.radians(3.0))
    assert banked < flat < off_camber

  def test_bank_credit_is_limited(self):
    t, _, v, yaw_rate = plan(25.0, curve_ahead(0.02))
    assert get_curve_gap(t, v, yaw_rate, NO_STD, 0.5) == get_curve_gap(t, v, yaw_rate, NO_STD, BANK_MAX)
    assert get_curve_gap(t, v, yaw_rate, NO_STD, 0.5) > 0.0

  def test_bank_sign_follows_the_curve_direction(self):
    t, _, v, yaw_rate = plan(25.0, curve_ahead(0.02))
    roll = np.radians(3.0)
    assert get_curve_gap(t, v, yaw_rate, NO_STD, roll) == get_curve_gap(t, v, -yaw_rate, NO_STD, -roll)

  def test_bank_and_lateral_accel_add_up(self):
    v, curv, roll = 25.0, 0.02, np.radians(2.0)
    t, _, vv, yaw_rate = plan(v, curve_ahead(curv))
    a_lat = A_LAT_MAX + ACCELERATION_DUE_TO_GRAVITY * np.sin(roll)
    first = T[T > 3.0][0]
    assert get_curve_gap(t, vv, yaw_rate, NO_STD, roll) == pytest.approx((v - np.sqrt(a_lat / curv)) / first)

  def test_one_bad_plan_point_has_no_gap(self):
    t, _, v, yaw_rate = plan(25.0, lambda t: 0.0 * t)
    yaw_rate[10] = 0.5  # one point with a huge yaw rate
    assert get_curve_gap(t, v, yaw_rate, NO_STD, 0.0) == 0.0

  def test_uncertain_curve_has_a_bigger_gap(self):
    t, _, v, yaw_rate = plan(20.0, curve_ahead(0.006))  # 2.4 m/s² at 20 m/s: below the limit if the model is sure
    assert get_curve_gap(t, v, yaw_rate, np.full_like(T, 0.001), 0.0) == 0.0
    assert get_curve_gap(t, v, yaw_rate, np.where(T > 3, 0.05, 0.001), 0.0) > 0.0


class TestCurveAccel:
  def test_no_gap_no_limit(self):
    assert get_curve_accel(1.0, 0.0) == NO_LIMIT

  def test_small_gap_only_stops_the_acceleration(self):
    assert get_curve_accel(1.0, COAST_GAP / 2) == 0.0
    assert get_curve_accel(-0.4, COAST_GAP / 2) == -0.4

  def test_big_gap_brakes_by_the_gap_on_top_of_the_model(self):
    assert get_curve_accel(0.5, 1.0) == pytest.approx(-0.5)
    assert get_curve_accel(0.0, 5.0) == CURVE_ACCEL_MIN


class TestCurveSpeed:
  def test_correction_starts_from_the_model_accel_with_limited_jerk(self):
    cs = CurveSpeed(DT_MDL)
    msg = model_msg(*plan(25.0, curve_ahead(0.02)))
    a_e2e = 0.5
    a_first = cs.update(msg, 0.0, a_e2e)
    assert a_first == pytest.approx(a_e2e - CURVE_JERK * DT_MDL)
    gap = get_curve_gap(T, np.full_like(T, 25.0), curve_ahead(0.02)(T) * 25.0, NO_STD, 0.0)
    assert run(cs, msg, 0.0, a_e2e, 100) == pytest.approx(max(a_e2e - gap, CURVE_ACCEL_MIN))

  def test_correction_goes_away_when_the_model_slows_down(self):
    cs = CurveSpeed(DT_MDL)
    curv = 0.02
    run(cs, model_msg(*plan(25.0, curve_ahead(curv))), 0.0, 0.0, 100)
    assert cs.a_curve < -1.0
    # now the model plans the curve speed at the curve by itself
    v_max = np.sqrt(A_LAT_MAX / curv)
    v = np.where(T > 3.0, v_max * 0.99, 25.0)
    slow = model_msg(T, np.cumsum(v) * 0.1, v, curve_ahead(curv)(T) * v)
    steps = 0
    while cs.update(slow, 0.0, -1.0) < NO_LIMIT:
      steps += 1
    # the correction ends with limited jerk, then there is no limit
    assert 0 < steps * DT_MDL <= 2.0 / CURVE_JERK + DT_MDL

  def test_straight_road_has_no_limit(self):
    cs = CurveSpeed(DT_MDL)
    assert run(cs, model_msg(*plan(30.0, lambda t: 0.0 * t)), 0.0, 1.0, 20) == NO_LIMIT

  def test_no_std_in_the_message_is_the_same_as_a_sure_model(self):
    t, x, v, yaw_rate = plan(25.0, curve_ahead(0.02))
    a, b = CurveSpeed(DT_MDL), CurveSpeed(DT_MDL)
    for _ in range(50):
      assert a.update(model_msg(t, x, v, yaw_rate), 0.0, 0.3) == b.update(model_msg(t, x, v, yaw_rate, NO_STD), 0.0, 0.3)

  def test_empty_model_message_has_no_limit(self):
    cs = CurveSpeed(DT_MDL)
    assert cs.update(model_msg([], [], [], []), 0.0, 0.5) == NO_LIMIT
