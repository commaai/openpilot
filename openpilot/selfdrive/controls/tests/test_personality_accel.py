from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.personality_accel import ACCEL_GAIN, GATE_BP, LIFT_PER_SPEED, Personality, PersonalityAccel


def make_sm(personality, v_ego=20.0, p_gas=1.0, p_brake=0.0):
  predictions = SimpleNamespace(gasPressProbs=[0.0, 0.0, 0.0, 0.0, p_gas, 0.0], brakePressProbs=[0.0, 0.0, 0.0, 0.0, p_brake, 0.0])
  return {
    'selfdriveState': SimpleNamespace(personality=personality),
    'carState': SimpleNamespace(vEgo=v_ego),
    'modelV2': SimpleNamespace(meta=SimpleNamespace(disengagePredictions=predictions)),
  }


def settle(accel, sm, a_e2e, seconds=5.0):
  out = 0.0
  for _ in range(int(seconds / DT_MDL)):
    out = accel.update(sm, a_e2e)
  return out


@pytest.mark.parametrize("a_e2e", [-1.5, 0.0, 0.8])
def test_relaxed_is_unchanged(a_e2e):
  assert settle(PersonalityAccel(DT_MDL), make_sm(Personality.relaxed), a_e2e) == a_e2e


@pytest.mark.parametrize("a_e2e", [-1.5, 0.0, 0.8])
def test_standard_scales_both_ways(a_e2e):
  assert settle(PersonalityAccel(DT_MDL), make_sm(Personality.standard), a_e2e) == pytest.approx(ACCEL_GAIN * a_e2e)


def test_aggressive_adds_lift_scaled_by_speed_and_probs():
  sm = make_sm(Personality.aggressive, v_ego=20.0, p_gas=0.9, p_brake=0.2)
  out = settle(PersonalityAccel(DT_MDL), sm, 0.5)
  gate = np.interp(0.9 * 0.8, GATE_BP, [0.0, 1.0])
  assert out == pytest.approx(ACCEL_GAIN * 0.5 + LIFT_PER_SPEED * 20.0 * gate, abs=1e-3)


def test_aggressive_has_no_lift_when_gas_prob_is_low():
  sm = make_sm(Personality.aggressive, v_ego=20.0, p_gas=0.5, p_brake=0.0)
  assert settle(PersonalityAccel(DT_MDL), sm, 0.5) == pytest.approx(ACCEL_GAIN * 0.5)


def test_aggressive_has_no_lift_while_model_brakes():
  assert settle(PersonalityAccel(DT_MDL), make_sm(Personality.aggressive), -0.1) == pytest.approx(ACCEL_GAIN * -0.1)


def test_aggressive_has_no_lift_at_standstill():
  assert settle(PersonalityAccel(DT_MDL), make_sm(Personality.aggressive, v_ego=0.0), 0.0) == 0.0


def test_lift_is_smoothed_and_resets():
  accel = PersonalityAccel(DT_MDL)
  sm = make_sm(Personality.aggressive)
  target = LIFT_PER_SPEED * 20.0
  assert 0.0 < accel.update(sm, 0.0) < target / 2
  settle(accel, sm, 0.0)
  accel.reset()
  assert accel.lift_filter.x == 0.0


def test_missing_predictions_give_no_lift():
  sm = make_sm(Personality.aggressive)
  sm['modelV2'].meta.disengagePredictions = SimpleNamespace(gasPressProbs=[], brakePressProbs=[])
  assert settle(PersonalityAccel(DT_MDL), sm, 0.5) == pytest.approx(ACCEL_GAIN * 0.5)


def test_launch_boost_from_standstill_fades_with_speed():
  stopped = settle(PersonalityAccel(DT_MDL), make_sm(Personality.relaxed, v_ego=0.0, p_gas=0.0), 0.5)
  moving = settle(PersonalityAccel(DT_MDL), make_sm(Personality.relaxed, v_ego=10.0, p_gas=0.0), 0.5)
  assert stopped == pytest.approx(0.5 * 1.6)
  assert moving == pytest.approx(0.5)


def test_launch_boost_not_applied_while_creeping_or_braking():
  for a_e2e in (0.1, -0.5):
    assert settle(PersonalityAccel(DT_MDL), make_sm(Personality.relaxed, v_ego=0.0, p_gas=0.0), a_e2e) == pytest.approx(a_e2e)
