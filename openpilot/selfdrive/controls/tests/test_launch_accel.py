import pytest

from openpilot.selfdrive.controls.lib.launch_accel import get_launch_accel, LAUNCH_GAIN_V


def test_launch_boost_from_standstill_fades_with_speed():
  assert get_launch_accel(0.5, 0.0) == pytest.approx(0.5 * LAUNCH_GAIN_V[0])
  assert get_launch_accel(0.5, 10.0) == pytest.approx(0.5)


@pytest.mark.parametrize("a_e2e", [0.1, 0.0, -0.5, -2.0])
def test_launch_boost_not_applied_while_creeping_or_braking(a_e2e):
  assert get_launch_accel(a_e2e, 0.0) == a_e2e


@pytest.mark.parametrize("a_e2e", [-1.5, 0.0, 0.8])
def test_model_accel_is_unchanged_at_speed(a_e2e):
  assert get_launch_accel(a_e2e, 20.0) == a_e2e
