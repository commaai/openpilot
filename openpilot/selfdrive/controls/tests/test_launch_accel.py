import pytest

from openpilot.selfdrive.controls.lib.launch_accel import get_launch_accel, LAUNCH_GAIN_V


def test_launch_boost_from_standstill_fades_with_speed():
  assert get_launch_accel(0.5, 0.0) == pytest.approx(0.5 * LAUNCH_GAIN_V[0])
  assert get_launch_accel(0.5, 10.0) == pytest.approx(0.5 * LAUNCH_GAIN_V[1])


@pytest.mark.parametrize("v_ego", [0.0, 20.0])
@pytest.mark.parametrize("a_e2e", [0.1, 0.0, -0.5, -2.0])
def test_no_boost_while_creeping_or_braking(a_e2e, v_ego):
  assert get_launch_accel(a_e2e, v_ego) == a_e2e


def test_small_boost_at_speed():
  assert get_launch_accel(0.8, 20.0) == pytest.approx(0.8 * LAUNCH_GAIN_V[1])
  assert LAUNCH_GAIN_V[1] <= 1.15
