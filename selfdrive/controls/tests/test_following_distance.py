#!/usr/bin/env python3
import unittest
from openpilot.common.params import Params
from cereal import log

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import desired_follow_distance, get_T_FOLLOW
from openpilot.selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver


def run_following_distance_simulation(v_lead, t_end=100.0, e2e=False):
  man = Maneuver(
    '',
    duration=t_end,
    initial_speed=float(v_lead),
    lead_relevancy=True,
    initial_distance_lead=100,
    speed_lead_values=[v_lead],
    breakpoints=[0.],
    e2e=e2e,
  )
  valid, output = man.evaluate()
  assert valid
  return output[-1,2] - output[-1,1]


class TestFollowingDistance(unittest.TestCase):
  def test_following_distance(self):
    params = Params()
    personalities = [log.LongitudinalPersonality.relaxed,
                     log.LongitudinalPersonality.standard,
                     log.LongitudinalPersonality.aggressive]
    for personality in personalities:
      params.put("LongitudinalPersonality", str(personality))
      for e2e in [False, True]:
        for speed in [0,10,35]:
          print(f'Testing {speed} m/s')
          v_lead = float(speed)
          simulation_steady_state = run_following_distance_simulation(v_lead, e2e=e2e)
          correct_steady_state = desired_follow_distance(v_lead, v_lead, get_T_FOLLOW(personality))
          err_ratio = 0.2 if e2e else 0.1
          self.assertAlmostEqual(simulation_steady_state, correct_steady_state, delta=(err_ratio * correct_steady_state + .5))


if __name__ == "__main__":
  unittest.main()
