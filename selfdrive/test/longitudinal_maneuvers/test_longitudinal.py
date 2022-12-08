#!/usr/bin/env python3
import os
from parameterized import parameterized
import unittest

from common.params import Params
from selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import STOP_DISTANCE
from selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver


# TODO: make new FCW tests
def create_maneuvers(e2e):
  return [
    Maneuver(
      'approach stopped car at 25m/s, initial distance: 120m',
      duration=20.,
      initial_speed=25.,
      lead_relevancy=True,
      initial_distance_lead=120.,
      speed_lead_values=[30., 0.],
      breakpoints=[0., 1.],
      e2e=e2e,
    ),
    Maneuver(
      'approach stopped car at 20m/s, initial distance 90m',
      duration=20.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=90.,
      speed_lead_values=[20., 0.],
      breakpoints=[0., 1.],
      e2e=e2e,
    ),
    Maneuver(
      'steady state following a car at 20m/s, then lead decel to 0mph at 1m/s^2',
      duration=50.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=35.,
      speed_lead_values=[20., 20., 0.],
      breakpoints=[0., 15., 35.0],
      e2e=e2e,
    ),
    Maneuver(
      'steady state following a car at 20m/s, then lead decel to 0mph at 2m/s^2',
      duration=50.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=35.,
      speed_lead_values=[20., 20., 0.],
      breakpoints=[0., 15., 25.0],
      e2e=e2e,
    ),
    Maneuver(
      'steady state following a car at 20m/s, then lead decel to 0mph at 3m/s^2',
      duration=50.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=35.,
      speed_lead_values=[20., 20., 0.],
      breakpoints=[0., 15., 21.66],
      e2e=e2e,
    ),
    Maneuver(
      'steady state following a car at 20m/s, then lead decel to 0mph at 3+m/s^2',
      duration=40.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=35.,
      speed_lead_values=[20., 20., 0.],
      prob_lead_values=[0., 1., 1.],
      cruise_values=[20., 20., 20.],
      breakpoints=[2., 2.01, 8.8],
      e2e=e2e,
    ),
    Maneuver(
      "approach stopped car at 20m/s, with prob_lead_values",
      duration=30.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=120.,
      speed_lead_values=[0.0, 0., 0.],
      prob_lead_values=[0.0, 0., 1.],
      cruise_values=[20., 20., 20.],
      breakpoints=[0.0, 2., 2.01],
      e2e=e2e,
    ),
    Maneuver(
      "approach slower cut-in car at 20m/s",
      duration=20.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=50.,
      speed_lead_values=[15., 15.],
      breakpoints=[1., 11.],
      only_lead2=True,
      e2e=e2e,
    ),
    Maneuver(
      "stay stopped behind radar override lead",
      duration=20.,
      initial_speed=0.,
      lead_relevancy=True,
      initial_distance_lead=10.,
      speed_lead_values=[0., 0.],
      prob_lead_values=[0., 0.],
      breakpoints=[1., 11.],
      only_radar=True,
      e2e=e2e,
    ),
    Maneuver(
      "NaN recovery",
      duration=30.,
      initial_speed=15.,
      lead_relevancy=True,
      initial_distance_lead=60.,
      speed_lead_values=[0., 0., 0.0],
      breakpoints=[1., 1.01, 11.],
      cruise_values=[float("nan"), 15., 15.],
      e2e=e2e,
    ),
    # controls relies on planner commanding to move for stock-ACC resume spamming
    Maneuver(
      "resume from a stop",
      duration=20.,
      initial_speed=0.,
      lead_relevancy=True,
      initial_distance_lead=STOP_DISTANCE,
      speed_lead_values=[0., 0., 2.],
      breakpoints=[1., 10., 15.],
      ensure_start=True,
      e2e=e2e,
    ),
    Maneuver(
      'cruising at 25 m/s while disabled',
      duration=20.,
      initial_speed=25.,
      lead_relevancy=False,
      enabled=False,
      e2e=e2e,
    ),
  ]


class LongitudinalControl(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    os.environ['SIMULATION'] = "1"
    os.environ['SKIP_FW_QUERY'] = "1"
    os.environ['NO_CAN_TIMEOUT'] = "1"

    params = Params()
    params.clear_all()
    params.put_bool("Passive", bool(os.getenv("PASSIVE")))
    params.put_bool("OpenpilotEnabledToggle", True)

  @parameterized.expand([(man,) for e2e in [True, False] for man in create_maneuvers(e2e)])
  def test_maneuver(self, maneuver):
    print(maneuver.title, f'in {"e2e" if maneuver.e2e else "acc"} mode')
    valid, _ = maneuver.evaluate()
    self.assertTrue(valid, msg=maneuver.title)


if __name__ == "__main__":
  unittest.main(failfast=True)
