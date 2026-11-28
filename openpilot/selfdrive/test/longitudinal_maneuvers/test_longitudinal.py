from openpilot.common.test import OpenpilotTestCase
from openpilot.common.parameterized import parameterized

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import STOP_DISTANCE
from openpilot.selfdrive.selfdrived.selfdrived import LONGITUDINAL_PERSONALITY_MAP
from openpilot.selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver


# TODO: make new FCW tests
def create_maneuvers(kwargs):
  maneuvers = [
    Maneuver(
      'approach stopped car at 25m/s, initial distance: 120m',
      duration=20.,
      initial_speed=25.,
      lead_relevancy=True,
      initial_distance_lead=120.,
      speed_lead_values=[30., 0.],
      breakpoints=[0., 1.],
      **kwargs,
    ),
    Maneuver(
      'approach stopped car at 20m/s, initial distance 90m',
      duration=20.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=90.,
      speed_lead_values=[20., 0.],
      breakpoints=[0., 1.],
      **kwargs,
    ),
    Maneuver(
      'steady state following a car at 20m/s, then lead decel to 0mph at 1m/s^2',
      duration=50.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=35.,
      speed_lead_values=[20., 20., 0.],
      breakpoints=[0., 15., 35.0],
      **kwargs,
    ),
    Maneuver(
      'steady state following a car at 20m/s, then lead decel to 0mph at 2m/s^2',
      duration=50.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=35.,
      speed_lead_values=[20., 20., 0.],
      breakpoints=[0., 15., 25.0],
      **kwargs,
    ),
    Maneuver(
      'steady state following a car at 20m/s, then lead decel to 0mph at 3m/s^2',
      duration=50.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=35.,
      speed_lead_values=[20., 20., 0.],
      breakpoints=[0., 15., 21.66],
      **kwargs,
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
      **kwargs,
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
      **kwargs,
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
      **kwargs,
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
      **kwargs,
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
      **kwargs,
    ),
    Maneuver(
      'cruising at 25 m/s while disabled',
      duration=20.,
      initial_speed=25.,
      lead_relevancy=False,
      enabled=False,
      **kwargs,
    ),
  ]
  if not kwargs['force_decel']:
    # controls relies on planner commanding to move for stock-ACC resume spamming
    maneuvers.append(Maneuver(
      "resume from a stop",
      duration=20.,
      initial_speed=0.,
      lead_relevancy=True,
      initial_distance_lead=STOP_DISTANCE,
      speed_lead_values=[0., 0., 7.],
      breakpoints=[1., 10., 15.],
      ensure_start=True,
      **kwargs,
    ))
  return maneuvers


def create_approach_stopped_adv_maneuvers(kwargs):
  return [
    Maneuver(
      f"approaching stopped lead that doesn't go ({v0:g}m/s from {d0:g}m), model wrongly predicts +{model_a:g}m/s^2",
      duration=duration,
      initial_speed=v0,
      lead_relevancy=True,
      initial_distance_lead=d0,
      speed_lead_values=[0., 0.],
      breakpoints=[0., duration],
      model_speed_lead_values=[0., model_a * (duration + 15.)],
      model_breakpoints=[0., duration + 15.],
      **kwargs,
    )
    for v0, d0, duration in ((30., 200., 25.), (15., 70., 20.), (8., 25., 18.))
    for model_a in (1., 2., 4.)
  ]


def create_stopped_adv_maneuvers(kwargs):
  return [
    Maneuver(
      f"stopped behind a lead that doesn't go ({d0:g}m), model wrongly predicts +{model_a:g}m/s^2",
      duration=20.,
      initial_speed=0.,
      lead_relevancy=True,
      initial_distance_lead=d0,
      speed_lead_values=[0., 0.],
      breakpoints=[0., 20.],
      model_speed_lead_values=[0., model_a * 35.],
      model_breakpoints=[0., 35.],
      **kwargs,
    )
    for d0 in (STOP_DISTANCE, 4.)
    for model_a in (1., 2., 4.)
  ]


def create_hard_braking_adv_maneuvers(kwargs):
  return [
    Maneuver(
      f"steady following ({v0:g}m/s) then lead brakes hard, model wrongly predicts 0m/s^2, lead -{decel:g}m/s^2",
      duration=15. + v0 / decel + 10.,
      initial_speed=v0,
      lead_relevancy=True,
      initial_distance_lead=d0,
      speed_lead_values=[v0, v0, 0.],
      breakpoints=[0., 15., 15. + v0 / decel],
      model_speed_lead_values=[v0, v0, v0],
      model_breakpoints=[0., 15., 15. + v0 / decel + 25.],
      **kwargs,
    )
    for v0, d0 in ((10., 20.), (20., 35.), (30., 50.))
    for decel in (2., 3., 3.5)
  ]


def create_adv_maneuvers(kwargs):
  return (
    create_approach_stopped_adv_maneuvers(kwargs)
    + create_stopped_adv_maneuvers(kwargs)
    + create_hard_braking_adv_maneuvers(kwargs)
  )


class TestLongitudinalControl(OpenpilotTestCase):
  @parameterized.expand([m for force_decel in (True, False) for m in create_maneuvers({"force_decel": force_decel})],
                        ids=lambda m: f"{m.title}_force_decel_{m.force_decel}")
  def test_maneuver(self, maneuver):
    valid, _ = maneuver.evaluate()
    assert valid

  @parameterized.expand([m for p in LONGITUDINAL_PERSONALITY_MAP for m in create_adv_maneuvers({"personality": p})],
                        ids=lambda m: f"{m.title}_personality_{LONGITUDINAL_PERSONALITY_MAP[m.personality]}")
  def test_adv_maneuver(self, maneuver):
    valid, _ = maneuver.evaluate()
    assert valid
