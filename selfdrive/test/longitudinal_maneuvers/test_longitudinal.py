#!/usr/bin/env python3
import os
from selfdrive.test.process_replay.test_processes import get_segment
from selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver
from selfdrive.manager.process_config import managed_processes
from common.file_helpers import mkdirs_exists_ok
from common.params import Params
from tools.lib.logreader import LogReader
import unittest



def check_no_collision(log):
  return min(log['d_rel']) > 0


def check_fcw(log):
  return any(log['fcw'])


def check_engaged(log):
  return log['controls_state_msgs'][-1][-1].active

def put_default_car_params():
  cp = [m for m in LogReader(get_segment('0982d79ebb0de295|2021-01-08--10-13-10--0')) if m.which() == 'carParams']
  Params().put("CarParams", cp[0].carParams.as_builder().to_bytes())



maneuvers = [
  Maneuver(
    "fcw: traveling at 30 m/s and approaching lead traveling at 20m/s",
    duration=15.,
    initial_speed=30.,
    lead_relevancy=True,
    initial_distance_lead=100.,
    speed_lead_values=[20.],
    speed_lead_breakpoints=[1.],
    cruise_button_presses=[],
    checks=[check_fcw],
  ),
  Maneuver(
    "fcw: traveling at 20 m/s following a lead that decels from 20m/s to 0 at 3m/s2",
    duration=13.,
    initial_speed=20.,
    lead_relevancy=True,
    initial_distance_lead=35.,
    speed_lead_values=[20., 0.],
    speed_lead_breakpoints=[3., 9.6],
    cruise_button_presses=[],
    checks=[check_fcw],
  ),
  Maneuver(
    "fcw: traveling at 20 m/s following a lead that decels from 20m/s to 0 at 5m/s2",
    duration=8.,
    initial_speed=20.,
    lead_relevancy=True,
    initial_distance_lead=35.,
    speed_lead_values=[20., 0.],
    speed_lead_breakpoints=[3., 7.],
    cruise_button_presses=[],
    checks=[check_fcw],
  ),
  Maneuver(
    'steady state following a car at 20m/s, then lead decel to 0mph at 1m/s^2',
    duration=50.,
    initial_speed=20.,
    lead_relevancy=True,
    initial_distance_lead=35.,
    speed_lead_values=[20., 20., 0.],
    speed_lead_breakpoints=[0., 15., 35.0],
    checks=[check_engaged, check_no_collision],
  ),
  Maneuver(
    'steady state following a car at 20m/s, then lead decel to 0mph at 2m/s^2',
    duration=50.,
    initial_speed=20.,
    lead_relevancy=True,
    initial_distance_lead=35.,
    speed_lead_values=[20., 20., 0.],
    speed_lead_breakpoints=[0., 15., 25.0],
    checks=[check_engaged, check_no_collision],
  ),
  Maneuver(
    'steady state following a car at 20m/s, then lead decel to 0mph at 3m/s^2',
    duration=50.,
    initial_speed=20.,
    lead_relevancy=True,
    initial_distance_lead=35.,
    speed_lead_values=[20., 20., 0.],
    speed_lead_breakpoints=[0., 15., 21.66],
    checks=[check_engaged, check_fcw],
  ),
  Maneuver(
    'steady state following a car at 20m/s, then lead decel to 0mph at 5m/s^2',
    duration=40.,
    initial_speed=20.,
    lead_relevancy=True,
    initial_distance_lead=35.,
    speed_lead_values=[20., 20., 0.],
    speed_lead_breakpoints=[0., 15., 19.],
    checks=[check_engaged, check_fcw],
  ),
  Maneuver(
    "accelerate from 20 while lead vehicle decelerates from 40 to 0 at 2m/s2",
    duration=30.,
    initial_speed=10.,
    lead_relevancy=True,
    initial_distance_lead=10.,
    speed_lead_values=[20., 0.],
    speed_lead_breakpoints=[1., 11.],
    checks=[check_engaged, check_no_collision],
  ),
]


def setup_output():
  output_dir = os.path.join(os.getcwd(), 'out/longitudinal')
  if not os.path.exists(os.path.join(output_dir, "index.html")):
    # write test output header

    css_style = """
    .maneuver_title {
      font-size: 24px;
      text-align: center;
    }
    .maneuver_graph {
      width: 100%;
    }
    """

    view_html = "<html><head><style>%s</style></head><body><table>" % (css_style,)
    for i, man in enumerate(maneuvers):
      view_html += "<tr><td class='maneuver_title' colspan=5><div>%s</div></td></tr><tr>" % (man.title,)
      for c in ['distance.svg', 'speeds.svg', 'acceleration.svg', 'pedals.svg', 'pid.svg']:
        view_html += "<td><img class='maneuver_graph' src='%s'/></td>" % (os.path.join("maneuver" + str(i + 1).zfill(2), c), )
      view_html += "</tr>"

    mkdirs_exists_ok(output_dir)
    with open(os.path.join(output_dir, "index.html"), "w") as f:
      f.write(view_html)


class LongitudinalControl(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    os.environ['SIMULATION'] = "1"
    os.environ['SKIP_FW_QUERY'] = "1"
    os.environ['NO_CAN_TIMEOUT'] = "1"

    setup_output()

    params = Params()
    params.clear_all()
    params.put_bool("Passive", bool(os.getenv("PASSIVE")))
    params.put_bool("OpenpilotEnabledToggle", True)
    params.put_bool("CommunityFeaturesToggle", True)

  # hack
  def test_longitudinal_setup(self):
    pass


def run_maneuver_worker(k):
  man = maneuvers[k]

  def run(self):
    print(man.title)
    put_default_car_params()
    managed_processes['plannerd'].start()

    _, valid = man.evaluate()

    managed_processes['plannerd'].stop()
    self.assertTrue(valid)

  return run


for k in range(len(maneuvers)):
  setattr(LongitudinalControl, "test_longitudinal_maneuvers_%d : " % (k + 1) + maneuvers[k].title, run_maneuver_worker(k))

if __name__ == "__main__":
  unittest.main(failfast=True)
