#!/usr/bin/env python3
import os
os.environ['OLD_CAN'] = '1'
os.environ['NOCRASH'] = '1'

import time
import unittest
import shutil
import matplotlib
matplotlib.use('svg')

from selfdrive.config import Conversions as CV
from selfdrive.car.honda.values import CruiseButtons as CB
from selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver
import selfdrive.manager as manager
from common.params import Params


def create_dir(path):
  try:
    os.makedirs(path)
  except OSError:
    pass


def check_no_collision(log):
  return min(log['d_rel']) > 0

def check_fcw(log):
  return any(log['fcw'])

def check_engaged(log):
  return log['controls_state_msgs'][-1][-1].active

maneuvers = [
  Maneuver(
    'while cruising at 40 mph, change cruise speed to 50mph',
    duration=30.,
    initial_speed = 40. * CV.MPH_TO_MS,
    cruise_button_presses = [(CB.DECEL_SET, 2.), (0, 2.3),
                             (CB.RES_ACCEL, 10.), (0, 10.1),
                             (CB.RES_ACCEL, 10.2), (0, 10.3)],
    checks=[check_engaged],
  ),
  Maneuver(
    'while cruising at 60 mph, change cruise speed to 50mph',
    duration=30.,
    initial_speed=60. * CV.MPH_TO_MS,
    cruise_button_presses = [(CB.DECEL_SET, 2.), (0, 2.3),
                             (CB.DECEL_SET, 10.), (0, 10.1),
                             (CB.DECEL_SET, 10.2), (0, 10.3)],
    checks=[check_engaged],
  ),
  Maneuver(
    'while cruising at 20mph, grade change +10%',
    duration=25.,
    initial_speed=20. * CV.MPH_TO_MS,
    cruise_button_presses = [(CB.DECEL_SET, 1.2), (0, 1.3)],
    grade_values = [0., 0., 1.0],
    grade_breakpoints = [0., 10., 11.],
    checks=[check_engaged],
  ),
  Maneuver(
    'while cruising at 20mph, grade change -10%',
    duration=25.,
    initial_speed=20. * CV.MPH_TO_MS,
    cruise_button_presses = [(CB.DECEL_SET, 1.2), (0, 1.3)],
    grade_values = [0., 0., -1.0],
    grade_breakpoints = [0., 10., 11.],
    checks=[check_engaged],
  ),
  Maneuver(
    'approaching a 40mph car while cruising at 60mph from 100m away',
    duration=30.,
    initial_speed = 60. * CV.MPH_TO_MS,
    lead_relevancy=True,
    initial_distance_lead=100.,
    speed_lead_values = [40.*CV.MPH_TO_MS, 40.*CV.MPH_TO_MS],
    speed_lead_breakpoints = [0., 100.],
    cruise_button_presses = [(CB.DECEL_SET, 1.2), (0, 1.3)],
    checks=[check_engaged, check_no_collision],
  ),
  Maneuver(
    'approaching a 0mph car while cruising at 40mph from 150m away',
    duration=30.,
    initial_speed = 40. * CV.MPH_TO_MS,
    lead_relevancy=True,
    initial_distance_lead=150.,
    speed_lead_values = [0.*CV.MPH_TO_MS, 0.*CV.MPH_TO_MS],
    speed_lead_breakpoints = [0., 100.],
    cruise_button_presses = [(CB.DECEL_SET, 1.2), (0, 1.3)],
    checks=[check_engaged, check_no_collision],
  ),
  Maneuver(
    'steady state following a car at 20m/s, then lead decel to 0mph at 1m/s^2',
    duration=50.,
    initial_speed = 20.,
    lead_relevancy=True,
    initial_distance_lead=35.,
    speed_lead_values = [20., 20., 0.],
    speed_lead_breakpoints = [0., 15., 35.0],
    cruise_button_presses = [(CB.DECEL_SET, 1.2), (0, 1.3)],
    checks=[check_engaged, check_no_collision],
  ),
  Maneuver(
    'steady state following a car at 20m/s, then lead decel to 0mph at 2m/s^2',
    duration=50.,
    initial_speed = 20.,
    lead_relevancy=True,
    initial_distance_lead=35.,
    speed_lead_values = [20., 20., 0.],
    speed_lead_breakpoints = [0., 15., 25.0],
    cruise_button_presses = [(CB.DECEL_SET, 1.2), (0, 1.3)],
    checks=[check_engaged, check_no_collision],
  ),
  Maneuver(
    'steady state following a car at 20m/s, then lead decel to 0mph at 3m/s^2',
    duration=50.,
    initial_speed = 20.,
    lead_relevancy=True,
    initial_distance_lead=35.,
    speed_lead_values = [20., 20., 0.],
    speed_lead_breakpoints = [0., 15., 21.66],
    cruise_button_presses = [(CB.DECEL_SET, 1.2), (0, 1.3)],
    checks=[check_engaged, check_fcw],
  ),
  Maneuver(
    'steady state following a car at 20m/s, then lead decel to 0mph at 5m/s^2',
    duration=40.,
    initial_speed = 20.,
    lead_relevancy=True,
    initial_distance_lead=35.,
    speed_lead_values = [20., 20., 0.],
    speed_lead_breakpoints = [0., 15., 19.],
    cruise_button_presses = [(CB.DECEL_SET, 1.2), (0, 1.3)],
    checks=[check_engaged, check_fcw],
  ),
  Maneuver(
    'starting at 0mph, approaching a stopped car 100m away',
    duration=30.,
    initial_speed = 0.,
    lead_relevancy=True,
    initial_distance_lead=100.,
    cruise_button_presses = [(CB.DECEL_SET, 1.2), (0, 1.3),
                             (CB.RES_ACCEL, 1.4), (0.0, 1.5),
                             (CB.RES_ACCEL, 1.6), (0.0, 1.7),
                             (CB.RES_ACCEL, 1.8), (0.0, 1.9)],
    checks=[check_engaged, check_no_collision],
  ),
  Maneuver(
    "following a car at 60mph, lead accel and decel at 0.5m/s^2 every 2s",
    duration=25.,
    initial_speed=30.,
    lead_relevancy=True,
    initial_distance_lead=49.,
    speed_lead_values=[30.,30.,29.,31.,29.,31.,29.],
    speed_lead_breakpoints=[0., 6., 8., 12.,16.,20.,24.],
    cruise_button_presses = [(CB.DECEL_SET, 1.2), (0, 1.3),
                             (CB.RES_ACCEL, 1.4), (0.0, 1.5),
                             (CB.RES_ACCEL, 1.6), (0.0, 1.7)],
    checks=[check_engaged, check_no_collision],
  ),
  Maneuver(
    "following a car at 10mph, stop and go at 1m/s2 lead dece1 and accel",
    duration=70.,
    initial_speed=10.,
    lead_relevancy=True,
    initial_distance_lead=20.,
    speed_lead_values=[10., 0., 0., 10., 0.,10.],
    speed_lead_breakpoints=[10., 20., 30., 40., 50., 60.],
    cruise_button_presses = [(CB.DECEL_SET, 1.2), (0, 1.3),
                             (CB.RES_ACCEL, 1.4), (0.0, 1.5),
                             (CB.RES_ACCEL, 1.6), (0.0, 1.7)],
    checks=[check_engaged, check_no_collision],
  ),
  Maneuver(
    "green light: stopped behind lead car, lead car accelerates at 1.5 m/s",
    duration=30.,
    initial_speed=0.,
    lead_relevancy=True,
    initial_distance_lead=4.,
    speed_lead_values=[0,  0 , 45],
    speed_lead_breakpoints=[0, 10., 40.],
    cruise_button_presses = [(CB.DECEL_SET, 1.2), (0, 1.3),
                             (CB.RES_ACCEL, 1.4), (0.0, 1.5),
                             (CB.RES_ACCEL, 1.6), (0.0, 1.7),
                             (CB.RES_ACCEL, 1.8), (0.0, 1.9),
                             (CB.RES_ACCEL, 2.0), (0.0, 2.1),
                             (CB.RES_ACCEL, 2.2), (0.0, 2.3)],
    checks=[check_engaged, check_no_collision],
  ),
  Maneuver(
    "stop and go with 1m/s2 lead decel and accel, with full stops",
    duration=70.,
    initial_speed=0.,
    lead_relevancy=True,
    initial_distance_lead=20.,
    speed_lead_values=[10., 0., 0., 10., 0., 0.] ,
    speed_lead_breakpoints=[10., 20., 30., 40., 50., 60.],
    cruise_button_presses = [(CB.DECEL_SET, 1.2), (0, 1.3),
                            (CB.RES_ACCEL, 1.4), (0.0, 1.5),
                             (CB.RES_ACCEL, 1.6), (0.0, 1.7)],
    checks=[check_engaged, check_no_collision],
  ),
  Maneuver(
    "stop and go with 1.5m/s2 lead accel and 3.3m/s^2 lead decel, with full stops",
    duration=45.,
    initial_speed=0.,
    lead_relevancy=True,
    initial_distance_lead=20.,
    speed_lead_values=[10., 0., 0., 10., 0., 0.] ,
    speed_lead_breakpoints=[10., 13., 26., 33., 36., 45.],
    cruise_button_presses = [(CB.DECEL_SET, 1.2), (0, 1.3),
                             (CB.RES_ACCEL, 1.4), (0.0, 1.5),
                             (CB.RES_ACCEL, 1.6), (0.0, 1.7)],
    checks=[check_engaged, check_no_collision],
  ),
  Maneuver(
    "accelerate from 20 while lead vehicle decelerates from 40 to 20 at 1m/s2",
    duration=30.,
    initial_speed=10.,
    lead_relevancy=True,
    initial_distance_lead=10.,
    speed_lead_values=[20., 10.],
    speed_lead_breakpoints=[1., 11.],
    cruise_button_presses = [(CB.DECEL_SET, 1.2), (0, 1.3),
                             (CB.RES_ACCEL, 1.4), (0.0, 1.5),
                             (CB.RES_ACCEL, 1.6), (0.0, 1.7),
                             (CB.RES_ACCEL, 1.8), (0.0, 1.9),
                             (CB.RES_ACCEL, 2.0), (0.0, 2.1),
                             (CB.RES_ACCEL, 2.2), (0.0, 2.3)],
    checks=[check_engaged, check_no_collision],
  ),
  Maneuver(
    "accelerate from 20 while lead vehicle decelerates from 40 to 0 at 2m/s2",
    duration=30.,
    initial_speed=10.,
    lead_relevancy=True,
    initial_distance_lead=10.,
    speed_lead_values=[20., 0.],
    speed_lead_breakpoints=[1., 11.],
    cruise_button_presses = [(CB.DECEL_SET, 1.2), (0, 1.3),
                             (CB.RES_ACCEL, 1.4), (0.0, 1.5),
                             (CB.RES_ACCEL, 1.6), (0.0, 1.7),
                             (CB.RES_ACCEL, 1.8), (0.0, 1.9),
                             (CB.RES_ACCEL, 2.0), (0.0, 2.1),
                             (CB.RES_ACCEL, 2.2), (0.0, 2.3)],
    checks=[check_engaged, check_no_collision],
  ),
  Maneuver(
    "fcw: traveling at 30 m/s and approaching lead traveling at 20m/s",
    duration=15.,
    initial_speed=30.,
    lead_relevancy=True,
    initial_distance_lead=100.,
    speed_lead_values=[20.],
    speed_lead_breakpoints=[1.],
    cruise_button_presses = [],
    checks=[check_fcw],
  ),
  Maneuver(
    "fcw: traveling at 20 m/s following a lead that decels from 20m/s to 0 at 1m/s2",
    duration=18.,
    initial_speed=20.,
    lead_relevancy=True,
    initial_distance_lead=35.,
    speed_lead_values=[20., 0.],
    speed_lead_breakpoints=[3., 23.],
    cruise_button_presses = [],
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
    cruise_button_presses = [],
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
    cruise_button_presses = [],
    checks=[check_fcw],
  )
]

# maneuvers = [maneuvers[-11]]
# maneuvers = [maneuvers[6]]

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
        view_html += "<td><img class='maneuver_graph' src='%s'/></td>" % (os.path.join("maneuver" + str(i+1).zfill(2), c), )
      view_html += "</tr>"

    create_dir(output_dir)
    with open(os.path.join(output_dir, "index.html"), "w") as f:
      f.write(view_html)

class LongitudinalControl(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    os.environ['NO_CAN_TIMEOUT'] = "1"

    setup_output()

    shutil.rmtree('/data/params', ignore_errors=True)
    params = Params()
    params.put("Passive", "1" if os.getenv("PASSIVE") else "0")
    params.put("OpenpilotEnabledToggle", "1")
    params.put("CommunityFeaturesToggle", "1")

    manager.gctx = {}
    manager.prepare_managed_process('radard')
    manager.prepare_managed_process('controlsd')
    manager.prepare_managed_process('plannerd')

  @classmethod
  def tearDownClass(cls):
    pass

  # hack
  def test_longitudinal_setup(self):
    pass

def run_maneuver_worker(k):
  man = maneuvers[k]
  output_dir = os.path.join(os.getcwd(), 'out/longitudinal')

  def run(self):
    print(man.title)
    manager.start_managed_process('radard')
    manager.start_managed_process('controlsd')
    manager.start_managed_process('plannerd')

    plot, valid = man.evaluate()
    plot.write_plot(output_dir, "maneuver" + str(k+1).zfill(2))

    manager.kill_managed_process('radard')
    manager.kill_managed_process('controlsd')
    manager.kill_managed_process('plannerd')
    time.sleep(5)

    self.assertTrue(valid)

  return run

for k in range(len(maneuvers)):
  setattr(LongitudinalControl, "test_longitudinal_maneuvers_%d" % (k+1), run_maneuver_worker(k))

if __name__ == "__main__":
  unittest.main(failfast=True)
