#!/usr/bin/env python
import os
os.environ['OLD_CAN'] = '1'
os.environ['NOCRASH'] = '1'

import time
import unittest
import shutil

import matplotlib
matplotlib.use('svg')

from selfdrive.config import Conversions as CV, CruiseButtons as CB
from selfdrive.test.plant.maneuver import Maneuver
import selfdrive.manager as manager

def create_dir(path):
  try:
    os.makedirs(path)
  except OSError:
    pass

maneuvers = [
  Maneuver(
    'while cruising at 40 mph, change cruise speed to 50mph',
    duration=30.,
    initial_speed = 40. * CV.MPH_TO_MS,
    cruise_button_presses = [(CB.DECEL_SET, 2.), (0, 2.3),
                            (CB.RES_ACCEL, 10.), (0, 10.1),
                            (CB.RES_ACCEL, 10.2), (0, 10.3)]
  ),
  Maneuver(
    'while cruising at 60 mph, change cruise speed to 50mph',
    duration=30.,
    initial_speed=60. * CV.MPH_TO_MS,
    cruise_button_presses = [(CB.DECEL_SET, 2.), (0, 2.3),
                            (CB.DECEL_SET, 10.), (0, 10.1),
                            (CB.DECEL_SET, 10.2), (0, 10.3)]
  ),
  Maneuver(
    'while cruising at 20mph, grade change +10%',
    duration=25.,
    initial_speed=20. * CV.MPH_TO_MS,
    cruise_button_presses = [(CB.DECEL_SET, 1.2), (0, 1.3)],
    grade_values = [0., 0., 1.0],
    grade_breakpoints = [0., 10., 11.]
  ),
  Maneuver(
    'while cruising at 20mph, grade change -10%',
    duration=25.,
    initial_speed=20. * CV.MPH_TO_MS,
    cruise_button_presses = [(CB.DECEL_SET, 1.2), (0, 1.3)],
    grade_values = [0., 0., -1.0],
    grade_breakpoints = [0., 10., 11.]
  ),
  Maneuver(
    'approaching a 40mph car while cruising at 60mph from 100m away',
    duration=30.,
    initial_speed = 60. * CV.MPH_TO_MS,
    lead_relevancy=True,
    initial_distance_lead=100.,
    speed_lead_values = [40.*CV.MPH_TO_MS, 40.*CV.MPH_TO_MS],
    speed_lead_breakpoints = [0., 100.],
    cruise_button_presses = [(CB.DECEL_SET, 1.2), (0, 1.3)]
  ),
  Maneuver(
    'approaching a 0mph car while cruising at 40mph from 150m away',
    duration=30.,
    initial_speed = 40. * CV.MPH_TO_MS,
    lead_relevancy=True,
    initial_distance_lead=150.,
    speed_lead_values = [0.*CV.MPH_TO_MS, 0.*CV.MPH_TO_MS],
    speed_lead_breakpoints = [0., 100.],
    cruise_button_presses = [(CB.DECEL_SET, 1.2), (0, 1.3)]
  ),
  Maneuver(
    'steady state following a car at 20m/s, then lead decel to 0mph at 1m/s^2',
    duration=50.,
    initial_speed = 20.,
    lead_relevancy=True,
    initial_distance_lead=35.,
    speed_lead_values = [20.*CV.MPH_TO_MS, 20.*CV.MPH_TO_MS, 0.*CV.MPH_TO_MS],
    speed_lead_breakpoints = [0., 15., 35.0],
    cruise_button_presses = [(CB.DECEL_SET, 1.2), (0, 1.3)]
  ),
  Maneuver(
    'steady state following a car at 20m/s, then lead decel to 0mph at 2m/s^2',
    duration=50.,
    initial_speed = 20.,
    lead_relevancy=True,
    initial_distance_lead=35.,
    speed_lead_values = [20.*CV.MPH_TO_MS, 20.*CV.MPH_TO_MS, 0.*CV.MPH_TO_MS],
    speed_lead_breakpoints = [0., 15., 25.0],
    cruise_button_presses = [(CB.DECEL_SET, 1.2), (0, 1.3)]
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
                            (CB.RES_ACCEL, 1.8), (0.0, 1.9)]
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
                            (CB.RES_ACCEL, 1.6), (0.0, 1.7)]
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
                            (CB.RES_ACCEL, 1.6), (0.0, 1.7)]
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
                            (CB.RES_ACCEL, 2.2), (0.0, 2.3)]
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
                            (CB.RES_ACCEL, 1.6), (0.0, 1.7)]
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
                            (CB.RES_ACCEL, 2.2), (0.0, 2.3)]
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
                            (CB.RES_ACCEL, 2.2), (0.0, 2.3)]
  )
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
        view_html += "<td><img class='maneuver_graph' src='%s'/></td>" % (os.path.join("maneuver" + str(i+1).zfill(2), c), )
      view_html += "</tr>"

    create_dir(output_dir)
    with open(os.path.join(output_dir, "index.html"), "w") as f:
      f.write(view_html)

class LongitudinalControl(unittest.TestCase):
  @classmethod
  def setUpClass(cls):

    setup_output()

    shutil.rmtree('/data/params', ignore_errors=True)

    manager.gctx = {}
    manager.prepare_managed_process('radard')
    manager.prepare_managed_process('controlsd')

    manager.start_managed_process('radard')
    manager.start_managed_process('controlsd')

  @classmethod
  def tearDownClass(cls):
    manager.kill_managed_process('radard')
    manager.kill_managed_process('controlsd')
    time.sleep(5)

  # hack
  def test_longitudinal_setup(self):
    pass

WORKERS = 8
def run_maneuver_worker(k):
  output_dir = os.path.join(os.getcwd(), 'out/longitudinal')
  for i, man in enumerate(maneuvers[k::WORKERS]):
    score, plot = man.evaluate()
    plot.write_plot(output_dir, "maneuver" + str(WORKERS * i + k+1).zfill(2))

for k in xrange(WORKERS):
  setattr(LongitudinalControl,
    "test_longitudinal_maneuvers_%d" % (k+1),
    lambda self, k=k: run_maneuver_worker(k))

if __name__ == "__main__":
  unittest.main()

