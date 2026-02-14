import io
import sys
import markdown
import numpy as np
import matplotlib.pyplot as plt
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.tests.test_following_distance import desired_follow_distance
from openpilot.selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver

TIME = 0
LEAD_DISTANCE= 2
EGO_V = 3
EGO_A = 5
D_REL = 6

axis_labels = ['Time (s)',
               'Ego position (m)',
               'Lead absolute position (m)',
               'Ego Velocity (m/s)',
               'Lead Velocity (m/s)',
               'Ego acceleration (m/s^2)',
               'Lead distance (m)'
               ]

def get_html_from_results(results, labels, AXIS):
  fig, ax = plt.subplots(figsize=(16, 8))
  for idx, speed in enumerate(list(results.keys())):
    ax.plot(results[speed][:, TIME], results[speed][:, AXIS], label=labels[idx])

  ax.set_xlabel('Time (s)')
  ax.set_ylabel(axis_labels[AXIS])
  ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
  ax.grid(True, linestyle='--', alpha=0.7)
  ax.text(-0.075, 0.5, '.', transform=ax.transAxes, color='none')

  fig_buffer = io.StringIO()
  fig.savefig(fig_buffer, format='svg', bbox_inches='tight')
  plt.close(fig)
  return fig_buffer.getvalue() + '<br/>'

def generate_mpc_tuning_report():
  htmls = []

  results = {}
  name = 'Resuming behind lead'
  labels = []
  for lead_accel in np.linspace(1.0, 4.0, 4):
    man = Maneuver(
      '',
      duration=11,
      initial_speed=0.0,
      lead_relevancy=True,
      initial_distance_lead=desired_follow_distance(0.0, 0.0),
      speed_lead_values=[0.0, 10 * lead_accel],
      cruise_values=[100, 100],
      prob_lead_values=[1.0, 1.0],
      breakpoints=[1., 11],
    )
    valid, results[lead_accel] = man.evaluate()
    labels.append(f'{lead_accel} m/s^2 lead acceleration')

  htmls.append(markdown.markdown('# ' + name))
  htmls.append(get_html_from_results(results, labels, EGO_V))
  htmls.append(get_html_from_results(results, labels, EGO_A))


  results = {}
  name = 'Approaching stopped car from 140m'
  labels = []
  for speed in np.arange(0,45,5):
    man = Maneuver(
      name,
      duration=30.,
      initial_speed=float(speed),
      lead_relevancy=True,
      initial_distance_lead=140.,
      speed_lead_values=[0.0, 0.],
      breakpoints=[0., 30.],
    )
    valid, results[speed] = man.evaluate()
    results[speed][:,2] = results[speed][:,2] - results[speed][:,1]
    labels.append(f'{speed} m/s approach speed')

  htmls.append(markdown.markdown('# ' + name))
  htmls.append(get_html_from_results(results, labels, EGO_A))
  htmls.append(get_html_from_results(results, labels, D_REL))


  results = {}
  name = 'Following 5s (triangular) oscillating lead'
  labels = []
  speed = np.int64(10)
  for oscil in np.arange(0, 10, 1):
    man = Maneuver(
      '',
      duration=30.,
      initial_speed=float(speed),
      lead_relevancy=True,
      initial_distance_lead=desired_follow_distance(speed, speed),
      speed_lead_values=[speed, speed, speed - oscil, speed + oscil, speed - oscil, speed + oscil, speed - oscil],
      breakpoints=[0.,2., 5, 8, 15, 18, 25.],
    )
    valid, results[oscil] = man.evaluate()
    labels.append(f'{oscil} m/s oscillation size')

  htmls.append(markdown.markdown('# ' + name))
  htmls.append(get_html_from_results(results, labels, D_REL))
  htmls.append(get_html_from_results(results, labels, EGO_V))
  htmls.append(get_html_from_results(results, labels, EGO_A))


  results = {}
  name = 'Following 5s (sinusoidal) oscillating lead'
  labels = []
  speed = np.int64(10)
  duration = float(30)
  f_osc = 1. / 5
  for oscil in np.arange(0, 10, 1):
    bps = DT_MDL * np.arange(int(duration / DT_MDL))
    lead_speeds = speed + oscil * np.sin(2 * np.pi * f_osc * bps)
    man = Maneuver(
      '',
      duration=duration,
      initial_speed=float(speed),
      lead_relevancy=True,
      initial_distance_lead=desired_follow_distance(speed, speed),
      speed_lead_values=lead_speeds,
      breakpoints=bps,
    )
    valid, results[oscil] = man.evaluate()
    labels.append(f'{oscil} m/s oscilliation size')

  htmls.append(markdown.markdown('# ' + name))
  htmls.append(get_html_from_results(results, labels, D_REL))
  htmls.append(get_html_from_results(results, labels, EGO_V))
  htmls.append(get_html_from_results(results, labels, EGO_A))


  results = {}
  name = 'Speed profile when converging to steady state lead at 30m/s'
  labels = []
  for distance in np.arange(20, 140, 10):
    man = Maneuver(
      '',
      duration=50,
      initial_speed=30.0,
      lead_relevancy=True,
      initial_distance_lead=distance,
      speed_lead_values=[30.0],
      breakpoints=[0.],
    )
    valid, results[distance] = man.evaluate()
    results[distance][:,2] = results[distance][:,2] - results[distance][:,1]
    labels.append(f'{distance} m initial distance')

  htmls.append(markdown.markdown('# ' + name))
  htmls.append(get_html_from_results(results, labels, EGO_V))
  htmls.append(get_html_from_results(results, labels, D_REL))


  results = {}
  name = 'Speed profile when converging to steady state lead at 20m/s'
  labels = []
  for distance in np.arange(20, 140, 10):
    man = Maneuver(
      '',
      duration=50,
      initial_speed=20.0,
      lead_relevancy=True,
      initial_distance_lead=distance,
      speed_lead_values=[20.0],
      breakpoints=[0.],
    )
    valid, results[distance] = man.evaluate()
    results[distance][:,2] = results[distance][:,2] - results[distance][:,1]
    labels.append(f'{distance} m initial distance')

  htmls.append(markdown.markdown('# ' + name))
  htmls.append(get_html_from_results(results, labels, EGO_V))
  htmls.append(get_html_from_results(results, labels, D_REL))


  results = {}
  name = 'Following car at 30m/s that comes to a stop'
  labels = []
  for stop_time in np.arange(4, 14, 1):
    man = Maneuver(
      '',
      duration=30,
      initial_speed=30.0,
      cruise_values=[30.0, 30.0, 30.0],
      lead_relevancy=True,
      initial_distance_lead=60.0,
      speed_lead_values=[30.0, 30.0, 0.0],
      breakpoints=[0., 5., 5 + stop_time],
    )
    valid, results[stop_time] = man.evaluate()
    results[stop_time][:,2] = results[stop_time][:,2] - results[stop_time][:,1]
    labels.append(f'{stop_time} seconds stop time')

  htmls.append(markdown.markdown('# ' + name))
  htmls.append(get_html_from_results(results, labels, EGO_A))
  htmls.append(get_html_from_results(results, labels, D_REL))


  results = {}
  name = 'Response to cut-in at half follow distance'
  labels = []
  for speed in np.arange(0, 40, 5):
    man = Maneuver(
      '',
      duration=20,
      initial_speed=float(speed),
      cruise_values=[speed, speed, speed],
      lead_relevancy=True,
      initial_distance_lead=desired_follow_distance(speed, speed)/2,
      speed_lead_values=[speed, speed, speed],
      prob_lead_values=[0.0, 0.0, 1.0],
      breakpoints=[0., 5.0, 5.01],
    )
    valid, results[speed] = man.evaluate()
    labels.append(f'{speed} m/s speed')

  htmls.append(markdown.markdown('# ' + name))
  htmls.append(get_html_from_results(results, labels, EGO_A))
  htmls.append(get_html_from_results(results, labels, D_REL))


  results = {}
  name = 'Follow a lead that accelerates at 2m/s^2 until steady state speed'
  labels = []
  for speed in np.arange(0, 40, 5):
    man = Maneuver(
      '',
      duration=60,
      initial_speed=0.0,
      lead_relevancy=True,
      initial_distance_lead=desired_follow_distance(0.0, 0.0),
      speed_lead_values=[0.0, 0.0, speed],
      prob_lead_values=[1.0, 1.0, 1.0],
      breakpoints=[0., 1.0, speed/2],
    )
    valid, results[speed] = man.evaluate()
    labels.append(f'{speed} m/s speed')

  htmls.append(markdown.markdown('# ' + name))
  htmls.append(get_html_from_results(results, labels, EGO_V))
  htmls.append(get_html_from_results(results, labels, EGO_A))


  results = {}
  name = 'From stop to cruise'
  labels = []
  for speed in np.arange(0, 40, 5):
    man = Maneuver(
      '',
      duration=50,
      initial_speed=0.0,
      lead_relevancy=True,
      initial_distance_lead=desired_follow_distance(0.0, 0.0),
      speed_lead_values=[0.0, 0.0],
      cruise_values=[0.0, speed],
      prob_lead_values=[0.0, 0.0],
      breakpoints=[1., 1.01],
    )
    valid, results[speed] = man.evaluate()
    labels.append(f'{speed} m/s speed')

  htmls.append(markdown.markdown('# ' + name))
  htmls.append(get_html_from_results(results, labels, EGO_V))
  htmls.append(get_html_from_results(results, labels, EGO_A))


  results = {}
  name = 'From cruise to min'
  labels = []
  for speed in np.arange(10, 40, 5):
    man = Maneuver(
      '',
      duration=50,
      initial_speed=float(speed),
      lead_relevancy=True,
      initial_distance_lead=desired_follow_distance(0.0, 0.0),
      speed_lead_values=[0.0, 0.0],
      cruise_values=[speed, 10.0],
      prob_lead_values=[0.0, 0.0],
      breakpoints=[1., 1.01],
    )
    valid, results[speed] = man.evaluate()
    labels.append(f'{speed} m/s speed')

  htmls.append(markdown.markdown('# ' + name))
  htmls.append(get_html_from_results(results, labels, EGO_V))
  htmls.append(get_html_from_results(results, labels, EGO_A))

  return htmls

if __name__ == '__main__':
  htmls = generate_mpc_tuning_report()

  if len(sys.argv) < 2:
    file_name = 'long_mpc_tune_report.html'
  else:
    file_name = sys.argv[1]

  with open(file_name, 'w') as f:
    f.write(markdown.markdown('# MPC longitudinal tuning report'))
    for html in htmls:
      f.write(html)
