#!/usr/bin/env python3
import argparse
import base64
import io
import math
from collections import defaultdict
import matplotlib.pyplot as plt
from openpilot.common.utils import tabulate

from openpilot.selfdrive.tuning.maneuver_helpers import init_report_builder, load_maneuver_route, write_report


def report(platform, route, _description, CP, ID, maneuvers):
  target_cross_times = defaultdict(list)
  builder = init_report_builder("Longitudinal maneuver report", platform, route, _description, CP, ID)

  for description, runs in maneuvers:
    print(f'plotting maneuver: {description}, runs: {len(runs)}')
    builder.append("<div style='border-top: 1px solid #000; margin: 20px 0;'></div>\n")
    builder.append(f"<h2>{description}</h2>\n")
    for run, msgs in enumerate(runs):
      t_carControl, carControl = zip(*[(m.logMonoTime, m.carControl) for m in msgs if m.which() == 'carControl'], strict=True)
      t_carOutput, carOutput = zip(*[(m.logMonoTime, m.carOutput) for m in msgs if m.which() == 'carOutput'], strict=True)
      t_carState, carState = zip(*[(m.logMonoTime, m.carState) for m in msgs if m.which() == 'carState'], strict=True)
      t_livePose, livePose = zip(*[(m.logMonoTime, m.livePose) for m in msgs if m.which() == 'livePose'], strict=True)
      t_longitudinalPlan, longitudinalPlan = zip(*[(m.logMonoTime, m.longitudinalPlan) for m in msgs if m.which() == 'longitudinalPlan'], strict=True)

      # make time relative seconds
      t_carControl = [(t - t_carControl[0]) / 1e9 for t in t_carControl]
      t_carOutput = [(t - t_carOutput[0]) / 1e9 for t in t_carOutput]
      t_carState = [(t - t_carState[0]) / 1e9 for t in t_carState]
      t_livePose = [(t - t_livePose[0]) / 1e9 for t in t_livePose]
      t_longitudinalPlan = [(t - t_longitudinalPlan[0]) / 1e9 for t in t_longitudinalPlan]

      # maneuver validity
      longActive = [m.longActive for m in carControl]
      maneuver_valid = all(longActive) and (not any(cs.cruiseState.standstill for cs in carState) or CP.autoResumeSng)

      _open = 'open' if maneuver_valid else ''
      title = f'Run #{int(run)+1}' + (' <span style="color: red">(invalid maneuver!)</span>' if not maneuver_valid else '')

      builder.append(f"<details {_open}><summary><h3 style='display: inline-block;'>{title}</h3></summary>\n")

      # get first acceleration target and first intersection
      aTarget = longitudinalPlan[0].aTarget
      target_cross_time = None
      builder.append(f'<h3 style="font-weight: normal">Initial aTarget: {round(aTarget, 2)} m/s^2')

      # Localizer is noisy, require two consecutive 20Hz frames above threshold
      prev_crossed = False
      for t, lp in zip(t_livePose, livePose, strict=True):
        crossed = (0 < aTarget < lp.accelerationDevice.x) or (0 > aTarget > lp.accelerationDevice.x)
        if crossed and prev_crossed:
          builder.append(f', <strong>crossed in {t:.3f}s</strong>')
          target_cross_time = t
          if maneuver_valid:
            target_cross_times[description].append(t)
          break
        prev_crossed = crossed
      else:
        builder.append(', <strong>not crossed</strong>')
      builder.append('</h3>')

      pitches = [math.degrees(m.orientationNED[1]) for m in carControl]
      builder.append(f'<h3 style="font-weight: normal">Average pitch: <strong>{sum(pitches) / len(pitches):0.2f} degrees</strong></h3>')

      plt.rcParams['font.size'] = 40
      fig = plt.figure(figsize=(30, 26))
      ax = fig.subplots(4, 1, sharex=True, gridspec_kw={'height_ratios': [5, 3, 1, 1]})

      ax[0].grid(linewidth=4)
      ax[0].plot(t_carControl, [m.actuators.accel for m in carControl], label='carControl.actuators.accel', linewidth=6)
      ax[0].plot(t_carOutput, [m.actuatorsOutput.accel for m in carOutput], label='carOutput.actuatorsOutput.accel', linewidth=6)
      ax[0].plot(t_longitudinalPlan, [m.aTarget for m in longitudinalPlan], label='longitudinalPlan.aTarget', linewidth=6)
      ax[0].plot(t_carState, [m.aEgo for m in carState], label='carState.aEgo', linewidth=6)
      ax[0].plot(t_livePose, [m.accelerationDevice.x for m in livePose], label='livePose.accelerationDevice.x', linewidth=6)
      # TODO localizer accel
      ax[0].set_ylabel('Acceleration (m/s^2)')
      #ax[0].set_ylim(-6.5, 6.5)
      ax[0].legend(prop={'size': 30})

      if target_cross_time is not None:
        ax[0].plot(target_cross_time, aTarget, marker='o', markersize=50, markeredgewidth=7, markeredgecolor='black', markerfacecolor='None')

      ax[1].grid(linewidth=4)
      ax[1].plot(t_carState, [m.vEgo for m in carState], 'g', label='vEgo', linewidth=6)
      ax[1].set_ylabel('Velocity (m/s)')
      ax[1].legend()

      ax[2].plot(t_carControl, longActive, label='longActive', linewidth=6)
      ax[3].plot(t_carState, [m.gasPressed for m in carState], label='gasPressed', linewidth=6)
      ax[3].plot(t_carState, [m.brakePressed for m in carState], label='brakePressed', linewidth=6)
      for i in (2, 3):
        ax[i].set_yticks([0, 1], minor=False)
        ax[i].set_ylim(-1, 2)
        ax[i].legend()

      ax[-1].set_xlabel("Time (s)")
      fig.tight_layout()

      buffer = io.BytesIO()
      fig.savefig(buffer, format='webp')
      plt.close(fig)
      buffer.seek(0)
      builder.append(f"<img src='data:image/webp;base64,{base64.b64encode(buffer.getvalue()).decode()}' style='width:100%; max-width:800px;'>\n")
      builder.append("</details>\n")

  summary = ["<h2>Summary</h2>\n"]
  cols = ['maneuver', 'crossed', 'runs', 'mean', 'min', 'max']
  table = []
  for description, runs in maneuvers:
    times = target_cross_times[description]
    l = [description, len(times), len(runs)]
    if len(times):
      l.extend([round(sum(times) / len(times), 2), round(min(times), 2), round(max(times), 2)])
    table.append(l)
  summary.append(tabulate(table, headers=cols, tablefmt='html', numalign='left') + '\n')

  write_report("longitudinal_reports", platform, route, builder, summary)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generate longitudinal maneuver report from route')
  parser.add_argument('route', type=str, help='Route name (e.g. 00000000--5f742174be)')
  parser.add_argument('description', type=str, nargs='?')

  args = parser.parse_args()

  platform, CP, ID, maneuvers = load_maneuver_route(args.route, lambda text: 'Maneuver Active' in text)
  report(platform, args.route, args.description, CP, ID, maneuvers)
