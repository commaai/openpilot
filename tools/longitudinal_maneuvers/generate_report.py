#!/usr/bin/env python3
import argparse
import base64
import io
import os
import pprint
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

from openpilot.tools.lib.logreader import LogReader
from openpilot.system.hardware.hw import Paths


def format_car_params(CP):
  return pprint.pformat({k: v for k, v in CP.to_dict().items() if not k.endswith('DEPRECATED')}, indent=2)


def report(platform, route, CP, maneuvers):
  output_path = Path(__file__).resolve().parent / "longitudinal_reports"
  output_fn = output_path / f"{platform}_{route.replace('/', '_')}.html"
  output_path.mkdir(exist_ok=True)
  target_cross_times = defaultdict(list)
  with open(output_fn, "w") as f:
    f.write("<h1>Longitudinal maneuver report</h1>\n")
    f.write(f"<h3>{platform}</h3>\n")
    f.write(f"<h3>{route}</h3>\n")
    f.write(f"<details><summary><h3 style='display: inline-block;'>CarParams</h3></summary><pre>{format_car_params(CP)}</pre></details>\n")
    for description, runs in maneuvers:
      print(f'plotting maneuver: {description}, runs: {len(runs)}')
      f.write("<div style='border-top: 1px solid #000; margin: 20px 0;'></div>\n")
      f.write(f"<h2>{description}</h2>\n")
      for run, msgs in enumerate(runs):
        t_carControl, carControl = zip(*[(m.logMonoTime, m.carControl) for m in msgs if m.which() == 'carControl'], strict=True)
        t_carOutput, carOutput = zip(*[(m.logMonoTime, m.carOutput) for m in msgs if m.which() == 'carOutput'], strict=True)
        t_carState, carState = zip(*[(m.logMonoTime, m.carState) for m in msgs if m.which() == 'carState'], strict=True)
        t_longitudinalPlan, longitudinalPlan = zip(*[(m.logMonoTime, m.longitudinalPlan) for m in msgs if m.which() == 'longitudinalPlan'], strict=True)

        # make time relative seconds
        t_carControl = [(t - t_carControl[0]) / 1e9 for t in t_carControl]
        t_carOutput = [(t - t_carOutput[0]) / 1e9 for t in t_carOutput]
        t_carState = [(t - t_carState[0]) / 1e9 for t in t_carState]
        t_longitudinalPlan = [(t - t_longitudinalPlan[0]) / 1e9 for t in t_longitudinalPlan]

        # maneuver validity
        longActive = [m.longActive for m in carControl]
        maneuver_valid = all(longActive)

        _open = 'open' if maneuver_valid else ''
        title = f'Run #{int(run)+1}' + (' <span style="color: red">(invalid maneuver!)</span>' if not maneuver_valid else '')

        f.write(f"<details {_open}><summary><h3 style='display: inline-block;'>{title}</h3></summary>\n")

        # get first acceleration target and first intersection
        aTarget = longitudinalPlan[0].aTarget
        target_cross_time = None
        f.write(f'<h3 style="font-weight: normal">Initial aTarget: {aTarget} m/s^2')
        for t, cs in zip(t_carState, carState, strict=True):
          if (0 < aTarget < cs.aEgo) or (0 > aTarget > cs.aEgo):
            f.write(f', <strong>crossed in {t:.3f}s</strong>')
            target_cross_time = t
            if maneuver_valid:
              target_cross_times[description].append(t)
            break
        else:
          f.write(', <strong>not crossed</strong>')
        f.write('</h3>')

        plt.rcParams['font.size'] = 40
        fig = plt.figure(figsize=(30, 26))
        ax = fig.subplots(4, 1, sharex=True, gridspec_kw={'height_ratios': [5, 3, 1, 1]})

        ax[0].grid(linewidth=4)
        ax[0].plot(t_carControl, [m.actuators.accel for m in carControl], label='carControl.actuators.accel', linewidth=6)
        ax[0].plot(t_carOutput, [m.actuatorsOutput.accel for m in carOutput], label='carOutput.actuatorsOutput.accel', linewidth=6)
        ax[0].plot(t_longitudinalPlan, [m.aTarget for m in longitudinalPlan], label='longitudinalPlan.aTarget', linewidth=6)
        ax[0].plot(t_carState, [m.aEgo for m in carState], label='carState.aEgo', linewidth=6)
        # TODO localizer accel
        ax[0].set_ylabel('Acceleration (m/s^2)')
        #ax[0].set_ylim(-6.5, 6.5)
        ax[0].legend()

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
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        f.write(f"<img src='data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}' style='width:100%; max-width:800px;'>\n")
        f.write("</details>\n")

    f.write("<h2>Summary</h2>\n")
    for description, runs in maneuvers:
      times = target_cross_times[description]
      f.write(f"<h3>{description}</h3>\n")
      f.write(f"<p>Target crossed {len(times)} out of {len(runs)} runs</p>\n")
      if len(times):
        f.write(f"<p>Mean time to cross: {sum(times) / len(times):.3f}s, min: {min(times):.3f}s, max: {max(times):.3f}s</p>\n")

  print(f"\nReport written to {output_fn}\n")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generate longitudinal maneuver report from route')
  parser.add_argument('route', type=str, help='Route name (e.g. 00000000--5f742174be)')

  args = parser.parse_args()

  if '/' in args.route or '|' in args.route:
    lr = LogReader(args.route)
  else:
    segs = [seg for seg in os.listdir(Paths.log_root()) if args.route in seg]
    lr = LogReader([os.path.join(Paths.log_root(), seg, 'rlog') for seg in segs])

  CP = lr.first('carParams')
  platform = CP.carFingerprint
  print('processing report for', platform)

  maneuvers: list[tuple[str, list[list]]] = []
  active_prev = False
  description_prev = None

  for msg in lr:
    if msg.which() == 'alertDebug':
      active = 'Maneuver Active' in msg.alertDebug.alertText1
      if active and not active_prev:
        if msg.alertDebug.alertText2 == description_prev:
          maneuvers[-1][1].append([])
        else:
          maneuvers.append((msg.alertDebug.alertText2, [[]]))
        description_prev = maneuvers[-1][0]
      active_prev = active

    if active_prev:
      maneuvers[-1][1][-1].append(msg)

  report(platform, args.route, CP, maneuvers)
