#!/usr/bin/env python3
import argparse
import base64
import io
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt

from openpilot.tools.lib.logreader import LogReader
from openpilot.system.hardware.hw import Paths


def report(platform, maneuvers):
  output_path = Path(__file__).resolve().parent / "longitudinal_reports"
  output_fn = output_path / f"{platform}_{time.strftime('%Y%m%d-%H_%M_%S')}.html"
  output_path.mkdir(exist_ok=True)
  with open(output_fn, "w") as f:
    f.write("<h1>Longitudinal maneuver report</h1>\n")
    f.write(f"<h3>{platform}</h3>\n")
    for description, runs in maneuvers:
      print('plotting maneuver:', description, 'runs:', len(runs))
      f.write("<div style='border-top: 1px solid #000; margin: 20px 0;'></div>\n")
      f.write(f"<h2>{description}</h2>\n")
      for run, msgs in enumerate(runs):
        t_carControl, carControl = zip(*[(m.logMonoTime, m.carControl) for m in msgs if m.which() == 'carControl'], strict=True)
        t_carOutput, carOutput = zip(*[(m.logMonoTime, m.carOutput) for m in msgs if m.which() == 'carOutput'], strict=True)
        t_carState, carState = zip(*[(m.logMonoTime, m.carState) for m in msgs if m.which() == 'carState'], strict=True)
        t_longitudinalPlan, longitudinalPlan = zip(*[(m.logMonoTime, m.longitudinalPlan) for m in msgs if m.which() == 'longitudinalPlan'], strict=True)

        longActive = [m.longActive for m in carControl]
        gasPressed = [m.gasPressed for m in carState]
        brakePressed = [m.brakePressed for m in carState]

        maneuver_valid = all(longActive) and not (any(gasPressed) or any(brakePressed))

        _open = 'open' if maneuver_valid else ''
        title = f'Run #{int(run)+1}' + (' <span style="color: red">(invalid maneuver!)</span>' if not maneuver_valid else '')

        f.write(f"<details {_open}><summary><h3 style='display: inline-block;'>{title}</h3></summary>\n")

        plt.rcParams['font.size'] = 40
        fig = plt.figure(figsize=(30, 25))
        ax = fig.subplots(4, 1, sharex=True, gridspec_kw={'hspace': 0, 'height_ratios': [5, 3, 1, 1]})

        ax[0].grid(linewidth=4)
        ax[0].plot(t_carControl, [m.actuators.accel for m in carControl], label='carControl.actuators.accel', linewidth=6)
        ax[0].plot(t_carOutput, [m.actuatorsOutput.accel for m in carOutput], label='carOutput.actuatorsOutput.accel', linewidth=6)
        ax[0].plot(t_longitudinalPlan, [m.aTarget for m in longitudinalPlan], label='longitudinalPlan.aTarget', linewidth=6)
        ax[0].plot(t_carState, [m.aEgo for m in carState], label='carState.aEgo', linewidth=6)
        # TODO localizer accel
        ax[0].set_ylabel('Acceleration (m/s^2)')
        #ax[0].set_ylim(-6.5, 6.5)
        ax[0].legend()

        ax[1].grid(linewidth=4)
        ax[1].plot(t_carState, [m.vEgo for m in carState], 'g', label='vEgo', linewidth=6)
        ax[1].set_ylabel('Velocity (m/s)')
        ax[1].legend()

        ax[2].plot(t_carControl, longActive, label='longActive', linewidth=6)
        ax[3].plot(t_carState, gasPressed, label='gasPressed', linewidth=6)
        ax[3].plot(t_carState, brakePressed, label='brakePressed', linewidth=6)
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

  report(platform, maneuvers)
