#!/usr/bin/env python3
import argparse
import base64
import io
import math
import numpy as np
import os
import webbrowser
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from openpilot.common.utils import tabulate

from cereal import car
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.controls.lib.latcontrol_torque import LP_FILTER_CUTOFF_HZ
from openpilot.tools.lib.logreader import LogReader
from openpilot.system.hardware.hw import Paths
from openpilot.common.constants import CV
from openpilot.tools.longitudinal_maneuvers.generate_report import format_car_params


def lat_accel(curvature, v):
  return curvature * max(v, 1.0) ** 2


def report(platform, route, _description, CP, ID, maneuvers):
  output_path = Path(__file__).resolve().parent / "lateral_reports"
  output_fn = output_path / f"{platform}_{route.replace('/', '_')}.html"
  output_path.mkdir(exist_ok=True)
  target_cross_times = defaultdict(list)

  builder = [
    "<style>summary { cursor: pointer; }\n td, th { padding: 8px; } </style>\n",
    "<h1>Lateral maneuver report</h1>\n",
    f"<h3>{platform}</h3>\n",
    f"<h3>{route}</h3>\n",
    f"<h3>{ID.gitCommit}, {ID.gitBranch}, {ID.gitRemote}</h3>\n",
  ]
  if _description is not None:
    builder.append(f"<h3>Description: {_description}</h3>\n")
  builder.append(f"<details><summary><h3 style='display: inline-block;'>CarParams</h3></summary><pre>{format_car_params(CP)}</pre></details>\n")
  builder.append('{ summary }')  # to be replaced below
  for description, runs in maneuvers:
    # filter incomplete runs
    completed_runs = [msgs for msgs in runs
                      if any(m.alertDebug.alertText1 == 'Complete' for m in msgs if m.which() == 'alertDebug')]
    print(f'plotting maneuver: {description}')
    if not completed_runs:
      continue
    builder.append("<div style='border-top: 1px solid #000; margin: 20px 0;'></div>\n")
    builder.append(f"<h2>{description}</h2>\n")
    for run, msgs in enumerate(completed_runs):
      t_carControl, carControl = zip(*[(m.logMonoTime, m.carControl) for m in msgs if m.which() == 'carControl'], strict=True)
      t_carState, carState = zip(*[(m.logMonoTime, m.carState) for m in msgs if m.which() == 'carState'], strict=True)
      t_controlsState, controlsState = zip(*[(m.logMonoTime, m.controlsState) for m in msgs if m.which() == 'controlsState'], strict=True)
      t_lateralPlan, lateralPlan = zip(*[(m.logMonoTime, m.lateralManeuverPlan) for m in msgs if m.which() == 'lateralManeuverPlan' and m.valid], strict=True)
      t_carOutput, carOutput = zip(*[(m.logMonoTime, m.carOutput) for m in msgs if m.which() == 'carOutput'], strict=True)

      # make time relative seconds
      t_carControl = [(t - t_carControl[0]) / 1e9 for t in t_carControl]
      t_carState = [(t - t_carState[0]) / 1e9 for t in t_carState]
      t_controlsState = [(t - t_controlsState[0]) / 1e9 for t in t_controlsState]
      t_lateralPlan = [(t - t_lateralPlan[0]) / 1e9 for t in t_lateralPlan]
      t_carOutput = [(t - t_carOutput[0]) / 1e9 for t in t_carOutput]

      # maneuver validity
      latActive = [m.latActive for m in carControl]
      maneuver_valid = all(latActive) and not any(cs.steeringPressed for cs in carState)

      _open = 'open' if maneuver_valid else ''
      title = f'Run #{int(run)+1}' + (' <span style="color: red">(invalid maneuver!)</span>' if not maneuver_valid else '')

      builder.append(f"<details {_open}><summary><h3 style='display: inline-block;'>{title}</h3></summary>\n")

      baseline_accel = lat_accel(controlsState[0].curvature, carState[0].vEgo)
      v_ego = [m.vEgo for m in carState]
      cross_markers = []

      if description.startswith('sine'):
        amplitude = max(abs(lat_accel(lp.desiredCurvature, v) - baseline_accel)
                        for lp, v in zip(lateralPlan, v_ego, strict=False))
        threshold = amplitude * 0.5
        builder.append('<h3 style="font-weight: normal">50% peak')
        for t, cs, v in zip(t_controlsState, controlsState, v_ego, strict=False):
          actual = lat_accel(cs.curvature, v) - baseline_accel
          if abs(actual) > threshold:
            builder.append(f', <strong>crossed in {t:.3f}s</strong>')
            cross_markers.append((t, actual + baseline_accel))
            if maneuver_valid:
              target_cross_times[description].append(t)
            break
        else:
          builder.append(', <strong>not crossed</strong>')
        builder.append('</h3>')
        if maneuver_valid:
          target_cross_times.setdefault(description, [])
      else:
        action_targets = [(0, lat_accel(lateralPlan[0].desiredCurvature, v_ego[0]) - baseline_accel)]
        for i in range(1, min(len(lateralPlan), len(v_ego))):
          if abs(lateralPlan[i].desiredCurvature - lateralPlan[i - 1].desiredCurvature) > 0.001:
            desired = lat_accel(lateralPlan[i].desiredCurvature, v_ego[i]) - baseline_accel
            action_targets.append((i, desired))

        for j, (start_i, act_target) in enumerate(action_targets):
          start_time = t_lateralPlan[start_i]
          end_time = t_lateralPlan[action_targets[j + 1][0]] if j + 1 < len(action_targets) else t_controlsState[-1]

          builder.append(f'<h3 style="font-weight: normal">aTarget: {round(act_target, 1)} m/s^2')
          prev_crossed = False
          for t, cs, v in zip(t_controlsState, controlsState, v_ego, strict=False):
            if not (start_time <= t <= end_time):
              continue
            actual_accel = lat_accel(cs.curvature, v) - baseline_accel
            crossed = (0 < act_target < actual_accel) or (0 > act_target > actual_accel)
            if crossed and prev_crossed:
              cross_time = t - start_time
              builder.append(f', <strong>crossed in {cross_time:.3f}s</strong>')
              cross_markers.append((t, act_target + baseline_accel))
              if maneuver_valid:
                target_cross_times[description].append(cross_time)
              break
            prev_crossed = crossed
          else:
            builder.append(', <strong>not crossed</strong>')
          builder.append('</h3>')
          if maneuver_valid:
            target_cross_times.setdefault(description, [])

      plt.rcParams['font.size'] = 40
      fig = plt.figure(figsize=(30, 30))
      ax = fig.subplots(4, 1, sharex=True, gridspec_kw={'height_ratios': [5, 3, 3, 3]})

      ax[0].grid(linewidth=4)
      desired_lat_accel = [lat_accel(m.desiredCurvature, v) for m, v in zip(lateralPlan, v_ego, strict=False)]
      if description.startswith('sine'):
        ax[0].plot(t_lateralPlan[:len(desired_lat_accel)], desired_lat_accel, label='desired lat accel', linewidth=6)
      else:
        t_desired = [t_lateralPlan[0]] + t_lateralPlan[:len(desired_lat_accel)]
        desired_lat_accel = [baseline_accel] + desired_lat_accel
        ax[0].step(t_desired, desired_lat_accel, label='desired lat accel', linewidth=6, where='post')
      actual_lat_accel = [lat_accel(cs.curvature, v) for cs, v in zip(controlsState, v_ego, strict=False)]
      ax[0].plot(t_controlsState[:len(actual_lat_accel)], actual_lat_accel, label='actual lat accel', linewidth=6)
      ax[0].set_ylabel('Lateral Accel (m/s^2)')

      for ct, cv in cross_markers:
        ax[0].plot(ct, cv, marker='o', markersize=50, markeredgewidth=7, markeredgecolor='black', markerfacecolor='None')

      ax2 = ax[0].twinx()
      if CP.steerControlType == car.CarParams.SteerControlType.angle:
        ax2.plot(t_carOutput, [-m.actuatorsOutput.steeringAngleDeg for m in carOutput], 'C2', label='steer angle', linewidth=6)
      else:
        ax2.plot(t_carOutput, [-m.actuatorsOutput.torque for m in carOutput], 'C2', label='steer torque', linewidth=6)

      h1, l1 = ax[0].get_legend_handles_labels()
      h2, l2 = ax2.get_legend_handles_labels()
      ax[0].legend(h1 + h2, l1 + l2, prop={'size': 30})

      ax[1].grid(linewidth=4)
      ax[1].plot(t_carState, [v * CV.MS_TO_MPH for v in v_ego], label='vEgo', linewidth=6)
      ax[1].set_ylabel('Velocity (mph)')
      ax[1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
      ax[1].legend()

      t_accel = np.array(t_controlsState[:len(actual_lat_accel)])
      raw_jerk = np.gradient(actual_lat_accel, t_accel)
      dt_avg = np.mean(np.diff(t_accel))
      jerk_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * LP_FILTER_CUTOFF_HZ), dt_avg)
      filtered_jerk = [jerk_filter.update(j) for j in raw_jerk]
      ax[2].grid(linewidth=4)
      ax[2].plot(t_accel, filtered_jerk, label='actual jerk', linewidth=6)
      if CP.steerControlType == car.CarParams.SteerControlType.torque:
        desired_jerk = [cs.lateralControlState.torqueState.desiredLateralJerk for cs in controlsState]
        ax[2].plot(t_controlsState[:len(controlsState)], desired_jerk, label='desired jerk', linewidth=6)
      ax[2].set_ylabel('Jerk (m/s^3)')
      ax[2].legend()

      ax[3].grid(linewidth=4)
      ax[3].plot(t_carControl, [math.degrees(m.orientationNED[0]) for m in carControl], label='roll', linewidth=6)
      ax[3].set_ylabel('Roll (deg)')
      ax[3].legend()

      ax[-1].set_xlabel("Time (s)")
      fig.tight_layout()

      buffer = io.BytesIO()
      fig.savefig(buffer, format='webp')
      plt.close(fig)
      buffer.seek(0)
      builder.append(f"<img src='data:image/webp;base64,{base64.b64encode(buffer.getvalue()).decode()}' style='width:100%; max-width:800px;'>\n")
      builder.append("</details>\n")

  summary = ["<h2>Summary</h2>\n"]
  cols = ['maneuver', 'crossed', 'mean', 'min', 'max']
  table = []
  for description, times in target_cross_times.items():
    l = [description, len(times)]
    if len(times):
      l.extend([round(sum(times) / len(times), 2), round(min(times), 2), round(max(times), 2)])
    table.append(l)
  summary.append(tabulate(table, headers=cols, tablefmt='html', numalign='left') + '\n')

  sum_idx = builder.index('{ summary }')
  builder[sum_idx:sum_idx + 1] = summary

  with open(output_fn, "w") as f:
    f.write(''.join(builder))

  print(f"\nOpening report: {output_fn}\n")
  webbrowser.open_new_tab(str(output_fn))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generate lateral maneuver report from route')
  parser.add_argument('route', type=str, help='Route name (e.g. 00000000--5f742174be)')
  parser.add_argument('description', type=str, nargs='?')

  args = parser.parse_args()

  if '/' in args.route or '|' in args.route:
    lr = LogReader(args.route, only_union_types=True)
  else:
    segs = [seg for seg in os.listdir(Paths.log_root()) if args.route in seg]
    lr = LogReader([os.path.join(Paths.log_root(), seg, 'rlog.zst') for seg in segs], only_union_types=True)

  CP = lr.first('carParams')
  ID = lr.first('initData')
  platform = CP.carFingerprint
  print('processing report for', platform)

  maneuvers: list[tuple[str, list[list]]] = []
  active_prev = False
  description_prev = None

  for msg in lr:
    if msg.which() == 'alertDebug':
      active = 'Active' in msg.alertDebug.alertText1 or msg.alertDebug.alertText1 == 'Complete'
      if active and not active_prev:
        if msg.alertDebug.alertText2 == description_prev:
          maneuvers[-1][1].append([])
        else:
          maneuvers.append((msg.alertDebug.alertText2, [[]]))
        description_prev = maneuvers[-1][0]
      active_prev = active

    if active_prev:
      maneuvers[-1][1][-1].append(msg)

  report(platform, args.route, args.description, CP, ID, maneuvers)
