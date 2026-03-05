#!/usr/bin/env python3
import argparse
import base64
import io
import os
import pprint
import webbrowser
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from openpilot.common.utils import tabulate

from openpilot.tools.lib.logreader import LogReader
from openpilot.system.hardware.hw import Paths
from openpilot.common.constants import CV


def format_car_params(CP):
  return pprint.pformat({k: v for k, v in CP.to_dict().items() if not k.endswith('DEPRECATED')}, indent=2)


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
    print(f'plotting maneuver: {description}, runs: {len(runs)}')
    builder.append("<div style='border-top: 1px solid #000; margin: 20px 0;'></div>\n")
    builder.append(f"<h2>{description}</h2>\n")
    for run, msgs in enumerate(runs):
      t_carControl, carControl = zip(*[(m.logMonoTime, m.carControl) for m in msgs if m.which() == 'carControl'], strict=True)
      t_carState, carState = zip(*[(m.logMonoTime, m.carState) for m in msgs if m.which() == 'carState'], strict=True)
      t_controlsState, controlsState = zip(*[(m.logMonoTime, m.controlsState) for m in msgs if m.which() == 'controlsState'], strict=True)
      t_lateralPlan, lateralPlan = zip(*[(m.logMonoTime, m.lateralManeuverPlan) for m in msgs if m.which() == 'lateralManeuverPlan' and m.valid], strict=True)

      # make time relative seconds
      t_carControl = [(t - t_carControl[0]) / 1e9 for t in t_carControl]
      t_carState = [(t - t_carState[0]) / 1e9 for t in t_carState]
      t_controlsState = [(t - t_controlsState[0]) / 1e9 for t in t_controlsState]
      t_lateralPlan = [(t - t_lateralPlan[0]) / 1e9 for t in t_lateralPlan]

      # maneuver validity
      latActive = [m.latActive for m in carControl]
      maneuver_valid = all(latActive) and not any(cs.steeringPressed for cs in carState)

      _open = 'open' if maneuver_valid else ''
      title = f'Run #{int(run)+1}' + (' <span style="color: red">(invalid maneuver!)</span>' if not maneuver_valid else '')

      builder.append(f"<details {_open}><summary><h3 style='display: inline-block;'>{title}</h3></summary>\n")

      baseline_accel = lat_accel(controlsState[0].curvature, carState[0].vEgo)
      cross_markers = []

      if description.startswith('sine'):
        amplitude = max(abs(lat_accel(lp.desiredCurvature, cs.vEgo) - baseline_accel)
                        for lp, cs in zip(lateralPlan, carState))
        threshold = amplitude * 0.5
        builder.append(f'<h3 style="font-weight: normal">50% peak')
        for t, cs, v in zip(t_controlsState, controlsState, [m.vEgo for m in carState], strict=False):
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
      else:
        action_targets = [(0, lat_accel(lateralPlan[0].desiredCurvature, carState[0].vEgo) - baseline_accel)]
        for i in range(1, min(len(lateralPlan), len(carState))):
          desired = lat_accel(lateralPlan[i].desiredCurvature, carState[i].vEgo) - baseline_accel
          if abs(desired - action_targets[-1][1]) > 0.2:
            action_targets.append((i, desired))

        for j, (start_i, act_target) in enumerate(action_targets):
          start_time = t_lateralPlan[start_i]
          end_time = t_lateralPlan[action_targets[j + 1][0]] if j + 1 < len(action_targets) else t_controlsState[-1]

          builder.append(f'<h3 style="font-weight: normal">aTarget: {round(act_target, 1)} m/s^2')
          prev_crossed = False
          for t, cs, v in zip(t_controlsState, controlsState, [m.vEgo for m in carState], strict=False):
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

      plt.rcParams['font.size'] = 40
      fig = plt.figure(figsize=(30, 18))
      ax = fig.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [5, 3]})

      ax[0].grid(linewidth=4)
      desired_lat_accel = [lat_accel(m.desiredCurvature, v) for m, v in zip(lateralPlan, [s.vEgo for s in carState], strict=False)]
      if description.startswith('sine'):
        ax[0].plot(t_lateralPlan[:len(desired_lat_accel)], desired_lat_accel, label='desired lat accel', linewidth=6)
      else:
        t_desired = [t_lateralPlan[0]] + t_lateralPlan[:len(desired_lat_accel)]
        desired_lat_accel = [baseline_accel] + desired_lat_accel
        ax[0].step(t_desired, desired_lat_accel, label='desired lat accel', linewidth=6, where='post')
      actual_lat_accel = [lat_accel(cs.curvature, v) for cs, v in zip(controlsState, [m.vEgo for m in carState], strict=False)]
      ax[0].plot(t_controlsState[:len(actual_lat_accel)], actual_lat_accel, label='actual lat accel', linewidth=6)
      ax[0].set_ylabel('Lateral Accel (m/s^2)')
      ax[0].legend(prop={'size': 30})

      for ct, cv in cross_markers:
        ax[0].plot(ct, cv, marker='o', markersize=50, markeredgewidth=7, markeredgecolor='black', markerfacecolor='None')

      ax[1].grid(linewidth=4)
      ax[1].plot(t_carState, [m.vEgo * CV.MS_TO_MPH for m in carState], 'g', label='vEgo', linewidth=6)
      ax[1].set_ylabel('Velocity (mph)')
      ax[1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
      ax[1].legend()

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
  for description, runs in maneuvers:
    times = target_cross_times[description]
    targets_per_run = 1 if description.startswith('sine') else 2
    total_targets = len(runs) * targets_per_run
    l = [description, f'{len(times)}/{total_targets}']
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
    lr = LogReader(args.route)
  else:
    segs = [seg for seg in os.listdir(Paths.log_root()) if args.route in seg]
    lr = LogReader([os.path.join(Paths.log_root(), seg, 'rlog.zst') for seg in segs])

  CP = lr.first('carParams')
  ID = lr.first('initData')
  platform = CP.carFingerprint
  print('processing report for', platform)

  maneuvers: list[tuple[str, list[list]]] = []
  active_prev = False
  description_prev = None

  for msg in lr:
    if msg.which() == 'alertDebug':
      active = 'Active' in msg.alertDebug.alertText1
      if active and not active_prev:
        if msg.alertDebug.alertText2 == description_prev:
          maneuvers[-1][1].append([])
        else:
          maneuvers.append((msg.alertDebug.alertText2, [[]]))
        description_prev = maneuvers[-1][0]
      active_prev = active

    if active_prev:
      maneuvers[-1][1][-1].append(msg)

  # filter out aborted runs (steering override)
  for i, (desc, runs) in enumerate(maneuvers):
    maneuvers[i] = (desc, [r for r in runs if not any(m.carState.steeringPressed for m in r if m.which() == 'carState')])
  maneuvers = [(desc, runs) for desc, runs in maneuvers if runs]

  report(platform, args.route, args.description, CP, ID, maneuvers)
