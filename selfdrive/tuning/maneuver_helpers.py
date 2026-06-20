import os
import pprint
import webbrowser
from collections.abc import Callable
from enum import IntEnum
from pathlib import Path

from openpilot.tools.lib.logreader import LogReader
from openpilot.system.hardware.hw import Paths


class Axis(IntEnum):
  TIME = 0
  EGO_POSITION = 1
  LEAD_DISTANCE= 2
  EGO_V = 3
  LEAD_V = 4
  EGO_A = 5
  D_REL = 6

axis_labels = {Axis.TIME: 'Time (s)',
               Axis.EGO_POSITION: 'Ego position (m)',
               Axis.LEAD_DISTANCE: 'Lead absolute position (m)',
               Axis.EGO_V: 'Ego Velocity (m/s)',
               Axis.LEAD_V: 'Lead Velocity (m/s)',
               Axis.EGO_A: 'Ego acceleration (m/s^2)',
               Axis.D_REL: 'Lead distance (m)'}


def format_car_params(CP):
  return pprint.pformat({k: v for k, v in CP.to_dict().items() if not k.endswith('DEPRECATED')}, indent=2)


def load_maneuver_route(route: str, active_fn: Callable, only_union_types: bool = False):
  if '/' in route or '|' in route:
    lr = LogReader(route, only_union_types=only_union_types)
  else:
    segs = [seg for seg in os.listdir(Paths.log_root()) if route in seg]
    lr = LogReader([os.path.join(Paths.log_root(), seg, 'rlog.zst') for seg in segs], only_union_types=only_union_types)

  CP = lr.first('carParams')
  ID = lr.first('initData')
  platform = CP.carFingerprint
  print('processing report for', platform)

  maneuvers: list[tuple[str, list[list]]] = []
  active_prev = False
  description_prev = None

  for msg in lr:
    if msg.which() == 'alertDebug':
      active = active_fn(msg.alertDebug.alertText1)
      if active and not active_prev:
        if msg.alertDebug.alertText2 == description_prev:
          maneuvers[-1][1].append([])
        else:
          maneuvers.append((msg.alertDebug.alertText2, [[]]))
        description_prev = maneuvers[-1][0]
      active_prev = active

    if active_prev:
      maneuvers[-1][1][-1].append(msg)

  return platform, CP, ID, maneuvers


def init_report_builder(title: str, platform: str, route: str, _description, CP, ID) -> list[str]:
  builder = [
    "<style>summary { cursor: pointer; }\n td, th { padding: 8px; } </style>\n",
    f"<h1>{title}</h1>\n",
    f"<h3>{platform}</h3>\n",
    f"<h3>{route}</h3>\n",
    f"<h3>{ID.gitCommit}, {ID.gitBranch}, {ID.gitRemote}</h3>\n",
  ]
  if _description is not None:
    builder.append(f"<h3>Description: {_description}</h3>\n")
  builder.append(f"<details><summary><h3 style='display: inline-block;'>CarParams</h3></summary><pre>{format_car_params(CP)}</pre></details>\n")
  builder.append('{ summary }')
  return builder


def write_report(output_dir: str, platform: str, route: str, builder: list[str], summary: list[str]):
  output_path = Path(__file__).resolve().parent / output_dir
  output_fn = output_path / f"{platform}_{route.replace('/', '_')}.html"
  output_path.mkdir(exist_ok=True)

  sum_idx = builder.index('{ summary }')
  builder[sum_idx:sum_idx + 1] = summary

  with open(output_fn, "w") as f:
    f.write(''.join(builder))

  print(f"\nOpening report: {output_fn}\n")
  webbrowser.open_new_tab(str(output_fn))
