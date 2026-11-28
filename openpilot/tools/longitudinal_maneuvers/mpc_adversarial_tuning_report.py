from collections import defaultdict
import sys
import markdown
import numpy as np

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import long_mpc
from openpilot.selfdrive.selfdrived.selfdrived import LONGITUDINAL_PERSONALITY_MAP
from openpilot.selfdrive.test.longitudinal_maneuvers.test_longitudinal import (
  create_approach_stopped_adv_maneuvers,
  create_hard_braking_adv_maneuvers,
  create_stopped_adv_maneuvers,
)
from openpilot.tools.longitudinal_maneuvers.maneuver_helpers import Axis
from openpilot.tools.longitudinal_maneuvers.mpc_longitudinal_tuning_report import get_html_from_results

REPORT_CSS = """<style>
table { border-collapse: collapse; margin: 12px 0; }
th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
</style>"""

PLOT_AXES = [Axis.EGO_A, Axis.D_REL, Axis.EGO_V]


def render_table(headers, rows):
  head = ''.join(f'<th>{h}</th>' for h in headers)
  body = ''.join('<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>' for row in rows)
  return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def render_section(heading, maneuvers):
  groups = defaultdict(list)
  for maneuver in maneuvers:
    groups[float(maneuver.v_ego[0]), float(maneuver.x_lead[0])].append(maneuver)

  htmls = [markdown.markdown(f'# {heading}')]
  table_rows = []
  for (v0, d0), group in groups.items():
    evaluated = [(m, *m.evaluate()) for m in group]
    results = {i: logs for i, (_, _, logs) in enumerate(evaluated)}
    labels = [m.title for m, _, _ in evaluated]
    styles = [{'linestyle': '--'} if m.only_radar else {} for m, _, _ in evaluated]

    for maneuver, valid, logs in evaluated:
      if maneuver.only_radar:
        continue
      base_logs = next(b_logs for b, _, b_logs in evaluated if b.only_radar and np.array_equal(b.x_lead, maneuver.x_lead))
      base_min_a, adv_min_a = float(np.min(base_logs[:, Axis.EGO_A])), float(np.min(logs[:, Axis.EGO_A]))
      base_min_d, adv_min_d = float(np.min(base_logs[:, Axis.D_REL])), float(np.min(logs[:, Axis.D_REL]))
      table_rows.append([
        maneuver.title,
        f'{base_min_a:+.2f}', f'{adv_min_a:+.2f}', f'{adv_min_a - base_min_a:+.2f}',
        f'{base_min_d:.2f}', f'{adv_min_d:.2f}', f'{adv_min_d - base_min_d:+.2f}',
        'PASS' if valid else 'CRASH',
      ])

    htmls.append(markdown.markdown(f'## v0 = {v0:g} m/s, d0 = {d0:g} m'))
    htmls.append(get_html_from_results(results, labels, PLOT_AXES, styles))

  headers = [
    'Case', 'Radar only min a (m/s^2)', 'Adv min a (m/s^2)', 'Δ min a (m/s^2)',
    'Radar only min d (m)', 'Adv min d (m)', 'Δ min d (m)', 'Result',
  ]
  htmls.insert(1, render_table(headers, table_rows))
  return htmls


def generate_adversarial_tuning_report(personality=log.LongitudinalPersonality.relaxed):
  kwargs = {'personality': personality}
  params_table = render_table(
    ['Parameter', 'Value'],
    [
      ['CLIP_MODEL_LEAD_ACCEL', f'{long_mpc.CLIP_MODEL_LEAD_ACCEL:g} m/s^2'],
      ['LEAD_FALLBACK_RANGE', f'{long_mpc.LEAD_FALLBACK_RANGE} m/s^2'],
      ['RADAR_RESIDUAL_DECAY_T', f'{long_mpc.RADAR_RESIDUAL_DECAY_T:g} s'],
      ['Personality', f'{LONGITUDINAL_PERSONALITY_MAP[personality]} (T_FOLLOW={long_mpc.get_T_FOLLOW(personality):g} s)'],
    ],
  )

  return [
    REPORT_CSS,
    markdown.markdown('# MPC adversarial lead-prediction tuning report'),
    markdown.markdown('## Active parameters'),
    params_table,
    *render_section(
      "1. Approaching a stopped lead that doesn't go, model wrongly predicts it will go",
      create_approach_stopped_adv_maneuvers(kwargs),
    ),
    *render_section(
      "2. Stopped behind a lead that doesn't go, model wrongly predicts it will go",
      create_stopped_adv_maneuvers(kwargs),
    ),
    *render_section(
      '3. Steady following then lead brakes hard, model wrongly predicts a = 0',
      create_hard_braking_adv_maneuvers(kwargs),
    ),
  ]


if __name__ == '__main__':
  file_name = sys.argv[1] if len(sys.argv) > 1 else 'mpc_adversarial_tune_report.html'
  htmls = generate_adversarial_tuning_report()

  with open(file_name, 'w') as f:
    for chunk in htmls:
      f.write(chunk)
  print(f'Wrote {file_name}')
