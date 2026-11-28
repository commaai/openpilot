import sys
import markdown
import matplotlib.pyplot as plt
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


def render_table(headers, rows):
  head = ''.join(f'<th>{h}</th>' for h in headers)
  body = ''.join('<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>' for row in rows)
  return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def summarize_pair(base_logs, adv_logs, adv_valid):
  base_min_a = float(np.min(base_logs[:, Axis.EGO_A]))
  adv_min_a = float(np.min(adv_logs[:, Axis.EGO_A]))
  base_min_d = float(np.min(base_logs[:, Axis.D_REL]))
  adv_min_d = float(np.min(adv_logs[:, Axis.D_REL]))
  return {
    'base_min_a': base_min_a,
    'adv_min_a': adv_min_a,
    'extra_brake': base_min_a - adv_min_a,
    'base_min_d': base_min_d,
    'adv_min_d': adv_min_d,
    'delta_d': adv_min_d - base_min_d,
    'creep': float(np.max(adv_logs[:, Axis.EGO_POSITION])),
    'max_v': float(np.max(adv_logs[:, Axis.EGO_V])),
    'max_a': float(np.max(adv_logs[:, Axis.EGO_A])),
    'valid': adv_valid,
  }


def render_section(heading, make_maneuvers, personality, axes, shared_baseline=True, is_stopped=False):
  colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
  maneuver_pairs = zip(
    make_maneuvers({'personality': personality}),
    make_maneuvers({'personality': personality, 'only_radar': True}),
    strict=True,
  )

  groups = {}
  for adv_man, base_man in maneuver_pairs:
    group_name, curve_label = adv_man.title.rsplit(', ', 1)
    groups.setdefault(group_name, []).append((curve_label, adv_man, base_man))

  htmls = [markdown.markdown(f'# {heading}')]
  table_rows = []
  for group_name, items in groups.items():
    results, labels, styles = {}, [], []
    base_cache = None
    if shared_baseline:
      _, base_cache = items[0][2].evaluate()
      results[len(results)] = base_cache
      labels.append('Baseline (aLeadK/aLeadTau)')
      styles.append({'color': '#111111', 'linestyle': '--', 'linewidth': 2.2})

    for idx, (curve_label, adv_man, base_man) in enumerate(items):
      color = colors[idx % len(colors)]
      if shared_baseline:
        base_valid, base_logs = True, base_cache
      else:
        base_valid, base_logs = base_man.evaluate()
        results[len(results)] = base_logs
        labels.append(f'Baseline · {curve_label}')
        styles.append({'color': color, 'linestyle': '--', 'linewidth': 1.5})

      adv_valid, adv_logs = adv_man.evaluate()
      results[len(results)] = adv_logs
      labels.append(curve_label)
      styles.append({'color': color})

      st = summarize_pair(base_logs, adv_logs, adv_valid)
      status = 'PASS' if st['valid'] else ('CRASH (base also crashed)' if not base_valid else 'CRASH')
      if is_stopped:
        row = [
          adv_man.title,
          f"{st['creep']:.2f}", f"{st['max_v']:.2f}", f"{st['max_a']:+.2f}",
          f"{st['base_min_a']:+.2f}", f"{st['adv_min_a']:+.2f}",
          f"{st['base_min_d']:.2f}", f"{st['adv_min_d']:.2f}", f"{st['delta_d']:+.2f}", status,
        ]
      else:
        row = [
          adv_man.title,
          f"{st['base_min_a']:+.2f}", f"{st['adv_min_a']:+.2f}", f"{st['extra_brake']:+.2f}",
          f"{st['base_min_d']:.2f}", f"{st['adv_min_d']:.2f}", f"{st['delta_d']:+.2f}", status,
        ]
      table_rows.append(row)

    htmls.append(markdown.markdown(f'## {group_name}'))
    htmls.append(get_html_from_results(results, labels, axes, styles))

  if is_stopped:
    headers = [
      'Case', 'Creep (m)', 'Max v (m/s)', 'Max a (m/s^2)', 'Baseline min a (m/s^2)', 'Adv min a (m/s^2)', 'Baseline min d (m)', 'Adv min d (m)', 'Δ min d (m)',
      'Result',
    ]
  else:
    headers = [
      'Case', 'Baseline peak a (m/s^2)', 'Adv peak a (m/s^2)', 'Extra braking (m/s^2)', 'Baseline min d (m)', 'Adv min d (m)', 'Δ min d (m)', 'Result',
    ]
  htmls.insert(1, render_table(headers, table_rows))
  return htmls


def generate_adversarial_tuning_report(personality=log.LongitudinalPersonality.relaxed):
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
      create_approach_stopped_adv_maneuvers, personality, [Axis.EGO_A, Axis.D_REL], shared_baseline=True,
    ),
    *render_section(
      "2. Stopped behind a lead that doesn't go, model wrongly predicts it will go",
      create_stopped_adv_maneuvers, personality, [Axis.EGO_A, Axis.D_REL, Axis.EGO_V], shared_baseline=True, is_stopped=True,
    ),
    *render_section(
      '3. Steady following then lead brakes hard, model wrongly predicts a = 0',
      create_hard_braking_adv_maneuvers, personality, [Axis.EGO_A, Axis.D_REL], shared_baseline=False,
    ),
  ]


if __name__ == '__main__':
  file_name = sys.argv[1] if len(sys.argv) > 1 else 'mpc_adversarial_tune_report.html'
  htmls = generate_adversarial_tuning_report()

  with open(file_name, 'w') as f:
    for chunk in htmls:
      f.write(chunk)
  print(f'Wrote {file_name}')
