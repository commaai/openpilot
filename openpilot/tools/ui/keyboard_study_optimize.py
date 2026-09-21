"""Constrained static target fitting with session-held-out replay and an audit report."""

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from openpilot.tools.ui.keyboard_study_quality import flagged_gesture
from openpilot.tools.ui.keyboard_study_report import load_sessions

KEYS = list('qwertyuiopasdfghjkl') + ['123'] + list('zxcvbnm') + [' ']
BASE_ADJUSTMENTS = {'l': (8, 0)}
MAX_SHIFT = 12
STEP = 4
MIN_TAPS = 20
MIN_SESSIONS = 3
MIN_FIX_SESSIONS = 2
MAX_ROUNDS = 12
STABILITY_FRACTION = 0.8


def layout_matches(geometry):
  rows = [''.join(key['char'].lower() for key in geometry if key['row'] == row) for row in range(3)]
  return rows == ['qwertyuiop', 'asdfghjkl', '123zxcvbnm '] and [key['char'].lower() for key in geometry] == KEYS


def correction_confirmed(records, index, record):
  """A later correct replacement at the same prefix, after deleting the wrong tap."""
  deleted = False
  for following in records[index + 1 :]:
    if following.get('trial') != record['trial']:
      continue
    if following['type'] == 'trial_end':
      break
    if following['type'] == 'backspace' and following.get('after') == record['before']:
      deleted = True
    if following['type'] == 'gesture' and following.get('before') == record['before'] and deleted:
      return following.get('committed') == record['expected']
  return False


def collect(sessions):
  taps, audit = [], []
  reference_centers = None
  for session in sessions:
    metadata = session['metadata']
    counts = Counter()
    for index, record in enumerate(session['records']):
      if record['type'] != 'gesture':
        continue
      geometry = record.get('geometry', [])
      if not layout_matches(geometry):
        counts['other_layout_or_symbols'] += 1
        continue
      if reference_centers is None:
        reference_centers = np.array([key['center'] for key in geometry])
      if not np.allclose(reference_centers, [key['center'] for key in geometry], atol=0.1):
        counts['different_positions'] += 1
        continue
      if (
        record.get('target') is None
        or len(record.get('committed', '')) != 1
        or record.get('kind') == 'layer_slide'
        or flagged_gesture(record, session['quality'])
      ):
        counts['unlabelled_slide_or_flagged'] += 1
        continue
      contacts = [sample for sample in record['samples'] if sample[3] != 'release']
      if not contacts or record['expected'].lower() not in KEYS:
        continue
      point = np.array(contacts[-1][1:3], dtype=float)
      recorded_centers = np.array([key['touch_center'] for key in geometry], dtype=float)
      recorded_selection = int(np.argmin(np.sum((recorded_centers - point) ** 2, axis=1)))
      if geometry[recorded_selection]['char'] != record['committed']:
        counts['replay_mismatch'] += 1
        continue
      centers = recorded_centers.copy()
      for key_index, key in enumerate(KEYS):
        centers[key_index] += np.array(BASE_ADJUSTMENTS.get(key, (0, 0))) - np.array(metadata.get('target_adjustments', {}).get(key, (0, 0)))
      target = KEYS.index(record['expected'].lower())
      nominal = np.array(geometry[target]['center'])
      pitch = min(np.linalg.norm(nominal - np.array(other['center'])) for other in geometry if other is not geometry[target])
      # Outliers remain in evaluation. This is an uncertainty rule, not an intent diagnosis.
      far = bool(np.linalg.norm(point - centers[target]) > pitch)
      corrected = correction_confirmed(session['records'], index, record) if record['committed'] != record['expected'] else False
      confidence = 0.0 if far else 1.0 if corrected or record['committed'] == record['expected'] else 0.25
      taps.append(
        {
          'session': metadata['id'],
          'finger': metadata['finger'],
          'complete': session['complete'],
          'point': point.tolist(),
          'centers': centers.tolist(),
          'target': target,
          'confidence': confidence,
          'far': far,
          'corrected': corrected,
          'trial': record['trial'],
          'time': record['press_time'],
          'committed': record['committed'],
          'expected': record['expected'],
        }
      )
      counts['evaluated'] += 1
      counts['far_training_excluded'] += far
      counts['confirmed_correction'] += corrected
    audit.append(
      {
        'id': metadata['id'],
        'condition': metadata['condition'],
        'finger': metadata['finger'],
        'complete': session['complete'],
        'malformed_lines': session['malformed_lines'],
        'quality': session['quality'],
        'counts': dict(counts),
      }
    )
  return taps, audit, reference_centers


def predict(taps, shifts):
  if not taps:
    return np.array([], dtype=int)
  points = np.array([tap['point'] for tap in taps])
  centers = np.array([tap['centers'] for tap in taps])
  return np.argmin(np.sum((centers + shifts - points[:, None, :]) ** 2, axis=2), axis=1)


def fit(taps):
  """Joint coordinate search; freeze controls, bound motion, forbid training regressions."""
  shifts = np.zeros((len(KEYS), 2))
  if not taps:
    return shifts
  points = np.array([tap['point'] for tap in taps])
  centers = np.array([tap['centers'] for tap in taps])
  targets = np.array([tap['target'] for tap in taps])
  identifiers = np.array([tap['session'] for tap in taps])
  baseline = predict(taps, shifts) == targets
  session_sizes = Counter(identifiers)
  weights = np.array([tap['confidence'] / session_sizes[tap['session']] for tap in taps])
  weights *= len(taps) / len(session_sizes)
  eligible = []
  for key_index, key in enumerate(KEYS):
    support = [tap for tap in taps if tap['target'] == key_index and tap['confidence'] > 0]
    if len(key) == 1 and key != ' ' and len(support) >= MIN_TAPS and len({tap['session'] for tap in support}) >= MIN_SESSIONS:
      eligible.append(key_index)
  current_correct = baseline.copy()
  for _ in range(MAX_ROUNDS):
    best = None
    best_gain = 0.0
    distances = np.sum((centers + shifts - points[:, None, :]) ** 2, axis=2)
    for key_index in eligible:
      for axis in range(2):
        for value in range(-MAX_SHIFT, MAX_SHIFT + 1, STEP):
          candidate = shifts[key_index].copy()
          candidate[axis] = value
          if np.linalg.norm(candidate) > MAX_SHIFT or np.array_equal(candidate, shifts[key_index]):
            continue
          candidate_distances = distances.copy()
          candidate_distances[:, key_index] = np.sum((centers[:, key_index] + candidate - points) ** 2, axis=1)
          correct = np.argmin(candidate_distances, axis=1) == targets
          if np.any(baseline & ~correct):
            continue
          improved = correct & ~current_correct & (weights > 0)
          if len(set(identifiers[improved])) < MIN_FIX_SESSIONS:
            continue
          movement_cost = 0.025 * (np.sum(candidate**2) - np.sum(shifts[key_index] ** 2)) / STEP**2
          gain = float(np.sum(weights * (correct.astype(float) - current_correct))) - movement_cost
          if gain > best_gain + 1e-9:
            best_gain = gain
            best = (key_index, candidate, correct)
    if best is None:
      break
    key_index, candidate, current_correct = best
    shifts[key_index] = candidate
  return shifts


def stable_fit(taps):
  candidate = fit(taps)
  identifiers = sorted({tap['session'] for tap in taps})
  if len(identifiers) < MIN_SESSIONS + 1:
    return np.zeros_like(candidate)
  inner_models = np.array([fit([tap for tap in taps if tap['session'] != identifier]) for identifier in identifiers])
  for key_index in range(len(KEYS)):
    for axis in range(2):
      direction = np.sign(candidate[key_index, axis])
      votes = np.sign(inner_models[:, key_index, axis]) == direction
      if direction and np.mean(votes) < STABILITY_FRACTION:
        candidate[key_index, axis] = 0
  # Dropping a coordinate can change interactions; retain the original no-regression guard.
  targets = np.array([tap['target'] for tap in taps])
  baseline = predict(taps, np.zeros_like(candidate)) == targets
  if np.any(baseline & (predict(taps, candidate) != targets)):
    return np.zeros_like(candidate)
  return candidate


def metrics(taps, shifts):
  targets = np.array([tap['target'] for tap in taps])
  before = predict(taps, np.zeros((len(KEYS), 2))) == targets
  after = predict(taps, shifts) == targets
  return {
    'taps': len(taps),
    'before_errors': int(sum(~before)),
    'after_errors': int(sum(~after)),
    'fixes': int(sum(~before & after)),
    'regressions': int(sum(before & ~after)),
    'far_taps': sum(tap['far'] for tap in taps),
  }


def adjustments(shifts):
  return {key: shift.tolist() for key, shift in zip(KEYS, shifts, strict=True) if np.any(shift)}


def run(directory, output):
  output.mkdir(parents=True, exist_ok=True)
  sessions = load_sessions(directory)
  taps, audit, nominal = collect(sessions)
  training = [tap for tap in taps if tap['complete']]
  fitted = stable_fit(training)
  folds = []
  for identifier in sorted({tap['session'] for tap in training}):
    train = [tap for tap in training if tap['session'] != identifier]
    held_out = [tap for tap in training if tap['session'] == identifier]
    fold_model = stable_fit(train)
    folds.append({'id': identifier, 'finger': held_out[0]['finger'], **metrics(held_out, fold_model), 'adjustments': adjustments(fold_model)})
  per_session = []
  for identifier in sorted({tap['session'] for tap in taps}):
    selected = [tap for tap in taps if tap['session'] == identifier]
    per_session.append({'id': identifier, 'finger': selected[0]['finger'], 'complete': selected[0]['complete'], **metrics(selected, fitted)})
  totals = {key: sum(fold[key] for fold in folds) for key in ('taps', 'before_errors', 'after_errors', 'fixes', 'regressions')}
  accepted = (
    totals['fixes'] > totals['regressions']
    and totals['regressions'] <= 1
    and all(fold['after_errors'] <= fold['before_errors'] + 1 for fold in folds)
    and all(
      sum(fold['after_errors'] - fold['before_errors'] for fold in folds if fold['finger'] == finger) <= 0 for finger in {tap['finger'] for tap in training}
    )
  )
  result = {
    'method': 'bounded joint center search; robust weighting; nested session-held-out stability selection',
    'accepted': accepted,
    'additional_adjustments': adjustments(fitted),
    'base_adjustments': BASE_ADJUSTMENTS,
    'limits': {
      'max_shift': MAX_SHIFT,
      'step': STEP,
      'min_taps': MIN_TAPS,
      'min_sessions': MIN_SESSIONS,
      'min_fix_sessions': MIN_FIX_SESSIONS,
      'max_rounds': MAX_ROUNDS,
      'stability_fraction': STABILITY_FRACTION,
    },
    'held_out': totals,
    'folds': folds,
    'fitted_replay': metrics(taps, fitted),
    'sessions': per_session,
    'audit': audit,
    'caveat': (
      'Sessions are not independent people. Prompt-prefix intent is inferred. '
      + 'Distant taps are excluded only from fitting, never from error totals. Historical replay cannot measure user adaptation.'
    ),
    'sources': {
      session['metadata']['id']: hashlib.sha256((directory / (session['metadata']['id'] + '.jsonl')).read_bytes()).hexdigest() for session in sessions
    },
  }
  frozen = dict(BASE_ADJUSTMENTS)
  for key, shift in adjustments(fitted).items():
    frozen[key] = (np.array(frozen.get(key, (0, 0))) + shift).tolist()
  (output / 'candidate-model.json').write_text(
    json.dumps(
      {
        'version': 'robust_static_v3_preview',
        'accepted': accepted,
        'target_adjustments': frozen,
        'held_out': totals,
        'source_sessions': result['sources'],
        'status': 'preview_only; not an assertion of optimality',
      },
      indent=2,
    )
    + '\n'
  )
  (output / 'optimization.json').write_text(json.dumps(result, indent=2) + '\n')
  (output / 'tap-audit.json').write_text(json.dumps(taps) + '\n')
  rows = ''.join(
    '<tr>'
    + ''.join(f'<td>{html.escape(str(row[key]))}</td>' for key in ('id', 'finger', 'complete', 'taps', 'before_errors', 'after_errors', 'fixes', 'regressions'))
    + '</tr>'
    for row in per_session
  )
  fold_rows = ''.join(
    '<tr>'
    + ''.join(f'<td>{html.escape(str(row[key]))}</td>' for key in ('id', 'finger', 'taps', 'before_errors', 'after_errors', 'fixes', 'regressions'))
    + '</tr>'
    for row in folds
  )
  document = f'''<!doctype html><meta charset="utf-8"><title>Automatic keyboard target optimization</title>
<style>body{{font:16px system-ui;max-width:1250px;margin:40px auto;padding:0 20px}}td,th{{padding:8px;border-bottom:1px solid
#ccc;text-align:left}}table{{border-collapse:collapse}}img{{max-width:100%}}</style>
<h1>Automatic keyboard target optimization</h1><p>{html.escape(result['method'])}</p>
<p><strong>Validation gate: {'PASS' if accepted else 'FAIL — keep deployed targets'}</strong></p>
<p>Held-out errors: {totals['before_errors']} → {totals['after_errors']} / {totals['taps']} taps; {totals['fixes']} fixes and {totals['regressions']}
regressions.</p>
<p>{html.escape(result['caveat'])}</p><p>Bounds ±12 px (Euclidean), 4 px search grid; minimum 20 nearby samples across three sessions per adjusted key
and improvements in two sessions per search step. Controls and space stay fixed. Correctable nearby mistakes without confirmed correction carry 0.25
weight; corrected and correct taps carry 1.0. Taps more than one local key pitch from intended center carry zero fitting weight. Sessions are
balanced. No baseline-correct training tap may regress.</p>
<p>Each retained shift direction must recur in at least 80% of inner leave-one-session-out fits.
This is repeated inside every outer fold; the scored session cannot influence its fitted model.</p>
<h2>Proposed additional shifts</h2><pre>{html.escape(json.dumps(adjustments(fitted), indent=2))}</pre>
<h2>Held-out sessions — independently refit for each
row</h2><table><tr><th>Session</th><th>Technique</th><th>Taps</th><th>Before</th><th>After</th><th>Fixes</th><th>Regressions</th></tr>{fold_rows}</table>
<h2>Final fitted model — all compatible recordings, including partial runs</h2><p>These rows reuse training data; do not interpret them as held-out
performance.</p><table><tr><th>Session</th><th>Technique</th><th>Complete</th><th>Taps</th><th>Before</th><th>After</th><th>Fixes</th><th>Regressions</th></tr>{rows}</table>
<h2>Actual boundaries</h2><img src="boundaries.png"><p><a href="optimization.json">Full audit and per-fold models</a> · <a
href="tap-audit.json">Per-tap inputs and confidence decisions</a></p>'''
  (output / 'report.html').write_text(document)
  if training:
    baseline_centers = np.array(training[0]['centers'])
    grid_x, grid_y = np.meshgrid(np.arange(536), np.arange(70, 240))
    grid = np.stack((grid_x, grid_y), axis=-1)
    figure = Figure(figsize=(14, 5), layout='constrained')
    FigureCanvasAgg(figure)
    for axis, shift, title in zip(figure.subplots(1, 2), (np.zeros_like(fitted), fitted), ('Deployed boundaries', 'Candidate boundaries'), strict=True):
      labels = np.argmin(np.sum((grid[:, :, None] - (baseline_centers + shift)) ** 2, axis=-1), axis=-1)
      axis.imshow(labels, extent=(0, 536, 240, 70), interpolation='nearest', cmap='tab20')
      edges = np.zeros_like(labels, dtype=bool)
      edges[:, 1:] |= labels[:, 1:] != labels[:, :-1]
      edges[1:, :] |= labels[1:, :] != labels[:-1, :]
      axis.imshow(np.ma.masked_where(~edges, np.zeros_like(labels)), extent=(0, 536, 240, 70),
                  interpolation='nearest', cmap='gray', vmin=0, vmax=1)
      for key, position in zip(KEYS, nominal, strict=True):
        axis.text(*position, 'space' if key == ' ' else key, ha='center', va='center')
      axis.set(title=title, xlim=(0, 536), ylim=(240, 70))
    figure.savefig(output / 'boundaries.png', dpi=300)
  print(json.dumps({key: result[key] for key in ('accepted', 'additional_adjustments', 'held_out', 'fitted_replay')}, indent=2))
  return result


def audit_reports(root, output, audit_directories):
  result = json.loads((output / 'optimization.json').read_text())
  shifts = np.array([result['additional_adjustments'].get(key, [0, 0]) for key in KEYS])
  known = {row['id']: row for row in result['sessions']}
  audit_known = {row['id']: row for row in result['audit']}
  extra = [session for directory in audit_directories for session in load_sessions(directory)]
  extra_taps, extra_audit, _ = collect(extra)
  extra_rows = []
  for session in extra:
    identifier = session['metadata']['id']
    selected = [tap for tap in extra_taps if tap['session'] == identifier]
    if selected:
      row = {'id': identifier, 'finger': session['metadata']['finger'], 'complete': session['complete'], **metrics(selected, shifts)}
      extra_rows.append(row)
      known[identifier] = row
  reports = []
  for path in sorted(root.glob('keyboard-*/**/summary.json')):
    summary = json.loads(path.read_text())
    entries = []
    for session in summary.get('sessions', []):
      identifier = session['id']
      if identifier in known:
        entries.append({'id': identifier, 'status': 'replayed', 'metrics': known[identifier]})
      elif identifier in audit_known and audit_known[identifier]['counts'].get('evaluated', 0) == 0:
        entries.append({'id': identifier, 'status': 'no compatible labelled letter taps'})
      else:
        entries.append({'id': identifier, 'status': 'older layout or deleted historical session; not calibration input'})
    reports.append({'report': str(path.parent / 'report.html'), 'sessions': entries})
  (output / 'report-coverage.json').write_text(json.dumps({'reports': reports, 'archived_same_layout_replay': extra_rows}, indent=2) + '\n')
  body = '''<h2>Verification against every existing report</h2><p>Sessions shared by reports are counted once.
Archived same-layout recordings are additional checks only; historical deleted exploratory sessions are not restored to training.</p>'''
  for report in reports:
    body += f"<h3>{html.escape(str(Path(report['report']).relative_to(root)))}</h3><ul>"
    for entry in report['sessions']:
      if entry['status'] == 'replayed':
        values = entry['metrics']
        text = (
          f"{values['taps']} taps: {values['before_errors']} → {values['after_errors']} errors; {values['fixes']} fixes, {values['regressions']} regressions"
        )
      else:
        text = entry['status']
      body += f"<li>{entry['id'][:8]}: {text}</li>"
    body += '</ul>'
  p = output / 'report.html'
  p.write_text(p.read_text() + body)
  return extra_rows


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('directory', type=Path)
  parser.add_argument('output', type=Path)
  parser.add_argument('--reports-root', type=Path)
  parser.add_argument('--audit-directory', type=Path, action='append', default=[])
  args = parser.parse_args()
  run(args.directory, args.output)
  if args.reports_root:
    audit_reports(args.reports_root, args.output, args.audit_directory)


if __name__ == '__main__':
  main()
