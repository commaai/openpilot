"""Analyze saved typing-study sessions and export a standalone HTML report."""
import argparse
import base64
from collections import defaultdict
import csv
import html
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from openpilot.tools.ui.keyboard_study_data import summarize
from openpilot.tools.ui.keyboard_study_quality import assess_quality, flagged_gesture
from openpilot.tools.ui.keyboard_study_overlay import make_keyboard_heatmaps


def load_sessions(directory, include_synthetic=False):
  sessions = []
  for path in sorted(directory.glob('*.jsonl')):
    records, malformed = [], 0
    counts = {'evdev_events': 0, 'python_samples': 0, 'frame_events': 0, 'ui_queue_dropped': 0, 'kernel_syn_dropped': 0, 'capture_errors': 0}
    # Raw streams stay intact on disk; avoid loading hundreds of MB of samples
    # into RAM just to summarize them.
    with path.open() as input_file:
      for line in input_file:
        try:
          record = json.loads(line)
        except json.JSONDecodeError:
          malformed += 1
          continue
        if record['type'] == 'evdev':
          counts['evdev_events'] += len(record['events'])
          counts['kernel_syn_dropped'] += sum(event[2:4] == [0, 3] for event in record['events'])
        elif record['type'] == 'python_samples':
          counts['python_samples'] += len(record['samples'])
          counts['ui_queue_dropped'] += record['ui_queue_dropped']
        elif record['type'] == 'frame':
          counts['frame_events'] += len(record['events'])
        else:
          counts['capture_errors'] += record['type'] == 'capture_error'
          records.append(record)
    metadata = next((record for record in records if record.get('type') == 'session'), None)
    if metadata is None or (metadata.get('synthetic') and not include_synthetic):
      continue
    # Keep the original session record intact; use later participant corrections
    # for grouping and retain their provenance in the raw records.
    metadata = dict(metadata)
    for record in records:
      if record.get('type') == 'session_annotation':
        metadata.update({key: value for key, value in record.get('changes', {}).items()
                         if key in ('finger', 'technique', 'intended_contacts')})
    excluded_trials = []
    for record in records:
      if record.get('type') == 'analysis_exclusion':
        excluded_trials = sorted(record.get('trials', []))
    records = [record for record in records if record.get('trial') not in excluded_trials]
    sessions.append({'metadata': metadata, 'records': records, 'malformed_lines': malformed,
                     'complete': any(record.get('type') == 'session_end' for record in records), 'raw_counts': counts,
                     'excluded_trials': excluded_trials, 'quality': assess_quality(records, metadata['prompts'])})
  return sessions


def labelled_taps(sessions):
  result = []
  for session in sessions:
    for record in session['records']:
      if record.get('type') != 'gesture' or record.get('target') is None or len(record.get('committed', '')) != 1:
        continue
      if flagged_gesture(record, session['quality']):
        continue
      contacts = [sample for sample in record['samples'] if sample[3] != 'release']
      if not contacts:
        continue
      target = record['target']
      first = np.array(contacts[0][1:3])
      last = np.array(contacts[-1][1:3])
      center = np.array(target['center'])
      result.append({'char': record['expected'], 'row': target['row'], 'session': session['metadata']['id'],
                     'finger': session['metadata']['finger'], 'first': first, 'last': last,
                     'offset': first - center, 'last_offset': last - center, 'mistap': record['committed'] != record['expected']})
  return result


def offset_summary(taps, field):
  groups = defaultdict(list)
  for tap in taps:
    groups[str(tap[field])].append(tap)
  result = []
  for key, values in sorted(groups.items()):
    offsets = np.array([value['offset'] for value in values])
    median = np.median(offsets, axis=0)
    last_median = np.median([value['last_offset'] for value in values], axis=0)
    result.append({'key': key, 'n': len(values), 'sessions': len({value['session'] for value in values}),
                   'first_dx': float(median[0]), 'first_dy': float(median[1]),
                   'last_dx': float(last_median[0]), 'last_dy': float(last_median[1]),
                   'p90_spread': float(np.percentile(np.linalg.norm(offsets - median, axis=1), 90))})
  return result


def spatial_patterns(taps):
  groups = defaultdict(list)
  for tap in taps:
    char = tap['char'].lower()
    names = [f'key {char if char != " " else "space"}']
    if char.isalpha():
      names.append(('top letters', 'middle letters', 'bottom letters')[tap['row']])
      if char not in 'qap':
        names.append('interior letters (excluding q/a/p)')
    for name in names:
      groups[(name, 'all')].append(tap)
      groups[(name, tap['finger'])].append(tap)
  result = []
  for (name, finger), selected in sorted(groups.items()):
    by_session = defaultdict(list)
    for tap in selected:
      by_session[tap['session']].append(tap['last_offset'])
    medians = np.array([np.median(offsets, axis=0) for offsets in by_session.values()])
    median = np.median([tap['last_offset'] for tap in selected], axis=0)
    result.append({'group': name, 'finger': finger, 'n': len(selected), 'sessions': len(by_session),
                   'dx': float(median[0]), 'dy': float(median[1]),
                   'sessions_left': int(sum(medians[:, 0] < 0)), 'sessions_right': int(sum(medians[:, 0] > 0)),
                   'sessions_below': int(sum(medians[:, 1] > 0))})
  return result


def make_report(directory, output_dir, include_synthetic=False, keyboard_image=None, condition=None):
  sessions = load_sessions(directory, include_synthetic)
  if condition is not None:
    sessions = [session for session in sessions if session['metadata'].get('condition') == condition]
  conditions = {session['metadata'].get('condition', 'unknown') for session in sessions}
  if len(conditions) > 1:
    raise ValueError(f'Mixed keyboard conditions {sorted(conditions)}. Use --condition to report each separately.')
  if not sessions:
    raise ValueError('No study sessions found. Synthetic smoke-test sessions are excluded by default.')
  taps = labelled_taps(sessions)
  metrics = [dict(**session['raw_counts'],
                  id=session['metadata']['id'], finger=session['metadata']['finger'], complete=session['complete'],
                  aborted=any(record['type'] == 'session_abort' for record in session['records']),
                  assignment_mode=session['metadata'].get('assignment_mode', 'not recorded'),
                  technique=session['metadata'].get('technique', 'not recorded'), condition=session['metadata'].get('condition'),
                  calibration_sha256=session['metadata'].get('calibration_sha256'),
                  excluded_phrases=[trial + 1 for trial in session['excluded_trials']],
                  synthetic=session['metadata'].get('synthetic', False), malformed_lines=session['malformed_lines'],
                  **summarize(session['records'])) for session in sessions]
  for metric, session in zip(metrics, sessions, strict=True):
    quality = session['quality']
    clean = summarize([record for record in session['records'] if record.get('trial') not in quality['flagged_trials']])
    metric.update(flagged_phrases=[trial + 1 for trial in quality['flagged_trials']], quality=quality,
                  comparison_trials=clean['trials'], comparison_wpm=clean['output_wpm'], comparison_error_rate=clean['final_error_rate'])
  key_offsets = offset_summary(taps, 'char')
  row_offsets = offset_summary(taps, 'row')
  finger_offsets = offset_summary(taps, 'finger')
  patterns = spatial_patterns(taps)
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / 'summary.json').write_text(json.dumps({'sessions': metrics, 'key_offsets': key_offsets,
                                                     'row_offsets': row_offsets, 'finger_offsets': finger_offsets,
                                                     'spatial_patterns': patterns}, indent=2))
  if key_offsets:
    with (output_dir / 'key_offsets.csv').open('w') as output:
      writer = csv.DictWriter(output, fieldnames=list(key_offsets[0]))
      writer.writeheader()
      writer.writerows(key_offsets)
  figures = make_keyboard_heatmaps(sessions, keyboard_image, output_dir) if keyboard_image is not None else []
  if taps:
    figure = Figure(figsize=(12, 4), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    first = np.array([tap['first'] for tap in taps])
    offsets = np.array([tap['offset'] for tap in taps])
    density = axes[0].hist2d(first[:, 0], first[:, 1], bins=(54, 24), range=((0, 536), (0, 240)))
    axes[0].set(xlabel='screen x (px)', ylabel='screen y (px)', title='First contact: all labelled taps', ylim=(240, 0))
    figure.colorbar(density[3], ax=axes[0], label='taps')
    density = axes[1].hist2d(offsets[:, 0], offsets[:, 1], bins=(40, 40))
    axes[1].axvline(0)
    axes[1].axhline(0)
    axes[1].set(xlabel='dx from intended glyph center (px)', ylabel='dy from intended glyph center (px)',
                title='First-contact offsets (positive y = below)')
    axes[1].invert_yaxis()
    figure.colorbar(density[3], ax=axes[1], label='taps')
    figure.savefig(output_dir / 'heatmap.png')
    figures.append('heatmap.png')

    figure = Figure(figsize=(15, 12), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(5, 6).ravel()
    for axis, char in zip(axes, 'abcdefghijklmnopqrstuvwxyz ', strict=False):
      points = np.array([tap['offset'] for tap in taps if tap['char'].lower() == char])
      if len(points):
        axis.hist2d(points[:, 0], points[:, 1], bins=(16, 16), range=((-60, 60), (-60, 60)))
      axis.axvline(0)
      axis.axhline(0)
      axis.set(title=f'{char if char != " " else "space"} (n={len(points)})', xlim=(-60, 60), ylim=(60, -60))
    for axis in axes[27:]:
      axis.set_visible(False)
    figure.suptitle('Offsets by intended key; first contact, ±60 px view')
    figure.savefig(output_dir / 'keys.png')
    figures.append('keys.png')

  def table(rows, columns):
    header = ''.join(f'<th>{html.escape(column)}</th>' for column in columns)
    body = ''
    for row in rows:
      cells = []
      for column in columns:
        value = row.get(column)
        value = f'{value:.2f}' if isinstance(value, float) else value
        cells.append(f'<td>{html.escape(str(value))}</td>')
      body += '<tr>' + ''.join(cells) + '</tr>'
    return f'<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>'

  title = 'Keyboard study' + (' — includes synthetic test data' if any(item['synthetic'] for item in metrics) else '')
  body = f'<h1>{title}</h1><p>{len(sessions)} sessions; {len(taps)} labelled character taps.</p>'
  if any(session['excluded_trials'] for session in sessions):
    body += '<p>Excluded phrases (numbered from 1 in the session table) are omitted from tap, error, speed, and spatial metrics. '
    body += 'Original records remain intact; raw capture-health counts include excluded input.</p>'
  body += '<p>Automatic quality flags identify rapid runs of at least four characters absent from the prompt while input is already off track '
  body += '(each gap at most 350 ms, median gap at most 200 ms). This flags possible non-compliance, not a diagnosis of intent. '
  body += 'Only flagged bursts are excluded from spatial samples; uncertain intended keys are already excluded by the prefix rule. '
  body += 'Comparison WPM and final error rates omit whole flagged phrases because their completion times and final text are contaminated. '
  body += 'Raw session totals and all original events remain available in summary.json and the JSONL logs.</p>'
  body += f'<p>Keyboard condition: {html.escape(next(iter(conditions)))}. These are observed copy-task samples.</p>'
  body += '''<p>Intent is inferred only when the text before a tap is an exact prefix of the prompt and the target is on the current layer.
  Wrong taps are included. After an uncorrected mismatch, further taps are unlabelled until the prefix matches again.
  Layer slides are counted separately: their initial contact targets a page button, so they are excluded from ordinary tap spatial estimates.
  Their transitions, destination geometry and inferred target remain recorded.
  Control-only taps are excluded from spatial estimates. This is not independent ground truth.</p>
  <p>Output WPM = final characters / 5 / input-span minutes (first touch through last input in each phrase).
  Reading before the first touch and next-button time are excluded.
  Final error rate = Levenshtein distance / prompt length. Backspaces count successful deletions.
  Labelled mistaps compare committed versus inferred intended character, not all-session error rate.
  Spatial summaries below use UI-delivered contacts. Files also retain every Linux input event with kernel timestamps,
  all slots, IDs, contact size/pressure, all Python samples (including unchanged polls), and complete frame batches.
  The native coordinate ranges and clock are stored in evdev_start; raw Linux axes may be rotated relative to the UI.
  No raw capacitance images are exposed by this stream.</p>'''
  if keyboard_image is not None:
    body += '<p><a href="keyboard-heatmap.html">Interactive heatmap over the recorded keyboard: filter technique/intended key and show exact taps</a></p>'
  body += '<h2>Sessions</h2>' + table(metrics, ['id', 'condition', 'technique', 'complete', 'aborted', 'trials', 'excluded_phrases', 'gestures',
                                              'labelled_taps', 'labelled_mistaps',
                                              'backspaces', 'layer_slides', 'layer_slide_mistaps', 'flagged_phrases',
                                              'comparison_trials', 'comparison_wpm', 'comparison_error_rate',
                                              'evdev_events', 'python_samples',
                                              'ui_queue_dropped', 'kernel_syn_dropped', 'capture_errors', 'malformed_lines'])
  for name in figures:
    encoded = base64.b64encode((output_dir / name).read_bytes()).decode()
    body += f'<img src="data:image/png;base64,{encoded}" alt="{name}">'
  body += '<h2>Spatial patterns across sessions</h2><p>Last-down offsets from nominal glyph anchors. Direction counts use each session median.</p>'
  selected_patterns = [row for row in patterns if row['finger'] == 'all' and
                       (not row['group'].startswith('key ') or row['group'] in ('key q', 'key a', 'key p', 'key space', 'key z', 'key m'))]
  body += table(selected_patterns, list(selected_patterns[0]) if selected_patterns else [])
  body += '<h2>Offsets by finger</h2>' + table(finger_offsets, list(finger_offsets[0]) if finger_offsets else [])
  body += '<h2>Offsets by row (0 = top)</h2>' + table(row_offsets, list(row_offsets[0]) if row_offsets else [])
  body += '<h2>Offsets by key</h2><p>Medians relative to the unanimated glyph center; positive dy is below it. '
  body += 'Small sample counts are descriptive only. No offsets are applied automatically; validate changes on new participants.</p>'
  body += table(key_offsets, list(key_offsets[0]) if key_offsets else [])
  document = '<!doctype html><meta charset="utf-8"><title>Keyboard study</title><style>'
  document += 'body{font:16px system-ui;margin:32px}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #ccc}'
  document += 'img{display:block;max-width:100%;margin:24px 0}p{max-width:1000px}</style>' + body
  (output_dir / 'report.html').write_text(document)
  return metrics


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('directory', type=Path)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--include-synthetic', action='store_true')
  parser.add_argument('--keyboard-image', type=Path, help='Keyboard screenshot matching the selected condition, aspect ratio 536:240')
  parser.add_argument('--condition', help='Report only this session condition; required if logs contain multiple conditions')
  args = parser.parse_args()
  make_report(args.directory, args.output_dir, args.include_synthetic, args.keyboard_image, args.condition)
  print(args.output_dir / 'report.html')
