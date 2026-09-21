"""Fit a frozen spatial model from completed stock sessions, keeping provenance."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from openpilot.tools.ui.keyboard_study_report import labelled_taps, load_sessions


def fit_calibration(directory):
  sessions = [session for session in load_sessions(directory) if session['complete'] and
              session['metadata'].get('condition') == 'stock_master' and not session['malformed_lines']]
  taps = labelled_taps(sessions)
  if not taps or {tap['row'] for tap in taps} != {0, 1, 2}:
    raise ValueError('Need labelled contacts in all three rows from completed stock sessions')

  def estimate(selected):
    return {'offset': np.median([tap['last_offset'] for tap in selected], axis=0).tolist(),
            'taps': len(selected), 'sessions': len({tap['session'] for tap in selected})}

  sources = []
  for session in sessions:
    identifier = session['metadata']['id']
    with (directory / f'{identifier}.jsonl').open('rb') as source:
      digest = hashlib.file_digest(source, 'sha256').hexdigest()
    sources.append({'session': identifier, 'sha256': digest})
  return {'version': 'hackathon_static_v1', 'point': 'last_down_event', 'metric': 'squared_euclidean',
          'method': 'per-key median dx/dy from nominal glyph anchor; case pooled; unseen characters use row medians; controls retain stock centers',
          'labels': 'next prompt character only while entered text is an exact prefix; target must be visible; include incorrect commits',
          'quality': 'keyboard_study_quality rapid out-of-prompt burst exclusion',
          'training_taps': len(taps), 'sources': sources,
          'rows': {str(row): estimate([tap for tap in taps if tap['row'] == row]) for row in range(3)},
          'keys': {char: estimate([tap for tap in taps if tap['char'].lower() == char]) for char in sorted({tap['char'].lower() for tap in taps})}}


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('directory', type=Path)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  model = fit_calibration(args.directory)
  args.output.write_text(json.dumps(model, indent=2) + '\n')
  print(f"{model['training_taps']} taps, {len(model['sources'])} sessions -> {args.output}")
