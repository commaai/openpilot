"""Local-only keyboard study records and explicitly defined metrics."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, UTC
import json
from pathlib import Path
from uuid import uuid4

from openpilot.common.realtime import drop_realtime, set_core_affinity

URL_PHRASE = 'openpilot.comma.ai'

PHRASES = (
  'hello comma',
  'the quick brown fox jumps over the lazy dog',
  'pack my box with five dozen liquor jugs',
  URL_PHRASE,
  'driver_42',
  'Blue7!river',
)


def writer_init():
  drop_realtime()
  set_core_affinity([0, 1, 2, 3])


class StudyWriter:
  def __init__(self, directory: Path, synthetic=False):
    self.directory = directory
    self.synthetic = synthetic
    self.path = None
    self.error = None
    self._worker = ThreadPoolExecutor(max_workers=1, initializer=writer_init)

  def start(self, finger, geometry, prompts, configuration):
    self.directory.mkdir(parents=True, exist_ok=True)
    identifier = uuid4().hex
    self.path = self.directory / f'{identifier}.jsonl'
    self.path.touch(exist_ok=False)
    self.write({'type': 'session', 'schema': 2, 'id': identifier, 'synthetic': self.synthetic,
                'created': datetime.now(UTC).isoformat(), 'finger': finger, 'prompts': prompts,
                'technique': 'one_index_finger' if finger == 'index' else 'two_thumbs',
                'intended_contacts': 1 if finger == 'index' else 2,
                **configuration, 'study_header': 'next_left_backspace_right', 'geometry': geometry,
                'coordinates': 'logical screen pixels; x right, y down',
                'timestamps': 'UI polling monotonic; Linux input clock declared in evdev_start'})

  def _append(self, path, record):
    try:
      with path.open('a') as output:
        output.write(json.dumps(record, separators=(',', ':')) + '\n')
    except OSError as error:
      self.error = str(error)

  def write(self, record, path=None):
    path = self.path if path is None else path
    if path is not None:
      self._worker.submit(self._append, path, record)

  def close(self):
    self._worker.shutdown(wait=True)


def edit_distance(expected, actual):
  previous = list(range(len(actual) + 1))
  for expected_index, expected_char in enumerate(expected, 1):
    current = [expected_index]
    for actual_index, actual_char in enumerate(actual, 1):
      current.append(min(current[-1] + 1, previous[actual_index] + 1,
                         previous[actual_index - 1] + (expected_char != actual_char)))
    previous = current
  return previous[-1]


def summarize(records):
  trials = [record for record in records if record['type'] == 'trial_end']
  gestures = [record for record in records if record['type'] == 'gesture']
  labelled = [record for record in gestures if record.get('target') is not None and len(record.get('committed', '')) == 1]
  slides = [record for record in gestures if record.get('kind') == 'layer_slide']
  labelled_slides = [record for record in slides if record.get('selection_target') is not None and len(record.get('committed', '')) == 1]
  duration = sum(record['duration'] for record in trials)
  output_characters = sum(len(record['text']) for record in trials)
  prompt_characters = sum(len(record['prompt']) for record in trials)
  corrections = sum(record['type'] == 'backspace' and record['before'] != record['after'] for record in records)
  errors = sum(edit_distance(record['prompt'], record['text']) for record in trials)
  return {
    'trials': len(trials), 'gestures': len(gestures), 'labelled_taps': len(labelled),
    'labelled_mistaps': sum(record['committed'] != record['expected'] for record in labelled),
    'layer_slides': len(slides), 'labelled_layer_slides': len(labelled_slides),
    'layer_slide_mistaps': sum(record['committed'] != record['expected'] for record in labelled_slides),
    'backspaces': corrections, 'duration_seconds': duration,
    'output_wpm': output_characters * 12 / duration if duration > 0 else None,
    'final_edit_errors': errors, 'prompt_characters': prompt_characters,
    'final_error_rate': errors / prompt_characters if prompt_characters else None,
  }
