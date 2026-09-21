"""Conservative, reviewable quality flags; never infer intent from the selected key."""
from statistics import median

MIN_BURST_CHARACTERS = 4
MAX_BURST_GAP_SECONDS = 0.35
MAX_BURST_MEDIAN_GAP_SECONDS = 0.20


def assess_quality(records, prompts):
  flags = []
  burst = []

  def finish_burst():
    if len(burst) >= MIN_BURST_CHARACTERS:
      intervals = [following['press_time'] - preceding['press_time'] for preceding, following in zip(burst, burst[1:], strict=False)]
      if median(intervals) <= MAX_BURST_MEDIAN_GAP_SECONDS:
        flags.append({'reason': 'rapid_out_of_prompt_burst', 'trial': burst[0]['trial'],
                      'start': burst[0]['press_time'], 'end': burst[-1]['samples'][-1][0],
                      'press_times': [gesture['press_time'] for gesture in burst],
                      'characters': ''.join(gesture['committed'] for gesture in burst),
                      'median_gap_seconds': median(intervals)})
    burst.clear()

  for record in records:
    if record['type'] not in ('gesture', 'backspace', 'trial_end'):
      continue
    if record['type'] != 'gesture':
      finish_burst()
      continue
    prompt = prompts[record['trial']]
    char = record.get('committed', '')
    suspicious = len(char) == 1 and char not in prompt and not prompt.startswith(record['before'])
    if burst and (record['trial'] != burst[-1]['trial'] or record['press_time'] - burst[-1]['press_time'] > MAX_BURST_GAP_SECONDS):
      finish_burst()
    if suspicious:
      burst.append(record)
    else:
      finish_burst()
  finish_burst()
  return {'rule_version': 1, 'flags': flags, 'flagged_trials': sorted({flag['trial'] for flag in flags}),
          'flagged_gestures': sum(len(flag['press_times']) for flag in flags)}


def flagged_gesture(record, quality):
  return any(record.get('trial') == flag['trial'] and record.get('press_time') in flag['press_times'] for flag in quality['flags'])
