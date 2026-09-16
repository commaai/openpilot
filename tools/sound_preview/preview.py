#!/usr/bin/env python3
"""Local soundd workbench: python -m tools.sound_preview.preview (no audio device needed)."""
import argparse
import base64
import io
import json
import math
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from openpilot.cereal import log, messaging
from openpilot.selfdrive.ui.soundd import SAMPLE_RATE, ESCALATION_ALERTS, Soundd, sound_list

# Representative event inputs from events.py, plus the UI timeout/preview paths.
# State delivery is simulated; selection, escalation, volume and PCM generation
# below all execute the production Soundd implementation.
ALERTS = {
  'distracted': ('Driver distracted', 'DISENGAGE IMMEDIATELY', 'Driver Distracted', 'critical', 'full', 'warningImmediate'),
  'unresponsive': ('Driver unresponsive', 'DISENGAGE IMMEDIATELY', 'Driver Unresponsive', 'critical', 'full', 'warningImmediate'),
  'fcw': ('Forward collision warning', 'BRAKE!', 'Risk of Collision', 'critical', 'full', 'warningSoft'),
  'fault': ('Immediate disengagement', 'TAKE CONTROL IMMEDIATELY', 'CAN Bus Disconnected', 'critical', 'full', 'warningImmediate'),
  'soft_red': ('Soft disengagement, red stage', 'TAKE CONTROL IMMEDIATELY', 'Camera Malfunction', 'critical', 'full', 'warningImmediate'),
  'soft_orange': ('Soft disengagement, orange stage', 'TAKE CONTROL IMMEDIATELY', 'Camera Malfunction', 'userPrompt', 'full', 'warningSoft'),
  'dm_orange': ('Driver monitoring, orange stage', 'Pay Attention', 'Driver Distracted', 'userPrompt', 'mid', 'promptDistracted'),
  'aeb': ('Emergency braking, silent', 'BRAKE!', 'Emergency Braking: Risk of Collision', 'critical', 'full', 'none'),
  'stock_aeb': ('Stock AEB, silent', 'BRAKE!', 'Stock AEB: Risk of Collision', 'critical', 'full', 'none'),
  'preview': ('Camera preview, fixed', 'DISENGAGE IMMEDIATELY', 'Driver Distracted · preview', 'critical', 'full', 'warningImmediate'),
  'old_preview': ('Camera preview, old metadata', 'Sound only', 'Old preview omitted severity and size', 'normal', 'none', 'warningImmediate'),
  'clear': ('No alert', 'No active alert', 'Timer resets', 'normal', 'none', 'none'),
  'timeout': ('System unresponsive', 'TAKE CONTROL IMMEDIATELY', 'System Unresponsive', 'critical', 'full', 'warningImmediate'),
}


def scenario(label, phases, expected, note=''):
  return {'label': label, 'phases': [{'alert': alert, 'seconds': seconds} for alert, seconds in phases],
          'expected': expected, 'note': note}

SCENARIOS = {
  **{name: scenario(ALERTS[name][0], [(name, 12)], 8.) for name in ['distracted', 'unresponsive', 'fcw', 'fault', 'preview']},
  'aeb': scenario('Silent AEB', [('aeb', 12)], None, 'Red with no assigned sound stays silent.'),
  'stock_aeb': scenario('Silent stock AEB', [('stock_aeb', 12)], None),
  'orange': scenario('Orange warning held', [('soft_orange', 12)], None, 'critical.wav alone does not mean a red alert.'),
  'old_preview': scenario('Reproduce old preview bug', [('old_preview', 12)], None, 'Sound-only metadata cannot start the red-alert timer.'),
  'short': scenario('Short collision warning', [('fcw', 2), ('clear', 10)], None, 'A brief warning does not reach eight seconds.'),
  'soft': scenario('Soft disable countdown', [('soft_orange', 2.5), ('soft_red', .5), ('clear', 9)], None,
                   'Typical countdown: only the final half-second is red.'),
  'reset': scenario('Clear and retrigger', [('distracted', 6), ('clear', 1), ('distracted', 10)], 15.),
  'silent_reset': scenario('Silent red interrupts timer', [('distracted', 6), ('aeb', 1), ('distracted', 10)], 15.),
  'orange_reset': scenario('Orange interrupts timer', [('distracted', 6), ('dm_orange', 1), ('distracted', 10)], 15.),
  'switch': scenario('Continuous red: DM → collision', [('distracted', 4), ('fcw', 8)], 8., 'Continuous audible red activity keeps the timer.'),
  'max_switch': scenario('Change sound family after max', [('distracted', 9), ('fcw', 3)], 8., 'Both max mappings currently use dm_critical_max.wav.'),
  'timeout': scenario('Lost selfdrive state', [('timeout', 16)], 13.05,
                      'Five-second timeout, eight seconds of red, then the existing timeout window ends at 15 seconds.'),
}


class SimulatedState:
  def __init__(self):
    self.state = messaging.new_message('selfdriveState').selfdriveState
    self.updated = {'selfdriveState': True}
    self.recv_time = {'selfdriveState': 0.}

  def __getitem__(self, name):
    return self.state


def simulate(phases, starting_volume=.2, include_audio=True):
  if not isinstance(phases, list) or not 1 <= len(phases) <= 12:
    raise ValueError('Use between one and twelve phases.')
  if not math.isfinite(starting_volume) or not 0 <= starting_volume <= 1:
    raise ValueError('Initial volume must be between zero and one.')
  steps = []
  for phase in phases:
    seconds = float(phase['seconds'])
    if phase['alert'] not in ALERTS or not math.isfinite(seconds) or not .05 <= seconds <= 30:
      raise ValueError('Each phase needs a known alert and a duration from 0.05 to 30 seconds.')
    steps.append((phase['alert'], round(seconds * 20)))
  if sum(count for _, count in steps) > 1200:
    raise ValueError('Keep the total duration at or below 60 seconds.')

  sd = Soundd()
  sd.current_volume = starting_volume
  sm = SimulatedState()
  trace, chunks = [], []
  frame = 0
  now = 0.
  # Only replace soundd's clock, not the process-wide time module.
  with patch('openpilot.selfdrive.ui.soundd.time', SimpleNamespace(monotonic=lambda: now)):
    for alert, count in steps:
      label, title, subtitle, status, size, sound = ALERTS[alert]
      phase_start = frame / 20
      for _ in range(count):
        now = frame / 20
        sm.state.alertSound = getattr(log.SelfdriveState.AudibleAlert, sound)
        sm.state.alertStatus = status
        sm.state.alertSize = size
        sm.state.enabled = True
        sm.updated['selfdriveState'] = alert != 'timeout'
        sm.recv_time['selfdriveState'] = phase_start if alert == 'timeout' else now
        if alert == 'timeout':
          sm.state.alertSound = 'none'
          sm.state.alertStatus = 'normal'
          sm.state.alertSize = 'none'
        sd.get_audible_alert(sm)
        sd.update_volume()
        filename = sound_list[sd.current_alert][0] if sd.current_alert else 'silence'
        red_time = 0 if sd.critical_start_time is None else now - sd.critical_start_time
        visible = status
        if alert == 'timeout':
          visible = 'critical' if 5 < now - phase_start < 15 else 'normal'
        trace.append({'time': now, 'label': label, 'title': title, 'subtitle': subtitle, 'status': visible,
                      'sound': filename, 'volume': sd.current_volume, 'redTime': red_time,
                      'escalated': sd.current_alert in ESCALATION_ALERTS.values()})
        # Exercise real looping and stop behavior even when no WAV is requested.
        pcm = sd.get_sound_data(SAMPLE_RATE // 20)
        if include_audio:
          chunks.append(pcm)
        frame += 1

  result = {'trace': trace, 'duration': frame / 20,
            'firstMax': next((row['time'] for row in trace if row['escalated']), None)}
  if include_audio:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav:
      wav.setnchannels(1)
      wav.setsampwidth(2)
      wav.setframerate(SAMPLE_RATE)
      wav.writeframes((np.clip(np.concatenate(chunks), -1, 1) * 32767).astype('<i2').tobytes())
    result['audio'] = base64.b64encode(buf.getvalue()).decode()
  return result


def run_checks():
  checks = []
  for key, spec in SCENARIOS.items():
    result = simulate(spec['phases'], include_audio=False)
    actual, expected = result['firstMax'], spec['expected']
    timing_ok = actual is None if expected is None else actual is not None and abs(actual - expected) < .051
    volume_ok = all(row['volume'] == 1. for row in result['trace'] if row['escalated'])
    checks.append({'id': key, 'label': spec['label'], 'passed': timing_ok and volume_ok,
                   'expected': expected, 'actual': actual})
  return checks


class Handler(BaseHTTPRequestHandler):
  def respond(self, body, content_type='application/json', status=200):
    data = json.dumps(body).encode() if content_type == 'application/json' else body
    self.send_response(status)
    self.send_header('Content-Type', content_type)
    self.send_header('Content-Length', str(len(data)))
    self.send_header('Cache-Control', 'no-store')
    self.end_headers()
    self.wfile.write(data)

  def do_GET(self):
    if self.path == '/':
      self.respond(Path(__file__).with_name('index.html').read_bytes(), 'text/html; charset=utf-8')
    elif self.path == '/api/catalog':
      self.respond({'alerts': {key: value[0] for key, value in ALERTS.items()}, 'scenarios': SCENARIOS})
    elif self.path == '/api/checks':
      self.respond(run_checks())
    else:
      self.respond({'error': 'Not found'}, status=404)

  def do_POST(self):
    if self.path != '/api/render':
      self.respond({'error': 'Not found'}, status=404)
      return
    try:
      length = int(self.headers.get('Content-Length', '0'))
      if not 0 < length <= 10000:
        raise ValueError('Invalid request size.')
      request = json.loads(self.rfile.read(length))
      self.respond(simulate(request['phases'], float(request.get('volume', .2))))
    except (ValueError, KeyError, TypeError) as e:
      self.respond({'error': str(e)}, status=400)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--port', type=int, default=8769)
  args = parser.parse_args()
  print(f'Open http://127.0.0.1:{args.port}', flush=True)
  HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
  main()
