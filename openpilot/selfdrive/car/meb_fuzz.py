# DO NOT MERGE: on-car hold fuzzing harness for VW MEB, controlled live from a JSON file
#
# card calls MebFuzz.update() right before CarInterface.apply(). It only replaces what openpilot itself sends
# (actuators.longControlState, actuators.accel, latActive), so the production MebLongStateMachine still decides
# every ACC_18 frame. Status and instructions are shown on screen through alertDebug.
#
# Files in FUZZ_DIR (/data/meb_fuzz), edited over ssh while driving:
#   control.json  read every 0.5 s when its mtime changes, see CONTROL_EXAMPLE
#   status.json   written at 2 Hz: phase, step, signals
#   results.jsonl one line per run: program, variants, outcome, signal change events
#   state.json    run counter and restart flag, survives card restarts and ignition cycles
#
# A run starts on every engagement (longActive rising edge) while control.json has "enabled" and a "program":
#   setup: P control to program.setup_speed until within 0.3 m/s for 1 s
#   steps: each step sends (state, accel) until its "until" expression is true or "timeout" seconds pass
#   hold:  stopping until disengage, then the run is written to results.jsonl
# "live": {"state": ..., "accel": ...} overrides everything immediately while engaged, for manual driving.
# Brake, gas or disengage hand back to the driver and end the run. TSK 6/7 ends the run.
#
# Expressions ("until", string "accel" or "state") are evaluated with: v a hms ss motion tsk cam req inhibit gear t T gas brake,
# plus abs/min/max/clip. t is seconds in the current step, T seconds in the run. Example: "hms == 1 or t > 4".

import json
import os
from datetime import datetime

import numpy as np

from opendbc.car import DT_CTRL, Bus
from opendbc.car.volkswagen.values import VolkswagenFlags
from openpilot.cereal import messaging
from openpilot.common.swaglog import cloudlog

FUZZ_DIR = os.environ.get('MEB_FUZZ_DIR', '/data/meb_fuzz')

CONTROL_EXAMPLE = {
  "enabled": True,
  "hold_source": "hms",       # opendbc held/braking_to_stop source: "hms" (#3861) or "standstill" (old master)
  "starting": "master",       # opendbc drive-off rule: "master" (Standstill) or "stock" (hold manager holding)
  "lat_active": False,        # send steering, False lets the driver steer in the lot
  "max_speed": 5.0,           # m/s, above this the run brakes to a hold
  "alert": None,              # ["line 1", "line 2"] to override the on-screen text
  "live": None,               # {"state": "pid", "accel": 0.3} to drive directly, ignores the program
  "program": {
    "name": "E1 aborted stop",
    "setup_speed": 2.2,
    "steps": [
      {"state": "stopping", "accel": -0.5, "until": "hms == 1", "timeout": 8},
      {"state": "pid", "accel": 0.1, "until": "ss == 1", "timeout": 4},
      {"state": "pid", "accel": 0.15, "until": "t >= 0.5"},
      {"state": "stopping", "accel": -0.5, "until": "t >= 3"},
    ],
    "hold_accel": "-0.5 if (motion == 1 and v > 0.05) else 0.0",
  },
}

STATES = ('off', 'pid', 'stopping', 'starting')
EVENT_SIGNALS = ('hms', 'ss', 'motion', 'tsk', 'cam', 'req', 'inhibit', 'gear', 'state')
ACCEL_LIMITS = (-2.5, 1.0)
SETUP_KP = 0.8
SETUP_LIMITS = (-1.5, 0.6)
FINISH_HOLD_TIME = 3.0  # s held at the end before the run counts as done
# brake only while still rolling forward: a negative accel at a standstill without a hold drives the car backwards
DEFAULT_HOLD_ACCEL = '-0.5 if (motion == 1 and v > 0.05) else 0.0'


def _read_json(path, default):
  try:
    with open(path) as f:
      return json.load(f)
  except Exception:
    return default


def _write_json(path, data):
  try:
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
      json.dump(data, f)
    os.replace(tmp, path)
  except Exception:
    cloudlog.exception('meb_fuzz: write failed')


class MebFuzz:
  def __init__(self, CI, pm):
    self.CI = CI
    self.pm = pm
    try:
      os.makedirs(FUZZ_DIR, exist_ok=True)
    except OSError:
      cloudlog.exception('meb_fuzz: cannot create fuzz dir')
    self.control_path = os.path.join(FUZZ_DIR, 'control.json')
    self.status_path = os.path.join(FUZZ_DIR, 'status.json')
    self.results_path = os.path.join(FUZZ_DIR, 'results.jsonl')
    self.state_path = os.path.join(FUZZ_DIR, 'state.json')

    self.control = {}
    self.control_mtime = None
    self.control_error = ''
    self.last_control_check = 0.
    self.last_status = 0.
    self.last_alert = 0.

    self.persist = _read_json(self.state_path, {'run_count': 0, 'restart_reason': '', 'last_outcome': ''})
    self.run = None
    self.prev_enabled = False
    self.armed = False  # engaged, waiting for the driver to release the pedals before the run starts
    self.restart_ok_time = 0.
    self.alert_text = ('', '')
    self.frame = 0  # all timing is in card cycles, so replays and sims run at log time

  @staticmethod
  def create(CI, pm):
    if CI.CP.brand == 'volkswagen' and CI.CP.flags & VolkswagenFlags.MEB and CI.CP.openpilotLongitudinalControl and CI.CC is not None:
      return MebFuzz(CI, pm)
    return None

  # ---- inputs

  def _load_control(self, now):
    if now - self.last_control_check < 0.5:
      return
    self.last_control_check = now
    try:
      mtime = os.stat(self.control_path).st_mtime
    except FileNotFoundError:
      self.control, self.control_mtime = {}, None
      return
    if mtime == self.control_mtime:
      return
    self.control_mtime = mtime
    try:
      with open(self.control_path) as f:
        self.control = json.load(f)
      self.control_error = ''
      cloudlog.event('meb_fuzz control', control=self.control)
    except Exception as e:
      self.control_error = f'control.json: {e}'[:120]

  def _signals(self, CS, CC):
    pt = self.CI.can_parsers[Bus.pt].vl
    cam = self.CI.can_parsers[Bus.cam].vl
    long_state = self.CI.CC.meb_long_state
    return {
      'v': CS.vEgo, 'a': CS.aEgo, 'gas': CS.gasPressed, 'brake': CS.brakePressed,
      'hms': int(pt['VMM_02']['HMS_Status']), 'inhibit': int(pt['VMM_02']['Long_Control_Inhibit']),
      'ss': int(pt['ESC_50']['Standstill']), 'motion': int(pt['ESC_50']['Motion_State']),
      'epb': int(pt['ESC_50']['EPB_Status']), 'tsk': int(pt['Motor_51']['TSK_Status']),
      'cam': int(cam['ACC_18']['ACC_Status_ACC']), 'req': int(long_state.prev_acc_hold_type),
      'gear': str(CS.gearShifter), 'acc_faulted': CS.accFaulted, 'can_valid': CS.canValid,
      'long_active': CC.longActive, 'enabled': CC.enabled,
    }

  # ---- run bookkeeping

  def _start_run(self, now, sig):
    program = self.control.get('program') or {}
    self.persist['run_count'] += 1
    self.run = {
      'id': self.persist['run_count'], 'start': now, 'program': program, 'phase': 'setup', 'step': -1,
      'step_start': now, 'setup_ok_since': None, 'events': [], 'last': {}, 'outcome': None, 'hold_since': None,
      'variants': {'hold_source': self.control.get('hold_source', 'hms'), 'starting': self.control.get('starting', 'master')},
      'start_v': sig['v'], 'max_v': sig['v'],
    }
    _write_json(self.state_path, self.persist)
    cloudlog.event('meb_fuzz start', run=self.run['id'], program=program.get('name', ''))

  def _end_run(self, outcome, sig):
    if self.run is None:
      return
    self.run['outcome'] = outcome
    result = {
      'id': self.run['id'], 'wall_time': datetime.now().isoformat(timespec='seconds'), 'program': self.run['program'], 'variants': self.run['variants'],
      'outcome': outcome, 'phase_at_end': self.run['phase'], 'step_at_end': self.run['step'],
      'duration': round(self.frame * DT_CTRL - self.run['start'], 2), 'start_v': round(self.run['start_v'], 2),
      'max_v': round(self.run['max_v'], 2), 'end_signals': {k: v for k, v in sig.items() if k not in ('gas', 'brake')},
      'events': self.run['events'],
    }
    try:
      with open(self.results_path, 'a') as f:
        f.write(json.dumps(result) + '\n')
    except Exception:
      cloudlog.exception('meb_fuzz: results write failed')
    cloudlog.event('meb_fuzz end', run=self.run['id'], outcome=outcome)
    self.persist['last_outcome'] = f"#{self.run['id']} {self.run['program'].get('name', '')}: {outcome}"
    _write_json(self.state_path, self.persist)
    self.run = None

  def _record(self, now, sig, state):
    values = dict(sig, state=state)
    t = round(now - self.run['start'], 3)
    for k in EVENT_SIGNALS:
      if self.run['last'].get(k) != values[k]:
        self.run['events'].append((t, k, values[k]))
        self.run['last'][k] = values[k]
    self.run['max_v'] = max(self.run['max_v'], sig['v'])

  @staticmethod
  def _eval(expr, env):
    if isinstance(expr, (int, float)):
      return expr
    return eval(expr, {'__builtins__': {}, 'abs': abs, 'min': min, 'max': max, 'int': int, 'float': float, 'round': round,
                       'clip': np.clip}, env)

  # ---- control

  def _program_cmd(self, now, sig):
    run = self.run
    program = run['program']
    env = dict(sig, t=now - run['step_start'], T=now - run['start'])
    max_speed = self.control.get('max_speed', 5.0)

    if run['phase'] != 'hold' and sig['v'] > max_speed + 1.0:
      self._event_note(now, 'too fast, holding')
      run['phase'] = 'hold'
    if run['phase'] != 'hold' and sig['motion'] == 2 and sig['v'] > 0.5:
      self._event_note(now, 'rolling back, holding')
      run['phase'] = 'hold'

    if run['phase'] == 'setup':
      target = min(program.get('setup_speed', 2.2), max_speed)
      accel = float(np.clip(SETUP_KP * (target - sig['v']), *SETUP_LIMITS))
      if abs(sig['v'] - target) < 0.3:
        run['setup_ok_since'] = run['setup_ok_since'] or now
        if now - run['setup_ok_since'] > 1.0:
          self._next_step(now)
      else:
        run['setup_ok_since'] = None
      if run['phase'] == 'setup':
        return 'pid', accel

    if run['phase'] == 'steps':
      steps = program.get('steps', [])
      step = steps[run['step']]
      env['t'] = now - run['step_start']
      try:
        done = bool(self._eval(step.get('until', 'False'), env))
        accel = float(self._eval(step.get('accel', 0.), env))
        state = step.get('state', 'pid')
        state = state if state in STATES else str(self._eval(state, env))
      except Exception as e:
        self._event_note(now, f'expr error step {run["step"]}: {e}'[:120])
        run['phase'] = 'hold'
        return 'stopping', 0.
      if done or env['t'] > step.get('timeout', 10.):
        if not done:
          self._event_note(now, f'step {run["step"]} timeout')
        self._next_step(now)
      if run['phase'] == 'steps':
        return state, accel

    # hold: stop and stay held until the driver disengages. done needs the hold manager holding a stopped car,
    # ESC_50.Standstill is also set while rolling backwards
    held = sig['hms'] == 1 and sig['motion'] == 3
    if not held:
      run['hold_since'] = None
    elif run['hold_since'] is None:
      run['hold_since'] = now
    if run['hold_since'] is not None and now - run['hold_since'] > FINISH_HOLD_TIME and run['outcome'] is None:
      run['outcome'] = 'done'
    try:
      accel = float(self._eval(program.get('hold_accel', DEFAULT_HOLD_ACCEL), env))
    except Exception:
      accel = 0.
    return 'stopping', accel

  def _next_step(self, now):
    run = self.run
    run['step'] += 1
    run['step_start'] = now
    run['phase'] = 'steps' if run['step'] < len(run['program'].get('steps', [])) else 'hold'
    self._event_note(now, f'phase {run["phase"]} step {run["step"]}')

  def _event_note(self, now, note):
    if self.run is not None:
      self.run['events'].append((round(now - self.run['start'], 3), 'note', note))

  # ---- main entry, every card cycle

  def update(self, CS, CC):
    self.frame += 1
    now = self.frame * DT_CTRL
    self._load_control(now)
    sig = self._signals(CS, CC)
    long_state = self.CI.CC.meb_long_state
    enabled = bool(self.control.get('enabled'))

    # variants are only applied while enabled, so a missing control file is plain #3861
    long_state.fuzz_hold_source = self.control.get('hold_source', 'hms') if enabled else 'hms'
    long_state.fuzz_starting = self.control.get('starting', 'master') if enabled else 'master'

    # bad states that need an ignition cycle
    if sig['tsk'] == 7 and not self.persist['restart_reason']:
      self.persist['restart_reason'] = 'TSK permanent fault (7)'
      _write_json(self.state_path, self.persist)
    if self.persist['restart_reason'] and sig['tsk'] in (2, 3) and not sig['acc_faulted']:
      self.restart_ok_time = self.restart_ok_time or now
      if now - self.restart_ok_time > 2.0:
        self.persist['restart_reason'] = ''
        _write_json(self.state_path, self.persist)
    else:
      self.restart_ok_time = 0.

    long_active = CC.longActive
    out = None
    if enabled:
      # engaging at a standstill needs the brake held, so a run arms on engagement and starts once the pedals are released
      # arm on engagement only: longActive also drops and returns around every gas override
      if CC.enabled and not self.prev_enabled:
        self.armed = True
      if not CC.enabled:
        self.armed = False
      if self.armed and long_active and not sig['brake'] and not sig['gas'] and self.run is None and self.control.get('program') and \
         not self.persist['restart_reason']:
        self.armed = False
        self._start_run(now, sig)

      if self.run is not None:
        if sig['tsk'] in (6, 7):
          self._end_run(f'TSK fault {sig["tsk"]}', sig)
        elif sig['gas'] or sig['brake'] or not long_active:
          reason = 'gas' if sig['gas'] else 'brake' if sig['brake'] else 'disengaged'
          self._end_run(self.run['outcome'] or f'aborted: {reason} in {self.run["phase"]}', sig)

      live = self.control.get('live')
      if long_active and not sig['gas'] and not sig['brake']:
        if live:
          out = (live.get('state', 'pid'), float(live.get('accel', 0.)))
        elif self.run is not None:
          out = self._program_cmd(now, sig)
      if self.run is not None:
        self._record(now, sig, out[0] if out else 'passthrough')
    elif self.run is not None:
      self._end_run('aborted: harness disabled', sig)
    self.prev_enabled = CC.enabled

    self._publish(now, sig, enabled)

    if out is None and (not enabled or self.control.get('lat_active', False)):
      return CC
    CC = CC.as_builder()
    if not self.control.get('lat_active', False):
      CC.latActive = False
    if out is not None:
      state, accel = out
      CC.actuators.longControlState = state if state in STATES else 'pid'
      CC.actuators.accel = float(np.clip(accel, *ACCEL_LIMITS))
    return CC.as_reader()

  # ---- outputs

  def _alert(self, sig, enabled):
    custom = self.control.get('alert')
    if custom:
      return str(custom[0]), str(custom[1]) if len(custom) > 1 else ''
    if self.control_error:
      return 'MEB fuzz: bad control.json', self.control_error
    if self.persist['restart_reason']:
      return 'RESTART CAR: off, wait 10 s, on', self.persist['restart_reason']
    if not enabled:
      return 'MEB fuzz off', 'write /data/meb_fuzz/control.json'
    if self.control.get('live') and sig['long_active']:
      live = self.control['live']
      return f"Live: {live.get('state', 'pid')} {float(live.get('accel', 0.)):+.2f}", self._sig_line(sig)
    name = (self.control.get('program') or {}).get('name', 'no program')
    if self.run is None:
      last = self.persist['last_outcome']
      return f'Next: {name}', f'engage when clear, last {last}' if last else 'engage when clear'
    if self.run['outcome']:
      return f"DONE #{self.run['id']}: {self.run['outcome']}", 'disengage, reverse, re-engage'
    steps = len(self.run['program'].get('steps', []))
    phase = self.run['phase']
    where = f"step {self.run['step'] + 1}/{steps}" if phase == 'steps' else phase
    return f"Run #{self.run['id']} {name}: {where}", self._sig_line(sig)

  @staticmethod
  def _sig_line(sig):
    return f"v {sig['v']:.2f} hms {sig['hms']} ss {sig['ss']} tsk {sig['tsk']} req {sig['req']} cam {sig['cam']}"

  def _publish(self, now, sig, enabled):
    if now - self.last_alert >= 0.1:
      self.last_alert = now
      self.alert_text = self._alert(sig, enabled)
      msg = messaging.new_message('alertDebug')
      msg.valid = True
      msg.alertDebug.alertText1, msg.alertDebug.alertText2 = self.alert_text
      self.pm.send('alertDebug', msg)

    if now - self.last_status >= 0.5:
      self.last_status = now
      run = None
      if self.run is not None:
        run = {k: self.run[k] for k in ('id', 'phase', 'step', 'outcome', 'variants')}
        run['program'] = self.run['program'].get('name', '')
      _write_json(self.status_path, {'time': datetime.now().isoformat(timespec='milliseconds'), 'enabled': enabled, 'alert': self.alert_text, 'run': run,
                                     'signals': sig, 'persist': self.persist, 'control_error': self.control_error})
