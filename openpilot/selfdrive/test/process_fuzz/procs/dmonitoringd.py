import copy
import math

import numpy as np

from openpilot.common.realtime import DT_DMON
from openpilot.selfdrive.monitoring.policy import DRIVER_MONITOR_SETTINGS
from openpilot.selfdrive.test.process_fuzz import procs as U, properties as P
from openpilot.selfdrive.test.process_fuzz.harness import Spec
from openpilot.selfdrive.test.process_fuzz.space import new_message

S = DRIVER_MONITOR_SETTINGS()
DMM = 'dmonitoringmodeld fill_driver_data slices a fixed 12-wide face_descs (dmonitoringmodeld.py:85-88)'
SIGMOID = 'sigmoid(safe_exp clip) is in [1.67e-5, 1] or NaN (parse_model_outputs.py:4-9, dmonitoringmodeld.py:81)'
STD = 'safe_exp is in [0, e^11] or NaN (dmonitoringmodeld.py:79)'
UP_LEN = 'dmonitoringmodeld: invariant driverStateV2 list lengths'
UP_PROB = 'dmonitoringmodeld: bounded driverStateV2 probabilities'

DD = [f'driverStateV2.{s}DriverData' for s in ('left', 'right')]
LEDGER = U.upstream('extrinsicsCalibration.rpyCalib[]') | {
  **{f'{d}.{f}.len': ('enforced', DMM, UP_LEN) for d in DD for f in ('faceOrientation', 'facePosition', 'faceOrientationStd', 'facePositionStd')},
  **{f'{d}.{f}:inf': ('enforced', SIGMOID, UP_PROB) for d in DD
     for f in ('faceProb', 'leftEyeProb', 'rightEyeProb', 'leftBlinkProb', 'rightBlinkProb', 'sunglassesProb', 'phoneProb', 'sleepProb')},
  'driverStateV2.wheelOnRightProb:inf': ('enforced', SIGMOID, UP_PROB),
  **{f'{d}.faceOrientationStd[]:inf': ('enforced', STD, 'dmonitoringmodeld: bounded driverStateV2 stds') for d in DD},
  'modelV2.meta.disengagePredictions.brakeDisengageProbs[]:inf': ('enforced', 'sigmoid, fill_model_msg.py:133 / parse_model_outputs.py:103',
                                                                  'modeld: bounded modelV2 probabilities'),
  'P DriverLockoutCount:negative': ('enforced', 'params_keys.h:41 default 0, only writer policy.py:322 increments', None),
}


def template(svc, g=None):
  m = U.shape(svc) if svc != 'modelV2' else new_message(svc)
  if svc == 'modelV2':
    m.modelV2.meta.disengagePredictions.brakeDisengageProbs = [0.] * 5
  return m


def cycles(r):
  # per driverMonitoringState output: times, outputs, valid flags
  outs = P.outs(r, 'driverMonitoringState')
  t = np.array([P.t_of(m) for m in outs])
  return t, [m.driverMonitoringState for m in outs], [m.valid for m in outs]


def demo(r):
  return bool(r.genome.params.get('IsDriverViewEnabled'))


def attentive_free(r):
  # per output: the driver gives DM nothing to restore awareness with: engaged, above the alert speed, no interaction,
  # and no confident face, or distracted (policy.py:348-369)
  t, dm, valid = cycles(r)
  en = [s is not None and s.enabled for s in P.held(r.ins, 'selfdriveState', t)]
  cs = P.held(r.ins, 'carState', t)
  ok = np.array([e and c is not None and c.vEgo >= S._ALERT_MIN_SPEED and not c.steeringPressed and not c.gasPressed and v and
                 (not d.visionPolicyState.faceDetected or d.visionPolicyState.isDistracted or not d.visionPolicyState.pose.uncertainty < S._HI_STD_THRESHOLD)
                 for e, c, d, v in zip(en, cs, dm, valid, strict=True)], dtype=bool)
  return t, dm, ok


# worst case to red: vision up to orange, then the wheeltouch timeout, after the hi-std fallback time
ESCALATE = S._VISION_POLICY_ALERT_3_TIMEOUT + S._WHEELTOUCH_POLICY_ALERT_3_TIMEOUT + S._HI_STD_FALLBACK_TIME * DT_DMON


@P.prop('escalates')
def escalates(r):
  # liveness: an unattended engaged drive always reaches the terminal alert
  if demo(r):
    return None
  t, dm, ok = attentive_free(r)
  held = P.held_for(t, ok, ESCALATE)
  bad = [k for k in np.flatnonzero(held) if dm[k].alertLevel != 'three']
  assert not bad, f'unattended for {ESCALATE:.0f}s but alertLevel {dm[bad[0]].alertLevel}, awareness ' + \
    P.around(t, [f'{d.alertLevel}/{d.activePolicy}/{d.visionPolicyState.awarenessPercent}/{d.wheeltouchPolicyState.awarenessPercent}' for d in dm], bad[0])
  return held.any() and 'unattended'


@P.prop('force decel sticky')
def force_decel_sticky(r):
  # once force decel is commanded it holds while the alert stays red (policy.py:398): no on/off pulse to the car
  t, dm, _ = cycles(r)
  for k in range(1, len(dm)):
    if dm[k - 1].noResponseForceDecel and dm[k].alertLevel == 'three':
      assert dm[k].noResponseForceDecel, f't={t[k]:.2f}s force decel dropped while red (lockout {dm[k - 1].lockout} -> {dm[k].lockout})'
  return any(d.noResponseForceDecel for d in dm) and 'force decel'


LEVEL = {'none': 0, 'one': 1, 'two': 2, 'three': 3}


@P.prop('alert steps')
def alert_steps(r):
  # alertLevel rises one level at a time unless the policy swapped (awareness swap, policy.py:192-203)
  t, dm, _ = cycles(r)
  for k in range(1, len(dm)):
    a, b = LEVEL[str(dm[k - 1].alertLevel)], LEVEL[str(dm[k].alertLevel)]
    assert b <= a + 1 or dm[k].activePolicy != dm[k - 1].activePolicy, f't={t[k]:.2f}s {dm[k - 1].alertLevel} -> {dm[k].alertLevel}'


def state_invariants(dm):
  v = dm.visionPolicyState
  active = v.awarenessPercent if dm.activePolicy == 'vision' else dm.wheeltouchPolicyState.awarenessPercent
  return ((dm.alertLevel != 'three' or active == 0) and (not dm.lockout or dm.lockoutMinutesRemaining >= 1) and
          (not v.isDistracted or v.faceDetected and (v.distractedTypes.pose or v.distractedTypes.eye or v.distractedTypes.phone or v.distractedTypes.sleep)) and
          (dm.alwaysOn or not dm.alwaysOnLockout) and (not dm.noResponseForceDecel or dm.alertLevel == 'three'))


STEPS = {0., np.float32(DT_DMON / S._VISION_POLICY_ALERT_3_TIMEOUT), np.float32(DT_DMON / S._WHEELTOUCH_POLICY_ALERT_3_TIMEOUT)}


@P.prop('disengaged resets')
def disengaged_resets(r):
  # not engaged and not always-on: no alert, full awareness (policy.py:335-340), for every step that ran
  if demo(r):
    return None
  t, dm, valid = cycles(r)
  en, n = P.held(r.ins, 'selfdriveState', t), 0
  for k, d in enumerate(dm):
    if valid[k] and en[k] is not None and not en[k].enabled and not d.alwaysOn:
      assert d.alertLevel == 'none', f't={t[k]:.2f}s disengaged but alertLevel {d.alertLevel}'
      n += 1
  return n and 'disengaged'


POSE = [f'{d}.faceOrientation[]' for d in DD]


@P.prop('recovers pose calibration')
def recovers(r):
  # a single non-finite pose sample must not stop the pose calibration for good (stat_live.py:61-65): rerun with a
  # one-frame NaN spike in the held pose; where the clean run's calibration still progresses afterwards, so must this one
  genes = [(p, g) for p, g in r.genome.genes.items() if p in POSE and g.shape == 'hold' and math.isfinite(g.a)]
  t, x = P.series(r.outs, 'driverMonitoringState.visionPolicyState.pose.pitchCalib.calibratedPercent')
  if r.err or not genes or len(x) < 2 or x[-1] == 0:
    return None
  t_s = t[np.searchsorted(x, x[-1] / 2)]  # spike half way through the clean run's progress
  g = copy.deepcopy(r.genome)
  for p, _ in genes:
    g.genes[p].shape, g.genes[p].b, g.genes[p].t0, g.genes[p].t1 = 'spike', math.nan, t_s, t_s + DT_DMON
  r2 = r.rerun(g)
  _, x2 = P.series(r2.outs, 'driverMonitoringState.visionPolicyState.pose.pitchCalib.calibratedPercent')
  k = int(np.searchsorted(t, t_s + 1.))
  if k >= min(len(x), len(x2)) or x[-1] <= x[k]:
    return None
  assert x2[-1] > x2[k], f'calibration frozen at {x2[k]}% after a NaN pose at t={t_s:.2f}s (clean run: {x[k]}% -> {x[-1]}%)'
  return 'progress'


TP = 'selfdrive/monitoring/policy.py:DriverMonitoring.'
SL = 'common/stat_live.py:RunningStat.'
UNREACHED = {
  'timer > duration: the pose filters stop counting at 7200 pushes, 360 s (policy.py:83)':
    [SL + 'push_data:"if self.max_trackable < 0 or self.n < self.max_trackable:"#1:F'],
  'dead: n was just incremented (max_trackable is -1 or 7200, never 0)': [SL + 'push_data:"if self.n == 0:":T'],
  'dead: only raw_stat.std() is called and every raw_stat starts from priors with n=2 (policy.py:93-94,137)': [SL + 'variance:"if self.n >= 2:":F'],
  U.H_LOCKSTEP: ['selfdrive/monitoring/dmonitoringd.py:dmonitoringd_thread:"if not sm.updated[\'driverStateV2\']:":F'],
  'dead: dmonitoringd passes no settings':
    [TP + '__init__:"self.settings = settings if settings is not None else DRIVER_MONITOR_SETTINGS()":notNone'],
  'dead: a face with high std is never distracted (distraction needs low_std, :286), so :309 already took it':
    [TP + '_update_states:"elif self.face_detected and self.pose.low_std:"#1:F',
     TP + '_update_states:"if self.face_detected and not self.pose.low_std and not self.driver_distracted:"#2:T'],
  'enforced driverStateV2.leftDriverData.faceOrientation.len':
    [TP + '_update_states:"if not all(len(x) > 0 for x in (driver_data.faceOrientation, driver_data.facePosition,":F'],
}
KNOWN = {
  'escalates': ('real', 'orange then face lost: wheeltouch refuses with step_change 0, awareness freezes, no red (policy.py:181-186)',
                'known_escalation_freeze.json'),
  'force decel sticky': ('real', 'a lockout ending while red resets cnt_since_alert_3: force decel drops 5 s (policy.py:327-331,398)',
                         'known_force_decel_drop.json'),
  'no_chatter driverMonitoringState.activePolicy': ('real', 'no hysteresis on _FACE_THRESHOLD: faceProb at 0.7 swaps vision/wheeltouch '
                                                    + '(policy.py:268,308)', 'known_policy_chatter.json'),
  'no_chatter driverMonitoringState.alertLevel': ('real', 'vEgo at _ALERT_MIN_SPEED toggles the lowspeed exemption: banner on/off '
                                                  + '(policy.py:345,359)', 'known_alert_chatter.json'),
  'no_chatter driverMonitoringState.isRHD': ('real', 'driver view filters the wheel side at M > 0.5 with no hysteresis, bypassing the engaged '
                                             + 'lock (policy.py:256-262)', 'known_rhd_chatter.json'),
  'finite driverMonitoringState': ('theoretical', 'a NaN/inf model pose reaches visionPolicyState.pose with valid=True (policy.py:269,415)',
                                   'known_nan_pose.json'),
  'recovers pose calibration': ('theoretical', 'one NaN pose makes the raw std NaN forever, pose/wheelpos calibration freezes (stat_live.py:61-65)',
                                'known_pose_calibrated.json'),
  'alive driverMonitoringState (DriverLockoutCount 127)': ('theoretical', 'a lockout at 127 overflows Int8 lockoutCount and kills dmonitoringd '
                                                           + '(policy.py:320,393), ~62 h of lockouts', 'known_lockout_overflow.json'),
}


def tag(sp, g, ins=None):
  return 'DriverLockoutCount 127' if g.params.get('DriverLockoutCount', 0) >= 127 else None  # lockoutCount is Int8


SPEC = Spec(
  proc='dmonitoringd',
  scope={
    'selfdrive/monitoring/dmonitoringd.py': None,
    'selfdrive/monitoring/policy.py': None,
    'common/stat_live.py': {'RunningStat', 'RunningStatFilter'},
  },
  ledger=LEDGER,
  unreached=UNREACHED,
  known=KNOWN,
  tag=tag,
  params={'IsRhdDetected': [False, True], 'AlwaysOnDM': [False, True], 'IsDriverViewEnabled': [False, True],
          'DriverTooDistracted': [False, True], 'DriverLockoutCount': [0, 1, 2, 3, 4, 5, 127]},
  template=template,
  duration=65.,  # lockout expiry from the params: 1200 frames + 1 (policy.py:44, :327)
  examples=16,
  bounds={
    'carState.vEgo': [S._ALERT_MIN_SPEED, S._WHEELPOS_CALIB_MIN_SPEED, S._POSE_CALIB_MIN_SPEED, 16., 16 + math.sqrt(0.4 / 0.00156),
                      16 - math.sqrt(0.4 / 0.00156)],  # policy.py:212 k1 peak and floor
    'carState.steeringAngleDeg': [S._POSE_YAW_MIN_STEER_DEG, -S._POSE_YAW_MIN_STEER_DEG, 180., -180.],
    **{f'{d}.{f}': v for d in DD for f, v in (
      ('faceProb', [S._FACE_THRESHOLD]), ('leftEyeProb', [S._EYE_THRESHOLD]), ('rightEyeProb', [S._EYE_THRESHOLD]),
      ('leftBlinkProb', [S._BLINK_THRESHOLD, 2 * S._BLINK_THRESHOLD - 1]), ('rightBlinkProb', [S._BLINK_THRESHOLD, 2 * S._BLINK_THRESHOLD - 1]),
      ('sunglassesProb', [S._SG_THRESHOLD]), ('phoneProb', [S._PHONE_THRESH]), ('sleepProb', [S._SLEEP_THRESH]),
      ('faceOrientationStd[]', [S._HI_STD_THRESHOLD, S._DCAM_UNCERTAIN_ALERT_THRESHOLD]))},
    # pose errors at facePosition 0 (no focal angle): pitch = o[0] - rpy[1], yaw = -o[1] - rpy[2] (policy.py:115-128, 221-242)
    **{f'{d}.faceOrientation[]': [S._PITCH_NATURAL_OFFSET - S._PITCH_NATURAL_THRESHOLD, S._PITCH_NATURAL_OFFSET - S._POSE_PITCH_THRESHOLD,
                                 S._PITCH_NATURAL_OFFSET - S._POSE_PITCH_THRESHOLD_SLACK,
                                 -S._YAW_NATURAL_OFFSET + S._POSE_YAW_THRESHOLD, -S._YAW_NATURAL_OFFSET - S._POSE_YAW_THRESHOLD,
                                 -S._YAW_NATURAL_OFFSET + S._POSE_YAW_THRESHOLD_SLACK, -S._YAW_NATURAL_OFFSET - S._POSE_YAW_THRESHOLD_SLACK,
                                 S._PITCH_MAX_OFFSET, S._PITCH_MIN_OFFSET, S._YAW_MAX_OFFSET, S._YAW_MIN_OFFSET] for d in DD},
    'driverStateV2.wheelOnRightProb': [S._WHEELPOS_THRESHOLD],
    'driverStateV2.frameId': [41, 6000, 12000, 2 ** 32 - 1],
    'modelV2.meta.disengagePredictions.brakeDisengageProbs[]': [0.1, 0.3, 0.5 * 0.6],
  },
  checks=[
    P.alive('driverMonitoringState'),
    P.finite('driverMonitoringState'),
    P.valid_from_inputs('driverMonitoringState'),
    P.invariant('driverMonitoringState', state_invariants, 'driverMonitoringState consistent'),
    P.invariant('driverMonitoringState', lambda d: d.visionPolicyState.awarenessStep in STEPS and d.wheeltouchPolicyState.awarenessStep in STEPS,
                'awarenessStep in its set'),
    P.bounded('driverMonitoringState.visionPolicyState.awarenessPercent', 0, 100),
    P.bounded('driverMonitoringState.wheeltouchPolicyState.awarenessPercent', 0, 100),
    P.monotone('driverMonitoringState.lockoutCount'),
    P.monotone('driverMonitoringState.rhdCalibration.calibratedPercent'),
    P.monotone('driverMonitoringState.visionPolicyState.pose.pitchCalib.calibratedPercent'),
    P.monotone('driverMonitoringState.visionPolicyState.pose.yawCalib.calibratedPercent'),
    P.monotone('driverMonitoringState.visionPolicyState.pose.calibrated'),
    escalates, force_decel_sticky, alert_steps, disengaged_resets, recovers,
    P.no_chatter('driverMonitoringState.activePolicy', 1.),
    P.no_chatter('driverMonitoringState.alertLevel', 1.),
    P.no_chatter('driverMonitoringState.isRHD', 1.),
  ],
)
