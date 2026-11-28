import math

import numpy as np

from openpilot.cereal import log
from openpilot.selfdrive.locationd import locationd as L
from openpilot.selfdrive.locationd.models.pose_kf import PoseKalman
from openpilot.selfdrive.test.process_fuzz import procs as U, properties as P, space as S
from openpilot.selfdrive.test.process_fuzz.harness import Spec

SENSORD = 'lsm6ds3_accel.py:72-82 / lsm6ds3_gyro.py:71-78: int16 * scale, init(acceleration / gyroUncalibrated), 3 elements'
SENSOR_TIME = 'sensord stamps the interrupt time, published right after'

LEDGER = U.upstream('cameraOdometry.', 'carState.vEgo', 'extrinsicsCalibration.rpyCalib') | {
  # sensor events are read from drain_sock (traced with no service prefix): sub_sock per service (locationd.py:273) fixes which();
  # logMonoTime and valid are the genome's timing and 'invalid' fault
  '.which()': ('enforced', 'sub_sock(which) only delivers that service, locationd.py:273', None),
  'logMonoTime': ('harness', 'message timing from the genome (rates, slow/drop/stale faults)', None),
  'valid': ('harness', 'the invalid fault', None),
  # not reduced: the union variant. process replay's routes reach the other-variant fall-through (older logs)
  **{f'{p}:nan|inf': ('enforced', SENSORD, 'sensord: finite') for p in ('accelerometer.acceleration.v[]', 'gyroscope.gyroUncalibrated.v[]')},
  # inf is enforced on these, so a value past float32 only comes from a list weight and is clamped to F32 in the message
  **{f'{p}:huge': ('theoretical', 'past float32: beyond int16 * scale', None) for p in ('accelerometer.acceleration.v[]', 'gyroscope.gyroUncalibrated.v[]')},
  'cameraOdometry.timestampEof:future': ('theoretical', 'frame end of exposure, same monotonic clock as logMonoTime, so never later '
                                         + '(camerad/modeld causality); < CAM_ODO_POSE_DELAY only without a VisionIPC timestamp', None),
  'accelerometer.timestamp:absolute': ('theoretical', SENSOR_TIME, None),
  'gyroscope.timestamp:absolute': ('theoretical', SENSOR_TIME, None),
  'P LocationFilterInitialState': ('theoretical', 'nothing in the tree writes it (only read at locationd.py:288)', None),
}


def template(svc, g=None):
  m = U.shape(svc)
  if svc == 'cameraOdometry':  # stds are safe_exp outputs, > 0 (parse_model_outputs.py:4-6)
    m.cameraOdometry.transStd, m.cameraOdometry.rotStd = [0.1] * 3, [0.01] * 3
  elif svc in ('accelerometer', 'gyroscope'):  # sensord's shape
    ev = getattr(m, svc)
    ev.source = log.SensorEventData.SensorSource.lsm6ds3
    ev.init('acceleration' if svc == 'accelerometer' else 'gyroUncalibrated').v = [0.] * 3
  return m


def initial_states():
  # LocationFilterInitialState (locationd.py:288-294): empty, the initial state, non-finite and wrong-length lists
  x, std = PoseKalman.initial_x.tolist(), np.sqrt(np.diag(PoseKalman.initial_P)).tolist()
  out = [None] * 6
  for v, s in [([], []), (x, std), ([math.nan] * 18, std), (x, [math.nan] * 18), (x[:3], std), (x, [0.] * 18), ([1e30] * 18, std)]:
    out.append({'event': {'deviceMotion': {'debugFilterState': {'value': v, 'std': s}}}})
  return out


def camodo_time_outside(g):
  # timestampEof is the frame's end of exposure on the clock modeld stamps logMonoTime with: never later than logMonoTime, never below
  # CAM_ODO_POSE_DELAY unless the VisionIPC buffer carries no timestamp. Older frames (modeld lag) are in the image
  ge, lo, hi = g.genes.get('cameraOdometry.timestampEof'), L.CAM_ODO_POSE_DELAY * 1e9, S.T0
  if ge is None:
    return True  # the template's 0
  vals = (ge.a, ge.b, ge.a + ge.b, ge.a - ge.b) if ge.shape == 'hover' else (ge.a, ge.b)
  return any((v < lo - S.T0 or v > 0) if ge.idx == 'lmt' else (v < lo or v > hi) for v in vals)


def tag(sp, g, ins=None):
  # sensor stamps are offsets from logMonoTime (any size: a stalled publisher is real), never absolute
  sensor_abs = any(g.genes.get(p) is None or g.genes[p].idx != 'lmt' for p in ('accelerometer.timestamp', 'gyroscope.timestamp'))
  other = sensor_abs or camodo_time_outside(g) or g.params.get('LocationFilterInitialState') is not None
  return P.theoretical(sp, g) or ('theoretical inputs' if other else None)


@P.prop('flags from inputs deviceMotion')
def flags_from_inputs(r):
  # inputsOK => every SubMaster input's last message valid (sm.all_valid, locationd.py:332) and the cameraOdometry it just used
  # finite (a NaN has to count as a bad input, not silently reset the filter); sensorsOK => the last accelerometer and gyroscope
  # messages valid and (sim) received in this cycle (:249-261)
  for c in P.cycles(r, ('deviceMotion',)):
    o = c.out['deviceMotion'].deviceMotion
    if o.inputsOK:
      for s in ('carState', 'extrinsicsCalibration', 'cameraOdometry'):
        assert s in c.last and c.last[s].valid, f't={c.t:.2f}s inputsOK with {s} {"invalid" if s in c.last else "never sent"}'
      co = c.trig.cameraOdometry
      bad = [k for k in ('trans', 'rot', 'transStd', 'rotStd') if not np.isfinite(list(getattr(co, k))).all()]
      assert not bad, f't={c.t:.2f}s inputsOK with non-finite cameraOdometry {bad}'
    if o.sensorsOK:
      for s in ('accelerometer', 'gyroscope'):
        assert s in c.last and c.last[s].valid, f't={c.t:.2f}s sensorsOK with {s} {"invalid" if s in c.last else "never sent"}'
        assert not r.genome.sim or c.recv[s] == c.trig.logMonoTime, f't={c.t:.2f}s sensorsOK (sim) without a {s} this cycle'


@P.prop('invariant deviceMotion valid consistent')
def valid_consistent(r):
  # msg.valid is filter_valid, which every measurement carries (locationd.py:221-227), and never goes back to False
  was = False
  for m in r.outs:
    dm = m.deviceMotion
    flags = {dm.orientationNED.valid, dm.velocityDevice.valid, dm.angularVelocityDevice.valid, dm.accelerationDevice.valid}
    assert flags == {m.valid}, f't={P.t_of(m):.2f}s measurement valid {flags} vs msg.valid {m.valid}'
    assert m.valid or not was, f't={P.t_of(m):.2f}s valid went back to False'
    was = m.valid


@P.prop('invariant posenetOK at low speed')
def posenet_low_speed(r):
  # posenetOK = not std_spike or car_speed <= 5 (locationd.py:242), car_speed = |vEgo| of the last carState (:149): paramsd's ledger
  t, ok = P.series(r.outs, 'deviceMotion.posenetOK')
  for k, v in enumerate(P.held(r.ins, 'carState.vEgo', t - 1e-6)):
    assert ok[k] or v is None or abs(v) > 5. or v != v, f't={t[k]:.2f}s posenetOK False at vEgo {v}'


@P.prop('monotonic deviceMotion.timestamp')
def monotonic_timestamp(r):
  # the filter time is what paramsd times its observations by (paramsd.py:68): it must not step backwards while valid
  ms = P.outs(r, 'deviceMotion')
  for a, b in zip(ms, ms[1:], strict=False):
    x, y = a.deviceMotion.timestamp, b.deviceMotion.timestamp
    assert not (a.valid and b.valid and y < x), f't={P.t_of(b):.2f}s timestamp went back {(x - y) / 1e9:.3f}s'


LOC = 'selfdrive/locationd/locationd.py:'
UNREACHED = {
  U.H_LOCKSTEP: [LOC + 'main:"if sm.updated["cameraOdometry"]:":F'],
  U.ENV_DEBUG: [LOC + 'LocationEstimator.get_msg:"if self.debug:":T', LOC + 'LocationEstimator.get_msg:"for k in self.observations.keys()":*'],
  'enforced: calibrationd always publishes 3 rpyCalib': [LOC + 'LocationEstimator.handle_log:"if len(msg.rpyCalib) > 0:":F'],
  'dead: new_x and new_P are assigned together': [LOC + 'LocationEstimator.handle_log:"if new_x is not None and new_P is not None:"#1:None'],
  'dead: the with body raises no Python exception': [LOC + 'main:"with log.Event.from_bytes(initial_pose_data) as lp_msg:":F'],
}

KNOWN = {
  'time_fields deviceMotion (theoretical inputs)': ('theoretical', 'a far-future cameraOdometry timestampEof moves the filter clock '
                                                    + '(locationd.py:162-164), only checked against the rewind', 'known_camodo_clock.json'),
  'finite deviceMotion (theoretical inputs)': ('theoretical', 'a huge restored LocationFilterInitialState (:292-294) overflows in '
                                               + 'prediction, unchecked (:205-206)', 'known_restore_huge_state.json'),
  'no_chatter deviceMotion.inputsOK': ('real', 'leaky counters start at 0 (:282-286, :319-332): a late sensor gives inputsOK True for '
                                       + '1-2 messages, then False', 'known_inputsok_startup.json'),
  'no_chatter deviceMotion.posenetOK': ('real', 'posenetOK = not std_spike or car_speed <= 5.0 has no speed hysteresis (locationd.py:239-242): '
                                        + 'soft disable at 5 m/s', 'known_posenet_speed_hover.json'),
  'alive deviceMotion (theoretical inputs)': ('theoretical', 'timestampEof < CAM_ODO_POSE_DELAY: negative UInt64 timestamp raises (:244); '
                                              + 'restored state of wrong length asserts (:41)', 'known_negative_filter_time.json'),
  'flags from inputs deviceMotion (theoretical inputs)': ('theoretical', 'NaN cameraOdometry passes the sanity checks (:169-178), resets '
                                                          + 'the filter and counts as SUCCESS (:325-326)', 'known_nan_camodo.json'),
  'monotonic deviceMotion.timestamp (theoretical inputs)': ('theoretical', 'the non-finite reset restarts the filter at the '
                                                            + 'cameraOdometry time (:98, :162), up to 0.8 s back', 'known_nan_camodo.json'),
  'no_chatter deviceMotion.inputsOK (theoretical inputs)': ('theoretical', 'cameraOdometry timestamps off the frame clock fail the '
                                                            + 'rewind check: the leaky counter hovers (:319-332)', 'known_inputsok_chatter_ts0.json'),
}

SPEC = Spec(
  proc='locationd',
  scope={'selfdrive/locationd/locationd.py': None},
  ledger=LEDGER,
  unreached=UNREACHED,
  known=KNOWN,
  tag=tag,
  log_clock=0.,  # sensor liveness reads time.monotonic (locationd.py:250)
  template=template,
  duration=10.,
  examples=24,
  bounds={
    'accelerometer.timestamp': [0., L.MAX_SENSOR_TIME_DIFF * 1e9, -L.MAX_SENSOR_TIME_DIFF * 1e9, -L.MAX_FILTER_REWIND_TIME * 1e9],
    'gyroscope.timestamp': [0., L.MAX_SENSOR_TIME_DIFF * 1e9, -L.MAX_SENSOR_TIME_DIFF * 1e9, -L.MAX_FILTER_REWIND_TIME * 1e9],
    'cameraOdometry.timestampEof': [0., L.CAM_ODO_POSE_DELAY * 1e9, (L.CAM_ODO_POSE_DELAY + L.MAX_FILTER_REWIND_TIME) * 1e9,
                                    (L.CAM_ODO_POSE_DELAY - L.MAX_FILTER_REWIND_TIME) * 1e9],
    'carState.vEgo': [5.0],
    'cameraOdometry.rotStd[]': [L.MIN_STD_SANITY_CHECK, 10 * L.ROTATION_SANITY_CHECK / math.sqrt(3)],
    'cameraOdometry.transStd[]': [L.MIN_STD_SANITY_CHECK, 10 * L.TRANS_SANITY_CHECK / math.sqrt(3), 7.0],
    'cameraOdometry.rot[]': [L.ROTATION_SANITY_CHECK / math.sqrt(3)],
    'cameraOdometry.trans[]': [L.TRANS_SANITY_CHECK / math.sqrt(3)],
    'extrinsicsCalibration.rpyCalib[]': [L.CALIB_RPY_SANITY_CHECK],
    'accelerometer.acceleration.v[]': [L.ACCEL_SANITY_CHECK / math.sqrt(3), 9.81],
    'gyroscope.gyroUncalibrated.v[]': [L.ROTATION_SANITY_CHECK / math.sqrt(3)],
  },
  params={'LocationFilterInitialState': initial_states()},
  weights={p: [[1.]] * 3 for p in ('cameraOdometry.rot[]', 'cameraOdometry.trans[]', 'gyroscope.gyroUncalibrated.v[]',
                                    'accelerometer.acceleration.v[]', 'extrinsicsCalibration.rpyCalib[]')},
  checks=[
    P.alive('deviceMotion', 'cameraOdometry'),
    P.finite('deviceMotion', allow=('debugFilterState',)),
    valid_consistent,
    flags_from_inputs,
    monotonic_timestamp,
    posenet_low_speed,
    P.no_chatter('deviceMotion.inputsOK', 1.),
    P.no_chatter('deviceMotion.posenetOK', 1.),
    # not sensorsOK: it is the last sensor message's presence/valid, a pass-through that must follow its input (locationd.py:249-261)
  ],
)
