import copy
import itertools
import math
import operator

import numpy as np

from openpilot.cereal import log
from openpilot.common.fuzzy import _INTEGER_RANGES as INT_RANGE
from openpilot.selfdrive.locationd import lagd as L
from openpilot.selfdrive.test.process_fuzz import procs as U, properties as P, space as S
from openpilot.selfdrive.test.process_fuzz.harness import Spec

MAX_INITIAL_LAG = max(S.get_car_params(fp, False).steerActuatorDelay for fp in S.interfaces if fp != 'MOCK') + 0.2  # lagd.py:185

Status = log.LateralDelay.Status
POSE = ('deviceMotion.angularVelocityDevice.valid', 'deviceMotion.posenetOK', 'deviceMotion.inputsOK')
FLIP = ('controlsState.desiredCurvature', 'deviceMotion.angularVelocityDevice.x', 'deviceMotion.angularVelocityDevice.y',
        'deviceMotion.angularVelocityDevice.z')
EST_HZ = 4.  # update_estimate runs every 5th deviceMotion (lagd.py:413)

LEDGER = {
  # restored LiveDelay is lagd's own output (lagd.py:419-420); anything else is a corrupt file or another version, tagged
  'P LiveDelay:outside image': ('theoretical', 'only lagd writes it (lagd.py:420), restore_in_image() is that image', None),
  'CP.carFingerprint': ('symmetry', 'only compared for equality with CarParamsPrevRoute (lagd.py:372), drawn as same/other', None),
  # every union member has `saturated` (asserted below): fixed to torqueState, whose saturated is fuzzed
  'controlsState.lateralControlState.which()': ('symmetry', 'renaming the active member, lagd.py:265', None),
  # logMonoTime starts at space.T0 = 1 s, so the initial last_*_t = 0 always blocks the first 1 s (lagd.py:206-209,302-305)
  '@logMonoTime[deviceMotion]:origin >= 2 s': ('harness', 'space.T0 fixed at 1 s', None),
}


def template(svc, g=None):
  m = U.shape(svc)
  if svc == 'deviceMotion':
    dm = m.deviceMotion
    dm.angularVelocityDevice.valid = dm.orientationNED.valid = dm.inputsOK = dm.sensorsOK = dm.posenetOK = True
  elif svc == 'extrinsicsCalibration':
    m.extrinsicsCalibration.calStatus = 'calibrated'
  elif svc == 'carControl':
    m.carControl.latActive = True
  elif svc == 'carState':
    m.carState.vEgo = 25.
  elif svc == 'controlsState':
    m.controlsState.lateralControlState.init('torqueState')
  return m


LCS = log.ControlsState.schema.fields['lateralControlState'].schema
assert all('saturated' in LCS.fields[u].schema.fields for u in LCS.union_fields)


def excite(rng, g):
  # desired lateral accel curv * v^2 swept by a sine, actual v * yaw following it after a true lag (0.15-0.60 is the ROI,
  # 0.625 and 0.7 are past it), with a gain and an offset (test_lagd's cos(10 t) * 0.3)
  curv = S.Gene('hover', 0., rng.choice([0.27, 0.3, 0.3, 0.4]) / 25. ** 2, t1=rng.choice([2 * math.pi / 10, 2 * math.pi / 10, 0.5, 0.8]))
  yaw = S.Gene('hold', rng.choice([0., 0., 0.01]), 0., t0=rng.choice([L.MIN_LAG, 0.2, 0.3, 0.45, 0.6, 0.625, 0.7]),
               rel='controlsState.desiredCurvature', k=25. * rng.choice([1., 1., 0.8, -1.]))
  return {'controlsState.desiredCurvature': curv, 'deviceMotion.angularVelocityDevice.z': yaw}


def restores():
  # restored LiveDelay over its classes (lagd.py:364-384); validBlocks -1 publishes a negative calPerc, <= -7 overflows its Int8
  out = [None, {'raw': 'deadbeef'}]
  lags = [L.MIN_LAG, 0.2, 0.3, 0.6, L.MAX_LAG, math.nan, 0., 1., math.inf]
  blocks = [0, 1, L.BLOCK_NUM_NEEDED - 1, L.BLOCK_NUM_NEEDED, L.BLOCK_NUM - 1, L.BLOCK_NUM, L.BLOCK_NUM + 1, -1, -7, INT_RANGE['int32'][0]]
  for k, (lag, vb) in enumerate(itertools.product(lags, blocks)):
    ld = {'lateralDelayEstimate': lag, 'validBlocks': vb, 'status': ['unestimated', 'estimated', 'estimated', 'invalid'][k % 4],
          'version': L.VERSION + (k % 11 == 4)}
    out.append({'event': {'lateralDelay': ld}})
  return out


def tag(sp, g, ins=None):
  # estimates are block means of lags in [MIN_LAG, MAX_LAG] or initial_lag (<= MAX_INITIAL_LAG)
  v = g.params.get('LiveDelay')
  ld = v.get('event', {}).get('lateralDelay') if v is not None else None
  ok = v is None or (ld is not None and L.MIN_LAG <= ld['lateralDelayEstimate'] <= MAX_INITIAL_LAG and 0 <= ld['validBlocks'] <= L.BLOCK_NUM)
  return None if ok else 'theoretical inputs'


@P.prop('invariant lateralDelay in lagd image')
def in_image(r):
  # what consumers use unconditionally: clip(mean, MIN_LAG, MAX_LAG) when estimated, else CP.steerActuatorDelay + 0.2 (lagd.py:237-240)
  init = float(np.float32(r.cp.steerActuatorDelay + 0.2))
  for m in P.outs(r, 'lateralDelay'):
    d = m.lateralDelay
    ok = L.MIN_LAG <= d.lateralDelay <= L.MAX_LAG if d.status == Status.estimated else d.lateralDelay == init
    assert ok, f't={P.t_of(m):.2f}s lateralDelay {d.lateralDelay} with status {d.status} (initial {init})'


@P.prop('invariant consistent')
def consistent(r):
  for m in P.outs(r, 'lateralDelay'):
    d = m.lateralDelay
    assert 0 <= d.validBlocks <= L.BLOCK_NUM and 0 <= d.calPerc <= 100, f't={P.t_of(m):.2f}s validBlocks {d.validBlocks} calPerc {d.calPerc}'
    assert (d.calPerc == 100) == (d.validBlocks >= L.BLOCK_NUM_NEEDED), f't={P.t_of(m):.2f}s calPerc {d.calPerc} at {d.validBlocks} blocks'
    assert d.status == Status.unestimated or d.validBlocks >= L.BLOCK_NUM_NEEDED, f't={P.t_of(m):.2f}s {d.status} at {d.validBlocks} blocks'
    assert d.lateralDelayEstimateStd >= 0 and d.version == L.VERSION, f't={P.t_of(m):.2f}s std {d.lateralDelayEstimateStd} version {d.version}'


@P.prop('invariant lateralDelay changes only at block completion')
def piecewise(r):
  # lateralDelay and status come from completed blocks only (lagd.py:155-162): validBlocks counts up, or the full ring moves on
  prev = None
  for m in P.outs(r, 'lateralDelay'):
    d = m.lateralDelay
    if prev is not None and (d.lateralDelay, d.status) != (prev.lateralDelay, prev.status):
      assert d.validBlocks != prev.validBlocks or d.validBlocks == L.BLOCK_NUM, \
        f't={P.t_of(m):.2f}s {prev.status} {prev.lateralDelay} -> {d.status} {d.lateralDelay} at {d.validBlocks} blocks'
    prev = d


def flip(g):
  # joint sign flip of desired curvature and the device angular velocity: every okay gate uses abs, the smoothing and NCC
  # are (bi)linear and IEEE negation commutes with them exactly, so the output is bit-identical
  g = copy.deepcopy(g)
  for p in FLIP:
    if ge := g.genes.get(p):  # unset fields stay at the template's 0
      ge.a, ge.b = -ge.a, -ge.b
      if ge.rel is not None and ge.rel not in FLIP:
        ge.k = -ge.k
  return g


@P.prop('metamorphic sign flip')
def sign_flip(r):
  outs = P.outs(r, 'lateralDelay')
  if r.err or len({(m.lateralDelay.calPerc, m.lateralDelay.lateralDelayEstimate) for m in outs}) < 2:
    return None  # nothing learned: the flip is vacuous, skip the rerun
  b = [m.lateralDelay.to_dict() for m in P.outs(r.rerun(flip(r.genome)), 'lateralDelay')]
  for k, (x, y) in enumerate(zip([m.lateralDelay.to_dict() for m in outs], b, strict=False)):
    assert str(x) == str(y), f't={P.t_of(outs[k]):.2f}s {x} != flipped {y}'
  return 'learned'


LG = 'selfdrive/locationd/lagd.py:'
EST, PTS = LG + 'LateralLagEstimator.update_estimate', LG + 'LateralLagEstimator.update_points'
SMOOTH = LG + 'masked_symmetric_moving_average:"assert k >= 1 and k % 2 == 1, "k must be positive and odd""'
UNREACHED = {
  'dead: both come from one Points window': [LG + 'LateralLagEstimator.actuator_delay:"assert len(expected_sig) == len(actual_sig)":F'],
  U.ENV_DEBUG: [LG + 'LateralLagEstimator.get_msg:"if debug:":T'],
  'dead: which is one of the 5 SubMaster services': [LG + 'LateralLagEstimator.handle_log:"elif which == "deviceMotion":":F'],
  'dead: only entered when times[0] <= last_estimate_t, so some t matches':
    [EST + '.<locals>.<genexpr>:"new_values_start_idx = next(-i for i, t in enumerate(reversed(times)) if t <= self.last_estimate_t)":exit'],
  'dead: Points is prefilled to window_len (lagd.py:114-117)': [EST + ':"if not self.points_enough():":F'],
  'dead: sensors_valid False sets last_pose_invalid_t = t (:299-300), so has_recovered is already False':
    [PTS + ':"okay = self.lat_active and not self.steering_pressed and not self.steering_saturated and \\"#7:F'],
  'dead: k = SMOOTH_K = 5': [SMOOTH + ':F', SMOOTH + '#1:F'],
}

KNOWN = {
  'invariant consistent (theoretical inputs)': ('theoretical', 'retrieve_initial_lag asserts validBlocks <= BLOCK_NUM but not >= 0 '
                                                + '(lagd.py:376): a restored -1 publishes calPerc -20', 'known_negative_blocks.json'),
  'alive lateralDelay (theoretical inputs)': ('theoretical', 'a restored validBlocks <= -7 overflows the Int8 calPerc in get_msg every start '
                                              + '(lagd.py:250)', 'known_negative_blocks_crash.json'),
}

SPEC = Spec(
  proc='lagd',
  scope={'selfdrive/locationd/lagd.py': None, 'selfdrive/locationd/helpers.py': {'parabolic_peak_interp', 'PoseCalibrator'}},
  ledger=LEDGER,
  unreached=UNREACHED,
  known=KNOWN,
  template=template,
  tag=tag,
  duration=(15., 15., 15., 40., 60.),  # one estimate needs >= 27 s, a block ~52 s
  examples=48,
  few=.5,
  excite=excite,
  params={'LiveDelay': restores(), 'CarParamsPrevRoute': [None, {'car': 'same'}, {'car': 'same'}, {'car': 'TOYOTA_PRIUS'}, {'raw': 'deadbeef'}]},
  bounds={
    'carState.vEgo': [L.MIN_VEGO],  # aliased as v_ego
    'deviceMotion.angularVelocityDevice.z': [L.MAX_YAW_RATE_SANITY_CHECK, -L.MAX_YAW_RATE_SANITY_CHECK,
                                             *(s * L.MAX_LAT_ACCEL / v for s in (1, -1) for v in (L.MIN_VEGO, 25.))],  # |v * yaw|
    'deviceMotion.angularVelocityDevice.zStd': [L.MAX_YAW_RATE_SANITY_CHECK],
    'controlsState.desiredCurvature': [s * a / 25. ** 2 for s in (1, -1) for a in (L.MAX_LAT_ACCEL_DIFF, L.MAX_LAT_ACCEL + L.MAX_LAT_ACCEL_DIFF)],
  },
  checks=[
    P.alive('lateralDelay', 'deviceMotion', every=5),
    P.finite('lateralDelay', degraded=lambda m: False),  # lateralDelayEstimate is persisted and restored as the next lag
    in_image,
    consistent,
    P.monotone('lateralDelay.validBlocks', name='monotone learner'),  # blocks only accumulate (lagd.py:150-152)
    P.monotone('lateralDelay.calPerc'),
    piecewise,
    P.transitions('lateralDelay.status', {('unestimated', 'estimated'), ('unestimated', 'invalid'), ('estimated', 'invalid'), ('invalid', 'estimated')}),
    P.no_chatter('lateralDelay.status', L.BLOCK_SIZE / EST_HZ),  # one block of estimates between status changes
    # the okay gates (lagd.py:287-307)
    P.gated('lateralDelay', {'carControl.latActive': bool, 'carState.steeringPressed': operator.not_, 'carState.vEgo': lambda v: v > L.MIN_VEGO,
                             'extrinsicsCalibration.calStatus': lambda s: s == log.ExtrinsicsCalibration.Status.calibrated,
                             'controlsState.lateralControlState.*.saturated': operator.not_, **dict.fromkeys(POSE, bool)},
            learned=lambda a, b: b.calPerc != a.calPerc),
    P.valid_from_inputs('lateralDelay', 'deviceMotion', by_time=True),
    sign_flip,
  ],
)
