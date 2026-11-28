import math
import operator

import numpy as np

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.locationd import torqued as T
from openpilot.selfdrive.locationd.lagd import MIN_LAG, MAX_LAG
from openpilot.selfdrive.test.process_fuzz import properties as P, space as S
from openpilot.selfdrive.test.process_fuzz.harness import Spec
from openpilot.selfdrive.test.process_fuzz.procs import lagd

FLAGS = ['deviceMotion.angularVelocityDevice.valid', 'deviceMotion.orientationNED.valid', 'deviceMotion.inputsOK',
         'deviceMotion.sensorsOK', 'deviceMotion.posenetOK']
RAW = ('latAccelFactorRaw', 'latAccelOffsetRaw', 'frictionCoefficientRaw')
LTP = 'lateralTorqueParameters'

LEDGER = {
  # lagd publishes clip(mean, MIN_LAG, MAX_LAG) when estimated, else CP.steerActuatorDelay + 0.2, always finite (lagd.py:229-240).
  # Outside that image torqued crashes in np.arange (torqued.py:195): NaN/inf "cannot compute length", 1e30 "Maximum allowed size"
  'lateralDelay.lateralDelay:outofimage': ('enforced', 'lagd.py:229-240', 'lagd: invariant lateralDelay in lagd image'),
  # calibrationd always publishes 3 angles; fewer crash rot_from_euler (helpers.py:183)
  'extrinsicsCalibration.rpyCalib[]:len': ('enforced', 'calibrationd.py:250 smooth_rpy.tolist()', 'calibrationd: invariant rpyCalib usable'),
  # restored LiveTorqueParameters is torqued's own earlier output (torqued.py:271-273): its points passed the gate (|y| <= 1,
  # x in a bucket, :202-204), decay in [50, 250], filtered values inside the sanity clips
  'P LiveTorqueParameters:outofimage': ('enforced', 'torqued.py:271-273 persists its own output', 'torqued: invariant consistent'),
  'P LiveTorqueParameters:nanpoints': ('theoretical', 'a NaN point can only come from a corrupt cache, kept to reach :156-158', None),
  'CP.carFingerprint': ('symmetry', 'only compared for equality with CarParamsPrevRoute (torqued.py:134), drawn as same/other', None),
  'CP.brand': ('symmetry', 'only `in ALLOWED_CARS` (torqued.py:76): one class per membership', None),
}


def template(svc, g=None):
  m = lagd.template(svc)
  if svc == 'carState':
    m.carState.vEgo = 20.
  elif svc == 'lateralDelay':
    m.lateralDelay.lateralDelay = 0.3
  return m


def cp_class(CP):
  w = CP.lateralTuning.which()
  return CP.brand in T.ALLOWED_CARS, w, (CP.lateralTuning.torque.friction, CP.lateralTuning.torque.latAccelFactor) if w == 'torque' else None


def offline(fp):
  CP = S.get_car_params(fp, False)
  t = CP.lateralTuning.torque if CP.lateralTuning.which() == 'torque' else None
  return (t.latAccelFactor, t.friction) if t else (0., 0.)


# *** restored LiveTorqueParameters: a small JSON descriptor per genome, expanded to points here ***

def restores():
  mins = T.MIN_BUCKET_POINTS
  scaled = (mins * (int(math.ceil(T.MIN_POINTS_TOTAL / mins.sum() * 100)) / 100)).astype(int)
  short = scaled.copy()
  short[3] = mins[3] - 1
  counts = [[0] * 8, [1] * 8, [1] * 7 + [0], mins.tolist(), scaled.tolist(), short.tolist(), [T.POINTS_PER_BUCKET] * 8,
            [T.POINTS_PER_BUCKET + 1] * 8, [T.POINTS_PER_BUCKET] * 4 + mins[4:].tolist()]  # last: skewed in x
  rng = np.random.default_rng(0)
  out = [None, {'raw': 'deadbeef'}]
  for k in range(40):
    out.append({'ltp': {
      'counts': counts[k % len(counts)], 'slope': float(rng.choice([2., 0.5, 0., -2., 10.])), 'offset': float(rng.choice([0., 0.3, -0.5, 1.])),
      'noise': float(rng.choice([0., 0.05, 0.3, 2.])), 'nan': int(k % 13 == 5), 'seed': k,
      'valid': bool(k % 3), 'version': T.VERSION + (k % 11 == 7),
      'decay': float(rng.choice([T.MIN_FILTER_DECAY, T.MIN_FILTER_DECAY + DT_MDL, 100., T.MAX_FILTER_DECAY - DT_MDL, T.MAX_FILTER_DECAY])),
      'factor': float(rng.choice([1., 1 - T.FACTOR_SANITY, 1 + T.FACTOR_SANITY])),
      'friction': float(rng.choice([1., 1 - T.FRICTION_SANITY, 1 + T.FRICTION_SANITY])),
      'latAccelOffsetFiltered': float(rng.choice([0., -T.LAT_ACC_THRESHOLD, T.LAT_ACC_THRESHOLD, 0.2])),
    }})
  return out


def expand(d, fp):
  # points as torqued's gate leaves them (x in a bucket, |x| > STEER_MIN_THRESHOLD, |y| <= LAT_ACC_THRESHOLD), y = slope*x + offset + noise
  rng = np.random.default_rng(d['seed'])
  pts = []
  for (lo, hi), n in zip(T.STEER_BUCKET_BOUNDS, d['counts'], strict=True):
    x = rng.uniform(max(lo, T.STEER_MIN_THRESHOLD) if lo >= 0 else lo, min(hi, -T.STEER_MIN_THRESHOLD) if hi <= 0 else hi, n)
    y = np.clip(d['slope'] * x + d['offset'] + d['noise'] * rng.standard_normal(n), -T.LAT_ACC_THRESHOLD, T.LAT_ACC_THRESHOLD)
    pts += np.stack([x, y], axis=1).tolist()
  if d['nan']:  # every 10th: the fit subsamples FIT_POINTS_TOTAL with an unseeded RNG (helpers.py:73,103), so one NaN is hit or miss
    for p in pts[::10]:
      p[1] = math.nan
  f, fr = offline(fp)
  ltp = {'version': d['version'], 'valid': d['valid'], 'decay': d['decay'], 'points': pts, 'latAccelFactorFiltered': d['factor'] * f,
         'frictionCoefficientFiltered': d['friction'] * fr, 'latAccelOffsetFiltered': d['latAccelOffsetFiltered']}
  return {'event': {LTP: ltp}}


def excite(rng, g):
  # torque sweeping every bucket, yaw following it (lat_acc = vEgo * yaw, steer = -torque) with a lag and an offset
  torque = S.Gene('hover', rng.choice([0., 0.05, -0.05]), rng.choice([0.45, 0.55, 0.3]), t1=rng.choice([1., 3., 6.]))
  k = rng.choice([0.1, 0.05, -0.1, 0.]) / 20.
  yaw = S.Gene('hold', rng.choice([0., 0.005, -0.03]), 0., t0=rng.choice([0., 0.1, 0.3]), rel='carOutput.actuatorsOutput.torque', k=-k)
  return {'carOutput.actuatorsOutput.torque': torque, 'deviceMotion.angularVelocityDevice.z': yaw}


# *** properties ***

@P.prop('invariant consistent')
def consistent(r):
  use = r.cp.brand in T.ALLOWED_CARS and r.cp.lateralTuning.which() == 'torque'
  for m in P.outs(r, LTP):
    p = m.lateralTorqueParameters
    assert T.MIN_FILTER_DECAY <= p.decay <= T.MAX_FILTER_DECAY, f't={P.t_of(m):.2f}s decay {p.decay}'
    assert 0 <= p.calPerc <= 100 and 0 <= p.totalBucketPoints <= T.POINTS_PER_BUCKET * 8, f't={P.t_of(m):.2f}s {p.calPerc} {p.totalBucketPoints}'
    assert p.valid == (p.calPerc == 100), f't={P.t_of(m):.2f}s valid {p.valid} at calPerc {p.calPerc}'  # a NaN fit resets to 0
    assert p.version == T.VERSION and p.useParams == use, f't={P.t_of(m):.2f}s version {p.version} useParams {p.useParams}'


@P.prop('bounded filtered within sanity')
def within_sanity(r):
  # consumers use the filtered values whenever valid (controlsd.py:88-90); the fit is clipped to [1 -+ sanity] * offline before
  # filtering (torqued.py:230-231), so the filter output stays inside too
  f, fr = offline(r.genome.fp)
  for m in P.outs(r, LTP):
    p = m.lateralTorqueParameters
    for name, v, off, s in (('latAccelFactor', p.latAccelFactorFiltered, f, T.FACTOR_SANITY),
                            ('friction', p.frictionCoefficientFiltered, fr, T.FRICTION_SANITY)):
      lo, hi = (1 - s) * off, (1 + s) * off
      assert lo - 1e-4 <= v <= hi + 1e-4, f't={P.t_of(m):.2f}s {name} {v} not in [{lo}, {hi}]'


@P.prop('invariant one filter step per message')
def decay_step(r):
  # one estimate per published message: decay (the filter time constant) steps by at most DT_MDL (torqued.py:162)
  for a, b in zip(P.outs(r, LTP), P.outs(r, LTP)[1:], strict=False):
    p, q = a.lateralTorqueParameters, b.lateralTorqueParameters
    assert q.maxResets != p.maxResets or q.decay - p.decay <= DT_MDL + 1e-4, f't={P.t_of(b):.2f}s decay {p.decay} -> {q.decay}'


TQ = 'selfdrive/locationd/torqued.py:'
INIT = TQ + 'TorqueEstimator.__init__:'
UNREACHED = {
  'dead: main() never passes decimated or track_all_points (torqued.py:256)':
    [INIT + '"if decimated:":T', TQ + 'TorqueEstimator.handle_log:"if self.track_all_points:":T'],
  'dead: which is one of the 6 SubMaster services': [TQ + 'TorqueEstimator.handle_log:"elif which == "deviceMotion":":F'],
  'dead: the with body is one attribute read or an assignment, it cannot raise':
    [INIT + '"with log.Event.from_bytes(torque_cache) as log_evt:":F', INIT + '"with car.CarParams.from_bytes(params_cache) as msg:":F'],
  'dead: points are finite or NaN (gate :202-204, restore), NaN makes svd raise LinAlgError':
    [TQ + 'TorqueEstimator.estimate_params:"except np.linalg.LinAlgError as e:":F'],
  'dead: estimate_params returns floats, never None': [TQ + 'TorqueEstimator.get_msg.<locals>.<genexpr>:"if any(val is None or np.isnan(val) '
                                                       + 'for val in [latAccelFactor, latAccelOffset, frictionCoeff]):"#1:T'],
}

KNOWN = {
  'bounded lateralTorqueParameters.latAccelOffsetFiltered': ('real', 'the TLS offset is not clipped (torqued.py:229-232, unlike factor/friction): '
                                                             + 'a y-spread cloud fits a near-vertical line',
                                                             'known_offset_unclipped.json'),
  'invariant one filter step per message': ('real', 'main() calls get_msg() twice on sm.frame % 240 frames (torqued.py:267-273): the filters '
                                            + 'step twice every 12 s', 'known_double_update.json'),
}

LAG_IMAGE = [MIN_LAG, MAX_LAG, float(np.nextafter(np.float32(MIN_LAG), 1)), float(np.nextafter(np.float32(MAX_LAG), 0)),
             *sorted({round(S.get_car_params(fp, False).steerActuatorDelay + 0.2, 6) for fp in S.interfaces if fp != 'MOCK'})]

SPEC = Spec(
  proc='torqued',
  scope={'selfdrive/locationd/torqued.py': None, 'selfdrive/locationd/helpers.py': {'PointBuckets', 'NPQueue', 'PoseCalibrator'}},
  ledger=LEDGER,
  unreached=UNREACHED,
  known=KNOWN,
  template=template,
  duration=(15., 15., 30.),  # > 12 s: the cache write at sm.frame % 240 (torqued.py:271)
  examples=96,
  encode=lambda v, g: expand(v['ltp'], g.fp) if isinstance(v, dict) and 'ltp' in v else v,
  cp_class=cp_class,
  image={'lateralDelay.lateralDelay': (MIN_LAG, lagd.MAX_INITIAL_LAG, LAG_IMAGE)},
  few=.5,
  excite=excite,
  params={'LiveTorqueParameters': restores(), 'CarParamsPrevRoute': [None, {'car': 'same'}, {'car': 'same'}, {'car': 'TOYOTA_PRIUS'}, {'raw': 'deadbeef'}]},
  bounds={
    'carState.vEgo': [T.MIN_VEL],  # aliased as vego
    'carOutput.actuatorsOutput.torque': sorted({s * v for s in (1, -1) for b in T.STEER_BUCKET_BOUNDS for v in (*b, T.STEER_MIN_THRESHOLD)}),
    'deviceMotion.angularVelocityDevice.z': [s * T.LAT_ACC_THRESHOLD / v for s in (1, -1) for v in (T.MIN_VEL, 20.)],  # lat_acc = vego * yaw
    'deviceMotion.orientationNED.x': [s * math.asin(T.LAT_ACC_THRESHOLD / 9.81) for s in (1, -1)],  # sin(roll) * g
    'deviceMotion.timestamp': [0, *(s * x for s in (1, -1) for x in (1e9 * T.MIN_ENGAGE_BUFFER, 1e9 * T.HISTORY, 1e9 * T.HISTORY + 5e7))],
  },
  checks=[
    P.alive(LTP, 'deviceMotion', every=5),
    P.finite(LTP, allow=RAW),
    P.invariant(LTP, lambda p: not p.valid or all(math.isfinite(getattr(p, k)) for k in RAW), 'raw finite when valid'),
    within_sanity,
    P.bounded(f'{LTP}.latAccelOffsetFiltered', -T.LAT_ACC_THRESHOLD, T.LAT_ACC_THRESHOLD),
    consistent,
    # resets only count up; between resets the learner only accumulates: points and decay never go down, valid stays valid
    P.monotone(f'{LTP}.maxResets'),
    *(P.monotone(f'{LTP}.{f}', reset=f'{LTP}.maxResets') for f in ('totalBucketPoints', 'decay', 'valid')),
    decay_step,
    # the gates (torqued.py:185-204); a reset may restart the count
    P.gated(LTP, {'carControl.latActive': bool, 'carState.steeringPressed': operator.not_, 'carState.vEgo': lambda v: v > T.MIN_VEL,
                  'carOutput.actuatorsOutput.torque': lambda v: abs(v) > T.STEER_MIN_THRESHOLD, **dict.fromkeys(FLAGS, bool)},
            learned=lambda a, b: b.totalBucketPoints > a.totalBucketPoints and b.maxResets <= 1),
    P.valid_from_inputs(LTP, 'deviceMotion', by_time=True),
  ],
)
