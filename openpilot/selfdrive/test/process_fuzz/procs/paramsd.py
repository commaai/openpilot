import math

import numpy as np

from openpilot.selfdrive.locationd import paramsd as PD
from openpilot.selfdrive.test.process_fuzz import procs as U, properties as P, space as S
from openpilot.selfdrive.test.process_fuzz.harness import Spec

CAN = 'card decodes it from fixed-width CAN signals (|vEgo| < 1e3 m/s, |angle| < 1e4 deg), KF at interfaces.py:302-311'
LEDGER = U.upstream('extrinsicsCalibration.rpyCalib[]', 'carState.vEgo') | {
  'carState.vEgo:beyond_can': ('theoretical', CAN, 'card: finite carState'),
  'carState.steeringAngleDeg:nan|inf|beyond_can': ('theoretical', CAN, 'card: finite carState'),
  **{f'deviceMotion.{f}:nan|inf': ('theoretical', 'locationd resets on non-finite x (locationd.py:94-98) but not after a rewind '
                                   + 'fast-forward (ekf_sym.cc:110-114)', 'locationd: finite deviceMotion')
     for f in ('angularVelocityDevice.z', 'orientationNED.x')},
  'P LiveParametersV2:outside image': ('theoretical', 'only paramsd writes it (paramsd.py:272), x finite after get_msg\'s reset (:137-140)', None),
  'deviceMotion.posenetOK:low_speed': ('theoretical', 'locationd forces posenetOK True at |vEgo| <= 5 (locationd.py:149,242)',
                                       'locationd: invariant posenetOK at low speed'),
  'CP.carFingerprint': ('symmetry', 'only compared for equality with CarParamsPrevRoute (paramsd.py:214), drawn as same/other', None),
  'deviceMotion.timestamp:far': ('theoretical', 'locationd filter time; far from logMonoTime only after an unvalidated cameraOdometry '
                                 + 'timestampEof jump (locationd.py:162)', 'locationd: monotonic deviceMotion.timestamp'),
}


def template(svc, g=None):
  m = U.shape(svc)
  if svc == 'deviceMotion':  # locationd's nominal output: valid measurements, positive stds (locationd.py:220-244)
    m.valid = True
    dm = m.deviceMotion
    for k in ('orientationNED', 'velocityDevice', 'accelerationDevice', 'angularVelocityDevice'):
      xyz = getattr(dm, k)
      xyz.xStd, xyz.yStd, xyz.zStd, xyz.valid = 0.01, 0.01, 0.01, True
    dm.inputsOK, dm.posenetOK, dm.sensorsOK = True, True, True
  return m


def restores():
  # LiveParametersV2 over its restore classes (paramsd.py:202-241): steerRatio around [0.5, 2] x CP.steerRatio (as a multiple),
  # angle offsets around the 8/10 deg hysteresis, stiffness around [0.2, 5]; non-finite = outside image
  sr = [1., 0.5, 2., 0.4999, 2.0001, 0.1, math.nan]
  sf = [1., 0.2, 5., 0.19, 5.1, 0., -1., math.nan, math.inf]
  off = [0., 7.9, 8., 9.99, 10., 10.01, -9.99, -10., 45., math.nan, math.inf]
  rng = np.random.default_rng(0)
  out = [None, None, {'raw': 'deadbeef'}]
  for _ in range(40):
    out.append({'event': {'vehicleParameters': {'steerRatio': float(rng.choice(sr)), 'stiffnessFactor': float(rng.choice(sf)),
                                                'angleOffsetAverageDeg': float(rng.choice(off))}}, 'cp_mult': {'steerRatio': 'steerRatio'}})
  return out


def encode(v, g):
  # cp_mult: {field: CP field} scales a restored field by this fingerprint's CP value
  if isinstance(v, dict) and 'cp_mult' in v:
    vp, CP = dict(v['event']['vehicleParameters']), S.get_car_params(g.fp, False, g.fpv)
    return {'event': {'vehicleParameters': vp | {k: float(np.float32(vp[k])) * getattr(CP, f) for k, f in v['cp_mult'].items()}}}
  return v


def tag(sp, g, ins=None):
  ts, v, ok = (g.genes.get(p) for p in ('deviceMotion.timestamp', 'carState.vEgo', 'deviceMotion.posenetOK'))
  # timestamp:far: absolute, or an offset beyond the 1 s rewind window
  far = ts is not None and (ts.idx != 'lmt' or not all(abs(x) <= 1e9 for x in (ts.a, ts.b, ts.a + ts.b, ts.a - ts.b)))
  can = any((ge := g.genes.get(p)) is not None and any(abs(x) > lim for x in (ge.a, ge.b, ge.a + ge.b))
            for p, lim in (('carState.vEgo', 1e3), ('carState.steeringAngleDeg', 1e4)))
  # posenetOK:low_speed: False while |vEgo| <= 5
  low = ok is not None and any(not ok.at(t) and not abs(v.at(t) if v is not None else 0.) > 5. for t in np.arange(0, g.dur, 0.05))
  vp = (g.params.get('LiveParametersV2') or {'event': {'vehicleParameters': {}}}).get('event', {}).get('vehicleParameters')
  restore = vp is None or not all(map(math.isfinite, vp.values()))
  return P.theoretical(sp, g) or ('theoretical inputs' if far or can or low or restore else None)


def rate_limited(path, max_delta):
  # consumers use these unconditionally (controlsd.py:77-83, longitudinal_planner.py:100), so the rate limit holds on every message
  @P.prop(f'rate_limited {path}')
  def check(r):
    t, x = P.series(r.outs, path)
    for k in range(1, len(x)):
      d = abs(x[k] - x[k - 1])
      ulp = np.spacing(np.float32(max(abs(x[k]), abs(x[k - 1]))))  # published as float32
      assert not d > max_delta + 2 * ulp, f'|delta| {d} > {max_delta}: {P.around(t, x, k)}'
  return check


def consistent(vp):
  # the flags are exactly the published values against their ranges (paramsd.py:170-181)
  return (vp.posenetValid and (not vp.valid or (vp.angleOffsetAverageValid and vp.angleOffsetValid and vp.steerRatioValid and vp.stiffnessFactorValid
          and abs(vp.roll) < PD.ROLL_MAX)) and vp.stiffnessFactorValid == (0.2 <= vp.stiffnessFactor <= 5.0))


@P.prop('invariant steerRatioValid')
def sr_valid(r):
  for m in r.outs:
    vp = m.vehicleParameters
    assert vp.steerRatioValid == (0.5 * r.cp.steerRatio <= vp.steerRatio <= 2. * r.cp.steerRatio), f't={P.t_of(m):.2f}s steerRatioValid {vp}'


PD_ = 'selfdrive/locationd/paramsd.py:'
UNREACHED = {
  U.H_LOCKSTEP: [PD_ + 'main:"if sm.updated[\'deviceMotion\']:":F'],
  'dead: only the three subscribed services reach handle_log': [PD_ + 'VehicleParamsLearner.handle_log:"elif which == \'carState\':":F'],
  U.ENV_DEBUG: [PD_ + 'VehicleParamsLearner.get_msg:"if debug:":T',
                PD_ + 'VehicleParamsLearner.__init__:"self.P_initial = P_initial if P_initial is not None else CarKalman.P_initial":notNone',
                PD_ + 'retrieve_initial_vehicle_params:"if debug and len(initial_filter_std) != 0:":T',
                PD_ + 'retrieve_initial_vehicle_params:"if debug and len(initial_filter_std) != 0:"#1:*'],
}

SNAP = 'a reset (non-finite KF state, :137-140) snaps the rate-limited value back to its initial value (:60-64) in one message'
KNOWN = {
  'finite vehicleParameters (theoretical inputs)': ('theoretical', 'stds are sqrt(P) unchecked (:136): a far-ahead deviceMotion timestamp '
                                                    + 'predicts P to inf/NaN while valid; no consumer reads them', 'known_nan_std_far_ts.json'),
  'no_chatter vehicleParameters.valid': ('real', 'roll_std < ROLL_STD_MAX has no hysteresis (paramsd.py:178): posenetOK False ~1.3 s '
                                         + 'blips valid False (selfdrived.py:414)', 'known_roll_std_blip.json'),
  'no_chatter vehicleParameters.valid (theoretical inputs)': ('theoretical', 'non-finite KF state (inf vEgo) resets the filter in get_msg '
                                                              + '(:137-140), valid drops for the reset messages', 'known_chatter_far_ts.json'),
  'finite vehicleParameters used unconditionally (theoretical inputs)': ('theoretical', 'a restored NaN angleOffsetAverageDeg/stiffnessFactor '
                                                                         + 'is unchecked (:223-227) and reused by every reset (:60-64)',
                                                                         'known_nan_restore.json'),
  **{f'rate_limited vehicleParameters.{k} (theoretical inputs)': ('theoretical', SNAP, 'known_reset_snap.json')
     for k in ('angleOffsetDeg', 'angleOffsetAverageDeg', 'roll')},
}

SPEC = Spec(
  proc='paramsd',
  scope={'selfdrive/locationd/paramsd.py': None},
  ledger=LEDGER,
  unreached=UNREACHED,
  known=KNOWN,
  tag=tag,
  template=template,
  encode=encode,
  duration=20.,
  examples=48,
  params={'LiveParametersV2': restores(), 'CarParamsPrevRoute': [None, {'car': 'same'}, {'car': 'same'}, {'car': 'TOYOTA_PRIUS'}]},
  bounds={
    'carState.vEgo': [PD.MIN_ACTIVE_SPEED, PD.LOW_ACTIVE_SPEED, 0.],
    'carState.steeringAngleDeg': [45., -45.],
    'deviceMotion.angularVelocityDevice.z': [1., -1., 0.],
    'deviceMotion.angularVelocityDevice.zStd': [0., 10.],
    'deviceMotion.orientationNED.x': [PD.ROLL_MIN, PD.ROLL_MAX, PD.ROLL_LOWERED_MAX, -PD.ROLL_LOWERED_MAX],
    'deviceMotion.orientationNED.xStd': [PD.ROLL_STD_MAX],
    'deviceMotion.timestamp': [0., 1e9, -1e9, 5e7, -5e7],  # offsets from logMonoTime: the KF rewinds 1 s (EKF_sym_pyx max_rewind_age)
  },
  # lateral accel check |v * (yaw + observed yaw)| < 4 (paramsd.py:151): observed yaw ~ -4 / v
  relations={'deviceMotion.angularVelocityDevice.z': [('carState.vEgo', -0.4), ('carState.vEgo', 0.4)]},
  checks=[
    P.alive('vehicleParameters', 'deviceMotion'),
    P.finite('vehicleParameters', degraded=lambda m: not m.valid or not m.vehicleParameters.valid),
    # by controlsd.py:77-83 and longitudinal_planner.py:100
    P.finite('vehicleParameters', allow=('steerRatioStd', 'stiffnessFactorStd', 'angleOffsetAverageStd', 'angleOffsetFastStd'),
             degraded=lambda m: False, name='finite vehicleParameters used unconditionally'),
    rate_limited('vehicleParameters.angleOffsetDeg', PD.MAX_ANGLE_OFFSET_DELTA),
    rate_limited('vehicleParameters.angleOffsetAverageDeg', PD.MAX_ANGLE_OFFSET_DELTA),
    rate_limited('vehicleParameters.roll', PD.ROLL_MAX_DELTA),
    P.hysteresis('vehicleParameters.angleOffsetValid', 'vehicleParameters.angleOffsetDeg', PD.OFFSET_MAX, PD.OFFSET_LOWERED_MAX),
    P.hysteresis('vehicleParameters.angleOffsetAverageValid', 'vehicleParameters.angleOffsetAverageDeg', PD.OFFSET_MAX, PD.OFFSET_LOWERED_MAX),
    P.invariant('vehicleParameters', consistent, 'flags consistent'),
    sr_valid,
    P.valid_from_inputs('vehicleParameters', 'deviceMotion', ['carState', 'deviceMotion', 'extrinsicsCalibration']),
    P.no_chatter('vehicleParameters.valid', 1.),
  ],
)
