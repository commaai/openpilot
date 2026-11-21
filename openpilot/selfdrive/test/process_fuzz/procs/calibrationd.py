import math

import numpy as np

from openpilot.cereal import log
from openpilot.common.fuzzy import _INTEGER_RANGES as INT_RANGE
from openpilot.selfdrive.locationd import calibrationd as C
from openpilot.selfdrive.locationd.locationd import CALIB_RPY_SANITY_CHECK
from openpilot.selfdrive.test.process_fuzz import procs as U, properties as P
from openpilot.selfdrive.test.process_fuzz.harness import Spec

Status = log.ExtrinsicsCalibration.Status
MODELD = 'fill_model_msg.py:185-192 slices fixed pose/road_transform/wide_from_device_euler shapes (constants.py:26-27)'

LEDGER = U.upstream('cameraOdometry.', 'carState.vEgo') | {
  **{f'cameraOdometry.{f}.len': ('enforced', MODELD, 'modeld: list lengths') for f in ('roadTransformTrans', 'roadTransformTransStd', 'wideFromDeviceEuler')},
  'P CalibrationParams:outside image': ('theoretical', 'only calibrationd writes it (calibrationd.py:172): restore_in_image() is that image', None),
  'CP.carFingerprint': ('symmetry', 'only compared for equality with CarParamsPrevRoute (calibrationd.py:280), drawn as same/other', None),
  'CP.brand': ('symmetry', 'only compared with "mock" (calibrationd.py:280); MOCK is its own cp_class', None),
}


def restores():
  # CalibrationParams over its restore classes (calibrationd.py:79-116): validBlocks around INPUTS_NEEDED/INPUTS_WANTED and
  # negative, rpyCalib/height/wideFromDeviceEuler of the wrong length or non-finite, rpy at and past the limits
  rng = np.random.default_rng(0)
  pitch = [0., C.PITCH_LIMITS[0], C.PITCH_LIMITS[1], C.PITCH_LIMITS[0] - .005, C.PITCH_LIMITS[1] + .005, 0.05, -0.05, math.nan]
  yaw = [0., C.YAW_LIMITS[0], C.YAW_LIMITS[1], C.YAW_LIMITS[0] - .005, C.YAW_LIMITS[1] + .005, 0.03, -0.03, math.inf]
  roll = [0., 0.1, CALIB_RPY_SANITY_CHECK, 1., -1.]
  # validBlocks > INPUTS_WANTED all index past the ring (:145); 10**6 stands for the rest of int32, which only adds an allocation
  # of range(validBlocks) at :136 (int32 max: ~80 GB, OOM, see KNOWN)
  blocks = [0, 1, C.INPUTS_NEEDED - 1, C.INPUTS_NEEDED, C.INPUTS_NEEDED + 1, C.INPUTS_WANTED - 1, C.INPUTS_WANTED, C.INPUTS_WANTED + 1, -1,
            INT_RANGE['int32'][0], 10 ** 6]
  out = [None, {'raw': 'deadbeef'}]
  for k in range(48):
    n = [3, 3, 3, 3, 2, 0][k % 6]
    rpy = [float(rng.choice(roll)), float(rng.choice(pitch)), float(rng.choice(yaw))][:n]
    cal = {'validBlocks': blocks[k % len(blocks)], 'rpyCalib': rpy,
           'height': [[1.22], [1.22], [math.nan], [], [1., 2.]][k % 5], 'wideFromDeviceEuler': [[0.] * 3, [0.] * 3, [0.01, math.nan, 0.], [0.]][k % 4]}
    out.append({'event': {'extrinsicsCalibration': cal}})
  return out


def f32(x):
  return float(np.float32(x))  # rpyCalib/rpyCalibSpread are float32 lists: a value at a float64 limit is published rounded


def clipped(c):
  pl, yl = C.PITCH_LIMITS, C.YAW_LIMITS
  return len(c.rpyCalib) != 3 or (f32(pl[0] - .005) <= c.rpyCalib[1] <= f32(pl[1] + .005) and f32(yl[0] - .005) <= c.rpyCalib[2] <= f32(yl[1] + .005))


def consistent(c):
  cal = c.calStatus in (Status.calibrated, Status.invalid)
  return (cal == (c.validBlocks >= C.INPUTS_NEEDED) and (c.calStatus != Status.recalibrating or c.validBlocks < C.INPUTS_NEEDED)
          and 0 <= c.validBlocks <= C.INPUTS_WANTED and 0 <= c.calPerc <= 100)


@P.prop('invariant calibrated within limits')
def calibrated_within_limits(r):
  # notCar is forced calibrated at rpy 0 (calibrationd.py:255-260); otherwise calibrated => the spread is below the reset
  # thresholds (calibrationd.py:157-168)
  for m in r.outs:
    c = m.extrinsicsCalibration
    if r.cp.notCar:
      assert c.calStatus == Status.calibrated and list(c.rpyCalib) == [0.] * 3 and c.calPerc == 100, f't={P.t_of(m):.2f}s notCar {c}'
    elif c.calStatus == Status.calibrated:
      assert c.rpyCalibSpread[1] <= f32(C.MAX_ALLOWED_PITCH_SPREAD) and c.rpyCalibSpread[2] <= f32(C.MAX_ALLOWED_YAW_SPREAD), \
        f't={P.t_of(m):.2f}s calibrated with spread {list(c.rpyCalibSpread)}'


def restore_in_image(v):
  # calibrationd writes its own published message (calibrationd.py:172): rpyCalib of length 3, finite, pitch/yaw clipped
  # (sanity_clip :54-59), roll any finite value, 0 <= validBlocks <= INPUTS_WANTED, height of length 1, wideFromDeviceEuler of
  # length 3 (non-finite height/wide are in the image: :220-223)
  if v is None:
    return True
  c = v.get('event', {}).get('extrinsicsCalibration')
  if c is None:
    return False
  rpy = c['rpyCalib']
  return (len(rpy) == 3 and all(math.isfinite(x) for x in rpy) and clipped(log.ExtrinsicsCalibration.new_message(rpyCalib=rpy))
          and 0 <= c['validBlocks'] <= C.INPUTS_WANTED and len(c['height']) == 1 and len(c['wideFromDeviceEuler']) == 3)


def tag(sp, g, ins=None):
  return P.theoretical(sp, g) or (None if restore_in_image(g.params.get('CalibrationParams')) else 'theoretical inputs')


TRANSITIONS = {(a, b) for a in ('uncalibrated', 'calibrated', 'invalid', 'recalibrating') for b in ('calibrated', 'invalid', 'recalibrating')}

CAL = 'selfdrive/locationd/calibrationd.py:'
UNREACHED = {
  U.ENV_DEBUG: [CAL + 'main:"if DEBUG and new_rpy is not None:":T', CAL + 'main:"if DEBUG and new_rpy is not None:"#1:*'],
  'dead: main always passes param_put=True (:283)': [CAL + 'Calibrator.__init__:"if param_put and calibration_params:":F',
                                                     CAL + 'Calibrator.update_status:"if self.param_put and write_this_cycle:":F'],
  'dead: the with body only reads fields, it cannot raise': [CAL + 'Calibrator.__init__:"with log.Event.from_bytes(calibration_params) as msg:":F',
                                                             CAL + 'main:"with car.CarParams.from_bytes(last_carparams_data) as last_CP:":F'],
  'dead: validBlocks is Int32, always finite': [CAL + 'Calibrator.reset:"if not np.isfinite(valid_blocks) or valid_blocks < 0:":F'],
  'enforced: modeld fixes the cameraOdometry list lengths': [CAL + 'Calibrator.handle_cam_odom:"if len(wide_from_device_euler) == 3:":F',
                                                             CAL + 'Calibrator.handle_cam_odom:"if (len(road_transform_trans) == 3):":F',
                                                             CAL + 'Calibrator.handle_cam_odom:"if len(road_transform_trans_std) == 3:":F'],
}

KNOWN = {
  'invariant rpyCalib inside locationd sanity range': ('real', 'roll is never clipped (sanity_clip :57) and composes each residual onto rpy '
                                                       + '(:207): it drifts past locationd.py:155 and persists', 'known_roll_drift.json'),
  'finite extrinsicsCalibration (theoretical inputs)': ('theoretical', 'a NaN roadTransformTrans/wideFromDeviceEuler sample poisons its ring '
                                                        + 'slot (0 * NaN, :62, :220-223)', 'known_nan_height.json'),
  'invariant rpyCalib usable (theoretical inputs)': ('theoretical', 'a restored rpyCalib is checked for finiteness, not length (:98-101): '
                                                     + 'crashes rot_from_euler downstream', 'known_restore_rpy_len.json'),
  'alive extrinsicsCalibration (theoretical inputs)': ('theoretical', 'restore raises on rpyCalib len != 3 (:51), validBlocks > 50 (:145), '
                                                       + 'OOMs on int32 max (:136)', 'known_restore_crash.json'),
}

SPEC = Spec(
  proc='calibrationd',
  cp_class=lambda CP: (CP.notCar, CP.brand == 'mock'),
  scope={'selfdrive/locationd/calibrationd.py': None},
  ledger=LEDGER,
  unreached=UNREACHED,
  known=KNOWN,
  tag=tag,
  template=U.shape,
  # trans weights: [1] for x, tan of the yaw/pitch limit ladder for y/z (yaw = atan2(t1, t0), pitch = -atan2(t2, t0), :204-206)
  weights={
    'cameraOdometry.trans[]': [[1.], [math.tan(y) for y in (C.YAW_LIMITS[0] - .005, *C.YAW_LIMITS, C.YAW_LIMITS[1] + .005, 0., 0.01, -0.01,
                                                             C.MAX_ALLOWED_YAW_SPREAD)],
                               [-math.tan(p) for p in (C.PITCH_LIMITS[0] - .005, *C.PITCH_LIMITS, C.PITCH_LIMITS[1] + .005, 0., 0.02, -0.02,
                                                       C.MAX_ALLOWED_PITCH_SPREAD)]],
    'cameraOdometry.rot[]': [[0.], [0.], [1.]],
    'cameraOdometry.transStd[]': [[0.], [1.], [0.]],
    'cameraOdometry.roadTransformTransStd[]': [[0.], [0.], [1.]],
    'cameraOdometry.roadTransformTrans[]': [[0.], [0.], [1.]],
    'cameraOdometry.wideFromDeviceEuler[]': [[1.], [1.], [1.]],
  },
  duration=40.,  # 6 blocks of 100 accepted 20 Hz samples from scratch; deeper phases come from restored CalibrationParams
  examples=32,
  params={'CalibrationParams': restores(),
          'CarParamsPrevRoute': [None, {'car': 'same'}, {'car': 'same'}, {'car': 'COMMA_BODY'}, {'car': 'TOYOTA_PRIUS'}, {'car': 'MOCK'}]},
  bounds={
    'carState.vEgo': [C.MIN_SPEED_FILTER],
    'cameraOdometry.transStd[]': [0.],
    'cameraOdometry.roadTransformTransStd[]': [C.MAX_HEIGHT_STD],
  },
  relations={'cameraOdometry.transStd[]': [('cameraOdometry.trans[]', math.tan(C.MAX_VEL_ANGLE_STD))]},  # arctan2(std[1], trans[0]), :194
  checks=[
    P.alive('extrinsicsCalibration', 'cameraOdometry', every=5),
    # consumers use rpyCalib unconditionally (locationd.py:153-158, helpers.py:181-183, modeld)
    P.invariant('extrinsicsCalibration', lambda c: len(c.rpyCalib) == 3 and all(map(math.isfinite, c.rpyCalib)), 'rpyCalib usable'),
    # outside it locationd silently keeps its old calibration (locationd.py:155-156)
    P.invariant('extrinsicsCalibration', lambda c: all(abs(x) <= CALIB_RPY_SANITY_CHECK for x in c.rpyCalib), 'rpyCalib inside locationd sanity range'),
    P.invariant('extrinsicsCalibration', clipped, 'pitch/yaw clipped'),
    P.finite('extrinsicsCalibration', degraded=lambda m: not m.valid or m.extrinsicsCalibration.calStatus != Status.calibrated),
    P.invariant('extrinsicsCalibration', consistent, 'status consistent with validBlocks'),
    calibrated_within_limits,
    P.transitions('extrinsicsCalibration.calStatus', TRANSITIONS),
    P.no_chatter('extrinsicsCalibration.calStatus', 2 * C.BLOCK_SIZE / 20.),
    P.valid_from_inputs('extrinsicsCalibration', 'cameraOdometry', ['cameraOdometry', 'carState'], by_time=True),  # sm.all_checks, :304
  ],
)
