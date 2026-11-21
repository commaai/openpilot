# dmonitoringmodeld through the raw-output front end in procs/modeld.py: each cabin frame carries the net's float16 output
# vector, the process's own slicing/parsing/packing runs on it. 'frame.real' genomes run the real net on seeded images.
import math

import numpy as np

import openpilot.selfdrive.modeld.dmonitoringmodeld as dmm
from openpilot.selfdrive.monitoring.policy import DRIVER_MONITOR_SETTINGS
from openpilot.selfdrive.test.process_fuzz import procs as U, properties as P
from openpilot.selfdrive.test.process_fuzz.harness import Spec
from openpilot.selfdrive.test.process_fuzz.procs.modeld import P_MIN, STD_MAX, VisionSpace, decode, model_meta, raw_ledger


class FuzzModelState(dmm.ModelState):
  # ModelState.run replaced by the frame's raw vector (the net's output dtype, float16, upcast like run() does)
  def __init__(self, cam_w, cam_h):
    self.cam, self.real = (cam_w, cam_h), None
    self.numpy_inputs = {'calib': np.zeros((1, 3), dtype=np.float32)}
    self.output_slices = DMSpace.META[0]

  def run(self, buf, calib, transform):
    raw = decode(buf.data, *self.cam, *DMSpace.META[1:])
    if raw is None:
      self.real = self.real or FuzzModelState.__base__(*self.cam)
      return self.real.run(buf, calib, transform)
    return raw.astype(np.float32), 0.


def front(launcher, once):
  dmm.ModelState = FuzzModelState
  return launcher


class DMSpace(VisionSpace):
  MAIN, META = 'cabinCameraState', model_meta('dmonitoring_model_tinygrad.pkl')
  SIZES, DEFAULT = {'1928x1208': (1928, 1208), '1344x760': (1344, 760)}, '1928x1208'
  UNUSED = ('features',)  # only in rawPredictions, which is empty without SEND_RAW_PRED


LEDGER = raw_ledger(DMSpace, 'the net outputs float16 (output_specs): any float16 vector over-approximates every image; '
                    + 'NaN/inf are not shown net-reachable') | U.upstream('extrinsicsCalibration.rpyCalib[]') | {
  'frame.size:other': ('enforced', 'camerad: 1928x1208 or 1344x760; other sizes have no dm_warp pkl (dmonitoringmodeld.py:47)', None),
  'cabinCameraState.frameId:first_dropped': ('harness', 'process replay creates no stream without a frame (process_replay.py:204)', None),
}

M = 'selfdrive/modeld/dmonitoringmodeld.py:main:'
UNREACHED = {
  U.H_CONNECT: [M + '"while not vipc_client.connect(False):"#1:*', M + '"while not vipc_client.connect(False):":F'],
  'dead: the connect loop only exits connected, and a connected client has its buffer size':
    [M + '"assert vipc_client.is_connected()":F', M + '"assert cam_w is not None*:None'],
  U.H_RECV: [M + '"if buf is None:":None'],
  'env: SEND_RAW_PRED unset': [M + '"raw_pred = model_output.tobytes() if SEND_RAW_PRED else b\'\'":T'],
}
KNOWN = {
  'valid_from_inputs driverStateV2': ('real', 'always valid=True (dmonitoringmodeld.py:99), before any calibration and with an invalid one',
                                      'known_valid_ignores_inputs.json'),
}


SIDES = ('leftDriverData', 'rightDriverData')
PROBS = ('faceProb', 'leftEyeProb', 'rightEyeProb', 'leftBlinkProb', 'rightBlinkProb', 'sunglassesProb', 'phoneProb', 'sleepProb')


def lengths(d):
  return all((len(s.faceOrientation), len(s.faceOrientationStd), len(s.facePosition), len(s.facePositionStd)) == (3, 3, 2, 2)
             for s in (d.leftDriverData, d.rightDriverData))


@P.prop('frameId passthrough')
def frame_ids(r):
  # driverStateV2.frameId is the frame's id, in order (dmonitoringd's %40 / %6000 triggers depend on it)
  sent = [m.cabinCameraState.frameId for m in r.ins if m.which() == 'cabinCameraState']
  got = [m.driverStateV2.frameId for m in P.outs(r, 'driverStateV2')]
  assert got == sent[:len(got)], f'frameIds {got[:8]} for frames {sent[:8]}'


@P.prop('stateless')
def stateless(r):
  # the net has no recurrent input and run() keeps no state but calib: the same image and calib give the same output,
  # whatever came before (real-net genomes repeat image seeds)
  real = r.genome.genes.get('frame.real')
  if real is None or r.err:
    return None
  seen = {}
  for c in P.cycles(r, ('driverStateV2',)):
    d = c.out['driverStateV2'].driverStateV2.to_dict()
    for k in ('frameId', 'modelExecutionTime', 'gpuExecutionTime'):
      d.pop(k, None)
    cal = c.last.get('extrinsicsCalibration')  # the last one sent before the frame, else zeros
    key = (int(real.at(c.t)), str(tuple(cal.extrinsicsCalibration.rpyCalib) if cal is not None else (0., 0., 0.)))
    assert str(seen.setdefault(key, d)) == str(d), f't={c.t:.2f}s output differs for the same image and calib'
  return 'real net'


S_ = DRIVER_MONITOR_SETTINGS()


def logit(p):
  return -math.log(1 / p - 1)


SPEC = Spec(
  proc='dmonitoringmodeld',
  scope={'selfdrive/modeld/dmonitoringmodeld.py': {'main', 'slice_outputs', 'parse_model_output', 'fill_driver_data', 'get_driverstate_packet'}},
  ledger=LEDGER,
  unreached=UNREACHED,
  known=KNOWN,
  template=U.shape,
  duration=3.,
  examples=16,
  space=DMSpace,
  front=front,
  bounds={
    'frameId': [1, 40, 41, 6000, 2 ** 32 - 1],
    # logits of the thresholds dmonitoringd applies to these (policy.py:49-54), the clip (-11) and float saturation
    **{f'out.{k}[]': [-11., 17., logit(S_._FACE_THRESHOLD), logit(S_._EYE_THRESHOLD), logit(S_._SG_THRESHOLD), logit(S_._BLINK_THRESHOLD), 0.]
       for k in DMSpace.META[0] if 'prob' in k or k == 'wheel_on_right'},
    **{f'out.face_descs_{s}[]': [11., math.log(S_._HI_STD_THRESHOLD), math.log(S_._DCAM_UNCERTAIN_ALERT_THRESHOLD), 65504.] for s in ('lhd', 'rhd')},
  },
  checks=[
    P.alive('driverStateV2', 'cabinCameraState'),
    P.invariant('driverStateV2', lengths, 'driverStateV2 list lengths'),
    # sigmoid(clip(x, -11)): [1/(1+e^11), 1] or NaN
    P.bounded(['driverStateV2.wheelOnRightProb'] + [f'driverStateV2.{s}.{k}' for s in SIDES for k in PROBS], P_MIN, 1., nan_ok=True,
              name='bounded driverStateV2 probabilities'),
    P.bounded([f'driverStateV2.{s}.{k}' for s in SIDES for k in ('faceOrientationStd', 'facePositionStd')], 0., STD_MAX, nan_ok=True,
              name='bounded driverStateV2 stds'),
    # valid is always True (dmonitoringmodeld.py:99): nothing excuses a non-finite output made from a finite raw vector
    P.recovers(['driverStateV2'], 0., bad=lambda r, c: 'frame.real' in r.genome.genes or r.space.raw_bad(r.genome, c.t),
               name='finite for finite raw outputs'),
    frame_ids, stateless,
    P.valid_from_inputs('driverStateV2', 'cabinCameraState', ['extrinsicsCalibration']),
  ],
)
