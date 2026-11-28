# Shared by the process specs: producer guarantees (ledger entries) several consumers rely on, producers' message shapes, and
# UNREACHED reasons. Opt-in: a spec uses upstream(...) for the ledger entries of what it reads; entries it doesn't read are unused.
import re

from openpilot.selfdrive.modeld.constants import ModelConstants as MC
from openpilot.selfdrive.test.process_fuzz.space import new_message

MODELD = 'fill_model_msg.py:90-128 fixes it'
SAFE_EXP = 'parse_model_outputs.py:4-6,52 safe_exp clips at e^11'
ODO_STDS = ('transStd', 'rotStd', 'roadTransformTransStd')
CALIB = 'calibrationd publishes 3 finite angles: sanity_clip :53-59, isfinite on restore :98-101, always 3 :250,259'
UPSTREAM = {
  'extrinsicsCalibration.rpyCalib[]:nan|inf|len': ('enforced', CALIB, 'calibrationd: invariant rpyCalib usable'),
  'extrinsicsCalibration.rpyCalib.len': ('enforced', CALIB, 'calibrationd: invariant rpyCalib usable'),
  'modelV2.velocity.x.len': ('enforced', MODELD, 'modeld: list lengths'),
  'selfdriveState.personality:outofrange': ('enforced', 'selfdrived.py:526 raises before :541', 'selfdrived: personality_range'),
  'vehicleParameters.roll:inf': ('enforced', 'paramsd.py:137-146', 'paramsd: finite vehicleParameters used unconditionally'),
  'carState.vEgo:nan|inf': ('theoretical', 'card KF over CAN-decoded ints, opendbc/car/interfaces.py:302-311', 'card: finite carState'),
  **{f'cameraOdometry.{f}[]:nan|inf': ('theoretical', 'no finiteness guard between the net output and fill_model_msg.py:185-188', 'modeld: finite')
     for f in ('trans', 'rot', 'wideFromDeviceEuler', 'roadTransformTrans')},
  **{f'cameraOdometry.{f}[]:nan': ('theoretical', 'safe_exp of a NaN net output is NaN, parse_model_outputs.py:4-6', 'modeld: finite')
     for f in ODO_STDS},
  **{f'cameraOdometry.{f}[]:inf': ('enforced', SAFE_EXP, 'modeld: stds in [0, e^11] or nan') for f in ODO_STDS},
  # inf is enforced, so a value past float32 only comes from a list weight and is clamped to F32 in the message
  **{f'cameraOdometry.{f}[]:huge': ('theoretical', SAFE_EXP, 'modeld: stds in [0, e^11] or nan') for f in ODO_STDS},
}


def upstream(*prefixes) -> dict:
  return {k: v for k, v in UPSTREAM.items() if k.startswith(prefixes)}


def shape(svc, g=None):
  # a producer's message shape: list lengths and time axes it always publishes, zeros elsewhere
  m = new_message(svc)
  if svc == 'modelV2':  # fill_model_msg.py:90-128
    md, n = m.modelV2, MC.IDX_N
    for xyzt in (md.position, md.velocity, md.orientationRate):
      xyzt.x, xyzt.y, xyzt.z, xyzt.t = [0.] * n, [0.] * n, [0.] * n, MC.T_IDXS
    md.init('laneLines', 4)
    for ll in md.laneLines:
      ll.x, ll.y, ll.z = MC.X_IDXS, [0.] * n, [0.] * n
    md.laneLineProbs = [0.] * 4
    md.init('leadsV3', 3)
    for ld in md.leadsV3:
      for k in ('x', 'y', 'v', 'a', 'xStd', 'yStd', 'vStd', 'aStd'):
        setattr(ld, k, [0.] * MC.LEAD_TRAJ_LEN)
      ld.t = MC.LEAD_T_IDXS
    md.meta.desirePrediction = [0.] * (MC.DESIRE_PRED_LEN * MC.DESIRE_PRED_WIDTH)
  elif svc == 'cameraOdometry':  # fill_model_msg.py:185-192
    co = m.cameraOdometry
    for k in ('trans', 'rot', 'transStd', 'rotStd', 'wideFromDeviceEuler', 'wideFromDeviceEulerStd', 'roadTransformTrans', 'roadTransformTransStd'):
      setattr(co, k, [0.] * 3)
  elif svc == 'extrinsicsCalibration':  # calibrationd.py:250,259
    m.extrinsicsCalibration.rpyCalib = [0.] * 3
  elif svc == 'driverStateV2':  # dmonitoringmodeld.py:85-88
    for d in (m.driverStateV2.leftDriverData, m.driverStateV2.rightDriverData):
      d.faceOrientation, d.faceOrientationStd, d.facePosition, d.facePositionStd = [0.] * 3, [0.] * 3, [0.] * 2, [0.] * 2
  return m


# UNREACHED reasons used by several specs
ENV_DEBUG = 'env: DEBUG unset'
H_LOCKSTEP = 'harness: lockstep always delivers the polled trigger'
H_CONNECT = 'harness: process replay starts the VisionIPC server before the process, connect succeeds at once'
H_RECV = 'harness: lockstep only unblocks recv with a frame'
DEAD_WITH = 'dead: with-exit suppresses nothing (only on an exception from the body)'
DEAD_EXC = 'dead: only an Exception subclass is raised there'
# structural: merged into every spec, may match nothing, must never match a reached id
STRUCTURAL = {DEAD_WITH: ['*:"with *:T'], DEAD_EXC: ['*:"except Exception*:"*:F']}


def match(pattern, branch_id):
  # '*' is the only wildcard: ids contain [ ] ? from the source
  return re.fullmatch('.*'.join(map(re.escape, pattern.split('*'))), branch_id) is not None
