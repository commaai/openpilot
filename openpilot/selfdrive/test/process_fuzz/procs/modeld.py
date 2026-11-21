# modeld and dmonitoringmodeld take camera frames through VisionIPC. No branch depends on pixels, only on the net's raw output
# vector, so the front replaces the net by the raw vector itself: each frame carries MAGIC + the vector in its first rows, and
# the process's own ModelState.run (desire pulse, slicing, the real Parser) runs on it. Any raw vector over-approximates the
# image space. A frame without MAGIC runs the real net ('frame.real' genomes).
import base64
import copy
import io
import math
import pickle
import struct

import numpy as np

import openpilot.selfdrive.modeld.modeld as md
from openpilot.common.realtime import DT_MDL
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.selfdrive.controls.lib.desire_helper import LANE_CHANGE_SPEED_MIN
from openpilot.selfdrive.modeld.constants import ModelConstants as MC
from openpilot.selfdrive.modeld.helpers import MODELS_DIR
from openpilot.selfdrive.test.process_fuzz import procs as U, properties as P, space as S
from openpilot.selfdrive.test.process_fuzz.harness import Spec
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

MAGIC = b'PFUZZRAW'
RAW_ROWS = 16  # a raw-vector frame only needs its first rows: the frame size only picks the warp/intrinsics
STD_MAX = float(np.float32(math.exp(11)))  # safe_exp clip, parse_model_outputs.py:4-6
P_MIN = float(np.float32(1 / (1 + math.exp(11))))


def model_meta(name):
  # output slices and dtype from the pickle's opcodes, without loading the weights (0.15 s instead of ~4 s)
  with open(MODELS_DIR / name, 'rb') as f:
    ops = f.read(struct.unpack('<q', f.read(8))[0])
  u = pickle.Unpickler(io.BytesIO(ops))
  u.persistent_load = lambda pid: None
  j = u.load()
  shape, dtype, _ = j['output_specs']['outputs']
  return pickle.loads(base64.b64decode(j['metadata']['metadata']['output_slices'])), int(np.prod(shape)), np.dtype(dtype)


def decode(data, w, h, n, dtype):
  # the raw vector from a frame's Y rows (stride-padded), or None for a real image
  stride = get_nv12_info(w, h)[0]
  k = len(MAGIC) + n * dtype.itemsize
  rows = -(-k // w)
  y = np.frombuffer(data, dtype=np.uint8, count=stride * rows).reshape(rows, stride)[:, :w].reshape(-1)[:k]
  return np.frombuffer(y[len(MAGIC):].tobytes(), dtype=dtype).copy() if y[:len(MAGIC)].tobytes() == MAGIC else None


class Frames:
  # process replay's FrameReader interface: frames are handed out in send order (run_step calls get() once per camera state)
  pix_fmt = 'nv12'

  def __init__(self, w, h):
    self.w, self.h, self.q = w, h, []

  def get(self, fid):
    kind, x = self.q.pop(0)
    img = np.zeros(self.w * self.h * 3 // 2, dtype=np.uint8)
    if kind == 'raw':
      b = MAGIC + x
      img[:len(b)] = np.frombuffer(b, dtype=np.uint8)
    else:  # a real image: seeded noise, flat gray or black
      r = np.random.default_rng(int(x))
      img[:] = r.integers(0, 256, img.size, dtype=np.uint8) if x % 3 == 0 else (x % 256 if x % 3 == 1 else 0)
    return img


class Msgs(list):
  frs: dict | None = None


def spread(v, n, idx):
  # one list gene -> n elements, as space.setpath does for capnp lists, plus 'at:k' (only element k)
  j = np.arange(n)
  with np.errstate(invalid='ignore'):
    if idx.startswith('at:'):
      return np.where(j == int(idx[3:]) % n, v, 0.)
    return {'lin': v * j / max(n - 1, 1), 'alt': v * (-1.) ** j, 'one': np.where(j == 0, v, 0.)}.get(idx, np.full(n, v))


class VisionSpace(S.Space):
  # genes on the raw output heads ('out.<head>[]'), frame counters/timestamps as offsets from a nominal clock, frame size
  MAIN, EXTRA, META, UNUSED = '', None, None, ()
  SIZE, SIZES, DEFAULT, FIXED = 'frame.size', {}, None, ()  # the frame size gene, its choices and default; FIXED: leaves set from it

  def __init__(self, spec):
    super().__init__(spec)
    self.slices, self.n, self.dtype = self.META
    self.counters = [f'{c}.{f}' for c in (self.MAIN, self.EXTRA) if c for f in ('frameId', 'timestampSof', 'timestampEof')]
    self.leaves |= {p: S.leaf_type(p) for p in self.counters + list(self.FIXED)}
    self.leaves |= {f'out.{k}[]': ('float32', None) for k in self.slices if k not in self.UNUSED}
    self.leaves[self.SIZE] = ('enum', list(self.SIZES))

  def boundaries(self, path, g=None):
    return super().boundaries(path, g) if path not in self.counters else self.spec.bounds.get(path.split('.')[-1], [])

  def value(self, rng, path, g=None):
    if path.startswith(('frame.', 'camera.')):  # harness choices, not capnp enums: no out-of-range values
      return rng.choice(self.leaves[path][1])
    if path in self.counters:  # offsets, signed, around the boundaries the pairing/drop logic uses
      b = self.boundaries(path)
      return rng.choice([0, 0, 1, -1, rng.choice(b or [0]) + rng.choice([-1, 0, 1]), rng.randint(-10 ** 9, 10 ** 9), 2 ** 32, -2 ** 40])
    return super().value(rng, path, g)

  def gene(self, rng, path, g=None):
    ge = super().gene(rng, path, g)
    if path.startswith('out.') and rng.random() < 0.4:
      sl = self.slices[path[4:-2]]
      ge.idx = f'at:{rng.randrange(sl.stop - sl.start)}'
    return ge

  def draw(self, rng):
    g = super().draw(rng)
    for p in self.FIXED:
      g.genes.pop(p, None)
    g.genes[self.SIZE] = S.Gene('hold', rng.choice(list(self.SIZES)), 0.)
    return g

  def choice(self, g):
    s = g.genes.get(self.SIZE)
    return s.a if s is not None else self.DEFAULT

  def at(self, g, p, t):
    v = super().at(g, p, t)
    if p in self.counters and isinstance(v, (int, float)) and math.isfinite(v):
      v = v + (round(t / DT_MDL) if p.endswith('frameId') else S.T0 + int(round(t * 1e9)))
    return v

  def raw(self, g, t):
    x = np.zeros(self.n)
    for k, sl in self.slices.items():
      p = f'out.{k}[]'
      pert = g.perturb and g.perturb[0] == p
      if p in g.genes or pert:
        v = self.at(g, p, t) if p in g.genes else 0.
        if pert:
          v += g.perturb[1] * (1 if int(round(t / DT_MDL)) % 2 == 0 else -1)
        x[sl] = spread(v, sl.stop - sl.start, g.genes[p].idx if p in g.genes else 'same')
    with np.errstate(over='ignore'):
      return x.astype(self.dtype).tobytes()

  def raw_bad(self, g, t):
    return 'frame.real' not in g.genes and not np.all(np.isfinite(np.frombuffer(self.raw(g, t), dtype=self.dtype)))

  def messages(self, g):
    g = copy.deepcopy(g)
    for p in self.counters:  # counters always run, a gene only offsets them
      g.genes.setdefault(p, S.Gene('hold', 0., 0.))
    for f in g.faults:  # process replay only creates a stream it sees a frame of (process_replay.py:204): the first frame always goes
      if f[0] == self.MAIN and f[1] in ('drop', 'slow'):
        f[2] = max(f[2], DT_MDL / 2)
    self.fix(g)
    out = Msgs(self.adjust(g, super().messages(g)))
    real = g.genes.get('frame.real')
    w, h = self.SIZES[self.choice(g)]
    out.frs = {c: Frames(w, h if real is not None else RAW_ROWS) for c in (self.MAIN, self.EXTRA) if c}
    for m in out:
      if m.which() in out.frs:
        t = P.t_of(m)
        out.frs[m.which()].q.append(('real', int(real.at(t))) if real is not None else ('raw', self.raw(g, t) if m.which() == self.MAIN else b''))
    return out

  def fix(self, g):
    pass

  def adjust(self, g, msgs):
    return msgs


def raw_ledger(space, why):
  return {f'out.{k}[]:nan|inf': ('theoretical', why, None) for k in space.META[0] if k not in space.UNUSED}


# *** modeld ***

META = model_meta('driving_tinygrad.pkl')
SVCS = ('modelV2', 'drivingModelData', 'cameraOdometry')


class FuzzModelState(md.ModelState):
  # the real ModelState.run with the net (copy_inputs/run_warp/run_model/copy_outputs) replaced by the frame's raw vector
  def __init__(self, cam_w, cam_h, chestnut):
    self.cam, self.chestnut, self.real = (cam_w, cam_h), chestnut, None
    self.vision_input_names = ('img', 'big_img')
    self.output_slices = META[0]
    self.prev_desire = np.zeros(MC.DESIRE_LEN, dtype=np.float32)
    stride, y_height, uv_height, _ = get_nv12_info(cam_w, cam_h)
    self.frame_copy_size = stride * (y_height + uv_height)
    self.frames = np.zeros((2, self.frame_copy_size), dtype=np.uint8)
    self.npy = {'tfm': np.zeros((2, 3, 3), np.float32), 'desire': np.zeros(MC.DESIRE_LEN, np.float32),
                'traffic_convention': np.zeros((1, 2), np.float32), 'action_t': np.zeros((1, 2), np.float32)}
    self.input_queues, self.warp_inputs, self.outputs = {}, {}, {}
    self.copy_inputs = lambda: None
    self.run_warp = self.run_model = lambda **kw: None
    self.copy_outputs = lambda: self
    self.parser = md.Parser()

  def numpy(self):
    return decode(self.frames[0], *self.cam, META[1], META[2]).reshape(1, -1)

  def run(self, bufs, *a, **kw):
    if decode(bufs['img'].data, *self.cam, 0, META[2]) is None:
      self.real = self.real or FuzzModelState.__base__(*self.cam, False)
      return self.real.run(bufs, *a, **kw)
    return super().run(bufs, *a, **kw)


def front(launcher, once):
  md.ModelState = FuzzModelState
  return launcher


class ModeldSpace(VisionSpace):
  MAIN, EXTRA, META = 'narrowRoadCameraState', 'wideRoadCameraState', META
  UNUSED = ('hidden_state', 'pad')  # never read by modeld (output_slices: only parsed heads are used)
  # the frame size is the sensor's: camera.pair (device/sensor) decides it, and sets deviceType and sensor
  SIZE, DEFAULT, FIXED = 'camera.pair', 'tici/ar0231', ('deviceState.deviceType', 'narrowRoadCameraState.sensor')
  SIZES = {f'{d}/{s}': (c.narrow_road.width, c.narrow_road.height) for (d, s), c in DEVICE_CAMERAS.items()
           if (c.narrow_road.width, c.narrow_road.height) in ((1928, 1208), (1344, 760))}

  def __init__(self, spec):
    super().__init__(spec)
    self.leaves['camera.streams'] = ('enum', ['dual', 'narrow'])

  def fix(self, g):
    d, s = self.choice(g).split('/')
    g.genes['deviceState.deviceType'] = S.Gene('hold', d, 0.)
    g.genes['narrowRoadCameraState.sensor'] = S.Gene('hold', s, 0.)

  def adjust(self, g, msgs):
    st = g.genes.get('camera.streams')
    if st is not None and st.a == 'narrow':
      return [m for m in msgs if m.which() != self.EXTRA]
    # camerad syncs the two sensors (spectra.cc:1515-1527): an extra frame is never >= 25 ms behind its main one
    main_sof = {m.logMonoTime: m.narrowRoadCameraState.timestampSof for m in msgs if m.which() == self.MAIN}
    extras = []
    for m in msgs:
      if m.which() == self.EXTRA:
        sof = main_sof.get(m.logMonoTime)
        extras.append(edit(m, timestampSof=sof - 25_000_000 + 1) if sof is not None and m.wideRoadCameraState.timestampSof + 25_000_000 <= sof else m)
    # lockstep sends frames per cycle, so modeld waiting for a newer extra frame (modeld.py:364-368) would wait forever; on a
    # device the next one just arrives, so it is sent early, before its main frame
    out, j, last = [], 0, None
    for m in msgs:
      if m.which() == self.EXTRA:
        continue
      if m.which() == self.MAIN:
        sof = m.narrowRoadCameraState.timestampSof
        while j < len(extras) and (extras[j].logMonoTime <= m.logMonoTime or last is None or last + 25_000_000 <= sof):
          e = extras[j]
          out.append(e if e.logMonoTime <= m.logMonoTime else edit(e, logMonoTime=m.logMonoTime))
          last, j = e.wideRoadCameraState.timestampSof, j + 1
        if last is None or last + 25_000_000 <= sof:
          break  # no new enough extra frame will come: modeld waits for one, as on a device
      out.append(m)
    return out


def edit(m, logMonoTime=None, timestampSof=None):
  b = m.as_builder()
  if logMonoTime is not None:
    b.logMonoTime = logMonoTime
  if timestampSof is not None:
    getattr(b, b.which()).timestampSof = timestampSof
  return b.as_reader()


LEDGER = raw_ledger(ModeldSpace, 'any float32 vector over-approximates every image; NaN/inf are float32-reachable, not shown net-reachable') | \
  U.upstream('extrinsicsCalibration.rpyCalib[]') | {
  'deviceState.deviceType': ('enforced', 'HARDWARE.get_device_type(); fuzzed jointly with the sensor as camera.pair', None),
  'narrowRoadCameraState.sensor': ('enforced', 'camerad publishes its detected sensor; fuzzed jointly with deviceType as camera.pair', None),
  'camera.pair:other': ('enforced', 'only DEVICE_CAMERAS pairs (else KeyError, modeld.py:391), frames of the sensor size (camerad)', None),
  'wideRoadCameraState.timestampSof:behind_25ms': ('enforced', 'camerad syncs the sensors (spectra.cc:1515-1527), ModeldSpace.adjust', None),
  'wideRoadCameraState.frameId:late': ('harness', 'lockstep: a missing/late extra frame is replaced by the next one sent early', None),
  'narrowRoadCameraState.frameId:first_dropped': ('harness', 'process replay creates no stream without a frame (process_replay.py:204)', None),
  'camera.streams:wide_only': ('harness', 'process replay locks on the narrow stream (process_replay.py:557), modeld would wait forever', None),
  'CP.brand': ('symmetry', 'only logged, modeld.py:341', None),
}

MD, MAIN, FILL = 'selfdrive/modeld/modeld.py:', 'selfdrive/modeld/modeld.py:main:', 'selfdrive/modeld/fill_model_msg.py:'
PMO, PARSE = 'selfdrive/modeld/parse_model_outputs.py:', 'selfdrive/modeld/parse_model_outputs.py:Parser.'
UNREACHED = {
  'env: no chestnut eGPU on a PC (modeld.py:245): the big model, its fallback and GPU state paths': [
    MAIN + '"CHESTNUT = *#*:T', MAIN + '"CHESTNUT = chestnut_compiled() and (chestnut_present() or (cable_connected() and not usb_host_connected()))":F',
    MAIN + '"if CHESTNUT:*:T', MAIN + '"*chestnut_state*:T', MAIN + '"pub_socks = *:T', MAIN + '"if chestnut_state is not None:":*',
    MAIN + '"if model is None and not params.get_bool(*', MAIN + '"if not model.chestnut:":*', MAIN + '"small_model = *#1:*',
    MAIN + '"small_model = *:notNone', MAIN + '"if model is None:":notNone', MAIN + '"if model_output is not None:":None',
    MAIN + '"except Exception:":T', MD + 'main.<locals>.load_big:"except Exception:":T', MD + 'ModelState.run:"if after_enqueue is not None:":notNone',
    MD + 'ModelState.run:"if self.chestnut*:T', MD + 'ModelState.run:"if self.chestnut*#1:F'],
  'env: SEND_RAW_PRED unset': [FILL + 'fill_model_msg:"if SEND_RAW_PRED:":T', MD + 'ModelState.run:"if SEND_RAW_PRED:":T'],
  U.H_CONNECT: [MAIN + '"if available_streams:":F', MAIN + '"while not vipc_client_main.connect(False):"#1:*',
                MAIN + '"while not vipc_client_main.connect(False):":F', MAIN + '*vipc_client_extra.connect(False):"#1:F',
                MAIN + '*vipc_client_extra.connect(False):"#2:*', MAIN + '*vipc_client_extra.connect(False):"#3:*'],
  U.H_RECV: [MAIN + '"if buf_*:None'],
  'harness: a wide-only stream is not replayable (LEDGER camera.streams)': [MAIN + '"*main_wide_camera else *:T'],
  'harness: the launcher calls main() without --demo': [MAIN + '"if demo:":T'],
  'harness: lockstep delivers each frame with its own camera state, so the SubMaster never has a newer one':
    [FILL + 'fill_model_msg:"frame_age = frame_id - vipc_frame_id if frame_id > vipc_frame_id else 0":T'],
  'harness: the first main frame always comes with its camera state (LEDGER first_dropped)': [MAIN + '"if sm.updated["extrinsicsCalibration"]*#1:F'],
  'enforced: extra frames are never >= 25 ms behind main ones (LEDGER), so the loop always runs once':
    [MAIN + '"while meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:":F'],
  'dead: this model has a non-MHP lead head (144 = 3x6x4x2) and no MHP plan, so in_N/out_N are 0 (parse_model_outputs.py:105-109)': [
    PARSE + 'parse_mdn:"for *', PARSE + 'parse_mdn:"if *N*:T', PARSE + 'parse_mdn:"if out_N == 1:":F',
    PARSE + 'is_mhp:"if outs[name].shape[1] == 2 * shape:":F', PARSE + 'parse_vision_outputs:*lead_mhp*:T'],
  'dead: every head the parser asks for is in output_slices': [PARSE + '*check_missing*:T', PARSE + 'check_missing:*#1:F'],
  'dead: the raw output is float32 (output_specs), softmax always works in place': [PMO + 'softmax:*#1:*', PMO + 'softmax:*":F'],
  'dead: every caller passes it': [FILL + 'fill_xyvat:"if *_std is not None:":None', MAIN + '"assert cam_w*:None',
                                   PARSE + 'parse_categorical_crossentropy:"if out_shape is not None:":None'],
  'dead: this model has the action head (output_slices action 2062:2066)': [MD + 'get_action_from_model:"if \'action\' not in model_output:":T'],
  'dead: DesireHelper.desire is none/laneChangeLeft/laneChangeRight (desire_helper.py:73-79)': [MAIN + '"if desire >= 0 *:F'],
  'dead: laneChangeStarting always has a direction (set on entering preLaneChange, desire_helper.py:39,43)':
    ['selfdrive/controls/lib/desire_helper.py:DesireHelper.update:"elif self.lane_change_direction == LaneChangeDirection.right:":F'],
}
KNOWN = {
  'finite': ('theoretical', 'one non-finite raw accel latches desiredAcceleration: smooth_value keeps (1-alpha)*NaN (modeld.py:70)',
             'known_nan_accel_latch.json'),
  'no_chatter modelV2.action.shouldStop': ('real', 'should_stop has no hysteresis (drive_helpers.py:17), raw unsmoothed accel (modeld.py:69)',
                                           'known_should_stop.json'),
  'valid_from_inputs modelV2': ('real', 'modelV2.valid only means calibration seen (modeld.py:458, fill_model_msg.py:79)',
                                'known_valid_ignores_inputs.json'),
}


@P.prop('alive modeld')
def alive(r):
  # no crash/hang; at most one output per main frame, and each names a main frame that was sent
  assert r.err is None, r.err
  sent = [m.narrowRoadCameraState.frameId for m in r.ins if m.which() == 'narrowRoadCameraState']
  n, sent = len(sent), set(sent)
  for svc in SVCS:
    o = P.outs(r, svc)
    assert len(o) <= n, f'{len(o)} {svc} for {n} frames'
    bad = [m for m in o if getattr(m, svc).frameId not in sent]
    assert not bad, f'{svc}.frameId {getattr(bad[0], svc).frameId} was never sent'
  return 'alive'


def lens(d):
  return (all(len(getattr(getattr(d, k), a)) == MC.IDX_N for k in ('position', 'velocity', 'acceleration', 'orientation', 'orientationRate') for a in 'xyzt')
          and len(d.laneLines) == 4 and len(d.laneLineProbs) == 4 and len(d.roadEdges) == 2 and all(len(ll.y) == MC.IDX_N for ll in d.laneLines)
          and len(d.leadsV3) == 3 and all(len(getattr(ld, a)) == MC.LEAD_TRAJ_LEN for ld in d.leadsV3 for a in ('x', 'y', 'v', 'a', 'xStd', 'yStd',
                                                                                                                 'vStd', 'aStd'))
          and len(d.meta.desireState) == MC.DESIRE_LEN and len(d.meta.desirePrediction) == MC.DESIRE_PRED_LEN * MC.DESIRE_PRED_WIDTH
          and len(d.meta.disengagePredictions.brakeDisengageProbs) == len(MC.META_T_IDXS))


@P.prop('list lengths')
def list_lengths(r):
  for m in P.outs(r, 'modelV2'):
    assert lens(m.modelV2), f't={P.t_of(m):.2f}s modelV2 list lengths'
  for m in P.outs(r, 'cameraOdometry'):
    c = m.cameraOdometry
    assert all(len(getattr(c, k)) == 3 for k in ('trans', 'rot', 'transStd', 'rotStd', 'wideFromDeviceEuler', 'roadTransformTrans')), 'cameraOdometry lengths'


@P.prop('time axes')
def time_axes(r):
  for m in P.outs(r, 'modelV2'):
    assert list(m.modelV2.position.t) == MC.T_IDXS and all(list(ld.t) == MC.LEAD_T_IDXS for ld in m.modelV2.leadsV3), f't={P.t_of(m):.2f}s'


def lane_direction(d):
  return (str(d.meta.laneChangeState) == 'off') == (str(d.meta.laneChangeDirection) == 'none')


DP = 'modelV2.meta.disengagePredictions.'
SPEC = Spec(
  proc='modeld',
  scope={
    'selfdrive/modeld/modeld.py': {'main', 'get_action_from_model', 'ModelState.run'},
    'selfdrive/modeld/fill_model_msg.py': None,
    'selfdrive/modeld/parse_model_outputs.py': None,
    'selfdrive/controls/lib/desire_helper.py': None,
    'selfdrive/controls/lib/drive_helpers.py': {'should_stop', 'smooth_value'},
  },
  ledger=LEDGER,
  unreached=UNREACHED,
  known=KNOWN,
  template=U.shape,
  duration=12.,
  examples=16,
  timeout=5,  # a hung pairing loop must fail in seconds, not process replay's 30
  space=ModeldSpace,
  front=front,
  bounds={
    'carState.vEgo': [0., 0.3, 2.5, LANE_CHANGE_SPEED_MIN],
    'frameId': [1, 2, 10, 11, 40, 2 ** 32 - 1],
    'timestampSof': [10_000_000, -10_000_000, 25_000_000, -25_000_000, -1_000_000_000],
    'timestampEof': [0, -50_000_000],
    'out.action[]': [0.1, -0.1],
    'out.desire_state[]': [-11., 11., 17.],
    'out.meta[]': [-11., 17., -math.log(1 / 0.05 - 1), -math.log(1 / 0.15 - 1), -math.log(1 / 0.7 - 1)],
    'out.lead_prob[]': [-11., 17.],
    'out.plan[]': [1e19, 1e38],
  },
  checks=[
    alive, list_lengths, time_axes,
    # no NaN latch: outputs are finite once the raw vector and the read inputs have been finite for 2 s
    P.recovers(SVCS, 2., bad=lambda r, c: r.space.raw_bad(r.genome, c.t) or any(map(r.space.bad, c.last.values())),
               trigger='narrowRoadCameraState', by_time=True, name='finite'),
    P.bounded([f'cameraOdometry.{k}' for k in ('transStd', 'rotStd', 'roadTransformTransStd', 'wideFromDeviceEulerStd')] +
            [f'modelV2.{k}' for k in ('position.xStd', 'laneLineStds', 'roadEdgeStds', 'leadsV3[].xStd', 'leadsV3[].vStd')], 0., STD_MAX,
            nan_ok=True, name='stds in [0, e^11] or nan'),
    P.bounded('modelV2.leadsV3[].prob', 0., 1., nan_ok=True, name='lead prob range'),
    P.bounded(['modelV2.meta.engagedProb', 'modelV2.meta.desireState', 'modelV2.meta.desirePrediction', 'modelV2.laneLineProbs'] +
            [DP + k for k in ('brakeDisengageProbs', 'gasDisengageProbs', 'steerOverrideProbs', 'brake3MetersPerSecondSquaredProbs',
                              'brake4MetersPerSecondSquaredProbs', 'brake5MetersPerSecondSquaredProbs')], 0., 1., nan_ok=True,
              name='bounded modelV2 probabilities'),
    P.invariant('modelV2', lane_direction, 'laneChangeDirection none iff off'),
    P.transitions('modelV2.meta.laneChangeState', {('off', 'preLaneChange'), ('preLaneChange', 'off'), ('preLaneChange', 'laneChangeStarting'),
                                                   ('laneChangeStarting', 'preLaneChange'), ('laneChangeStarting', 'off')}),
    P.bounded('modelV2.frameDropPerc', 0., 100. * 10 / 11),
    P.valid_from_inputs('modelV2', 'narrowRoadCameraState', ['carState', 'carControl', 'controlsState', 'extrinsicsCalibration', 'deviceState',
                                                             'driverMonitoringState', 'lateralDelay', 'narrowRoadCameraState']),
    P.no_chatter('modelV2.action.shouldStop', 1.),
  ],
)
