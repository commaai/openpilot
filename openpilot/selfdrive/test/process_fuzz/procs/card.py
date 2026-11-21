import functools
import math
import threading

import numpy as np

import openpilot.cereal.messaging as messaging
import openpilot.selfdrive.test.process_replay.process_replay as pr
from openpilot.common.constants import CV
from opendbc.car.can_definitions import CanData
from opendbc.car.car_helpers import interfaces
from opendbc.car.structs import car
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_MIN, V_CRUISE_UNSET
from openpilot.selfdrive.test.process_fuzz import probe, properties as P, space as S
from openpilot.selfdrive.test.process_fuzz.harness import Spec

# card's input is CAN; its logic (card.py, cruise.py) sees CAN only through CI.update's CarState. Front end:
# carState is fuzzed directly over its capnp type and carried to the child in an extra frame on bus FUZZ_SRC of the same
# tick's can message, where CI.update's result is replaced by it. That's a superset of what CI.update can return; a
# violation it causes is triaged for CAN reachability. The rest of each can message is raw frames on the platform's own
# addresses (garbage counters/checksums/lengths; valid-checksum CAN through CANPacker is opendbc's tier, not built here).
# card builds its CarParams from CAN, so it ignores Genome.fpv.
FUZZ_SRC = 200
DT = int(pr.get_process_config('card').processing_time * 1e9)

CAN_DECODED = ('enforced', 'CANParser: raw int * factor + offset (opendbc/can/parser.py:60), then the linear speed KF (interfaces.py:306-311)', None)
SCHEMA = ('enforced', 'capnp enum: publishers use the same cereal/opendbc schema', None)
LEDGER = {
  # controlsd zeroes every non-finite actuator before publishing (controlsd.py:139-147); joystickd (np.clip) is a debug tool.
  # CI.apply hands the whole struct to the platform's CarController (card.py:238), traced as 'carControl.**'
  'carControl.actuators.*:nan|inf': ('enforced', 'controlsd.py:139-147', 'controlsd: invariant actuators finite'),
  # setSpeed = vCruiseCluster (controlsd.py:166), card's own finite output (check 'finite carState')
  'carControl.hudControl.setSpeed:nan|inf': ('enforced', 'controlsd.py:166', 'controlsd: finite outputs (finite inputs)'),
  # controlsd sets them from 3-vectors once it has a calibrated pose (controlsd.py:156-158), else leaves them empty
  **{f'carControl.{f}.len:shape': ('enforced', 'controlsd.py:156-158 .xyz.tolist() of a 3-vector, or unset', None)
     for f in ('orientationNED', 'angularVelocity')},
  # passive: carControl.enabled = selfdriveState.enabled (controlsd.py:96) and selfdrived never steps its state machine for a
  # passive car (selfdrived.py:564). Still fuzzed (card in isolation); set_speed_needs_button skips it
  'carControl.enabled:passive': ('enforced', 'controlsd.py:96, selfdrived.py:564', 'selfdrived: passive_never_enabled'),
  # huge: |raw| < 2^64 and every DBC factor/offset is small, so |value| < 1e30 (float32 overflow of speed * 3.6 needs > 9e37)
  **{f'carState.{f}:nan|inf|huge': CAN_DECODED for f in ('vEgo', 'cruiseState.speed', 'cruiseState.speedCluster')},
  'can[].src:bus200': ('harness', 'bus FUZZ_SRC carries the fuzzed CarState: raw frames on it move to the next bus', None),
  'P AlphaLongitudinalEnabled': ('harness', 'fuzzed as Genome.alpha_long, written by fingerprint_callback', None),
  'P ObdMultiplexingChanged': ('harness', "only waited on (card.py:38); fingerprint_callback plays pandad's side (panda_safety.cc:36)", None),
  # card is its only writer (card.py:149, the CarParams it built) and Params writes are atomic (tmp file + rename)
  'P CarParamsCache:garbage': ('enforced', 'card.py:149 only writer; params.cc writes via tmp + rename', None),
  # enums: every publisher (card's CarState, controlsd, selfdrived) sets them from the same compiled schema
  **{f'{p}:outofrange': SCHEMA for p in ('carState.buttonEvents[].type', 'carControl.actuators.longControlState', 'carControl.hudControl.visualAlert',
                                         'carControl.hudControl.audibleAlert', 'onroadEvents[].name')},
}

CARD = 'selfdrive/car/card.py:'
UNREACHED = {
  'harness: main() never passes a CI (card.py:277), only tests do': [CARD + 'Car.__init__:"if CI is None:":notNone'],
  'harness: card_fingerprint_callback sends a non-empty can first (process_replay.py:336); an empty one only loops':
    [CARD + 'Car.__init__:"if len(can.can) > 0:":F'],
  'dead: the with body is a plain assignment': [CARD + 'Car.__init__:"with car.CarParams.from_bytes(cached_params_raw) as _cached_params:":F'],
  'dead: every interface has a CarController (interfaces.py:127 builds it unconditionally)':
    [CARD + 'Car.__init__:"controller_available = self.CI.CC is not None and openpilot_enabled_toggle and not self.CP.dashcamOnly":F'],
  'env: /cache/params/SecOCKey is an absolute device path, absent off-device':
    [CARD + 'Car.__init__:"with open("/cache/params/SecOCKey") as f:":F', CARD + 'Car.__init__:"if len(user_key) == 32:":*'],
  'harness: the loop ends only on process exit and replay kills card': [CARD + 'Car.params_thread:"while not evt.is_set():":T'],
  'harness: replay steps card once per can message, so the 20 ms can_sock timeout (card.py:67) never returns an empty drain':
    [CARD + 'Car.state_update:"if not can_rcv_valid:":F', CARD + 'Car.state_update:"if can_rcv_valid and REPLAY:":F'],
}

KNOWN = {
  'alive carState (SecOCKey not hex)': ('real', 'bytes.fromhex on a non-hex SecOCKey raises in Car.__init__ (card.py:132); only len == 32 '
                                        + 'is checked (:125)', 'known_secoc_not_hex.json'),
  'alive carState (short frame)': ('theoretical', 'a frame shorter than its DBC message on a checksummed address: the checksum indexes past '
                                   + 'it (mqbcan.py:180)', 'known_short_frame.json'),
  'vcruise (engaging, cruise unavailable)': ('theoretical', 'initialize_v_cruise runs on the enable edge and ignores cruiseState.available '
                                             + '(card.py:183-186, cruise.py:125-135)', 'known_vcruise_engaging_unavailable.json'),
}


# *** child side ***

def front(launcher, once):
  # CI.update's CarState is replaced by the fuzzed one when the can message carries it
  def run(*a, **kw):
    from opendbc.car.interfaces import CarInterfaceBase
    import openpilot.selfdrive.car.card as card

    def update(self, can_packets, _orig=CarInterfaceBase.update):
      fuzz = [f[1] for _, fs in can_packets for f in fs if f[2] == FUZZ_SRC]  # (address, dat, src) tuples
      ret = _orig(self, [(t, [f for f in fs if f[2] != FUZZ_SRC]) for t, fs in can_packets])
      if not fuzz:
        return ret
      with car.CarState.from_bytes(fuzz[-1]) as cs:
        return probe.wrap_value(cs.as_builder(), 'carState', once) if once else cs.as_builder()

    def apply(self, c, now_nanos=None, _orig=CarInterfaceBase.apply):  # each platform's CarController reads its own subset
      if once:
        once('carControl.**')
      return _orig(self, c, now_nanos)

    def state_publish(self, CS, RD, _orig=card.Car.state_publish):
      _orig(self, getattr(CS, '_r', CS), RD)

    CarInterfaceBase.update, CarInterfaceBase.apply, card.Car.state_publish = update, apply, state_publish
    launcher(*a, **kw)
  return run


def fingerprint_callback(alpha_long):
  # process replay's card_fingerprint_callback, with alpha long from its param and pandad's side of the OBD multiplexing
  # handshake (car_helpers.py:122 -> card.py:34-38, panda_safety.cc:36)
  def cb(rc, pm, msgs, fingerprint):
    from openpilot.common.params import Params
    Params().put_bool('AlphaLongitudinalEnabled', alpha_long, block=True)
    # FINGERPRINT is set, so 201 empty can messages end the CAN fingerprint (car_helpers.py:74); the genome replays from t=0
    pre = []
    for k in range(210):
      m = messaging.new_message('can', 0)
      m.logMonoTime = (k + 1) * 1_000_000  # before the genome (S.T0)
      pre.append(m.as_reader())

    # its own first messages are stamped with the wall clock, which would put the child's log-time clocks days ahead
    def new_message(*a, _orig=messaging.new_message, **kw):
      m = _orig(*a, **kw)
      m.logMonoTime = 1
      return m
    messaging.new_message, done = new_message, threading.Event()

    def obd():
      params = Params()
      while not done.wait(0.01):
        if not params.get_bool('ObdMultiplexingChanged'):
          params.put_bool('ObdMultiplexingChanged', True)
    threading.Thread(target=obd, daemon=True).start()
    try:
      pr.card_fingerprint_callback(rc, pm, pre, fingerprint)
    finally:
      done.set()
      messaging.new_message = new_message.__kwdefaults__['_orig']
  return cb


def encode(v, g):  # BYTES params by name: this car's CarParams, or garbage
  return {'car': 'same'} if v == 'cp' else {'raw': b'\x00\xffgarbage'.hex()} if v == 'garbage' else v


# *** space: fingerprint classes, carState pseudo-service folded into can, raw frames ***

CS_FIELDS = ['carState.cruiseState.available', 'carState.cruiseState.speed', 'carState.cruiseState.speedCluster',
             'carState.cruiseState.standstill', 'carState.cruiseState.enabled', 'carState.buttonEvents.len',
             'carState.buttonEvents[].type', 'carState.buttonEvents[].pressed', 'carState.gasPressed', 'carState.vEgo',
             'carState.canValid', 'carState.canTimeout']
DAT = ['', '00', 'ff' * 8, '01' * 8, 'ff' * 64, '00' * 65, 'a5' * 3]  # empty, DLC-sized, CAN FD max, over-long, short


def cp(g):
  return S.get_car_params(g.fp, g.alpha_long)


def cp_class(CP):
  CI = interfaces[CP.carFingerprint]
  return (CP.brand, CP.pcmCruise, CP.dashcamOnly, CP.secOcRequired, CP.openpilotLongitudinalControl, CI.CarController is None,
          len(CP.safetyConfigs), str(CP.steerControlType))


@functools.cache
def addresses(fp, alpha):
  CI = interfaces[fp](S.get_car_params(fp, alpha))
  CI.update([(0, [])])  # parsers add their messages on first access
  return sorted({(a, p.bus) for p in CI.can_parsers.values() if p is not None for a in p.addresses})


class Space(S.Space):
  def __init__(self, spec):
    super().__init__(spec)
    self.pubs = sorted(set(self.pubs) | {'carState'})
    self.trigger = 'can'
    for p in sorted(set(CS_FIELDS) | {r for r in spec.reads if r.startswith('carState.')}):
      if p not in spec.ledger:
        self.leaves[p] = S.leaf_type(p)
    for p in ('can.len', 'can[].address', 'can[].src', 'can[].dat'):
      self.leaves[p] = S.leaf_type(p)

  def value(self, rng, path, g=None):
    if path == 'can[].dat':
      return rng.choice(DAT) if rng.random() < 0.7 else bytes(rng.getrandbits(8) for _ in range(rng.choice([1, 8, 32]))).hex()
    if path == "can[].address" and g and addresses(g.fp, False) and rng.random() < 0.8:
      return float(rng.choice(addresses(g.fp, False))[0])
    if path == 'can[].src':
      return float(rng.choice([0, 1, 2, 3, 128, 130, 255]))
    if path == 'can.len':
      return float(rng.choice([0, 1, 2, 5, 20]))
    if path == 'onroadEvents[].name' and rng.random() < 0.5:
      return 'selfdriveInitializing'
    return super().value(rng, path, g)

  def gene(self, rng, path, g=None):
    if path == 'can[].dat':
      t0 = rng.uniform(0, self.dur)
      return S.Gene(rng.choice(S.SHAPES[:4]), self.value(rng, path), self.value(rng, path), t0, t0 + rng.choice([0.01, 0.2, 1., 3.]))
    return super().gene(rng, path, g)

  def messages(self, g):
    # fold each tick's carState into that tick's can message (can is last in the tick: it's the trigger)
    out, cs = [], None
    for m in super().messages(g):
      if m.which() == 'carState':
        cs = m
        continue
      fold = m.which() == 'can' and cs is not None and cs.logMonoTime == m.logMonoTime
      if fold or (m.which() == 'can' and any(f.src == FUZZ_SRC for f in m.can)):
        b = m.as_builder()
        frames = [CanData(f.address, f.dat, FUZZ_SRC + 1 if f.src == FUZZ_SRC else f.src) for f in m.can]
        frames += [CanData(0, cs.carState.as_builder().to_bytes(), FUZZ_SRC)] if fold else []
        b.init('can', len(frames))
        for f, (a, d, s) in zip(b.can, frames, strict=True):
          f.address, f.dat, f.src = a, d, s
        m = b.as_reader()
      out.append(m)
    return out


# *** properties ***

@functools.cache
def frame_crashes(fp, alpha, frame):
  # a checksum function indexing past a short frame raises in CANParser.update, whatever the parser state
  if (frame[0], frame[2]) not in addresses(fp, alpha):
    return False
  try:
    CI = interfaces[fp](S.get_car_params(fp, alpha))
    CI.update([(0, [])])  # parsers add their messages on first access
    CI.update([(1, [frame])])
  except IndexError:
    return True
  return False


def tag(sp, g, ins=None):
  # inputs that hit a KNOWN crash: a non-hex SecOCKey (card.py:132), a short frame on a checksummed address
  key = g.params.get('SecOCKey')
  if cp(g).secOcRequired and key is not None and any(c not in '0123456789abcdefABCDEF' for c in key.strip()):
    return 'SecOCKey not hex'
  frames = {(f.address, bytes(f.dat), f.src) for m in ins or () if m.which() == 'can' for f in m.can if f.src != FUZZ_SRC}
  return 'short frame' if any(frame_crashes(g.fp, g.alpha_long, f) for f in frames) else None


def steps(r):
  # (cycle, fuzzed carState of its can message or None)
  for c in P.cycles(r, ('carState', 'carOutput'), 'can', by_time=True):
    fz = [f.dat for f in c.trig.can if f.src == FUZZ_SRC]
    yield c, car.CarState.from_bytes(fz[-1]).__enter__() if fz else None


KNOCKOUT = 1.25  # s: CI.init blocks card for an ECU knockout: disable_ecu retries 10x, 0.1 s timeout on the log clock = 120 can (card.py:229 TODO)


@P.prop('alive carState')
def alive(r):
  # one carState and one carOutput per can message (card is driven by can), except while CI.init blocks on each initialized
  # rising edge: no selfdriveInitializing in the last onroadEvents (card.py:228-233, 248-249)
  assert r.err is None, r.err
  stamps = {w: {m.logMonoTime - DT for m in P.outs(r, w)} for w in ('carState', 'carOutput')}
  init, ev, edge_t, n = False, None, -1e9, 0
  for m in r.ins:
    if m.which() == 'onroadEvents':
      ev = m
    if m.which() != 'can':
      continue
    n += 1
    now = ev is not None and not any(e.name == 'selfdriveInitializing' for e in ev.onroadEvents)
    if now and not init:
      edge_t = P.t_of(m)
    init = now
    for w, st in stamps.items():
      assert m.logMonoTime in st or P.t_of(m) - edge_t <= KNOCKOUT, f't={P.t_of(m):.2f}s no {w} for this can'
  for w in stamps:
    k = len(P.outs(r, w))
    assert k <= n + 1, f'{k} {w} for {n} can'
  return 'checked'


def vcruise(init_edge):
  # VCruiseHelper's image: unset (255) while cruise is unavailable; pcm mirrors cruiseState.speed (0 -> 255, -1 -> -1);
  # card's own set speed is 255 or in [V_CRUISE_MIN, V_CRUISE_MAX] and equals the cluster speed (cruise.py:43-63, 111).
  # Steps with carControl.enabled and cruise unavailable are their own check: initialize_v_cruise runs after
  # update_v_cruise and ignores cruiseState.available (card.py:183-186, cruise.py:125-135), a KNOWN bug
  @P.prop('vcruise' + (' (engaging, cruise unavailable)' if init_edge else ''))
  def check(r):
    for c, cs in steps(r):
      cc = c.last.get('carControl')
      if cs is None or init_edge != (cc is not None and cc.carControl.enabled and not cs.cruiseState.available):
        continue
      v, vc = c.out['carState'].carState.vCruise, c.out['carState'].carState.vCruiseCluster
      if not cs.cruiseState.available:
        assert v == vc == V_CRUISE_UNSET, f't={c.t:.2f}s vCruise {v}/{vc} with cruise unavailable'
      elif cp(r.genome).pcmCruise:
        sp = cs.cruiseState.speed
        want = V_CRUISE_UNSET if sp == 0 else -1 if sp == -1 else np.float32(sp * CV.MS_TO_KPH)
        assert v == want or (math.isnan(v) and math.isnan(sp)), f't={c.t:.2f}s pcm vCruise {v} for speed {sp}'
      else:
        assert v == vc and (v == V_CRUISE_UNSET or V_CRUISE_MIN <= v <= V_CRUISE_MAX), f't={c.t:.2f}s vCruise {v}/{vc}'
  return check


def passive(g):
  # card.py:112-114
  return cp(g).dashcamOnly or interfaces[g.fp].CarController is None or g.params.get('OpenpilotEnabledToggle') is False


@P.prop('set_speed_needs_button')
def set_speed_needs_button(r):
  # card's own set speed changes only on a cruise button while engaged: held with no accel/decel button event or held button,
  # carControl.enabled on in this and the previous step, and cruise available, it doesn't move (cruise.py:65-123, card.py:183-186)
  if cp(r.genome).pcmCruise or passive(r.genome):  # carControl.enabled:passive is enforced False, see LEDGER
    return None
  held, prev, n = {}, None, 0
  for c, cs in steps(r):
    en = 'carControl' in c.last and c.last['carControl'].carControl.enabled
    quiet = cs is not None and cs.cruiseState.available and not any(str(b.type) in ('accelCruise', 'decelCruise') for b in cs.buttonEvents)
    for b in cs.buttonEvents if cs is not None else ():
      held[str(b.type)] = b.pressed
    quiet = quiet and not any(held.get(b) for b in ('accelCruise', 'decelCruise'))
    v = c.out['carState'].carState.vCruise
    if prev is not None and prev[0] and prev[1] and en and quiet:
      n += 1
      assert v == prev[2], f't={c.t:.2f}s set speed {prev[2]} -> {v} without a button, carControl.enabled held'
    prev = (en, quiet, v)
  return 'checked' if n else None


@P.prop('outputs_valid')
def outputs_valid(r):
  # carState.valid == canValid (card.py:212); sendcan only once controls run, never when passive
  for m in r.outs:
    if m.which() == 'carState':
      assert m.valid == m.carState.canValid, f't={P.t_of(m):.2f}s carState.valid {m.valid} canValid {m.carState.canValid}'
  sent = [m for m in P.outs(r, 'sendcan') if len(m.sendcan)]
  assert not (passive(r.genome) and sent), f'{len(sent)} sendcan with a passive car'


SPEC = Spec(
  proc='card',
  scope={'selfdrive/car/card.py': None, 'selfdrive/car/cruise.py': None},
  ledger=LEDGER,
  known=KNOWN,
  unreached=UNREACHED,
  tag=tag,
  duration=8.,
  examples=16,
  space=Space,
  cp_class=cp_class,
  front=front,
  huge=1e30,  # see the carState ledger entries
  log_clock=0.,  # Ratekeeper and CI.apply's now_nanos (sim=False) read time.monotonic
  init=lambda g: fingerprint_callback(g.alpha_long),
  encode=encode,
  params={
    'IsMetric': [False, True], 'OpenpilotEnabledToggle': [False, True, True, True], 'IsReleaseBranch': [False, True],
    'InitialCruiseSpeedKph': [-1, 0, 8, 39, 40, 41, 100, 145, 146, 255, 2 ** 31 - 1],
    'SecOCKey': ['', '0' * 32, '0123456789abcdef' * 2, 'zz' * 16, 'ab' * 15, 'ab' * 17],
    'CarParamsCache': ['cp'], 'CarParamsPersistent': ['cp', 'garbage'], 'ObdMultiplexingEnabled': [False, True],
  },
  bounds={'carControl.orientationNED.len': [0, 3], 'carControl.angularVelocity.len': [0, 3],
          'carState.vEgo': [0., 8 * CV.KPH_TO_MS, 40 * CV.KPH_TO_MS, 145 * CV.KPH_TO_MS],
          'carState.cruiseState.speed': [V_CRUISE_MIN * CV.KPH_TO_MS, V_CRUISE_MAX * CV.KPH_TO_MS, V_CRUISE_UNSET * CV.KPH_TO_MS]},
  # not canErrorCounter: lockstep never drains empty (UNREACHED)
  checks=[alive, P.finite('carState', allow=('cumLagMs',)), P.finite('carOutput', degraded=lambda m: False), P.finite('radarTracks'),
          vcruise(False), vcruise(True), set_speed_needs_button, outputs_valid],
)
