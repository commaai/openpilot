# The input space of one process and genomes over it. A genome is a JSON-able description of one example: a CarParams
# (fingerprint, alpha long, fingerprint variant), params, SubMaster mode, one temporal gene per fuzzed read field, and message
# faults. Only fields the process reads get genes; everything else keeps its template value. Values cover each field's whole
# capnp type: specials, partition boundaries (harvested + Spec.bounds) with float32 neighbours, random magnitudes.
import functools
import math
import random
from dataclasses import asdict, dataclass, field

import capnp
import numpy as np

import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from openpilot.cereal.services import SERVICE_LIST
from openpilot.common.fuzzy import _INTEGER_RANGES as INT_RANGE, Fuzzy
from opendbc.car import structs
from opendbc.car.car_helpers import interfaces
from opendbc.car.fw_versions import FW_QUERY_CONFIGS, FW_VERSIONS, MODEL_TO_BRAND
from opendbc.car.interfaces import gen_empty_fingerprint
from openpilot.selfdrive.test.process_fuzz.consts import harvest

T0 = 1_000_000_000  # logMonoTime of tick 0, must be > 0
DT = 0.01
F32 = float(np.finfo(np.float32).max)
TEXT = ['', 'Active', 'maneuver Active', 'x', 'modeld', 'a' * 200]
SHAPES = ('hold', 'step', 'osc', 'spike', 'ramp', 'hover')
FAULTS = ('drop', 'invalid', 'stale', 'dup', 'slow', 'trigger_first', 'late')  # late: content k periods old, fresh logMonoTime
LIST_MODES = ('same', 'same', 'lin', 'alt', 'one')
DISCRETE = ('bool', 'enum', 'union', 'len', 'text', 'data')
FEW = (0, 1, 1, 2, 2, 3, 4, 6, 10, 20)  # genes kept by a sparse draw


@dataclass
class Gene:
  shape: str
  a: float
  b: float
  t0: float = 0.
  t1: float = 0.
  # list elements: same value, scaled by index, alternating sign, only the first, 'w:a,b,c' per-element weights;
  # 'lmt': the value is an offset from the message's own logMonoTime
  idx: str = 'same'
  rel: str | None = None  # value is k * rel's value + this gene (cross-field classes like v_cruise - v_ego)
  k: float = 1.

  def at(self, t):
    s, a, b = self.shape, self.a, self.b
    if s == 'hold':
      return a
    if s == 'step':
      return a if t < self.t0 else b
    if s == 'spike':
      return b if self.t0 <= t < self.t1 else a
    if s == 'osc':  # square wave, half period t1
      return a if (t // max(self.t1, DT)) % 2 == 0 else b
    if s == 'ramp':
      return a + (b - a) * min(max((t - self.t0) / max(self.t1 - self.t0, DT), 0.), 1.)
    if s == 'hover':  # a +- b, period t1
      return a + b * math.sin(2 * math.pi * t / max(self.t1, DT))
    raise ValueError(s)


@dataclass
class Genome:
  fp: str
  alpha_long: bool = False
  sim: bool = True  # SIMULATION=1: SubMaster alive = seen, freq_ok = True
  params: dict = field(default_factory=dict)
  dur: float = 10.
  genes: dict = field(default_factory=dict)   # path -> Gene
  faults: list = field(default_factory=list)  # [svc, kind, t0, t1, k]
  perturb: list | None = None  # [path, delta]: +-delta alternating each trigger cycle, for hf_gain
  fpv: str = ''  # fingerprint variant, see cp_variants

  def to_dict(self):
    return asdict(self)

  @staticmethod
  def from_dict(d):
    d = dict(d)
    d['genes'] = {p: Gene(**g) for p, g in d['genes'].items()}
    return Genome(**d)


def leaf_type(path):
  # capnp type of a read path like 'modelV2.leadsV3[].x[]': ('float32', None), ('enum', [names]), ('len', None), ...
  if path.endswith('.len'):
    return 'len', None
  s = log.Event.schema
  toks = path.split('.')
  for k, tok in enumerate(toks):
    if tok.endswith('()'):
      return 'union', list(s.union_fields)
    f = s.fields[tok.removesuffix('[]')]
    if f.proto.which() == 'group':
      s = f.schema
      continue
    t = f.proto.slot.type
    w = t.which()
    sch = f.schema if w in ('struct', 'list', 'enum') else None
    if w == 'list':
      w = t.list.elementType.which()
      sch = sch.elementType if w in ('struct', 'enum') else None
    if w == 'struct':
      s = sch
      continue
    assert k == len(toks) - 1, path
    return w, (list(sch.enumerants) if w == 'enum' else None)
  return 'struct', None


def deep_fields(path, depth=None):
  # every leaf under a struct ('carControl.**': a CarController reads what its platform needs), deprecated ones aside;
  # depth=1: its direct fields ('.*')
  s = log.Event.schema
  for tok in path.split('.'):
    s = s.fields[tok.removesuffix('[]')].schema
    s = getattr(s, 'elementType', s)
  if depth == 1:
    return [f'{path}.{n}' for n in s.fields]
  out = []
  for n, f in s.fields.items():
    if 'deprecated' in n.lower():
      continue
    q = f'{path}.{n}'
    w = f.proto.which() == 'slot' and f.proto.slot.type.which()
    if w == 'list':
      e = f.proto.slot.type.list.elementType.which()
      out += [q + '.len'] + (deep_fields(q + '[]') if e == 'struct' else [q + '[]'])
    else:
      out += deep_fields(q) if w in (False, 'struct') else [q]
  return out


def expand(r):
  # a whole-struct read: '.*' each field, '.**' each leaf
  if r.startswith('CP.'):
    return {r}
  if r.endswith('.**'):
    return set(deep_fields(r[:-3]))
  return set(deep_fields(r[:-2], 1)) if r.endswith('.*') else {r}


def svc_of(path):
  return path.split('.')[0].removesuffix('[]')


def is_time(path):
  k = path.rsplit('.', 1)[-1].split('[')[0]
  return k.startswith('timestamp') or k.endswith(('Delay', 'ExecutionTime', 'MonoTime'))


# *** CarParams: every platform, alpha long, and fingerprint variants (a fingerprint-derived CP an empty fingerprint misses) ***

def extra_ecus(fp):
  cfg = FW_QUERY_CONFIGS.get(MODEL_TO_BRAND.get(fp))
  return cfg.extra_ecus if cfg else []


def cp_variants(fp):
  # '': empty CAN and FW; 'can': every address on every bus; 'fw': every ECU with its first version; 'fw:...': that one version;
  # 'extra': the brand's extra_ecus present (any version: get_params keys on presence, toyota/interface.py:57)
  out = ['', 'can', 'fw', 'can+fw'] + (['fw+extra'] if extra_ecus(fp) else [])
  for (ecu, addr, sub), vers in sorted(FW_VERSIONS.get(fp, {}).items(), key=str):
    out += [f'fw:{int(ecu)}:{addr}:{sub}:{v.hex()}' for v in vers]
  return out


def car_fw(fp, fpv):
  fws = []
  for part in fpv.split('+'):
    if part == 'fw':
      fws += [(ecu, addr, sub, vers[0]) for (ecu, addr, sub), vers in FW_VERSIONS.get(fp, {}).items()]
    elif part == 'extra':
      fws += [(ecu, addr, sub, b'\x00') for ecu, addr, sub in extra_ecus(fp)]
    elif part.startswith('fw:'):
      ecu, addr, sub, ver = part[3:].split(':')
      fws.append((int(ecu), int(addr), None if sub == 'None' else int(sub), bytes.fromhex(ver)))
  return [structs.CarParams.CarFw.new_message(ecu=int(e), address=a, subAddress=s or 0, fwVersion=v) for e, a, s, v in fws]


@functools.cache
def _car_params(fp, alpha_long, fpv=''):
  fingerprint = {b: dict.fromkeys(range(0x800), 8) for b in range(8)} if 'can' in fpv.split('+') else gen_empty_fingerprint()
  return interfaces[fp].get_params(fp, fingerprint, car_fw(fp, fpv), alpha_long, False, False)


def get_car_params(fp, alpha_long, fpv=''):
  return _car_params(fp, alpha_long, fpv).copy()


def cp_get(CP, path):
  v = CP
  for tok in path.split('.'):
    if tok == 'len':
      v = len(v)
    elif tok == 'which()':
      v = v.which()
    elif tok == '*':
      v = str(v)
    elif isinstance(v, (list, tuple)):
      v = tuple(getattr(x, tok.removesuffix('[]')) for x in v)
    else:
      try:
        v = getattr(v, tok.removesuffix('[]'))
      except capnp.KjException:  # a union member that isn't set
        return None
      v = list(v) if tok.endswith('[]') else v
  return v


@functools.cache
def cp_groups(key):
  # every (fp, alpha_long, fpv), MOCK included, grouped by key(CP) (or by the values of a tuple of CP fields)
  groups = {}
  for fp in sorted(interfaces):
    for alpha in (False, True):
      for fpv in cp_variants(fp):
        try:
          CP = _car_params(fp, alpha, fpv)
        except Exception:  # not a CarParams get_car can return
          continue
        k = key(CP) if callable(key) else tuple(str(cp_get(CP, f)) for f in key)
        groups.setdefault(k, []).append((fp, alpha, fpv))
  return sorted(groups.values())



def classes_of(v, huge):
  # the value classes a ledger can name: nan, inf (or rounds to inf in float32), huge, nonpositive
  out = {'nan' if v != v else 'inf' if abs(v) > F32 else 'finite'}
  if v == v and abs(v) > huge:
    out.add('huge')
  if v <= 0 or v != v:
    out.add('nonpositive')
  return out


def values(obj, path):
  # every leaf under path of a struct: '[]' flattens, '*' is the active union member, '.len' has none
  toks = path.split('.') if isinstance(path, str) else path
  if not toks:
    return list(obj) if isinstance(obj, (list, capnp.lib.capnp._DynamicListReader)) else [obj]
  t = toks[0]
  if t == 'len':
    return []
  if t == '*':
    return values(getattr(obj, obj.which()), toks[1:])
  if t.removesuffix('[]') in obj.schema.union_fields and obj.which() != t.removesuffix('[]'):
    return []
  if t.endswith('[]'):
    lst = getattr(obj, t[:-2])
    return [x for el in lst for x in values(el, toks[1:])] if len(toks) > 1 else list(lst)
  return values(getattr(obj, t), toks[1:])


class Space:
  def __init__(self, spec):
    from openpilot.selfdrive.test.process_replay.process_replay import get_process_config
    self.spec = spec
    cfg = get_process_config(spec.proc)
    self.pubs = sorted(set(cfg.pubs) | set(spec.extra_pubs) | set(spec.optional_pubs))
    self.trigger = getattr(cfg.should_recv_callback, 'trigger_msg_type', None)
    self.durs = spec.duration if isinstance(spec.duration, tuple) else (spec.duration,)
    self.dur = self.durs[0]
    self.weights = spec.weights
    # path -> value classes an upstream guarantee rules out ('nan'/'inf'/'huge'/'nonpositive'/'outofrange'/...), and the ones
    # it files as theoretical (still fuzzed)
    self.excluded, self.theo = {}, {}
    for k, v in spec.ledger.items():
      if ':' in k:
        p, cs = k.split(':')
        for q in expand(p):
          (self.theo if v[0] == 'theoretical' else self.excluded).setdefault(q, set()).update(cs.split('|'))
    reads = set().union(*map(expand, spec.reads))  # a whole-struct use reads every field of it
    fixed = {k for k, v in spec.ledger.items() if ':' not in k and v[0] != 'theoretical'}
    self.leaves = {r: leaf_type(r) for r in sorted(reads) if svc_of(r) in self.pubs and r not in fixed}
    self.lmt = {p for p, (k, _) in self.leaves.items() if k.startswith(('int', 'uint')) and is_time(p)}
    self.glob, self.by_name = harvest(tuple(spec.scope))
    self.cp_fields = tuple(sorted(r[3:] for r in spec.reads if r.startswith('CP.') and r not in spec.ledger))
    self.groups = cp_groups(spec.cp_class or self.cp_fields)
    self.cp_consts = {}

  def boundaries(self, path, g=None):
    name = path.split('.')[-1].removesuffix('[]')
    b = set(self.by_name.get(name, ())) | set(self.spec.bounds.get(path, ())) | set(self.cp_bounds(g))
    return sorted(x for x in b if math.isfinite(x))  # a CP value can be inf: specials cover it

  def cp_bounds(self, g):
    # CP values are boundaries too (vEgo < CP.minEnableSpeed + 0.5)
    if g is None:
      return ()
    k = (g.fp, g.alpha_long, g.fpv)
    if k not in self.cp_consts:
      CP, small = _car_params(*k), [c for c in self.glob if 0 < c <= 5]
      vals = {float(v) for f in self.cp_fields if '.' not in f and isinstance(v := getattr(CP, f), (int, float)) and not isinstance(v, bool)}
      self.cp_consts[k] = sorted(vals | {v + c for v in vals for c in small} | {v - c for v in vals for c in small})
    return self.cp_consts[k]

  # *** values: specials, partition boundaries +- eps, random magnitudes ***
  def value(self, rng, path, g=None):
    if path in self.spec.image:  # enforced by the producer: its image, boundaries and interior
      lo, hi, pts = self.spec.image[path]
      return float(rng.choice(pts)) if rng.random() < 0.7 else rng.uniform(lo, hi)
    kind, extra = self.leaves[path]
    ex = self.excluded.get(path, set())
    if kind == 'bool':
      return rng.random() < 0.5
    if kind == 'union':
      return rng.choice(extra)
    if kind == 'enum':
      return rng.choice(extra if 'outofrange' in ex or rng.random() < 0.9 else [len(extra), 2 ** 16 - 1])
    if kind == 'len':  # 'shape' (ledger): only the producer's lengths, listed in Spec.bounds
      return rng.choice(self.spec.bounds[path] if 'shape' in ex else [0, 1, 2, 3, 8, 64])
    if kind == 'text':
      return rng.choice(TEXT)
    if kind == 'data':
      return rng.choice(['', '00', 'ff' * 8, bytes(rng.getrandbits(8) for _ in range(rng.choice([1, 8, 64]))).hex()])
    for _ in range(20):  # finite classes a ledger excludes ('huge', 'nonpositive') are redrawn
      v = self.number(rng, path, g, kind, ex)
      if not (classes_of(v, self.spec.huge) & ex):
        return v
    return 1.

  def number(self, rng, path, g, kind, ex):
    local = self.boundaries(path, g)
    pool = local if local and rng.random() < 0.6 else sorted(self.glob | set(local))
    r = rng.random()
    fz = Fuzzy(rng.getrandbits(64), rng.randrange(1, 33))  # type edges and random bit patterns
    if kind.startswith(('int', 'uint')):
      lo, hi = INT_RANGE[kind]
      if r < 0.3:
        return fz.integer(lo, hi)
      if r < 0.8 and pool:
        return int(min(max(round(rng.choice(pool)) + rng.choice([-1, 0, 0, 1]), lo), hi))
      return rng.randint(max(lo, -1000), min(hi, 1000))
    if r < 0.25:
      return fz.floating(32, allow_nan='nan' not in ex, allow_infinity='inf' not in ex)
    if r < 0.75 and pool:
      c = rng.choice(pool)
      return float(rng.choice([c, c, np.nextafter(np.float32(c), np.float32(math.inf)), np.nextafter(np.float32(c), np.float32(-math.inf)),
                               c * 1.01 + 1e-3, c * 0.99 - 1e-3, -c]))
    return rng.choice([-1, 1]) * 10 ** rng.uniform(-3, 4)

  def gene(self, rng, path, g=None):
    kind = self.leaves[path][0]
    numeric = kind not in DISCRETE
    shape = rng.choice(SHAPES if numeric else SHAPES[:4])
    t0 = rng.uniform(0, self.dur)
    f = SERVICE_LIST[svc_of(path)].frequency if svc_of(path) in SERVICE_LIST else 0
    per = 1. / f if f > 0 else 1.  # one message wide: a single-sample blip on any service
    t1 = rng.choice([DT, 0.02, 0.1, 0.5, 2., per]) if shape in ('osc', 'hover') else t0 + rng.choice([DT, 0.05, 0.2, 1., 3., per])
    ge = Gene(shape, self.value(rng, path, g), self.value(rng, path, g), t0, t1)
    if shape == 'hover':
      ge.b = abs(ge.a) * rng.choice([1e-4, 1e-2, 0.1]) + rng.choice([1e-3, 0.05]) if math.isfinite(ge.a) else 1.
    if '[]' in path:
      ge.idx = rng.choice(LIST_MODES)
    rels = self.spec.relations.get(path, [])
    if numeric and rels and rng.random() < 0.4:
      small = [c for c in self.glob if abs(c) <= 5] + [-c for c in self.glob if 0 < c <= 5] + [0.]
      ge.rel, ge.k = rng.choice(rels)
      ge.a, ge.b = rng.choice(small), rng.choice(small)
    if path in self.weights and rng.random() < 0.6:
      ws = [self.weight(rng, pool) for pool in self.weights[path]]
      if self.excluded.get(path, set()) & {'nan', 'inf'}:  # an upstream guarantee rules out nan/inf: no non-finite weights either
        ws = [w if math.isfinite(w) else 1. for w in ws]
      ge.idx = 'w:' + ','.join(repr(float(w)) for w in ws)
    if path in self.lmt and rng.random() < 0.9:
      ge.idx = 'lmt'
    return ge

  def weight(self, rng, pool):
    r = rng.random()
    if r < 0.6 and pool:
      c = rng.choice(pool)
      return rng.choice([c, c, np.nextafter(np.float32(c), np.float32(math.inf)), np.nextafter(np.float32(c), np.float32(-math.inf)), -c])
    if r < 0.8:
      return rng.choice([0., 1., -1., math.nan, math.inf, -math.inf, 1e-6])
    return rng.choice([-1, 1]) * 10 ** rng.uniform(-4, 1)

  def fault(self, rng):
    t0 = rng.choice([0., rng.uniform(0, self.dur)])
    return [rng.choice(self.pubs), rng.choice(FAULTS), t0, t0 + rng.choice([0.05, 0.3, 1., 3., self.dur]), rng.choice([2, 3, 10])]

  def draw(self, rng: random.Random):
    # half the draws of a spec with a tag are redrawn until untagged (inside every producer's image), so checks on untagged
    # runs get power: most whole-space draws touch some theoretical class
    g = self.draw_one(rng)
    if self.spec.tag and rng.random() < 0.5:
      for _ in range(100):
        if not self.spec.tag(self, g, None):
          break
        g = self.draw_one(rng)
    return g

  def draw_one(self, rng):
    fp, alpha, fpv = rng.choice(rng.choice(self.groups))
    g = Genome(fp, alpha, rng.random() < 0.5, {k: rng.choice(v) for k, v in self.spec.params.items()}, rng.choice(self.durs), fpv=fpv)
    for p in rng.sample(list(self.leaves), rng.randint(0, len(self.leaves))):
      g.genes[p] = self.gene(rng, p, g)
    g.faults = [self.fault(rng) for _ in range(rng.choice([0, 0, 1, 2, 3]))]
    for p in sorted(self.lmt):  # timestamps: mostly offsets from logMonoTime (absolute values stay reachable through genes)
      if p not in g.genes and rng.random() < 0.9:
        g.genes[p] = Gene('hold', 0., 0., idx='lmt')
    if rng.random() < self.spec.few:  # sparse: a few genes on the producers' nominal template, plus an excitation
      keep = set(rng.sample(sorted(g.genes), min(len(g.genes), rng.choice(FEW))))
      g.genes = {p: ge for p, ge in g.genes.items() if p in keep or ge.idx == 'lmt'}
      for p, ge in (self.spec.excite(rng, g) if self.spec.excite else {}).items():
        g.genes.setdefault(p, ge)
      g.faults = g.faults[:rng.choice([0, 0, 1])]
    return g

  def theoretical(self, g):
    # a float gene takes a class the ledger files as theoretical for its path (list weights included)
    for p, ge in g.genes.items():
      cs = self.theo.get(p)
      if cs and isinstance(ge.a, float):
        w = [float(x) for x in ge.idx[2:].split(',')] if ge.idx.startswith('w:') else [1.]
        if any(classes_of(v * x, self.spec.huge) & cs for v in (ge.a, ge.b, ge.a + ge.b, ge.a - ge.b) for x in w):
          return True
    return False

  def bad(self, m):
    # a float this process reads from m is non-finite or past Spec.huge
    w = m.which()
    return any(not math.isfinite(x) or abs(x) > self.spec.huge for p, (k, _) in self.leaves.items() if svc_of(p) == w and k.startswith('float')
               for x in values(getattr(m, w), p.split('.')[1:]))

  # *** genome -> messages ***
  def messages(self, g: Genome):
    n = int(round(g.dur / DT))
    named = {svc_of(p) for p in g.genes} | {f[0] for f in g.faults}
    pubs = [s for s in self.pubs if s not in self.spec.optional_pubs or s in named]  # optional: only published if a genome names it
    period = {s: max(1, int(round(1. / (SERVICE_LIST[s].frequency * DT)))) if SERVICE_LIST[s].frequency > 0 else 100 for s in pubs}
    tmpl = {s: self.spec.template(s, g) for s in pubs}
    by_svc = {}
    for p in g.genes:
      if p in self.leaves:
        by_svc.setdefault(svc_of(p), []).append(p)
    if g.perturb and g.perturb[0] not in g.genes and g.perturb[0] in self.leaves:
      by_svc.setdefault(svc_of(g.perturb[0]), []).append(g.perturb[0])
    for v in by_svc.values():
      v.sort(key=lambda p: (not p.endswith('()'), not p.endswith('.len')))  # union tags, then lengths, then values
    tags = {p.rsplit('.', 1)[0] for p in g.genes if p.endswith('()')}  # unions whose variant a gene picks
    out, stale = [], {}
    trig_dt = period.get(self.trigger, 1) * DT
    for i in range(n):
      t = i * DT
      frame = []
      for svc in sorted(pubs, key=lambda s: s == self.trigger):
        kinds = {f[1]: f for f in g.faults if f[0] == svc and f[2] <= t < f[3]}
        if i % (period[svc] * (kinds['slow'][4] if 'slow' in kinds else 1)) or 'drop' in kinds:
          continue
        tt = stale.setdefault(svc, t) if 'stale' in kinds else t
        if 'stale' not in kinds:
          stale.pop(svc, None)
        if 'late' in kinds:
          tt = max(0., tt - kinds['late'][4] * period[svc] * DT)
        ev = tmpl[svc].copy()  # a fresh builder per message: re-initialised lists/unions never pile up in one arena
        ev.logMonoTime, ev.valid = T0 + i * int(DT * 1e9), 'invalid' not in kinds
        for p in by_svc.get(svc, ()):
          kind, ge = self.leaves[p], g.genes.get(p)
          v = self.at(g, p, tt)
          if g.perturb and g.perturb[0] == p:
            v = v + g.perturb[1] * (1 if int(round(tt / trig_dt)) % 2 == 0 else -1)
          idx = ge.idx if ge is not None else 'same'
          if idx == 'lmt' and math.isfinite(v):
            v, idx = ev.logMonoTime + v, 'same'
          if kind[0] == 'float32' and 'inf' in self.excluded.get(p, ()) and math.isfinite(v):  # no finite value rounds to inf
            v = min(max(v, -F32), F32)
          setpath(ev, p.split('.'), v, kind, idx, tags)
        m = log.Event.from_bytes(ev.to_bytes()).__enter__()
        frame += [m, m] if 'dup' in kinds else [m]
        if svc == self.trigger and 'trigger_first' in kinds:
          frame = frame[-1:] + frame[:-1]
      out += frame
    return out

  def at(self, g, p, t):
    ge = g.genes.get(p)
    if ge is None:
      return 0.
    if ge.rel is not None:  # a held relation lags its source by t0 (a response lagging its command)
      src = g.genes.get(ge.rel)
      v = ge.k * (src.at(t - ge.t0 if ge.shape == 'hold' else t) if src is not None else 0.) + ge.at(t)
    else:
      v = ge.at(t)
    if p in self.spec.image:
      v = min(max(v, self.spec.image[p][0]), self.spec.image[p][1])
    return v


def coerce(v, kind):
  if kind[0] in ('enum', 'union', 'bool', 'text'):
    return v
  if kind[0] == 'data':
    return bytes.fromhex(v) if isinstance(v, str) else v
  if kind[0] == 'len':
    return max(0, int(v)) if math.isfinite(v) else 0
  if kind[0].startswith(('int', 'uint')):
    lo, hi = INT_RANGE[kind[0]]
    return int(min(max(v, lo), hi)) if math.isfinite(v) else 0
  return float(v)


def element(v, j, n, idx, kind):
  # element j of n of a struct list (radar points, leads): ids/counters distinct, floats scaled (j+1)/n or alternating, bools alternating
  if kind == 'bool' and idx == 'alt':
    return v if j % 2 == 0 else not v
  if idx in ('same', 'one') or not kind.startswith(('int', 'uint', 'float')) or not math.isfinite(v):
    return v
  if kind.startswith(('int', 'uint')):
    return v + j if idx == 'lin' else v + j % 2 if idx == 'alt' else v
  return v * (j + 1) / n if idx == 'lin' else v * (-1) ** j if idx == 'alt' else v


def item(v, j, n, idx, kind):
  # element j of n of a primitive list, None = keep the template's
  if idx.startswith('w:'):
    w = [float(x) for x in idx[2:].split(',')]
    return v * w[j] if j < len(w) else None
  if idx == 'one' and j > 0:
    return None
  if not kind.startswith('float'):
    return v
  return v * j / max(n - 1, 1) if idx == 'lin' else v * (-1) ** j if idx == 'alt' else v


def setpath(obj, toks, v, kind, idx='same', tags=frozenset(), path=''):
  # the list mode applies at the innermost list of the path; outer struct lists get the same value per element
  name = toks[0]
  here = f'{path}.{name}' if path else name
  if name.endswith('[]'):
    lst = getattr(obj, name[:-2])
    n = len(lst)
    if len(toks) > 1:
      inner = any(t.endswith('[]') for t in toks[1:])
      for j, el in enumerate(lst):
        if inner:
          setpath(el, toks[1:], v, kind, idx, tags, here)
        elif idx != 'one' or j == 0:
          setpath(el, toks[1:], element(v, j, n, idx, kind[0]), kind, 'same', tags, here)
      return
    vals = [item(v, j, n, idx, kind[0]) for j in range(n)]
    setattr(obj, name[:-2], [coerce(x if x is not None else lst[j], kind) for j, x in enumerate(vals)])
  elif name.endswith('()'):  # union tag: keep the variant's content when unchanged, set (not init) primitive variants
    if obj.which() != v:
      try:
        obj.init(v)
      except Exception:
        setattr(obj, v, [] if obj.schema.fields[v].proto.slot.type.which() == 'list' else 0)
  elif len(toks) == 2 and toks[1] == 'len':
    obj.init(name, coerce(v, kind))
  elif len(toks) == 1:
    setattr(obj, name, coerce(v, kind))
  else:
    if name in obj.schema.union_fields and obj.which() != name:
      if path in tags:  # a gene picked another variant: this field is moot
        return
      obj.init(name)
    setpath(getattr(obj, name), toks[1:], v, kind, idx, tags, here)


def new_message(svc, g=None):
  try:
    return messaging.new_message(svc)
  except Exception:
    return messaging.new_message(svc, 1)
