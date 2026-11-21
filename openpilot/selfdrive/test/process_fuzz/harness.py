# Process fuzz: drive a process's real main() over its whole input space through process replay, assert generic
# properties for every input. A process is procs/<name>.py defining `SPEC = Spec(...)`.
import functools
import importlib
import json
import os
import tempfile
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field

import openpilot.selfdrive.test.process_replay.process_replay as pr
from openpilot.cereal import log
from openpilot.common.params import Params
from openpilot.selfdrive.test.process_fuzz import probe, space as S
from openpilot.selfdrive.test.process_fuzz.consts import module_of

PROCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'procs')


@dataclass
class Spec:
  proc: str                                # process replay config name: its pubs are the inputs, its subs the outputs
  scope: dict[str, set | None]             # {file: None (whole file) | {qualnames}}: whose branches count toward coverage
  checks: list[Callable]                   # properties (properties.py)
  ledger: dict[str, tuple] = field(default_factory=dict)     # {'path[:cls|cls]': (kind, why, upstream 'proc: check' | None)}
  unreached: dict[str, list] = field(default_factory=dict)   # {reason: [branch id | '*' pattern]}
  known: dict[str, tuple] = field(default_factory=dict)      # {failing check name: ('real' | 'theoretical', why, genome file)}
  template: Callable = S.new_message       # (svc, g=None) -> builder: the producer's nominal message
  duration: float | tuple = 10.            # tuple: draws pick one; [0] is nominal and sets gene timing
  examples: int = 64                       # fresh random examples in CI
  params: dict[str, list] = field(default_factory=dict)      # param domains, None = unset
  bounds: dict[str, list] = field(default_factory=dict)      # extra boundaries per field, for aliases harvesting misses
  relations: dict[str, list] = field(default_factory=dict)   # {field: [(other field, k)]}: genes on field - k * other
  weights: dict[str, list] = field(default_factory=dict)     # per-element list weights {path: [pool per element]}
  image: dict[str, tuple] = field(default_factory=dict)      # {path: (lo, hi, points)}: drawn from and clamped to a producer's image
  few: float = 0.                          # P(a draw keeps a few genes + excite + <= 1 fault)
  excite: Callable | None = None           # (rng, g) -> {path: Gene}, added on sparse draws
  cp_class: Callable | None = None         # CarParams -> group key for draws; default: the read CP fields' values
  tag: Callable | None = None              # (space, g, ins | None) -> label | None: failures on a tagged run are '<check> (<label>)'
  huge: float = S.F32                      # |x| above this is 'huge': ledger classes and Space.bad
  extra_pubs: tuple = ()                   # services a device publishes to it that CONFIGS omits
  optional_pubs: tuple = ()                # published only if a gene or a fault names the service
  log_clock: float | None = None           # child time.monotonic = latest replayed logMonoTime + this (sim=False: always, 0)
  passthrough: tuple = ()                  # struct reads assigned wholesale into a builder: traced as '<path>.*'
  timeout: int | None = None               # process replay's per-step timeout, seconds
  space: Callable = S.Space                # Space class for processes whose input isn't a plain message stream
  front: Callable | None = None            # child: front(launcher, once) -> launcher
  init: Callable | None = None             # genome -> process replay init_callback (default: write its CarParams)
  encode: Callable | None = None           # (value, genome) -> value, before encode_param's shared forms

  def __post_init__(self):
    from openpilot.selfdrive.test.process_fuzz import properties as P
    self.checks = self.checks + [P.time_fields(s) for s in pr.get_process_config(self.proc).subs if any(map(S.is_time, S.deep_fields(s)))]

  @functools.cached_property
  def reads(self) -> set:
    # the traced read set, written by fuzz.py update (a new process has none yet)
    path = os.path.join(PROCS, f'{self.proc}_reads.json')
    return set(json.load(open(path))) if os.path.exists(path) else set()


@dataclass
class Result:
  spec: Spec
  genome: S.Genome
  ins: list
  outs: list
  err: str | None
  branches: set
  reads: set
  tag: str | None = None
  cache: dict = field(default_factory=dict)  # memo for P.cycles and reruns

  @property
  def space(self) -> S.Space:
    return space(self.spec)

  @functools.cached_property
  def cp(self):
    return S.get_car_params(self.genome.fp, self.genome.alpha_long, self.genome.fpv)

  def rerun(self, g=None, transform=None) -> 'Result':
    # the same genome (or g) again, messages optionally transformed: differential and metamorphic checks
    g = g or self.genome
    k = (json.dumps(g.to_dict(), sort_keys=True), transform and transform.__qualname__)
    if k not in self.cache:
      self.cache[k] = run(self.spec, g, transform=transform)
    return self.cache[k]


SPACES: dict = {}
TABLES: dict = {}


def space(spec: Spec) -> S.Space:
  if spec.proc not in SPACES:
    SPACES[spec.proc] = spec.space(spec)
  return SPACES[spec.proc]


def config(spec: Spec, g: S.Genome):
  cfg = pr.get_process_config(spec.proc)
  cfg.pubs = sorted(set(cfg.pubs) | set(spec.extra_pubs) | set(spec.optional_pubs))  # not the space's: card folds carState into can
  cfg.timeout = spec.timeout or cfg.timeout
  cfg.simulation = g.sim
  cfg.init_callback = spec.init(g) if spec.init is not None else car_params_callback(S.get_car_params(g.fp, g.alpha_long, g.fpv))
  return cfg


def scope_files(spec: Spec) -> dict:
  return {importlib.import_module(module_of(f)).__file__: qn for f, qn in spec.scope.items()}


def branches(spec: Spec) -> tuple[dict, set]:
  # ({(file, qualname, firstlineno, offset): (jump id, fall id, fall offset)}, every in-scope direction id)
  if spec.proc not in TABLES:
    table, rels = {}, {v: k for k, v in zip(spec.scope, scope_files(spec), strict=True)}
    for f, qn in scope_files(spec).items():
      src = open(f).read()
      for (q, ln, off), ids in probe.branch_table(compile(src, f, 'exec'), rels[f], src.splitlines()).items():
        if probe.in_scope({f: qn}, f, q):
          table[(f, q, ln, off)] = ids
    TABLES[spec.proc] = table, {i for ids in table.values() for i in ids[:2]}
  return TABLES[spec.proc]


def car_params_callback(CP):
  # get_car_params_callback with any CarParams (alpha long, passive, ...)
  def cb(rc, pm, msgs, fingerprint):
    Params().put("CarParams", CP.to_bytes(), block=True)
  return cb


def encode_param(v, g, spec):
  # JSON genome param -> what Params.put takes. Byte params: {'event': {svc: fields}}, {'car': 'same' | platform} (CarParams),
  # {'raw': hex}; None = unset
  if spec.encode is not None:
    v = spec.encode(v, g)
  if isinstance(v, dict) and 'event' in v:
    return log.Event.new_message(**v['event']).to_bytes()
  if isinstance(v, dict) and 'car' in v:
    return (S.get_car_params(g.fp, g.alpha_long, g.fpv) if v['car'] == 'same' else S.get_car_params(v['car'], False)).to_bytes()
  if isinstance(v, dict) and 'raw' in v:
    return bytes.fromhex(v['raw'])
  return v


def run(spec: Spec, g: S.Genome, cov: bool = False, reads: bool = False, transform=None) -> Result:
  cfg, msgs = config(spec, g), space(spec).messages(g)
  msgs = transform(msgs) if transform else msgs
  fd, out = tempfile.mkstemp(prefix=f'process_fuzz_{spec.proc}_', dir='/dev/shm')
  os.close(fd)
  cfg.launcher_wrapper = functools.partial(probe.wrap, out=out, files=scope_files(spec) if cov else None, reads=reads, sim=g.sim,
                                           spec=spec, processing_time=cfg.processing_time)
  params = {k: v for k, v in ((k, encode_param(v, g, spec)) for k, v in g.params.items()) if v is not None}
  outs, err, store = [], None, {}
  try:
    outs = pr._replay_multi_process([cfg], msgs, getattr(msgs, 'frs', None), g.fp, params, store, True)
  except Exception:
    err = traceback.format_exc(limit=-2)
    if cfg.proc_name in store:
      err += '\nprocess stderr:\n' + '\n'.join(store[cfg.proc_name]['err'].strip().splitlines()[-12:])
  br, rd = parse(spec, out, cov)
  os.unlink(out)
  return Result(spec, g, msgs, outs, err, br, rd, spec.tag(space(spec), g, msgs) if spec.tag else None)


def parse(spec, path, cov):
  br, rd = set(), set()
  table = branches(spec)[0] if cov else {}
  with open(path) as f:
    lines = f.read().splitlines()
  for ln in lines:
    k, _, rest = ln.partition(' ' if ln.startswith('R') else '\t')
    if k == 'R':
      rd.add(rest)
    elif k == 'B':
      f, q, first, off, dest = rest.split('\t')
      ids = table.get((f, q, int(first), int(off)))
      if ids is not None:
        br.add(ids[1] if int(dest) == ids[2] else ids[0])
  return br, rd
