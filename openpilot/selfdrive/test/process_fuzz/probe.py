# Child-side instrumentation, installed through ProcessConfig.launcher_wrapper. Everything goes to an append-only file, one
# line per new item, so a crash or SIGKILL loses nothing:
#   B <file> <qualname> <firstlineno> <offset> <dest>   a branch direction (sys.monitoring BRANCH)
#   R <path>                                         a read: message field, SubMaster metadata (@...), CP.<field>, P <param>
# sim=False (or Spec.log_clock) puts time.monotonic on the replayed logMonoTime, so SubMaster's alive/freq_ok are real instead
# of SIMULATION's "seen".
# Messages the process builds are stamped on the replayed clock too (last input + processing time, as process replay restamps
# its outputs), so times the process derives from its own stamps (processingDelay) are what they'd be on a device.
import dis
import os
import sys

import capnp

from openpilot.selfdrive.test.process_fuzz.consts import module_of

M = sys.monitoring
TOOL = 4
JUMPS = {'POP_JUMP_IF_FALSE': ('F', 'T'), 'POP_JUMP_IF_TRUE': ('T', 'F'), 'POP_JUMP_IF_NONE': ('None', 'notNone'),
         'POP_JUMP_IF_NOT_NONE': ('notNone', 'None'), 'FOR_ITER': ('exit', 'iter')}


def branch_table(code, rel: str, lines: list[str]) -> dict:
  # {(qualname, firstlineno, offset): (id if jumped, id if fell through, fall-through offset)}, for one code object and
  # everything nested in it. Ids name the source line, not its number, so they survive line drift. Module bodies run at
  # import in the parent, before the probe, so they have no entries.
  out, n = {}, {}
  ins = list(dis.get_instructions(code))
  for k, i in enumerate(ins):
    if i.opname in JUMPS and code.co_name != '<module>':
      key = f'{rel}:{code.co_qualname}:"{lines[i.positions.lineno - 1].strip()}"'
      n[key] = n.get(key, -1) + 1
      key += f'#{n[key]}' if n[key] else ''
      j, f = JUMPS[i.opname]
      out[(code.co_qualname, code.co_firstlineno, i.offset)] = (f'{key}:{j}', f'{key}:{f}', ins[k + 1].offset)
  for c in code.co_consts:
    if hasattr(c, 'co_code'):
      out |= branch_table(c, rel, lines)
  return out


def wrap(launcher, out, files, reads, sim, spec, processing_time):
  # files: {absolute path: qualnames in scope, or None for the whole file}
  def run(*args, **kwargs):
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_APPEND)

    def w(s):
      os.write(fd, (s + '\n').encode())
    last = install_stamps(processing_time)
    install_env(sim, spec)
    if not sim or spec.log_clock is not None:
      install_log_clock(last, spec.log_clock or 0.)
    if files:
      install_branches(w, files)
    once = install_reads(w, spec.passthrough) if reads else None
    (spec.front(launcher, once) if spec.front else launcher)(*args, **kwargs)
  return run


def install_env(sim, spec):
  # sim=False is a device: no REPLAY. Scope modules read REPLAY/SIMULATION at import, which ran in the parent without them
  if not sim:
    os.environ.pop('REPLAY', None)
  for f in spec.scope:
    mod = sys.modules.get(module_of(f))
    for k in ('REPLAY', 'SIMULATION'):
      if isinstance(getattr(mod, k, None), bool):
        setattr(mod, k, sim)


def install_log_clock(last, offset):
  # time.monotonic on the replayed log time: the latest logMonoTime received (raw drains too: card's can), strictly increasing
  import time
  import openpilot.cereal.messaging as messaging
  clock = [0.]

  def monotonic():
    clock[0] = max(clock[0], last[0] / 1e9) + 1e-9
    return clock[0] + offset

  def drain_sock_raw(*a, _orig=messaging.drain_sock_raw, **kw):
    out = _orig(*a, **kw)
    for d in out:
      messaging.log_from_bytes(d)
    return out
  time.monotonic, messaging.drain_sock_raw = monotonic, drain_sock_raw


def install_stamps(processing_time):
  import openpilot.cereal.messaging as messaging
  last = [0]

  def lfb(dat, *a, _orig=messaging.log_from_bytes, **kw):
    m = _orig(dat, *a, **kw)
    if not a and not kw:  # an Event, not CarParams
      last[0] = max(last[0], m.logMonoTime)
    return m

  def new_message(*a, _orig=messaging.new_message, **kw):
    return _orig(*a, **({'logMonoTime': last[0] + int(processing_time * 1e9)} if last[0] else {}) | kw)
  messaging.log_from_bytes, messaging.new_message = lfb, new_message
  return last


def in_scope(files, filename, qualname):
  qn = files.get(filename, False)
  return qn is None or (qn is not False and any(qualname == q or qualname.startswith(q + '.') for q in qn))


def install_branches(w, files):
  # disable a location once both directions were seen (3.12 can only disable a whole branch instruction),
  # or right away if it's out of scope
  seen: dict = {}

  def branch(code, off, dest):
    d = seen.get((code, off))
    if d is None:
      if not in_scope(files, code.co_filename, code.co_qualname):
        return M.DISABLE
      d = seen[(code, off)] = set()
    if dest not in d:
      d.add(dest)
      w(f'B\t{code.co_filename}\t{code.co_qualname}\t{code.co_firstlineno}\t{off}\t{dest}')
    if len(d) == 2:
      return M.DISABLE

  M.use_tool_id(TOOL, 'process_fuzz')
  M.register_callback(TOOL, M.events.BRANCH, branch)
  M.set_events(TOOL, M.events.BRANCH)


# *** reads: SubMaster.__getitem__, recv_one/recv_sock/drain_sock and CarParams return recording proxies ***

PASSTHROUGH: set = set()  # struct paths assigned wholesale into a builder (radard's radarErrors = rr.errors): capnp rejects a proxy
STRUCTS = (capnp.lib.capnp._DynamicStructReader, capnp.lib.capnp._DynamicStructBuilder)
LISTS = (capnp.lib.capnp._DynamicListReader, capnp.lib.capnp._DynamicListBuilder)


class Struct:
  __slots__ = ('_r', '_p', '_w')

  def __init__(self, r, p, w):
    object.__setattr__(self, '_r', r)
    object.__setattr__(self, '_p', p)
    object.__setattr__(self, '_w', w)

  def __getattr__(self, k):
    v = getattr(self._r, k)
    if (q := f'{self._p}.{k}' if self._p else k) in PASSTHROUGH:
      self._w(q + '.*')
      return v
    if callable(v):  # which() is the union tag, anything else (to_dict(), as_builder(), ...) uses the whole struct
      self._w(f'{self._p}.{k}()' if k == 'which' else f'{self._p}.*')
      return v
    return wrap_value(v, f'{self._p}.{k}' if self._p else k, self._w)

  def __setattr__(self, k, v):  # a builder (card's CarState): writes go through
    setattr(self._r, k, v)

  def __eq__(self, o):
    return self._r == (o._r if isinstance(o, Struct) else o)

  def __hash__(self):
    return hash(self._r)


class List:
  __slots__ = ('_r', '_p', '_w')

  def __init__(self, r, p, w):
    self._r, self._p, self._w = r, p, w

  def __len__(self):
    self._w(f'{self._p}.len')
    return len(self._r)

  def __getitem__(self, i):
    if isinstance(i, slice):
      return [wrap_value(v, f'{self._p}[]', self._w) for v in self._r[i]]
    return wrap_value(self._r[i], f'{self._p}[]', self._w)

  def __iter__(self):
    self._w(f'{self._p}.len')
    return (wrap_value(v, f'{self._p}[]', self._w) for v in self._r)

  def __array__(self, dtype=None, copy=None):
    import numpy as np
    self._w(f'{self._p}[]')
    return np.array(list(self._r), dtype=dtype)


def wrap_value(v, p, w):
  if isinstance(v, STRUCTS):
    return Struct(v, p, w)
  if isinstance(v, LISTS):
    return List(v, p, w)
  w(p)
  return v


def install_reads(w, passthrough=()):
  import openpilot.cereal.messaging as messaging
  from openpilot.common.params import Params
  from opendbc.car.structs import car
  seen = set()
  PASSTHROUGH.update(passthrough)

  def once(s):
    if s not in seen:
      seen.add(s)
      w('R ' + s)

  def param(orig):
    def g(self, key, *a, **kw):
      once(f'P {key}')
      return orig(self, key, *a, **kw)
    return g
  Params.get, Params.get_bool = param(Params.get), param(Params.get_bool)

  def lfb(dat, struct=None, _orig=messaging.log_from_bytes):
    r = _orig(dat) if struct is None else _orig(dat, struct)
    return Struct(r, 'CP', once) if struct is car.CarParams else r
  messaging.log_from_bytes = lfb

  def recv(orig):  # sockets read outside SubMaster (selfdrived's carState, card's can, locationd's sensors)
    def r(*a, **kw):
      m = orig(*a, **kw)
      return None if m is None else [Struct(x, '', once) for x in m] if isinstance(m, list) else Struct(m, '', once)
    return r
  messaging.recv_one, messaging.recv_sock, messaging.drain_sock = recv(messaging.recv_one), recv(messaging.recv_sock), recv(messaging.drain_sock)

  def outside():  # not SubMaster's own use
    return not sys._getframe(2).f_code.co_filename.endswith('messaging/__init__.py')

  class Rec(dict):  # SubMaster metadata dicts, read by key
    name = ''

    def __getitem__(self, s):
      if outside():
        once(f'@{self.name}[{s}]')
      return dict.__getitem__(self, s)

  sm = messaging.SubMaster

  def update_msgs(self, *a, _orig=sm.update_msgs):
    _orig(self, *a)
    for n in ('valid', 'alive', 'freq_ok', 'updated', 'seen', 'logMonoTime', 'recv_frame', 'recv_time'):
      d = Rec(getattr(self, n))
      d.name = n
      setattr(self, n, d)
  sm.update_msgs = update_msgs
  sm.__getitem__ = lambda self, s: wrap_value(self.data[s], s, once)

  def check(orig):
    def f(self, service_list=None):
      if outside():
        once(f'@{orig.__name__}({",".join(sorted(service_list or self.services))})')
      return orig(self, service_list)
    return f
  for name in ('all_alive', 'all_freq_ok', 'all_valid', 'all_checks'):
    setattr(sm, name, check(getattr(sm, name)))
  return once
