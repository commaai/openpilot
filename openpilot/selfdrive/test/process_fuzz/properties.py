# Generic properties. A property is `check(r: Result)` with a .name: it raises AssertionError on a violation and may
# return a truthy label of what it applied to (falsy = vacuous); corpus selection keeps one example per label.
# Acceptable output for any input is correct or flagged degraded (valid=False); never crash, hang, NaN, unbounded
# commands, chatter, or valid output derived from invalid/stale inputs.
import bisect
import copy
import math
from collections import namedtuple

import numpy as np

from openpilot.cereal.services import SERVICE_LIST
from openpilot.selfdrive.test.process_fuzz.space import T0, values

# one trigger cycle: t (s), the trigger message, the last message of every service sent up to and incl. it (send order), the
# logMonoTime of the trigger that delivered each of those, and this cycle's output message of each output service
Cycle = namedtuple('Cycle', 't trig last recv out')


def prop(name):
  def deco(f):
    f.name = name
    return f
  return deco


def t_of(m) -> float:
  return (m.logMonoTime - T0) / 1e9


def get(m, path):
  for p in path.split('.'):
    m = getattr(m, m.which()) if p == '*' else getattr(m, p)
  return m


def outs(r, svc):
  return [m for m in r.outs if m.which() == svc]


def series(msgs, path):
  ms = [m for m in msgs if m.which() == path.split('.')[0]]
  return np.array([t_of(m) for m in ms]), [get(m, path) for m in ms]


def held(msgs, path, t):
  # value of `path` as last received at each time in t (zero-order hold), None before the first
  ts, xs = series(msgs, path)
  k = np.searchsorted(ts, t, side='right') - 1
  return [xs[j] if j >= 0 else None for j in k]


def held_for(t, mask, secs):
  # mask has been true continuously for at least secs at each t
  out, start = np.zeros(len(t), dtype=bool), None
  for k in range(len(t)):
    start = (start if start is not None else t[k]) if mask[k] else None
    out[k] = start is not None and t[k] - start >= secs
  return out


def around(t, x, k, n=3):
  return ', '.join(f'{t[j]:.2f}s={x[j]}' for j in range(max(0, k - n), min(len(x), k + n + 1)))


def nonfinite(d, path=''):
  if isinstance(d, float):
    return [] if math.isfinite(d) else [path]
  if isinstance(d, dict):
    return [p for k, v in d.items() for p in nonfinite(v, f'{path}.{k}')]
  if isinstance(d, list):
    return [p for j, v in enumerate(d) for p in nonfinite(v, f'{path}[{j}]')]
  return []


def ever(r, path, f):
  return any(f(x) for x in series(r.ins, path)[1])


def gated(service, gates, learned):
  # no learning unless every gate {input path: open(value)} is open at least once: else learned(first output, output) stays False
  @prop('gated learning')
  def check(r):
    shut, os_ = [p for p, f in gates.items() if not ever(r, p, f)], outs(r, service)
    if not shut or not os_:
      return None
    for m in os_:
      assert not learned(getattr(os_[0], service), getattr(m, service)), f't={t_of(m):.2f}s learned with {shut} never open'
    return 'shut'
  return check


def triggers(r, trigger=None):
  # [(trigger message, last message per service, logMonoTime of the trigger that delivered it)]: what each cycle received is
  # everything sent before its trigger, in send order (so trigger_first is handled)
  trigger = trigger or r.space.trigger
  if ('triggers', trigger) not in r.cache:
    last, recv, pending, out = {}, {}, set(), []
    for m in r.ins:
      last[m.which()] = m
      pending.add(m.which())
      if m.which() == trigger:
        recv |= dict.fromkeys(pending, m.logMonoTime)
        pending = set()
        out.append((m, dict(last), dict(recv)))
    r.cache[('triggers', trigger)] = out
  return r.cache[('triggers', trigger)]


def trigger_of(trig, os_, by_time):
  # index of the trigger each output is from: the k-th output is from the k-th trigger (process replay's lockstep), or with
  # by_time, for processes that publish on some triggers only, the last trigger at or before its stamp (trigger + processing time)
  ts = [m.logMonoTime for m, _, _ in trig]
  return [bisect.bisect_right(ts, o.logMonoTime) - 1 if by_time else k for k, o in enumerate(os_)]


def cycles(r, svcs, trigger=None, by_time=False, skip=0) -> list[Cycle]:
  # the cycles in which every service in svcs has an output; skip drops each service's first outputs
  key = ('cycles', tuple(svcs), trigger, by_time, skip)
  if key not in r.cache:
    trig = triggers(r, trigger)
    per = {}
    for s in svcs:
      os_ = outs(r, s)[skip:]
      per[s] = dict(zip(trigger_of(trig, os_, by_time), os_, strict=True))
    r.cache[key] = [Cycle(t_of(m), m, last, recv, {s: per[s][k] for s in svcs}) for k, (m, last, recv) in enumerate(trig)
                    if all(k in per[s] for s in svcs)]
  return r.cache[key]


def alive(service, trigger=None, every=1, name=None):
  # no exception or hang, and one output per `every` triggers (the last one may still be in flight)
  @prop(name or f'alive {service}')
  def check(r):
    assert r.err is None, r.err
    trig = trigger or r.space.trigger
    n, k = len(outs(r, service)), sum(m.which() == trig for m in r.ins)
    assert abs(n - math.ceil(k / every)) <= 1, f'{n} {service} for {k} {trig}'
  return check


def finite(service, allow=(), degraded=lambda m: not m.valid, name=None):
  @prop(name or f'finite {service}')
  def check(r):
    for m in outs(r, service):
      if not degraded(m):
        bad = [p for p in nonfinite(getattr(m, service).to_dict(), service) if not any(p.startswith(f'{service}.{a}') for a in allow)]
        assert not bad, f't={t_of(m):.2f}s nonfinite {bad[:5]}'
  return check


def bounded(paths, lo, hi, nan_ok=False, name=None):
  # every value under each path (lists flattened) in [lo, hi], or NaN if nan_ok
  paths = [paths] if isinstance(paths, str) else paths

  @prop(name or f'bounded {paths[0]}')
  def check(r):
    for path in paths:
      for m in outs(r, path.split('.')[0]):
        xs = np.array(values(m, path), dtype=float)
        bad = xs[~(((xs >= lo) & (xs <= hi)) | (np.isnan(xs) & nan_ok))]
        assert not len(bad), f't={t_of(m):.2f}s {path} not in [{lo}, {hi}]: {bad[:5].tolist()}'
  return check


def invariant(service, fn, name):
  @prop(f'invariant {name}')
  def check(r):
    for m in outs(r, service):
      assert fn(getattr(m, service)), f't={t_of(m):.2f}s {name}'
  return check


def valid_from_inputs(service, trigger=None, inputs=None, by_time=False, name=None):
  # one-directional: output valid => every input seen, last one valid and (sim=False) alive per SubMaster
  @prop(name or f'valid_from_inputs {service}')
  def check(r):
    trig = triggers(r, trigger)
    os_ = outs(r, service)
    for o, k in zip(os_, trigger_of(trig, os_, by_time), strict=True):
      if not o.valid or not 0 <= k < len(trig):
        continue
      m, seen, got = trig[k]
      t = t_of(m)
      for s in inputs or r.space.pubs:
        assert s in seen, f't={t:.2f}s valid before any {s}'
        assert seen[s].valid, f't={t:.2f}s valid with {s} invalid'
        age = (m.logMonoTime - got[s]) / 1e9  # SubMaster's alive bound is 10 periods
        assert r.genome.sim or age < 10. / SERVICE_LIST[s].frequency + 1e-3, f't={t:.2f}s valid with {s} {age:.2f}s old'
  return check


def recovers(services, settle, bad=None, trigger=None, by_time=False, name=None):
  # no latch: every float of these outputs is finite at cycles whose held inputs have all been good for > settle s (from the
  # start, or since the last bad cycle). bad(r, c) default: a float the process reads from a held input is non-finite or huge.
  # settle=inf checks the prefix before the first bad input, so it runs untagged
  @prop(name or f'finite {"/".join(services)} ' + ('(finite inputs)' if math.isinf(settle) else f'({settle:g} s after non-finite inputs)'))
  def check(r):
    bad_t, n = None, 0
    for c in cycles(r, services, trigger, by_time):
      if bad(r, c) if bad else any(r.space.bad(m) for m in c.last.values()):
        bad_t = c.t
      elif bad_t is None or c.t - bad_t > settle:
        n += 1
        for s in services:
          nf = nonfinite(getattr(c.out[s], s).to_dict(), s)
          assert not nf, f't={c.t:.2f}s nonfinite {nf[:5]}' + (f', last bad input at {bad_t:.2f}s' if bad_t is not None else '')
    return n and ('recovered' if bad_t is not None else 'clean')
  check.untagged = math.isinf(settle)
  return check


def hysteresis(valid_path, value_path, hi, lo, scale=1.):
  # valid -> invalid only at |x| >= hi, invalid -> valid only at |x| < lo (paramsd check_valid_with_hysteresis)
  @prop(f'hysteresis {valid_path}')
  def check(r):
    t, v = series(r.outs, valid_path)
    _, x = series(r.outs, value_path)
    for k in range(1, len(v)):
      a = abs(x[k] * scale)
      assert not (v[k - 1] and not v[k] and a < hi), f'{valid_path} dropped at |x|={a}: {around(t, x, k)}'
      assert not (not v[k - 1] and v[k] and not a < lo), f'{valid_path} recovered at |x|={a}: {around(t, x, k)}'
  return check


def no_chatter(path, window, steady=None, name=None):
  # bool/enum: no A -> B -> A within window; with steady(r) (keys aligned with the outputs), only while the keys hold
  @prop(name or f'no_chatter {path}')
  def check(r):
    t, x = series(r.outs, path)
    key = steady(r) if steady else None
    ch = [k for k in range(1, len(x)) if x[k] != x[k - 1]]
    for a, b in zip(ch, ch[1:], strict=False):
      if x[b] == x[a - 1] and t[b] - t[a] < window and (key is None or len(set(key[a - 1:b + 1])) == 1):
        raise AssertionError(f'{x[a - 1]} -> {x[a]} -> {x[b]} in {t[b] - t[a]:.2f}s: {around(t, x, a)}')
  return check


def monotone(path, reset=None, name=None):
  # never decreases (bools: False < True) while the reset path's value is unchanged
  @prop(name or f'monotone {path}')
  def check(r):
    t, x = series(r.outs, path)
    rs = series(r.outs, reset)[1] if reset else [0] * len(x)
    for k in range(1, len(x)):
      assert rs[k] != rs[k - 1] or float(x[k]) >= float(x[k - 1]), f'decreased: {around(t, x, k)}'
  return check


def transitions(path, allowed):
  @prop(f'transitions {path}')
  def check(r):
    t, x = series(r.outs, path)
    for k in range(1, len(x)):
      assert x[k] == x[k - 1] or (str(x[k - 1]), str(x[k])) in allowed, f'{x[k - 1]} -> {x[k]}: {around(t, x, k)}'
  return check


def hf_gain(inp, out, max_gain, delta=0.05, when=None, min_samples=20, name=''):
  # differential: rerun with +-delta alternating each cycle on `inp`; the alternating part of `out`'s difference, per
  # delta, must stay <= max_gain wherever `when` holds in both runs. Commands must not pass input noise through.
  @prop(f'hf_gain {inp} -> {out}{name}')
  def check(r):
    ok1 = when(r) if when else None
    if r.err or (ok1 is not None and ok1.sum() < min_samples):
      return None
    g = copy.deepcopy(r.genome)
    g.perturb = [inp, delta]
    r2 = r.rerun(g)
    t, x = series(r.outs, out)
    _, x2 = series(r2.outs, out)
    n = min(len(x), len(x2))
    d = np.array(x2[:n], dtype=float) - np.array(x[:n], dtype=float)
    ok = np.ones(n, dtype=bool) if when is None else ok1[:n] & when(r2)[:n]
    ok = ok[1:] & ok[:-1] & np.isfinite(d[1:]) & np.isfinite(d[:-1])
    dd = np.diff(d)[ok]
    if len(dd) < min_samples:
      return None
    gain = np.sqrt(np.mean(dd ** 2)) / (2 * delta)
    k = int(np.flatnonzero(ok)[np.argmax(np.abs(dd))]) + 1
    assert gain <= max_gain, f'gain {gain:.3f} > {max_gain} over {len(dd)} cycles; diff at {around(t, d, k)}'
    return True
  return check


def leaves(d, path=''):
  if isinstance(d, dict):
    return [x for k, v in d.items() if k != 'deprecated' for x in leaves(v, f'{path}.{k}' if path else k)]
  if isinstance(d, list):
    return [x for j, v in enumerate(d) for x in leaves(v, f'{path}[{j}]')]
  return [(path, d)]


def time_fields(service, degraded=lambda m: not m.valid):
  # every output, chosen by schema field name: durations (*Delay, *ExecutionTime) are seconds in [0, 10); a timestamp (timestamp*)
  # of a valid output is a time between the first input and now, or passed through from an input's timestamp* field (a 0 only
  # from a same-named field of the latest input); a reference to an input (*MonoTime) is 0 (none yet) or not in the future.
  # Catches unit mixing and sentinel/defaulted times claimed valid
  @prop(f'time_fields {service}')
  def check(r):
    lo = min((m.logMonoTime for m in r.ins), default=T0) - int(0.5e9)
    def stamps(ins):
      return {(q.rsplit('.', 1)[-1].split('[')[0], y) for i in ins if hasattr(w := getattr(i, i.which()), 'to_dict')
              for q, y in leaves(w.to_dict()) if q.rsplit('.', 1)[-1].startswith('timestamp')}
    every = None
    for m in r.outs:
      if m.which() != service or degraded(m) or not hasattr(v := getattr(m, service), 'to_dict'):
        continue
      for p, x in leaves(v.to_dict()):
        k, t = p.rsplit('.', 1)[-1].split('[')[0], t_of(m)
        if k.endswith(('Delay', 'ExecutionTime')) and isinstance(x, float):
          assert 0 <= x < 10, f't={t:.2f}s {service}.{p}={x} is not a duration in seconds'
        elif k.startswith('timestamp') and isinstance(x, int) and not lo <= x <= m.logMonoTime + int(1e9):
          if x:
            every = every if every is not None else {y for _, y in stamps(r.ins)}
            ok = x in every
          else:
            ok = (k, 0) in stamps({i.which(): i for i in r.ins if i.logMonoTime <= m.logMonoTime}.values())
          assert ok, f't={t:.2f}s {service}.{p}={x} outside the input times'
        elif k.endswith('MonoTime') and isinstance(x, int):
          assert x == 0 or lo <= x <= m.logMonoTime, f't={t:.2f}s {service}.{p}={x} outside the input times'
  return check


def theoretical(sp, g, ins=None, bad=None):
  # a tag: a gene takes a class the ledger files as theoretical, or with the run's messages, a read input is bad (default Space.bad)
  return 'theoretical inputs' if (sp.theoretical(g) if ins is None else any(map(bad or sp.bad, ins))) else None

