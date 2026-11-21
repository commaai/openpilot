#!/usr/bin/env python3
# Process fuzz CLI, see README.md
import argparse
import glob
import hashlib
import importlib
import json
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

from openpilot.selfdrive.test.process_fuzz import harness as H
from openpilot.selfdrive.test.process_fuzz.space import Genome

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('PROCESS_FUZZ_OUT', '/tmp/process_fuzz')
WORKERS = int(os.environ.get('PROCESS_FUZZ_WORKERS', len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1))


def load(name) -> H.Spec:
  return importlib.import_module(f'openpilot.selfdrive.test.process_fuzz.procs.{name}').SPEC


def execute(item, cov=False, reads=False):
  # one example in a worker: (name, genome) -> picklable result (a KjException isn't)
  name, g = item
  try:
    spec = load(name)
    r = H.run(spec, g, cov, reads)
    fails, applies = {}, set()
    for c in spec.checks:
      name_ = c.name + (f' ({r.tag})' if r.tag and not getattr(c, 'untagged', False) else '')
      try:
        if c(r):
          applies.add(name_)
      except AssertionError as e:
        fails[name_] = str(e)[:2000]
      except Exception:
        fails[name_] = 'check raised: ' + traceback.format_exc(limit=-3)
    return {'genome': g, 'branches': r.branches, 'reads': r.reads, 'applies': applies, 'fails': fails, 'tag': r.tag}
  except Exception:
    raise RuntimeError(traceback.format_exc()) from None


def known(spec, r, c):
  # a check known on any run is known on a tagged one too
  return c in spec.known or (r['tag'] is not None and c.removesuffix(f' ({r["tag"]})') in spec.known)


def cov_execute(item):
  return execute(item, cov=True, reads=True)


def pmap(name, fn, genomes, workers=WORKERS):
  # workers fork from here with the spec, space and branch table loaded
  spec = load(name)
  H.space(spec), H.branches(spec)
  with ProcessPoolExecutor(min(workers, max(len(genomes), 1))) as ex:
    yield from ex.map(fn, [(name, g) for g in genomes])


def ci(names, examples=1., workers=WORKERS, seed=''):
  # CI for many processes in one pool, longest first: {(name, 'corpus' | 'random'): [result]}. The corpus runs with coverage
  # and read tracing, the fresh examples (spec.examples * examples, from the seed) with properties only
  items = []
  for name in names:
    spec = load(name)
    sp, rng = H.space(spec), random.Random(f'{seed}:{name}')
    H.branches(spec)
    items += [(name, 'corpus', g) for g in corpus(name)]
    items += [(name, 'random', sp.draw(rng)) for _ in range(int(spec.examples * examples))]
  items.sort(key=lambda it: -it[2].dur * (3 if it[1] == 'corpus' else 1))  # a rough cost: the tail of the queue is short examples
  out = {(n, k): [] for n in names for k in ('corpus', 'random')}
  with ProcessPoolExecutor(min(workers, max(len(items), 1))) as ex:
    futs = [ex.submit(execute, (n, g), k == 'corpus', k == 'corpus') for n, k, g in items]
    for (name, kind, _), f in zip(items, futs, strict=True):
      out[(name, kind)].append(f.result())
  return out


def ci_key(r):
  # what the CI gates on: branch directions, reads, and which checks apply
  return set(r['branches']) | {('R', x) for x in r['reads']} | {('applies', n) for n in r['applies']}


def slug(s):
  return ''.join(c if c.isalnum() else '_' for c in s)[:60]


def save(name, g, prefix='', d=OUT):
  s = json.dumps({'proc': name, **g.to_dict()}, sort_keys=True)
  os.makedirs(d, exist_ok=True)
  path = os.path.join(d, f'{prefix}{"_" if prefix else ""}{hashlib.sha1(s.encode()).hexdigest()[:10]}.json')
  with open(path, 'w') as f:
    f.write(s)
  return path


def load_genome(path) -> tuple[str, Genome]:
  with open(path) as f:
    d = json.load(f)
  return d.pop('proc'), Genome.from_dict(d)


def corpus_files(name):
  return sorted(glob.glob(os.path.join(HERE, 'corpus', name, '*.json')))


def corpus(name) -> list[Genome]:
  return [load_genome(p)[1] for p in corpus_files(name)]


def coverage(name, results):
  den = H.branches(load(name))[1]
  return set().union(*(r['branches'] for r in results)) & den, den


def gaps(spec, den, got):
  # branch directions neither reached nor matched by an UNREACHED entry
  from openpilot.selfdrive.test.process_fuzz.procs import STRUCTURAL, match
  pats = [p for ps in [*spec.unreached.values(), *STRUCTURAL.values()] for p in ps]
  return {b for b in den - got if not any(match(p, b) for p in pats)}


def update(name, budget, seed):
  # add-only: fresh draws with coverage, then a greedy cover (shorter first) of what the corpus doesn't reach yet
  spec, rng = load(name), random.Random(seed)
  sp = H.space(spec)
  old = list(pmap(name, cov_execute, corpus(name)))
  have, new, t0 = set().union(*map(ci_key, old)), [], time.monotonic()
  while time.monotonic() - t0 < budget:
    for r in pmap(name, cov_execute, [sp.draw(rng) for _ in range(WORKERS * 4)]):
      if all(known(spec, r, c) for c in r['fails']):
        new.append((r, ci_key(r) - have))
  added = []
  left = set().union(*(k for _, k in new))
  while left:
    r, k = max(new, key=lambda rk: (len(rk[1] & left), -rk[0]['genome'].dur))
    added.append(r)
    left -= k
    print('added', save(name, r['genome'], d=os.path.join(HERE, 'corpus', name)))
  reads = sorted(set().union(*(r['reads'] for r in old + added)))
  with open(os.path.join(HERE, 'procs', f'{name}_reads.json'), 'w') as f:
    json.dump(reads, f, indent=1)
    f.write('\n')
  got, den = coverage(name, old + added)
  print(f'{len(got)}/{len(den)} branch directions, {len(added)} genomes added; unexplained:')
  for b in sorted(gaps(spec, den, got)):
    print('  ', b)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('cmd', choices=['test', 'run', 'update', 'repro'])
  ap.add_argument('target', nargs='*')
  ap.add_argument('--seed', type=int, default=int(time.monotonic() * 1e6))
  ap.add_argument('--budget', type=float, default=60.)
  a = ap.parse_args()

  if a.cmd == 'test':  # the unittest, for these processes only (default all)
    import unittest
    os.environ['PROCESS_FUZZ_PROCS'] = ','.join(a.target)
    r = unittest.main(module='openpilot.selfdrive.test.process_fuzz.test_process_fuzz', argv=[sys.argv[0], '-v'], exit=False)
    sys.exit(not r.result.wasSuccessful())
  if a.cmd == 'repro':
    name, g = load_genome(a.target[0])
    res = execute((name, g), cov=True, reads=True)
    print(json.dumps(res['fails'], indent=2) if res['fails'] else 'all checks pass')
  elif a.cmd == 'update':
    update(a.target[0], a.budget, a.seed)
  else:
    name, spec = a.target[0], load(a.target[0])
    rng, fails, n, t0 = random.Random(a.seed), {}, 0, time.monotonic()
    while time.monotonic() - t0 < a.budget:
      for r in pmap(name, execute, [H.space(spec).draw(rng) for _ in range(WORKERS * 2)]):
        n += 1
        for c, msg in r['fails'].items():
          if c not in fails:
            fails[c] = msg
            print(f'FAIL {c}: {msg.splitlines()[0][:200]}\n  genome: {save(name, r["genome"], f"fail_{slug(c)}")}', flush=True)
    print(f'seed {a.seed}: {n} runs, failing checks: {sorted(fails)}')


if __name__ == '__main__':
  main()
