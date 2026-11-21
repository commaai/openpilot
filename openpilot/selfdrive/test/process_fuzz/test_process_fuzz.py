import os
import pkgutil

from openpilot.common.fuzzy import FUZZ_SEED
from openpilot.common.parameterized import parameterized
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.test.process_fuzz import fuzz as F
from openpilot.selfdrive.test.process_fuzz import harness as H
from openpilot.selfdrive.test.process_fuzz import procs
from openpilot.selfdrive.test.process_fuzz import space as S

NAMES = [m.name for m in pkgutil.iter_modules(procs.__path__)]
KINDS = {'enforced', 'theoretical', 'symmetry', 'env', 'harness'}
DEEP = os.environ.get('PROCESS_FUZZ_DEEP') == '1'
SELECTED = [n for n in os.environ.get('PROCESS_FUZZ_PROCS', '').split(',') if n] or NAMES  # fuzz.py test <procs>
RESULTS: dict = {}


def results(name, kind):
  # every selected process's corpus and fresh examples run once, in one pool (fuzz.ci), on the first test that needs them
  if not RESULTS:
    RESULTS.update(F.ci(SELECTED, examples=10 if DEEP else 1, seed=FUZZ_SEED))
  return RESULTS[(name, kind)]


def report(name, spec, results):
  # every failure not in spec.known, with its genome as the repro
  return [f'{c}: {msg}\n  repro: python -m openpilot.selfdrive.test.process_fuzz.fuzz repro {F.save(name, r["genome"], f"fail_{F.slug(c)}")}'
          for r in results for c, msg in r['fails'].items() if not F.known(spec, r, c) or DEEP]


class TestProcessFuzz(OpenpilotTestCase):
  def setUp(self):
    kind, k = self._testMethodName.rsplit('_', 1)
    if kind != 'test_ledger' and NAMES[int(k)] not in SELECTED:
      self.skipTest('not in PROCESS_FUZZ_PROCS')

  @parameterized.expand(NAMES)
  def test_ledger(self, name):
    spec = F.load(name)
    sp = H.space(spec)
    reads = set().union(*map(S.expand, spec.reads))
    paths = {k.split(':')[0] for k in spec.ledger}
    for r in reads:  # every read is fuzzed or has a ledger entry
      ok = r in sp.leaves or r in paths or r.startswith('@') or r == 'P CarParams' or (r.startswith('CP.') and r[3:] in sp.cp_fields) \
        or r[2:] in spec.params
      self.assertTrue(ok, f'{name}: read {r} is neither fuzzed nor in the ledger')
    for k, (kind, why, upstream) in spec.ledger.items():
      self.assertTrue(k in procs.UPSTREAM or {k.split(':')[0]} & (spec.reads | reads | set(sp.leaves)), f'{name}: stale ledger entry {k}')
      self.assertIn(kind, KINDS, k)
      self.assertTrue(why, k)
      up = upstream and upstream.split(':')[0]
      if up in NAMES:  # an enforced reduction is a property on its producer
        self.assertIn(upstream.split(':', 1)[1].strip(), [c.name for c in F.load(up).checks], f'{name}: {k} needs upstream {upstream}')
    for c, (kind, _, genome) in spec.known.items():
      self.assertIn(kind, ('real', 'theoretical'), c)
      self.assertTrue(os.path.exists(os.path.join(F.HERE, 'corpus', name, genome)), f'{name}: known {c} has no genome')

  @parameterized.expand(NAMES)
  def test_corpus(self, name):
    # replay the committed corpus with the probe: read set, branch coverage gate, properties
    spec = F.load(name)
    res = results(name, 'corpus')
    self.assertTrue(res, f'{name}: no corpus, run fuzz.py update {name}')

    reads = set().union(*(r['reads'] for r in res))
    self.assertEqual(reads - spec.reads, set(), f'{name}: new reads, fuzz them or add a ledger entry (fuzz.py update {name})')
    self.assertEqual(spec.reads - reads, set(), f'{name}: reads the corpus no longer reaches')

    got, den = F.coverage(name, res)
    missed = den - got
    for reason, pats in spec.unreached.items():
      for p in pats:
        self.assertFalse([b for b in got if procs.match(p, b)], f'{name}: unreached {p} ({reason}) is reached')
        self.assertTrue([b for b in missed if procs.match(p, b)], f'{name}: unreached {p} ({reason}) matches nothing unreached')
    for reason, pats in procs.STRUCTURAL.items():
      for p in pats:
        self.assertFalse([b for b in got if procs.match(p, b)], f'{name}: structural {p} ({reason}) is reached')
    self.assertEqual(F.gaps(spec, den, got), set(), f'{name}: branch directions neither reached nor explained (fuzz.py update {name})')

    fails = report(name, spec, res)
    self.assertFalse(fails, '\n'.join(fails))

  @parameterized.expand(NAMES)
  def test_random(self, name):
    # fresh whole-space examples, properties only
    fails = report(name, F.load(name), results(name, 'random'))
    self.assertFalse(fails, f'FUZZ_SEED={FUZZ_SEED}\n' + '\n'.join(fails))
