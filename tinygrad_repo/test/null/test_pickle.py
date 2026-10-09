import unittest, pickle, types
from tinygrad import Tensor, dtypes
from tinygrad.helpers import ContextVar, Context
from tinygrad.uop.ops import PatternMatcher, UPat, UOp, deconstruct_function

class TestPickle(unittest.TestCase):
  def test_pickle_code_object(self):
    y = lambda x: x*2  # noqa: E731
    code_str = pickle.dumps(y.__code__)
    fxn = types.FunctionType(pickle.loads(code_str), globals())
    self.assertEqual(fxn(2), 4)

  def test_deconstruct_function_nested_comprehension(self):
    # pre PEP 709, each comprehension is its own code object, so dtypes here is referenced two code objects deep
    def fxn(): return [[dtypes.int for _ in range(2)] for _ in range(2)]
    self.assertEqual(types.FunctionType(*deconstruct_function(fxn))(), fxn())

  def test_pickle_pattern_matcher(self):
    pm = PatternMatcher([(UPat.cvar('x'), lambda x: x*2)])
    sink = UOp.const(2)
    tt = pm.rewrite(sink)
    pm_str = pickle.dumps(pm)
    pm2 = pickle.loads(pm_str)
    self.assertEqual(pm2.rewrite(sink).key, tt.key)

  def test_pickle_main_pattern_matcher(self):
    from tinygrad.uop.symbolic import sym
    ssym = pickle.dumps(sym)
    dsym = pickle.loads(ssym)
    self.assertEqual(dsym.patterns[0][0].location, sym.patterns[0][0].location)

  def test_pickle_context_var(self):
    v = ContextVar("test_var", 0)
    with Context(test_var=1):
      vs = pickle.dumps(v)
    v2 = pickle.loads(vs)
    self.assertEqual(v2.value, 1)

  def test_pickle_schedule(self):
    a = Tensor([1,2])
    out = a + 2
    sched = out.schedule_linear()
    pk = pickle.dumps(sched)
    sched_pk = pickle.loads(pk)
    self.assertEqual(sched_pk.src[-1].src[0], sched.src[-1].src[0])

  def test_pickle_renderer(self):
    from tinygrad.device import Device
    pk = pickle.dumps(Device.default.renderer)
    pickle.loads(pk)

if __name__ == '__main__':
  unittest.main()
