import unittest, pickle, types
import numpy as np
from tinygrad import Tensor, TinyJit, Variable, dtypes
from tinygrad.helpers import GlobalCounters, ContextVar, Context
from tinygrad.uop.ops import PatternMatcher, UPat, UOp, Ops

class TestPickle(unittest.TestCase):
  def test_pickle_code_object(self):
    y = lambda x: x*2  # noqa: E731
    code_str = pickle.dumps(y.__code__)
    fxn = types.FunctionType(pickle.loads(code_str), globals())
    self.assertEqual(fxn(2), 4)

  def test_pickle_pattern_matcher(self):
    pm = PatternMatcher([(UPat.cvar('x'), lambda x: x*2)])
    sink = UOp.const(dtypes.int, 2)
    tt = pm.rewrite(sink)
    pm_str = pickle.dumps(pm)
    pm2 = pickle.loads(pm_str)
    self.assertEqual(pm2.rewrite(sink).key, tt.key)

  def test_pickle_main_pattern_matcher(self):
    from tinygrad.codegen.devectorizer import sym
    ssym = pickle.dumps(sym)
    dsym = pickle.loads(ssym)
    self.assertEqual(dsym.patterns[0][0].location, sym.patterns[0][0].location)

  def test_pickle_realized_tensor(self):
    print("** init")
    t = Tensor.rand(10, 10).realize()
    st = pickle.dumps(t)
    t_values = t.numpy()
    del t # free buffers
    print("** post pickle")
    init = GlobalCounters.kernel_count
    t2:Tensor = pickle.loads(st)
    np.testing.assert_equal(t_values, t2.numpy())
    # expect at most one COPY kernel
    self.assertLessEqual(GlobalCounters.kernel_count-init, 1)

  def test_pickle_realized_tensor_alt(self):
    print("** init")
    t = Tensor.rand(10, 10).to("CPU").realize()
    st = pickle.dumps(t)
    t_values = t.numpy()
    del t # free buffers
    print("** post pickle")
    init = GlobalCounters.kernel_count
    t2:Tensor = pickle.loads(st)
    np.testing.assert_equal(t_values, t2.numpy())
    self.assertEqual(GlobalCounters.kernel_count-init, 0)

  def test_pickle_realized_tensor_alt2(self):
    print("** init")
    t = Tensor.rand(10, 10).to("CPU").realize()
    tensor_uop = t.uop
    assert tensor_uop.is_realized, f"expected {tensor_uop} to be realized"
    t_values = t.numpy()
    # pickle
    st = pickle.dumps(t)
    # free buffers
    del t
    del tensor_uop
    print("** post pickle")
    t2:Tensor = pickle.loads(st)
    assert t2.uop.is_realized, f"expected {t2.uop} to be realized"
    np.testing.assert_equal(t_values, t2.numpy())

  # NOTE: currently Buffer exists on the uop, not tensor
  def test_pickle_buffer_uop(self):
    t = Tensor.arange(4).realize()
    a = t.uop
    assert a.op is Ops.BUFFER
    self.assertIsNotNone(buffer:=a.realized)
    s = pickle.dumps(a)
    # free buffers
    del a
    del buffer
    a2:UOp = pickle.loads(s)
    self.assertListEqual(a2.realized.as_buffer().cast("I").tolist(), [0, 1, 2, 3])

  def test_pickle_unrealized_tensor(self):
    t = Tensor.ones(10, 10)
    st = pickle.dumps(t)
    t2:Tensor = pickle.loads(st)
    np.testing.assert_equal(t.numpy(), t2.numpy())

  def test_pickle_variable(self):
    v = Variable("i", 1, 20).bind(10)
    t1 = Tensor.ones(10, v).contiguous()
    t2 = Tensor.ones(10, v).contiguous()
    ret = (t1+t2).sum(1)
    st = pickle.dumps(ret)
    del ret
    vt2 = pickle.loads(st)
    np.testing.assert_equal(vt2.numpy(), 20)

  def test_pickle_buffer_view(self):
    t = Tensor.arange(10, device="CPU").contiguous().realize()
    vt = t[3:5].contiguous().realize()
    assert hasattr(vt.uop.buffer, 'base')
    ref_value = vt.tolist()
    st = pickle.dumps(vt)
    del t, vt
    vt2 = pickle.loads(st)
    assert hasattr(vt2.uop.buffer, 'base')
    assert ref_value == vt2.tolist()

  def test_pickle_numpy(self):
    t = Tensor(np.array([1,2,3,4.]), dtype=dtypes.float32)
    st = pickle.dumps(t)
    t2:Tensor = pickle.loads(st)
    np.testing.assert_equal(t.numpy(), t2.numpy())

  def test_pickle_jit(self):
    @TinyJit
    def add(a, b): return a.sum()+b+1
    for _ in range(3): add(Tensor.rand(10, 10), Tensor.rand(10, 10))
    st = pickle.dumps(add)
    del add

    add_fxn = pickle.loads(st)
    x = Tensor.ones(10, 10).contiguous().realize()
    y = Tensor.ones(10, 10).contiguous().realize()
    print("post jit")
    out = add_fxn(x, y)
    np.testing.assert_equal(out.numpy(), 102)

  def test_pickle_context_var(self):
    v = ContextVar("test_var", 0)
    with Context(test_var=1):
      vs = pickle.dumps(v)
    v2 = pickle.loads(vs)
    self.assertEqual(v2.value, 1)

  def test_pickle_schedule(self):
    a = Tensor([1,2])
    out = a + 2
    sched = out.schedule()
    pk = pickle.dumps(sched)
    sched_pk = pickle.loads(pk)
    self.assertEqual(sched_pk[-1].ast, sched[-1].ast)

  def test_pickle_renderer(self):
    from tinygrad.device import Device
    pk = pickle.dumps(Device.default.renderer)
    pickle.loads(pk)

class TestPickleJIT(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    N = 10
    @TinyJit
    def add(a, b): return a.sum()+b+1
    for _ in range(3): add(Tensor.rand(N, N), Tensor.rand(N, N))
    cls.st = pickle.dumps(add)
    del add

  def test_inspect(self):
    import io
    class FakeClass:
      def __init__(self, *args, **kwargs):
        print(self.module, self.name)
    class InspectUnpickler(pickle.Unpickler):
      def find_class(self, module, name): return type("SpecializedFakeClass", (FakeClass,), {"name": name, "module": module})
    InspectUnpickler(io.BytesIO(self.st)).load()

  @unittest.skip("we are still saving intermediate buffers")
  def test_size(self):
    # confirm no intermediate buffers are saved
    self.assertLess(len(self.st), 1_000_000)

if __name__ == '__main__':
  unittest.main()
