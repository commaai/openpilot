import multiprocessing, unittest
from concurrent.futures import ProcessPoolExecutor
from tinygrad import Context, Tensor, Variable, Device
from tinygrad.device import host_of
from tinygrad.helpers import HCQ_RUNTIME_DEV
from tinygrad.uop.uir import render_uir, parse_uir
from tinygrad.codegen import create_linear
from tinygrad.codegen.compile import compile_linear

@Context(ALLOW_DEVICE_USAGE=0)
def compile_uir(src:str, dev:str, host:str) -> str:
  with Context(DEV=dev, HCQ_RUNTIME_DEV=host): return render_uir(Tensor.compile(parse_uir(src)))

class TestExportRun(unittest.TestCase):
  def test_export_compile_run(self):
    # export -> external compile -> run, the function and the LINEAR cross the wire as UIR
    def model(xs, n):
      v = Variable("v", 1, 4).bind(n)
      return (Tensor(xs).realize() + 1).contiguous()[:v].contiguous() * v
    a, b = model([1., 2., 3., 4.], 3), model([10., 20., 30., 40.], 2)
    fn, first = a.export()
    _, second = b.export()
    # the compiler only gets the targets, it opens no device
    dev = ";".join(repr(Device.target_for(d)) for d in dict.fromkeys([Device.DEFAULT, host_of(Device.DEFAULT)]))
    host = repr(Device.target_for(HCQ_RUNTIME_DEV.device, HCQ_RUNTIME_DEV))
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as compiler:
      linear = parse_uir(compiler.submit(compile_uir, render_uir(fn), dev, host).result())
    Tensor.run(linear, first)
    Tensor.run(linear, second)
    self.assertEqual(a.shrink(((0, 3),)).tolist(), [6., 9., 12.])
    self.assertEqual(b.shrink(((0, 2),)).tolist(), [22., 42.])

  def test_link_scalar_extern(self):
    scale = Variable("scale", 1, 4)
    out = Tensor([1, 2, 3]).realize() * scale.bind(2)
    fn, externs = out.export()
    compiled = compile_linear(create_linear(fn))
    for n in (2, 3):
      inputs = [scale.bind(n) if x.is_bound_var else x for x in externs]
      Tensor.run(compiled, tuple(inputs))
      self.assertEqual(out.tolist(), [n, 2*n, 3*n])

  def test_capture_export(self):
    # a capturer gets the export, a function with no runtime state and its externs. nothing is compiled or run until it does
    from tinygrad.engine.realize import capturing
    from tinygrad.helpers import GlobalCounters
    from tinygrad.uop.ops import Ops
    class Capture:
      def __init__(self): self.exports = []
      def add_export(self, fn, externs): self.exports.append((fn, externs))
    a = Tensor([1, 2, 3]).realize()
    capturing.append(cap:=Capture())
    try:
      kernel_count = GlobalCounters.kernel_count
      out = (a + 1).realize()
      with Context(CAPTURING=0): ran = (a + 2).realize()
    finally: capturing.clear()
    (fn, externs), = cap.exports
    self.assertFalse(any(u.op is Ops.BUFFER for u in fn.toposort(enter_calls=True)))
    self.assertEqual(GlobalCounters.kernel_count, kernel_count+1)
    self.assertFalse(out.uop.base.buffer.is_allocated())
    self.assertEqual(ran.tolist(), [3, 4, 5])
    Tensor.run(Tensor.compile(fn), externs)
    self.assertEqual(out.tolist(), [2, 3, 4])

  def test_run_checks_externs(self):
    a, b = Tensor([1., 2.]).realize(), Tensor([1., 2., 3.]).realize()
    fn, externs = Tensor.export((a + 1).contiguous())
    _, other = Tensor.export((b + 1).contiguous())
    linear = Tensor.compile(fn)
    with self.assertRaisesRegex(RuntimeError, "must match its EXTERN"): Tensor.run(linear, other)
    with self.assertRaisesRegex(RuntimeError, "got nothing"): Tensor.run(linear, externs[:1])

if __name__ == "__main__":
  unittest.main()
