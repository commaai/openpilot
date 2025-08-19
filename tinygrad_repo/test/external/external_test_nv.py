import unittest, struct, array, ctypes
from tinygrad import Device, dtypes, Tensor
from tinygrad.helpers import to_mv
from tinygrad.runtime.ops_nv import NVDevice, HWQueue
from tinygrad.codegen.opt.search import Opt, OptOps
from tinygrad.engine.realize import get_runner, CompiledRunner, get_program
from test.external.fuzz_linearizer import get_fuzz_rawbufs

from tinygrad.codegen.opt.kernel import Kernel
from tinygrad.uop.ops import LazyOp, Ops, ReduceOps, BufferOps, MemBuffer
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View

@unittest.skipUnless(Device.DEFAULT == "NV", "NV specific tests/fixes")
class TestNV(unittest.TestCase):
  @classmethod
  def setUpClass(self):
    TestNV.d0: NVDevice = Device["NV"]
    TestNV.a = Tensor([0.,1.], device="NV").realize()
    TestNV.b = self.a + 1
    si = self.b.schedule()[-1]
    TestNV.d0_runner = get_runner(TestNV.d0.device, si.ast)
    TestNV.b.uop.buffer.allocate()
    TestNV.addr = struct.pack("QQ", TestNV.b.uop.buffer._buf.va_addr, TestNV.a.uop.buffer._buf.va_addr)

  def test_error_on_huge_dims(self):
    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=Ops.CAST, src=(LazyOp(op=Ops.MUL, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.half, st=ShapeTracker(views=(View(shape=(1, 1, 1024, 683), strides=(0, 0, 0, 1), offset=0, mask=None, contiguous=False),)))), LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=2, dtype=dtypes.half, st=ShapeTracker(views=(View(shape=(1, 1, 1024, 683), strides=(0, 0, 683, 1), offset=0, mask=None, contiguous=True),))))), arg=None),), arg=dtypes.float),), arg=(3,)),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 1, 1024, 1), strides=(0, 0, 1, 0), offset=0, mask=None, contiguous=True),)))) # noqa: E501
    opts = [Opt(op=OptOps.GROUP, axis=0, arg=0), Opt(op=OptOps.PADTO, axis=1, arg=32), Opt(op=OptOps.UNROLL, axis=0, arg=4), Opt(op=OptOps.LOCAL, axis=0, arg=2), Opt(op=OptOps.LOCAL, axis=0, arg=2)] # noqa: E501
    with self.assertRaises(RuntimeError) as cm:
      lin = Kernel(ast)
      lin.apply_opts(opts)
      rawbufs = get_fuzz_rawbufs(lin)
      prg = CompiledRunner(get_program(lin.get_optimized_ast(), lin.opts))
      prg(rawbufs, {}, wait=True)
    self.assertEqual(str(cm.exception), "This is a runtime error message")

  def test_buf4_usage(self):
    TestNV.along = Tensor([105615], device="NV").realize()
    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=Ops.SIN, src=(LazyOp(op=Ops.CAST, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.ulong, st=ShapeTracker(views=(View(shape=(3,), strides=(1,), offset=0, mask=None, contiguous=True),)))),), arg=dtypes.float),), arg=None),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3,), strides=(1,), offset=0, mask=None, contiguous=True),)))) # noqa: E501
    temp_runner = get_runner(TestNV.d0.device, (ast,))
    temp_runner([TestNV.b.uop.buffer, TestNV.along.uop.buffer], var_vals={})
    val = TestNV.b.uop.buffer.as_buffer().cast("f")[0]
    assert abs(val - 0.80647) < 0.001, f"got val {val}"

  def test_kernargs_no_oob_access(self):
    kernargs_start = TestNV.d0._gpu_alloc((2 << 20), map_to_cpu=True).va_addr
    kernargs = kernargs_start + ((2 << 20) - TestNV.d0_runner._prg.kernargs_alloc_size)
    to_mv(kernargs, 0x160).cast('I')[:] = array.array('I', TestNV.d0_runner._prg.constbuffer_0)
    ctypes.memmove(kernargs + TestNV.d0_runner._prg.kernargs_offset, TestNV.addr, len(TestNV.addr))

    q = HWQueue()
    q.exec(TestNV.d0_runner._prg, kernargs, TestNV.d0_runner.global_size, TestNV.d0_runner.local_size)
    q.signal(TestNV.d0.timeline_signal, TestNV.d0.timeline_value).submit(TestNV.d0)
    TestNV.d0._wait_signal(TestNV.d0.timeline_signal, TestNV.d0.timeline_value)
    TestNV.d0.timeline_value += 1
    val = TestNV.b.uop.buffer.as_buffer().cast("f")[0]
    assert val == 1.0, f"got val {val}"

if __name__ == "__main__":
  unittest.main()
